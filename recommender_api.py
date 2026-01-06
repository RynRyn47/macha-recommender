import os
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
import pickle
from fastapi.responses import RedirectResponse
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
import shutil

# --- CẤU HÌNH ---
load_dotenv()
# Render cung cấp biến môi trường này, nếu chạy local thì dùng localhost
MONGO_URI = os.getenv("MONGO_URI") 
DB_NAME = os.getenv("DB_NAME", "MACha")

app = FastAPI(title="MACha Recommender System")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")

# Biến toàn cục lưu Model
meta_df = None
cosine_sim = None
db = None

# --- HÀM TRAINING (ĐƯA VÀO TRONG API) ---
def train_model_task():
    global meta_df, cosine_sim
    print(f"🔄 [Scheduler] Bắt đầu training lại model lúc {datetime.now()}...")
    
    try:
        # 1. Kết nối DB lấy dữ liệu mới nhất
        client = MongoClient(MONGO_URI)
        database = client[DB_NAME]
        cursor = database['campaigns'].find({"status": "active"}) # Chỉ train bài active hoặc completed tùy bạn
        
        campaigns = list(cursor)
        if not campaigns:
            print("⚠️ Không có dữ liệu để train.")
            return

        # 2. Xử lý dữ liệu
        df = pd.DataFrame(campaigns)
        # Tạo cột 'content' để AI học
        df['content'] = df['title'] + " " + df['description'] + " " + df['category']
        df['content'] = df['content'].fillna('')

        # 3. Tính toán TF-IDF
        tfidf = TfidfVectorizer(stop_words='english') # Với tiếng Việt nên dùng thư viện xử lý từ stopword riêng
        tfidf_matrix = tfidf.fit_transform(df['content'])

        # 4. Tính Cosine Similarity
        new_cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # 5. Cập nhật vào RAM (Không cần lưu file pkl nữa vì Render file system là tạm thời)
        meta_df = df[['_id', 'title', 'category', 'current_amount']]
        # Chuyển _id sang string để dễ tìm kiếm sau này
        meta_df['_id'] = meta_df['_id'].astype(str)
        cosine_sim = new_cosine_sim
        
        print(f"✅ [Scheduler] Training hoàn tất! Đã cập nhật {len(df)} chiến dịch.")
        
    except Exception as e:
        print(f"❌ [Scheduler] Lỗi training: {e}")

# --- API EVENTS ---
@app.on_event("startup")
def startup_event():
    global db
    # 1. Kết nối DB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    # 2. Chạy training ngay lập tức khi khởi động server
    train_model_task()
    
    # 3. Lên lịch chạy định kỳ (Ví dụ: 60 phút/lần)
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_model_task, 'interval', minutes=60)
    scheduler.start()
    print("⏰ Đã khởi động Scheduler training định kỳ.")

# --- CÁC HÀM GỢI Ý (GIỮ NGUYÊN LOGIC CŨ) ---
def recommend_by_history(last_viewed_id, limit=10):
    global meta_df, cosine_sim
    if meta_df is None or cosine_sim is None: return []
    try:
        # Tìm index
        indices = meta_df[meta_df['_id'] == str(last_viewed_id)].index
        if len(indices) == 0: return []
        idx = indices[0]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:limit+1]]
        return meta_df.iloc[top_indices]['_id'].tolist()
    except: return []

def get_trending_campaigns(limit=10):
    try:
        cursor = db['campaigns'].find({"status": "active"}).sort("current_amount", -1).limit(limit)
        return [str(doc['_id']) for doc in cursor]
    except: return []

# --- API ENDPOINT ---
@app.post("/api/v1/recommend")
def get_recommendations(payload: dict = Body(...)):
    user_id = payload.get("user_id")
    limit = payload.get("limit", 10)
    recommended_ids = []
    strategy = "trending"

    if user_id:
        try:
            user = db['users'].find_one({"_id": ObjectId(user_id)})
            if user:
                # 1. Content-based History
                history = user.get("recently_viewed_campaigns", [])
                if history:
                    recommended_ids = recommend_by_history(str(history[-1]), limit)
                    if recommended_ids: strategy = "content_based_history"
                
                # 2. Onboarding Interest (Logic đơn giản hóa)
                if not recommended_ids:
                    interests = user.get("interests", [])
                    if meta_df is not None and interests:
                         filtered = meta_df[meta_df['category'].isin(interests)]
                         if not filtered.empty:
                             recommended_ids = filtered.head(limit)['_id'].tolist()
                             strategy = "onboarding_interest"
        except: pass

    if not recommended_ids:
        recommended_ids = get_trending_campaigns(limit)
        strategy = "cold_start_trending"

    return {
        "status": "success", 
        "strategy": strategy, 
        "count": len(recommended_ids),
        "campaign_ids": recommended_ids
    }