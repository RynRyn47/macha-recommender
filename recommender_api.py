import os
import re
import pickle
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Body
from fastapi.responses import RedirectResponse
from pymongo import MongoClient
from bson import ObjectId

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from apscheduler.schedulers.background import BackgroundScheduler

# --- 1. CẤU HÌNH HỆ THỐNG ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "MACha")

app = FastAPI(title="MACha Recommender System")

# Biến toàn cục (Lưu model trong RAM)
meta_df = None
cosine_sim = None
db = None

# Danh sách từ vô nghĩa (Stopwords) - Copy từ script training của bạn
VIETNAMESE_STOPWORDS = [
    "là", "của", "và", "những", "các", "trong", "khi", "cho", "để", "với", 
    "người", "tại", "cũng", "từ", "làm", "được", "ra", "vào", "về", "này",
    "kêu", "gọi", "ủng", "hộ", "quyên", "góp", "chung", "tay", "em", "cháu", "gia", "đình"
]

# --- 2. CÁC HÀM XỬ LÝ TEXT & MODEL ---

def clean_text(text):
    """Làm sạch văn bản: Chuyển thường, bỏ ký tự đặc biệt"""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def train_model_task():
    """Hàm này sẽ chạy ngầm định kỳ để retrain model"""
    global meta_df, cosine_sim
    print(f"🔄 [Scheduler] Bắt đầu Retrain Model lúc {datetime.now()}...")
    
    try:
        # Kết nối DB mới nhất (để tránh timeout connection cũ)
        client = MongoClient(MONGO_URI)
        database = client[DB_NAME]
        
        # 1. Lấy dữ liệu (Active + Completed)
        cursor = database['campaigns'].find(
            {"status": {"$in": ["active", "completed"]}},
            {"title": 1, "description": 1, "category": 1, "status": 1, "current_amount": 1}
        )
        
        campaigns = list(cursor)
        if not campaigns:
            print("⚠️ [Scheduler] Không có dữ liệu để train.")
            return

        df = pd.DataFrame(campaigns)

        # 2. Feature Engineering (Áp dụng công thức trọng số tối ưu)
        # Category x8, Title x4, Description x1
        df['combined_text'] = (
            (df['category'].fillna('') + " ") * 8 + 
            (df['title'].fillna('') + " ") * 4 + 
            df['description'].fillna('')
        )
        
        df['clean_text'] = df['combined_text'].apply(clean_text)

        # 3. TF-IDF (N-gram 1,2)
        tfidf = TfidfVectorizer(
            stop_words=VIETNAMESE_STOPWORDS,
            min_df=2,
            ngram_range=(1, 2),
            max_features=5000 
        )
        
        tfidf_matrix = tfidf.fit_transform(df['clean_text'])

        # 4. Tính Cosine Similarity (Dùng linear_kernel cho nhanh)
        new_cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # 5. Cập nhật vào RAM
        # Chuyển _id sang string để API dễ tra cứu
        df['_id'] = df['_id'].astype(str)
        
        # Chỉ giữ lại các cột cần thiết cho việc hiển thị/lọc
        meta_df = df[['_id', 'title', 'category', 'status', 'current_amount']]
        cosine_sim = new_cosine_sim
        
        print(f"✅ [Scheduler] Training XONG! Đã học {len(df)} chiến dịch. Matrix shape: {tfidf_matrix.shape}")
        
    except Exception as e:
        print(f"❌ [Scheduler] Lỗi training: {e}")

# --- 3. CÁC HÀM GỢI Ý (LOGIC HYBRID) ---

def get_trending_campaigns(limit=10):
    """Chiến lược Cold Start: Lấy bài Active & Tiền nhiều nhất"""
    if db is None: return []
    try:
        cursor = db['campaigns'].find(
            {"status": "active"}
        ).sort("current_amount", -1).limit(limit)
        return [str(doc['_id']) for doc in cursor]
    except: return []

def recommend_by_onboarding(categories, limit=10):
    """Chiến lược Interest: Lọc theo Category"""
    if meta_df is None: return []
    
    # Lọc category & Chỉ lấy bài Active
    filtered = meta_df[
        (meta_df['category'].isin(categories)) & 
        (meta_df['status'] == 'active')
    ]
    
    if filtered.empty: return []
    # Trả về các ID đầu tiên (hoặc có thể sample ngẫu nhiên)
    return filtered.head(limit)['_id'].tolist()

def recommend_by_history(last_viewed_id, limit=10):
    """Chiến lược Content-Based: Tìm bài tương tự"""
    if meta_df is None or cosine_sim is None: return []
    try:
        # Tìm index
        indices = meta_df[meta_df['_id'] == str(last_viewed_id)].index
        if len(indices) == 0: return []
        idx = indices[0]
        
        # Lấy điểm tương đồng
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Lấy top k (bỏ qua chính nó)
        top_indices = [i[0] for i in sim_scores[1:limit+1]]
        
        # Lọc: Chỉ gợi ý bài ACTIVE từ danh sách tương đồng
        recs = meta_df.iloc[top_indices]
        active_recs = recs[recs['status'] == 'active']
        
        return active_recs['_id'].tolist()
    except: return []

# --- 4. KHỞI ĐỘNG APP & SCHEDULER ---

@app.on_event("startup")
def startup_event():
    global db
    # 1. Kết nối DB
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        print(f"🔌 Connected DB: {DB_NAME}")
    except Exception as e:
        print(f"❌ DB Error: {e}")
    
    # 2. Train model ngay khi khởi động
    train_model_task()
    
    # 3. Lên lịch train lại mỗi 60 phút
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_model_task, 'interval', minutes=60)
    scheduler.start()
    print("⏰ Scheduler đã chạy (60 phút/lần).")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")

# --- 5. API ENDPOINT CHÍNH ---

@app.post("/api/v1/recommend")
def get_recommendations(payload: dict = Body(...)):
    """
    API Hybrid Recommendation
    """
    user_id = payload.get("user_id")
    limit = payload.get("limit", 10)
    
    recommended_ids = []
    strategy_used = "trending"

    # A. CÓ USER ID -> TRA CỨU DB
    if user_id and db is not None:
        try:
            user = db['users'].find_one({"_id": ObjectId(user_id)})
            if user:
                # 1. Lấy dữ liệu user
                history = user.get("recently_viewed_campaigns", [])
                interests = user.get("interests", [])
                
                history_recs = []
                interest_recs = []

                # 2. Logic History (Lấy 50%)
                if history:
                    last_seen_id = str(history[-1]) # Lấy bài mới xem nhất
                    h_limit = max(1, limit // 2)
                    history_recs = recommend_by_history(last_seen_id, limit=h_limit)

                # 3. Logic Interest (Lấy phần còn thiếu)
                if interests:
                    remaining = limit - len(history_recs)
                    if remaining > 0:
                        interest_recs = recommend_by_onboarding(interests, limit=remaining)

                # 4. Trộn & Khử trùng lặp (Giữ thứ tự: History -> Interest)
                combined = list(dict.fromkeys(history_recs + interest_recs))
                
                if combined:
                    recommended_ids = combined
                    if history_recs and interest_recs: strategy_used = "hybrid_mixed"
                    elif history_recs: strategy_used = "content_based_history"
                    elif interest_recs: strategy_used = "onboarding_interest"
                    
        except Exception as e:
            print(f"⚠️ Error user logic: {e}")

    # B. FALLBACK (TRENDING)
    if not recommended_ids:
        recommended_ids = get_trending_campaigns(limit)
        strategy_used = "cold_start_trending"

    # Giới hạn đúng số lượng yêu cầu
    recommended_ids = recommended_ids[:limit]

    return {
        "status": "success",
        "strategy": strategy_used,
        "count": len(recommended_ids),
        "campaign_ids": recommended_ids
    }