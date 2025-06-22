import mysql.connector
from mysql.connector import Error
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
import numpy as np

pd.options.mode.chained_assignment = None
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

app = Flask(__name__)

def create_connection():
    try:
        engine = create_engine(
            'mysql+mysqlconnector://vivutour_tour:voducviet2003@vi-vu-tour.pro.vn:3306/vivutour_datn_tour_new'
        )
        print("✅ Kết nối tới MySQL database thành công!")
        return engine
    except Error as e:
        print(f"❌ Lỗi khi kết nối MySQL: {e}")
        return None

def close_connection(engine):
    if engine:
        engine.dispose()
        print("🔒 Đã đóng kết nối.")

def combine_features(row):
    features = ['title', 'time', 'destination', 'domain']
    return ' '.join(str(row[f]) for f in features if f in row and pd.notnull(row[f]))

@app.route('/api/user-recommendations', methods=['GET'])
def get_user_recommendations():
    user_id = request.args.get('user_id')
    if not user_id or not user_id.isdigit():
        return jsonify({"error": "Invalid or missing 'user_id' parameter"}), 400
    user_id = int(user_id)

    engine = create_connection()
    if engine is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # 1. Lấy tất cả tour user đã từng đặt
        user_bookings = pd.read_sql(
            "SELECT DISTINCT tourId FROM tbl_booking WHERE userId = %(user_id)s;",
            engine, params={"user_id": user_id}
        )

        if user_bookings.empty:
            return jsonify({"recommended_tours": None, "message": "User has no booking history"}), 200

        # 2. Lấy tất cả tour đang hoạt động
        active_tours = pd.read_sql("""
            SELECT t.*, AVG(s.priceAdult) AS priceAdult
            FROM tbl_tour t
            LEFT JOIN tbl_start_end_date s ON t.tourId = s.tourId
            WHERE t.acStatus = 'y'
            GROUP BY t.tourId
        """, engine)

        # 3. Tạo đặc trưng kết hợp
        active_tours['combineFeatures'] = active_tours.apply(combine_features, axis=1)

        # 4. Tính TF-IDF và cosine similarity
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=2)
        tfidf_matrix = tfidf.fit_transform(active_tours['combineFeatures'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 5. Lấy index các tour user đã đặt
        booked_tour_ids = user_bookings['tourId'].tolist()
        booked_indices = active_tours[active_tours['tourId'].isin(booked_tour_ids)].index.to_numpy()

        # 6. Lấy index candidate tours (chưa đặt)
        candidate_indices = active_tours[~active_tours['tourId'].isin(booked_tour_ids)].index.to_numpy()

        if len(booked_indices) == 0 or len(candidate_indices) == 0:
            return jsonify({"message": "Not enough data to compute recommendations."}), 200

        # 7. Tính điểm cosine similarity trung bình giữa các tour đã đặt và candidate tours
        sim_scores = cosine_sim[np.ix_(booked_indices, candidate_indices)].mean(axis=0)

        candidate_tours = active_tours.iloc[candidate_indices].copy()
        candidate_tours['similarity'] = sim_scores

        # 8. Lấy top 4 tour similarity cao nhất làm đề xuất
        candidate_tours = candidate_tours.sort_values(by='similarity', ascending=False)
        recommended_tours = candidate_tours.head(4)

        # 9. Trả về danh sách tourId
        result = recommended_tours[['tourId']].to_dict(orient='records')

        return jsonify({"recommended_tours": result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    finally:
        close_connection(engine)



@app.route('/api/tour-recommendations', methods=['GET'])
def get_tour_recommendations():
    tour_id = request.args.get('tour_id')
    if not tour_id or not tour_id.isdigit():
        return jsonify({"error": "Invalid or missing 'tour_id' parameter"}), 400
    tour_id = int(tour_id)

    engine = create_connection()
    if engine is None:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        # 1. Chỉ lấy các tour đang hoạt động (acStatus = 'y')
        all_tours = pd.read_sql("""
            SELECT t.*, AVG(s.priceAdult) AS priceAdult
            FROM tbl_tour t
            LEFT JOIN tbl_start_end_date s ON t.tourId = s.tourId
            WHERE t.acStatus = 'y'
            GROUP BY t.tourId
        """, engine)

        # 2. Kiểm tra tour có tồn tại không
        if tour_id not in all_tours['tourId'].values:
            return jsonify({"error": "Tour not found."}), 404

        # 3. Tạo đặc trưng kết hợp
        all_tours['combineFeatures'] = all_tours.apply(combine_features, axis=1)

        # 4. TF-IDF và cosine similarity
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=2)
        tfidf_matrix = tfidf.fit_transform(all_tours['combineFeatures'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 5. Lấy chỉ số của tour đầu vào
        tour_idx = all_tours[all_tours['tourId'] == tour_id].index[0]

        # 6. Lấy top tour tương tự
        sim_scores = list(enumerate(cosine_sim[tour_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_similar_indices = [i for i, score in sim_scores if i != tour_idx][:10]

        similar_tours = all_tours.iloc[top_similar_indices].copy()
        similarities = [score for i, score in sim_scores if i in top_similar_indices]
        similar_tours['similarity'] = similarities

        # 7. Sắp xếp theo độ tương đồng và lấy top 4
        similar_tours = similar_tours.sort_values(by='similarity', ascending=False)
        result = similar_tours[['tourId']].head(4).to_dict(orient='records')

        return jsonify({"recommended_tours": result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    finally:
        close_connection(engine)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
