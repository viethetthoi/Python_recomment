import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
import time

# ⚙️ Cấu hình Pandas
pd.options.mode.chained_assignment = None
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

app = Flask(__name__)

# ✅ Kết nối MySQL
def create_connection():
    try:
        engine = create_engine(
            'mysql+mysqlconnector://vivutour_tour:voducviet2003@vi-vu-tour.pro.vn:3306/vivutour_datn_tour_new'
        )
        return engine
    except Exception as e:
        print(f"❌ Lỗi kết nối MySQL: {e}")
        return None

def close_connection(engine):
    if engine:
        engine.dispose()

# ✅ Kết hợp đặc trưng bằng vector hóa
def combine_features_df(df):
    return (
        df['title'].astype(str) + ' ' +
        df['description'].astype(str) + ' ' +
        df['time'].astype(str) + ' ' +
        df['priceAdult'].astype(str) + ' ' +
        df['destination'].astype(str) + ' ' +
        df['domain'].astype(str)
    )

@app.route('/api/user-recommendations', methods=['GET'])
def get_user_recommendations():
    start = time.time()
    user_id = request.args.get('user_id')

    if not user_id or not user_id.isdigit():
        return jsonify({"error": "Invalid or missing 'user_id' parameter"}), 400

    user_id = int(user_id)
    engine = create_connection()

    try:
        # 📦 Gộp dữ liệu người dùng
        query_user_data = f"""
            SELECT * FROM tbl_booking WHERE userId = {user_id};
            SELECT * FROM tbl_reviews WHERE userId = {user_id};
        """
        with engine.connect() as conn:
            bookings = pd.read_sql("SELECT * FROM tbl_booking WHERE userId = %s", conn, params=(user_id,))
            reviews = pd.read_sql("SELECT * FROM tbl_reviews WHERE userId = %s", conn, params=(user_id,))

        # Gộp các tour đã tương tác
        if bookings.empty and reviews.empty:
            return jsonify({"recommended_tours": None}), 200

        interacted_ids = pd.concat([bookings['tourId'], reviews['tourId']]).unique()

        # 📦 Truy vấn danh sách tour + đánh giá
        with engine.connect() as conn:
            all_tours = pd.read_sql("""
                SELECT t.*, AVG(s.priceAdult) AS priceAdult
                FROM tbl_tour t
                LEFT JOIN tbl_start_end_date s ON t.tourId = s.tourId
                GROUP BY t.tourId
            """, conn)

            all_reviews = pd.read_sql("SELECT tourId, rating FROM tbl_reviews", conn)

        # Tạo đặc trưng
        all_tours['combineFeatures'] = combine_features_df(all_tours)
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(all_tours['combineFeatures'])

        # 🎯 Lấy chỉ số các tour ứng viên (chưa xem)
        candidate_tours = all_tours[~all_tours['tourId'].isin(interacted_ids)].copy()
        if candidate_tours.empty:
            return jsonify({"message": "No tours available for recommendations."}), 404

        # 🎯 Tính similarity
        interacted_idx = all_tours[all_tours['tourId'].isin(interacted_ids)].index
        candidate_idx = candidate_tours.index
        sim_scores = cosine_similarity(tfidf_matrix[interacted_idx], tfidf_matrix[candidate_idx]).mean(axis=0)
        candidate_tours['similarity'] = sim_scores

        # 🎯 Gộp điểm đánh giá
        avg_rating = all_reviews.groupby('tourId')['rating'].mean().reset_index()
        avg_rating.columns = ['tourId', 'avg_rating']
        candidate_tours = candidate_tours.merge(avg_rating, on='tourId', how='left')
        candidate_tours['avg_rating'] = candidate_tours['avg_rating'].fillna(avg_rating['avg_rating'].mean())

        # 🎯 Sắp xếp và chọn top 4
        top_tours = candidate_tours.sort_values(by=['similarity', 'avg_rating'], ascending=[False, False]).head(4)
        result = top_tours[['tourId']].to_dict(orient='records')

        # print("✅ Xử lý recommendation mất:", round(time.time() - start, 2), "giây")
        return jsonify({"recommended_tours": result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500
    finally:
        close_connection(engine)

@app.route('/api/tour-recommendations', methods=['GET'])
def get_tour_recommendations():
    tour_id = request.args.get('tour_id')

    if not tour_id or not tour_id.isdigit():
        return jsonify({"error": "Invalid or missing 'tour_id' parameter"}), 400

    tour_id = int(tour_id)
    engine = create_connection()

    try:
        with engine.connect() as conn:
            all_tours = pd.read_sql("""
                SELECT t.*, AVG(s.priceAdult) AS priceAdult
                FROM tbl_tour t
                LEFT JOIN tbl_start_end_date s ON t.tourId = s.tourId
                GROUP BY t.tourId
            """, conn)

            all_reviews = pd.read_sql("SELECT tourId, rating FROM tbl_reviews", conn)

        if tour_id not in all_tours['tourId'].values:
            return jsonify({"error": "Tour not found."}), 404

        # Tính TF-IDF + cosine
        all_tours['combineFeatures'] = combine_features_df(all_tours)
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(all_tours['combineFeatures'])
        cosine_sim = cosine_similarity(tfidf_matrix)

        idx = all_tours[all_tours['tourId'] == tour_id].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        similar_indices = [i for i, _ in sim_scores if i != idx][:10]

        similar_tours = all_tours.iloc[similar_indices].copy()
        similar_tours['similarity'] = [cosine_sim[idx][i] for i in similar_indices]

        avg_rating = all_reviews.groupby('tourId')['rating'].mean().reset_index()
        avg_rating.columns = ['tourId', 'avg_rating']
        similar_tours = similar_tours.merge(avg_rating, on='tourId', how='left')
        similar_tours['avg_rating'] = similar_tours['avg_rating'].fillna(avg_rating['avg_rating'].mean())

        top = similar_tours.sort_values(by=['similarity', 'avg_rating'], ascending=[False, False]).head(4)
        result = top[['tourId']].to_dict(orient='records')
        return jsonify({"recommended_tours": result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500
    finally:
        close_connection(engine)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=False)
