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
        user_bookings = pd.read_sql("SELECT * FROM tbl_booking WHERE userId = %(user_id)s;", engine, params={"user_id": user_id})
        user_reviews = pd.read_sql("SELECT * FROM tbl_reviews WHERE userId = %(user_id)s;", engine, params={"user_id": user_id})
        all_tours = pd.read_sql("""
            SELECT t.*, AVG(s.priceAdult) AS priceAdult
            FROM tbl_tour t
            LEFT JOIN tbl_start_end_date s ON t.tourId = s.tourId
            GROUP BY t.tourId
        """, engine)

        if user_bookings.empty and user_reviews.empty:
            return jsonify({"recommended_tours": None, "message": "No interaction history for user"}), 200

        interacted_tour_ids = pd.concat([
            user_bookings['tourId'] if not user_bookings.empty else pd.Series(dtype=int),
            user_reviews['tourId'] if not user_reviews.empty else pd.Series(dtype=int)
        ]).unique()

        candidate_tours = all_tours[~all_tours['tourId'].isin(interacted_tour_ids)].copy()
        if candidate_tours.empty:
            return jsonify({"message": "No tours available for recommendations."}), 404

        all_tours['combineFeatures'] = all_tours.apply(combine_features, axis=1)
        candidate_tours['combineFeatures'] = candidate_tours.apply(combine_features, axis=1)

        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=2)
        tfidf_matrix_all = tfidf.fit_transform(all_tours['combineFeatures'])
        cosine_sim = cosine_similarity(tfidf_matrix_all, tfidf_matrix_all)

        interacted_indices = all_tours[all_tours['tourId'].isin(interacted_tour_ids)].index.to_numpy()
        candidate_indices = candidate_tours.index.to_numpy()

        sim_scores = cosine_sim[np.ix_(interacted_indices, candidate_indices)].mean(axis=0)
        candidate_tours['similarity'] = sim_scores

        all_reviews = pd.read_sql("SELECT * FROM tbl_reviews;", engine)
        avg_rating = all_reviews.groupby('tourId')['rating'].mean().reset_index()
        avg_rating.columns = ['tourId', 'avg_rating']

        candidate_tours['tourId'] = candidate_tours['tourId'].astype('Int64')
        avg_rating['tourId'] = avg_rating['tourId'].astype('Int64')
        candidate_tours = pd.merge(candidate_tours, avg_rating, on='tourId', how='left')

        mean_rating = avg_rating['avg_rating'].mean() if not avg_rating.empty else 3.0
        candidate_tours['avg_rating'] = candidate_tours['avg_rating'].fillna(mean_rating)

        candidate_tours = candidate_tours.sort_values(by=['similarity', 'avg_rating'], ascending=[False, False])
        recommended_tours = candidate_tours.head(4)
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
        all_tours = pd.read_sql("""
            SELECT t.*, AVG(s.priceAdult) AS priceAdult
            FROM tbl_tour t
            LEFT JOIN tbl_start_end_date s ON t.tourId = s.tourId
            GROUP BY t.tourId
        """, engine)

        if tour_id not in all_tours['tourId'].values:
            return jsonify({"error": "Tour not found."}), 404

        all_tours['combineFeatures'] = all_tours.apply(combine_features, axis=1)

        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=2)
        tfidf_matrix = tfidf.fit_transform(all_tours['combineFeatures'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        tour_idx = all_tours[all_tours['tourId'] == tour_id].index[0]
        sim_scores = list(enumerate(cosine_sim[tour_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_similar_indices = [i for i, score in sim_scores if i != tour_idx][:10]

        similar_tours = all_tours.iloc[top_similar_indices].copy()
        similarities = [score for i, score in sim_scores if i in top_similar_indices]
        similar_tours['similarity'] = similarities

        all_reviews = pd.read_sql("SELECT * FROM tbl_reviews;", engine)
        avg_rating = all_reviews.groupby('tourId')['rating'].mean().reset_index()
        avg_rating.columns = ['tourId', 'avg_rating']

        similar_tours['tourId'] = similar_tours['tourId'].astype('Int64')
        avg_rating['tourId'] = avg_rating['tourId'].astype('Int64')
        similar_tours = pd.merge(similar_tours, avg_rating, on='tourId', how='left')

        mean_rating = avg_rating['avg_rating'].mean() if not avg_rating.empty else 3.0
        similar_tours['avg_rating'] = similar_tours['avg_rating'].fillna(mean_rating)

        similar_tours = similar_tours.sort_values(by=['similarity', 'avg_rating'], ascending=[False, False])
        result = similar_tours[['tourId']].head(4).to_dict(orient='records')

        return jsonify({"recommended_tours": result})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    finally:
        close_connection(engine)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
