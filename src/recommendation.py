import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

predicted_ratings_df = pd.read_csv('csv/predicted_ratings.csv', index_col=0)
ratings_df = pd.read_csv('csv/ratings.csv')

predicted_ratings = predicted_ratings_df.values
movie_ids = predicted_ratings_df.columns.astype(int)
user_ids = predicted_ratings_df.index.astype(int)

ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.reindex(columns=movie_ids, fill_value=np.nan)
original_ratings_matrix = ratings_matrix.copy() 

ratings_matrix = ratings_matrix.fillna(0)


def recommend_by_prediction(user_id, top_n=10):
    """
    predicted_ratings.csv 기반으로 특정 사용자가 평가하지 않은 영화 중 예상 평점이 높은 상위 N개 추천
    """
    user_idx = np.where(user_ids == user_id)[0][0]
    user_ratings = predicted_ratings[user_idx]
    
    rated_movies = original_ratings_matrix.loc[user_id][original_ratings_matrix.loc[user_id].notna()].index
    unrated_movies = [movie for movie in movie_ids if movie not in rated_movies]
    
    recommendations = {movie: user_ratings[np.where(movie_ids == movie)[0][0]] for movie in unrated_movies}
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    return [int(movie_id) for movie_id, _ in sorted_recommendations[:top_n]]


# def recommend_by_knn(user_id, top_n=10):
#     """
#     KNN을 통해 예측 평점(predicted_ratings.csv)을 기반으로 유사한 사용자가 높게 평가한 영화를 추천
#     """
#     knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
#     knn_model.fit(predicted_ratings)
#     user_idx = np.where(user_ids == user_id)[0][0]
#     _, indices = knn_model.kneighbors([predicted_ratings[user_idx]], n_neighbors=5)
#     similar_users = user_ids[indices.flatten()]
#     recommendations = {}
#     for similar_user in similar_users:
#         similar_user_idx = np.where(user_ids == similar_user)[0][0]
#         similar_user_ratings = predicted_ratings[similar_user_idx]
#         for movie_idx, rating in enumerate(similar_user_ratings):
#             movie_id = movie_ids[movie_idx]
#             if pd.isna(original_ratings_matrix.loc[user_id, movie_id]):
#                 if movie_id not in recommendations:
#                     recommendations[movie_id] = rating
#                 else:
#                     recommendations[movie_id] += rating
#     recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
#     return [int(movie_id) for movie_id, _ in recommended_movies[:top_n]]

def recommend_by_knn(user_id, top_n=10):
    """
    KNN을 통해 실제 평점(ratings.csv)을 기반으로 유사한 사용자가 높게 평가한 영화를 추천
    """
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
    knn_model.fit(ratings_matrix.values)
    
    user_idx = ratings_matrix.index.get_loc(user_id)
    
    _, indices = knn_model.kneighbors([ratings_matrix.iloc[user_idx].values], n_neighbors=5)
    similar_users = ratings_matrix.index[indices.flatten()]
    
    recommendations = {}
    for similar_user in similar_users:
        similar_user_ratings = ratings_matrix.loc[similar_user]
        
        for movie_id, rating in similar_user_ratings.items():
            if pd.isna(original_ratings_matrix.loc[user_id, movie_id]) and not pd.isna(rating):
                if movie_id not in recommendations:
                    recommendations[movie_id] = rating
                else:
                    recommendations[movie_id] += rating
    
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    return [int(movie_id) for movie_id, _ in recommended_movies[:top_n]]



if __name__ == "__main__":

    # 예제 실행
    user_id = 2  # 추천을 받을 사용자 ID
    print("Recommendations for User 1 by Prediction:", recommend_by_prediction(user_id=user_id))
    print("Recommendations for User 1 by KNN:", recommend_by_knn(user_id=user_id))