import pandas as pd
import numpy as np

latent_dim = 30
learning_rate = 0.005
lambda_reg = 0.1
epochs = 30

ratings_df = pd.read_csv('csv/ratings.csv') 
movies_df = pd.read_csv('csv/movies.csv') 

all_movie_ids = movies_df['movieId'].unique()

user_ids = ratings_df['userId'].unique()
user_id_map = {id_: idx for idx, id_ in enumerate(user_ids)}
movie_id_map = {id_: idx for idx, id_ in enumerate(all_movie_ids)}  

ratings_df['user_idx'] = ratings_df['userId'].map(user_id_map)
ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_map)

num_users = len(user_ids)
num_movies = len(all_movie_ids)

# P = np.random.normal(scale=1.0 / latent_dim, size=(num_users, latent_dim)) 
# Q = np.random.normal(scale=1.0 / latent_dim, size=(num_movies, latent_dim))
P = np.random.uniform(-np.sqrt(6 / latent_dim), np.sqrt(6 / latent_dim), size=(num_users, latent_dim))
Q = np.random.uniform(-(np.sqrt(6 / latent_dim)), np.sqrt(6 / latent_dim), size=(num_movies, latent_dim))

RATING_MIN = ratings_df['rating'].min()
RATING_MAX = ratings_df['rating'].max()
ratings_df['rating_normalized'] = (ratings_df['rating'] - RATING_MIN) / (RATING_MAX - RATING_MIN)

for epoch in range(epochs):
    total_loss = 0
    for _, row in ratings_df.iterrows():
        user_idx = int(row['user_idx'])
        movie_idx = int(row['movie_idx'])
        rating = row['rating_normalized']

        pred_rating = np.dot(P[user_idx], Q[movie_idx])

        error = rating - pred_rating

        P[user_idx] += learning_rate * (error * Q[movie_idx] - lambda_reg * P[user_idx])
        Q[movie_idx] += learning_rate * (error * P[user_idx] - lambda_reg * Q[movie_idx])

        total_loss += error**2 + lambda_reg * (np.linalg.norm(P[user_idx])**2 + np.linalg.norm(Q[movie_idx])**2)
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

print("Calculating all predicted ratings...")
predicted_ratings = np.dot(P, Q.T)  

predicted_ratings = predicted_ratings * (RATING_MAX - RATING_MIN) + RATING_MIN


predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_ids, columns=all_movie_ids)

predicted_ratings_df = predicted_ratings_df.clip(lower=0, upper=5)

predicted_ratings_df.to_csv('predicted_ratings_test.csv')
print("Predicted ratings saved to 'predicted_ratings_test.csv'.")
