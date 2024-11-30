from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd
import os
import requests
from src.Content_based_Filtering_Keyword import get_similar_movies_by_keyword
from src.Filtering_by_Director import get_movies_by_director
from src.recommendation import recommend_by_knn, recommend_by_prediction
from src.Content_based_Filtering_Title import get_similar_movies
from src.weather_on_position import movie_weather_recommender
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 5  # 동시에 처리할 최대 스레드 수

def fetch_movie_details(movie_id):
    """Fetch movie details from OMDB API with a timeout."""
    try:
        response = requests.get(
            poster_api.format(API_KEY, movie_id),
            timeout=2  # 타임아웃 2초
        )
        data = response.json()
        return {
            "title": data.get("Title", "Error Title"),
            "poster": data.get("Poster", placeholder)
        }
    except requests.exceptions.Timeout:
        print(f"Timeout occurred for movie ID {movie_id}")
        return {"title": "Timeout Error", "poster": placeholder}
    except Exception as e:
        print(f"Error fetching movie details for ID {movie_id}: {e}")
        return {"title": "Error Title", "poster": placeholder}

def fetch_movie_details_concurrent(movie_ids):
    """Fetch movie details concurrently using ThreadPoolExecutor."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_movie = {executor.submit(fetch_movie_details, movie_id): movie_id for movie_id in movie_ids}
        for future in as_completed(future_to_movie):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error in concurrent processing: {e}")
                results.append({"title": "Error Title", "poster": placeholder})
    return results


app = Flask(__name__)

# Load environment variables and CSV files
load_dotenv()
API_KEY = os.getenv("OMDB_API_KEY")
poster_api = "http://www.omdbapi.com/?apikey={0}&i=tt{1:0>7}"
placeholder = "https://via.placeholder.com/200x300"

ratings_df = pd.read_csv("csv/ratings.csv")
movies_df = pd.read_csv("csv/movies.csv")
directors_df = pd.read_csv("csv/directors.csv")

def fetch_movie_details(movie_id):
    """Fetch movie details from OMDB API."""
    try:
        response = requests.get(poster_api.format(API_KEY, movie_id))
        data = response.json()
        return {
            "title": data.get("Title", "Error Title"),
            "poster": data.get("Poster", placeholder)
        }
    except Exception as e:
        print(f"Error fetching movie details for ID {movie_id}: {e}")
        return {"title": "Error Title", "poster": placeholder}

@app.route('/login', methods=['POST'])
def login():
    """Handle user login."""
    data = request.json
    user_id = int(data.get("user_id"))

    response = {"existing_user": user_id in ratings_df['userId'].unique()}
    if response["existing_user"]:
        print(f"Existing user logged in: {user_id}")

        # User-based prediction
        try:
            results = recommend_by_prediction(user_id, top_n=10)
            response["predicted_movies"] = fetch_movie_details_concurrent(results)
        except Exception as e:
            print(f"Error in recommend_by_prediction: {e}")

        # Collaborative filtering (KNN)
        try:
            results = recommend_by_knn(user_id, top_n=10)
            response["knn_movies"] = fetch_movie_details_concurrent(results)
        except Exception as e:
            print(f"Error in recommend_by_knn: {e}")

        # Weather-based recommendation
        try:
            results = movie_weather_recommender(recommend_num=10, min_rate_num=50)
            response["weather_movies"] = fetch_movie_details_concurrent(results)
        except Exception as e:
            print(f"Error in movie_weather_recommender: {e}")
    else:
        print(f"New user logged in: {user_id}")

    return jsonify(response)


@app.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.json
    user_id = int(data.get("user_id"))
    favorite_movie = data.get("favorite_movie")
    favorite_director = data.get("favorite_director")

    response = {"new_user": user_id}

    # Content-based filtering by movie
    try:
        results = get_similar_movies(favorite_movie, top_n=10)
        response["similar_movies"] = fetch_movie_details_concurrent(results)
    except Exception as e:
        print(f"Error in get_similar_movies: {e}")

    # Content-based filtering by director
    try:
        results = get_movies_by_director(favorite_director)
        response["director_movies"] = fetch_movie_details_concurrent(results)
    except Exception as e:
        print(f"Error in get_movies_by_director: {e}")

    # Weather-based recommendation
    try:
        results = movie_weather_recommender(recommend_num=10, min_rate_num=50)
        response["weather_movies"] = fetch_movie_details_concurrent(results)
    except Exception as e:
        print(f"Error in movie_weather_recommender: {e}")

    return jsonify(response)


@app.route('/search', methods=['POST'])
def search():
    """Search for movies based on a keyword."""
    data = request.json
    keyword = data.get("keyword")

    response = {}
    try:
        results = get_similar_movies_by_keyword(keyword)
        response["keyword_movies"] = [fetch_movie_details(movie_id) for movie_id in results]
    except Exception as e:
        print(f"Error in get_similar_movies_by_keyword: {e}")
        response["keyword_movies"] = []

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)
