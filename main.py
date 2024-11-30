from src.Content_based_Filtering_Keyword import get_similar_movies_by_keyword
from src.Filtering_by_Director import get_movies_by_director
from src.recommendation import recommend_by_knn, recommend_by_prediction
from src.Content_based_Filtering_Title import get_similar_movies
from src.weather_on_position import movie_weather_recommender
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
import os
import requests

def main():
    ratings_df = pd.read_csv("csv/ratings.csv")
    movies_df = pd.read_csv("csv/movies.csv")
    directors_df = pd.read_csv("csv/directors.csv")
# user_id 입력    
    user_id = input("userId를 입력하세요: ")

#기존 사용자    
    if int(user_id) in ratings_df['userId'].unique():
        print("\n기존 사용자입니다.")
        
        # recommend_by_prediction
        try:
            predicted_movies = recommend_by_prediction(int(user_id), top_n=10)
            if predicted_movies:
                print("사용자가 좋아할만한 영화 ID:", predicted_movies)
            else:
                print("추천 결과가 없습니다.")
        except Exception as e:
            print(f"추천 오류: {e}")
        
        # recommend_by_knn
        try:
            knn_movies = recommend_by_knn(int(user_id), top_n=10)
            if knn_movies:
                print("유사한 사용자가 좋아한 영화 ID:", knn_movies)
            else:
                print("추천 결과가 없습니다.")
        except Exception as e:
            print(f"KNN 기반 추천 오류: {e}")
        
        # movie_weather_recommender
        try:
            weather_movies = movie_weather_recommender(recommend_num=10, min_rate_num=50)
            if weather_movies:
                print("날씨에 어울리는 영화 ID:", weather_movies)
            else:
                print("추천 결과가 없습니다.")
        except Exception as e:
            print(f"날씨 기반 추천 오류: {e}")

# 콜드 스타트            
    else:
        print("\n콜드 스타트(새로운 사용자)입니다.")
        
        # 좋아하는 영화와 감독 정보 받기
        favorite_movie = input("좋아하는 영화 제목을 입력하세요: ")

        favorite_director = input("좋아하는 감독 이름을 입력하세요: ")
        
        # get_similar_movies
        try:
            similar_movies = get_similar_movies(favorite_movie, top_n=10)
            if isinstance(similar_movies, str):
                print(similar_movies)
            elif similar_movies:
                print("사용자가 좋아하는 영화와 유사한 영화 ID:", similar_movies)
            else:
                print("추천 결과가 없습니다.")
        except Exception as e:
            print(f"영화 유사도 기반 추천 오류: {e}")
        
        # get_movies_by_director
        try:
            director_movies = get_movies_by_director(favorite_director)
            if director_movies:
                print("사용자가 좋아하는 감독의 영화 ID:", director_movies)
            else:
                print(f"감독 '{favorite_director}'의 영화를 찾을 수 없습니다.")
        except Exception as e:
            print(f"감독 기반 추천 오류: {e}")
        
        # movie_weather_recommender
        try:
            weather_movies = movie_weather_recommender(recommend_num=10, min_rate_num=50)
            if weather_movies:
                print("날씨에 어울리는 영화 ID:", weather_movies)
            else:
                print("추천 결과가 없습니다.")
        except Exception as e:
            print(f"날씨 기반 추천 오류: {e}")

# 키워드 검색        
    # get_similar_movies_by_keyword
    keyword = input("검색 키워드를 입력하세요: ")
    try:
        keyword_movies = get_similar_movies_by_keyword(keyword)
        if keyword_movies:
            print("키워드로 검색된 영화 ID:", keyword_movies)
        else:
            print(f"키워드 '{keyword}'와 관련된 영화를 찾을 수 없습니다.")
    except Exception as e:
        print(f"키워드 검색 추천 오류: {e}")

if __name__ == "__main__":
    main()
