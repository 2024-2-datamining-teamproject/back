# input: 감독 이름
# 해당 감독의 영화ID 리스트 찾기
# 영화ID 리스트 중 평점평균 높은 순으로 sorting
# output: 상위 10개 영화ID

import pandas as pd

ratings_df = pd.read_csv("csv/ratings.csv")
directors_df = pd.read_csv("csv/directors.csv")

def get_movies_by_director(director_name):
    try:
        filtered_movies = directors_df[directors_df['director'] == director_name]
        
        if filtered_movies.empty:
            print(f"감독 '{director_name}'의 영화를 찾을 수 없습니다.")
            return []
        
        movie_ids = filtered_movies['movieId'].tolist()
        filtered_movie_ratings = ratings_df[ratings_df['movieId'].isin(movie_ids)]
        
        if filtered_movie_ratings.empty:
            print(f"감독 '{director_name}'의 영화에 대한 평점 데이터를 찾을 수 없습니다.")
            return []
        
        average_ratings = (
            filtered_movie_ratings
            .groupby('movieId')['rating']
            .mean()
            .reset_index()
            .rename(columns={'rating': 'avg_rating'})
            .sort_values(by='avg_rating', ascending=False)
        )
        
        top_10_movie_ids = average_ratings['movieId'].tolist()[:10]
        return top_10_movie_ids

    except FileNotFoundError as e:
        print("CSV 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return []
    except KeyError as e:
        print("CSV 파일에 필요한 열이 없습니다. 'director'와 'movieId' 또는 'rating' 열을 확인하세요.")
        return []


if __name__ == "__main__":
    director_name = input("감독 이름을 입력하세요: ")
    top_10_movies = get_movies_by_director(director_name)

    if top_10_movies:
        print(f"감독 '{director_name}'의 상위 10개 영화 ID: {top_10_movies}")
    else:
        print(f"감독 '{director_name}'의 영화를 찾을 수 없습니다.")
