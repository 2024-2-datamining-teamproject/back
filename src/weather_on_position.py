from dotenv import load_dotenv
import os
import requests
import pandas as pd

# position of Ajou University
lat = 37.280
lon = 127.044

load_dotenv()
API_KEY = os.getenv("API_KEY")

api_url = "https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}".format(lat, lon, API_KEY)

weather_genre_dic = {
        0:["Adventure", "Animation", "Children", "Comedy", "Musical"],
        1:["Drama", "Romance", "Fantasy", "Mystery"],
        2:["Romance", "Drama", "Crime", "Film-Noir"],
        3:["Fantasy", "Romance", "Adventure", "IMAX"],
        4:["Action", "Thriller", "Horror", "Sci-Fi"],
        5:["Mystery", "Horror", "Film-Noir", "Western"]
        }

movies_df = pd.read_csv("csv/movies.csv")
ratings_df = pd.read_csv("csv/ratings.csv")

def get_weather_info(lat, lon):
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        #print(data)
        #print(data.get('weather'))
        weather_code = data.get('weather')[0].get('id')
        return weather_code
    else:
        print("Request failed:", response.status_code)

def translate_weather_code(weather_code):
    if weather_code == 800:
        # 맑은 날씨의 경우
        return 0
    elif weather_code // 100 == 8:
        # 구름 낀 날씨의 경우
        return 1
    elif weather_code // 100 == 3 or weather_code // 100 == 4:
        # 비오는 날씨의 경우
        return 2
    elif weather_code // 100 == 6:
        # 눈오는 날씨의 경우
        return 3
    elif weather_code // 100 == 2 or weather_code == 771 or weather_code == 781:
        # 폭풍치는 날씨의 경우
        return 4
    elif weather_code // 100 == 7:
        # 안개 낀 날씨의 경우
        return 5

def filter_movies_by_weather(translated_weather_code, movie_genre_dic):
    # filter movies from given list, using weather code
    recommended_genre = weather_genre_dic.get(translated_weather_code)
    #print("recommended list:", recommended_genre)
    output = []
    for movie_id, genre_list in movie_genre_dic.items():
        #print("current list:", genre_list)
        if bool(set(genre_list) & set(recommended_genre)):
            #print("true for", movie_id)
            output.append(movie_id)
        #else:
            #print("false for", movie_id)

    return output


def extract_movie_genre(movie_id_list):
    given_movies_genre = {}
    # extract movie's genre information, and refine it in list form
    movie_data = movies_df[movies_df['movieId'].isin(movie_id_list)]
    #print("data:", movie_data)
    for index, row in movie_data.iterrows():
        #print(row['genres'])
        genre_list = row['genres'].split('|')  # '|'를 기준으로 장르 분리
        #print("genres:", genre_list)
        given_movies_genre[row['movieId']] = genre_list

    #print(given_movies_genre)
    return given_movies_genre

def sort_by_avg_rates(movie_id_list):
    # avg on every movie first,
    filtered_movie_list = ratings_df[ratings_df['movieId'].isin(movie_id_list)]
    #print(filtered_movie_list)
    average_ratings = filtered_movie_list.groupby('movieId')['rating'].mean().reset_index()
    average_ratings.rename(columns={'rating':'avg_rating'}, inplace=True)
    # and sort it in descending order
    average_ratings = average_ratings.sort_values(by='avg_rating', ascending=False)['movieId'].tolist()
    #print(average_ratings)

    return average_ratings

def movie_filterer(movie_id_list):
    # filter movie by checking weather of AJOU university
    weather_code = get_weather_info(lat, lon)
    translated_weather_code = translate_weather_code(weather_code)
    filtered_movies = filter_movies_by_weather(translated_weather_code, extract_movie_genre(movie_id_list))

    return filtered_movies

def movie_weather_recommender(recommend_num, min_rate_num):
    #print("whole movie id:", whole_movie_id_list)
    rating_counts = ratings_df.groupby('movieId').size().reset_index(name='count')
    valid_movie_list = rating_counts[rating_counts['count'] >= min_rate_num]['movieId'].tolist()
    filtered_movie_df = movies_df[movies_df['movieId'].isin(valid_movie_list)]
    result = movie_filterer(filtered_movie_df['movieId'].tolist())
    #print("filtered movie id:", filtered_movie_list)
    sorted_result = sort_by_avg_rates(result)
    

    return sorted_result[:recommend_num]

def main():
    recommend_num = 30
    min_rate_num = 50
    #test_movie_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    #result = movie_filterer(test_movie_list)
    
    result = movie_weather_recommender(recommend_num, min_rate_num)


    print(result)

if __name__ == "__main__":
    main()