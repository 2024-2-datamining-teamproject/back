import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv("csv/movies.csv")
tags_df = pd.read_csv("csv/tags.csv")

# tags.csv에서 movieId 별로 태그를 하나의 문자열로 결합
tags_df = tags_df.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()

# movies.csv와 tags.csv를 movieId를 기준으로 병합
movies_data = pd.merge(movies_df, tags_df, on="movieId", how="inner")

# TF-IDF 벡터화 (태그)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data["tag"])

# 영화 간 코사인 유사도 계산
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 특정 영화와 유사한 영화 찾기
def get_similar_movies(movie_title, top_n=10):
    # 영화 제목으로 인덱스 찾기
    try:
        movie_index = movies_data[movies_data["title"] == movie_title].index[0]
    except IndexError:
        return "해당 영화가 데이터에 없습니다."

    # 코사인 유사도 계산 결과에서 상위 유사 영화 찾기
    similarity_scores = list(enumerate(cosine_similarities[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movie_indices = [
        i[0] for i in similarity_scores[1 : top_n + 1]
    ]  # 자기 자신 제외

    # 상위 30개 유사 영화 movieId List 반환
    similar_movies = movies_data.iloc[similar_movie_indices]

    return list(similar_movies["movieId"])

if __name__ == "__main__":
    # Test
    movie_title = "Toy Story (1995)"
    similar_movies = get_similar_movies(movie_title)
    # 상위 30개 movieId list 출력
    print(similar_movies)