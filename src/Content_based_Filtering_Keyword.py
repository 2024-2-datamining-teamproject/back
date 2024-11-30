import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv("csv/movies.csv")
tags_df = pd.read_csv("csv/tags.csv")

# tags.csv에서 movieId 별로 태그를 하나의 문자열로 결합
tags_df = tags_df.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()

# movies.csv와 tags.csv를 movieId를 기준으로 병합
movies_data = pd.merge(movies_df, tags_df, on="movieId", how="inner")

# title과 tag 열을 결합
movies_data["tag"] = movies_data["title"] + " " + movies_data["tag"]

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data["tag"])


# 키워드와 유사한 영화 찾기
def get_similar_movies_by_keyword(keyword, top_n=10):
    # 키워드를 TF-IDF 벡터화하여 코사인 유사도 계산
    keyword_vec = tfidf_vectorizer.transform([keyword])
    cosine_similarities_keyword = cosine_similarity(keyword_vec, tfidf_matrix).flatten()

    # 유사도 상위 영화 추출
    similarity_scores = list(enumerate(cosine_similarities_keyword))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_movie_indices = [
        i[0] for i in similarity_scores[:top_n]
    ]  # 상위 top_n개만 선택

    # 상위 30개 유사 영화 movieId List 반환
    similar_movies = movies_data.iloc[similar_movie_indices]

    return list(similar_movies["movieId"])

if __name__ == "__main__":
    # Test
    keyword = "adventure"  # 키워드 입력
    similar_movies = get_similar_movies_by_keyword(keyword)
    # 상위 30개 movieId list 출력
    print(similar_movies)