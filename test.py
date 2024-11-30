import requests
from io import BytesIO

API_KEY = "c711702e"
movie_id = "83658"

poster_api = "http://img.omdbapi.com/?apikey={0}&i=tt{1:0>7}"

request_result = requests.get(poster_api.format(API_KEY, movie_id))
print(BytesIO(request_result.content))
