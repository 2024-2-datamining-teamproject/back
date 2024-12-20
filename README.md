# 설명

AJOUFLIX의 백엔드입니다.

# API endpoint

`login`: 로그인을 위한 api입니다. user id를 받아서, 기존 유저는 바로 추천 정보를, 신규 유저는 신규 유저라는 알림을 보냅니다.

`register`: 신규 유저의 등록을 위한 api입니다. 좋아하는 감독과 영화의 정보를 받아서, 이에 대응하는 추천 정보를 보냅니다.

`search`: 키워드 검색을 위한 api입니다. keyword를 받아서, 해당 키워드에 대응하는 추천 정보를 보냅니다.

# 구현 세부 사항

프론트로 보내는 영화의 정보는 영화의 이름과 포스터 이미지의 url입니다.

해당 정보들은 기존 구현체의 결과인 movieId들을 OMDb라는 api에 요청을 보내고, 결과에서 이름과 포스터 정보를 뽑아옵니다.

단일 스레드로 구현해보니까, 응답 속도가 너무 떨어져서, 다중 스레드를 이용하여 한 번에 여러 정보들을 가져오도록 하고, 여기에 기존보다 더 짧은 timeout을 적용해서 더 빠른 반응 속도를 갖도록 설계했습니다.

# 더 개선할 수 있을까?
OMDb로 부터 받는 영화들의 정보를 캐싱하게 코드를 수정한다면, 반응성을 더 올릴 수 있을 거 같긴 합니다
