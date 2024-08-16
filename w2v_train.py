#word2Vec 훈련
from gensim.models import Word2Vec  #word2Vec 학습 모듈
import json  #훈련 데이터 불러오기

with open('tokenized.json', 'r') as json_file:
    result = json.load(json_file)

#vector_size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
#window = 컨텍스트 윈도우 크기
#min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
#workers = 학습을 위한 프로세스 수
#sg = 0은 CBOW, 1은 Skip-gram.

model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)

model.wv.save_word2vec_format('eng_w2v') # 모델 저장