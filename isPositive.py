import pickle
import numpy as np

# 객체 복원
#모델이름 삽입
def reviewPrediction(text):
    with open('bedpipe.dat', 'rb') as fp:
        pipe = pickle.load(fp)

        # #test_data가 새로 들어온 리뷰의 텍스트여야함
        # for i in range(len(test_data)):
        #     text = test_data[i]
        str=[text]
        #str = ["매트리스가 높네요ㅠ"]

        # 예측 정확도 ->
        r1 = np.max(pipe.predict_proba(str) * 100)
         # 예측 결과
        r2 = pipe.predict(str)[0]

        if r2 == 0:
            print(str)
            print('정확도 : %.3f' % r1,'로 부정적 리뷰입니다.')
        elif r2== 1:
            print(str)
            print('정확도 : %.3f' % r1, '로 긍정적 리뷰입니다.')
