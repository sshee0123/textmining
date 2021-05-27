import operator
import pickle
import pymysql
from konlpy.tag import Okt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

HOST = 'database-1.cfi9ak8locdw.ap-northeast-2.rds.amazonaws.com'
PORT = 3306
USER = 'admin'
PASSWORD = 'dzbz2021'

okt = Okt()

# MySQL Connection 연결
conn = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, db='boardback', charset='utf8')

# Connection 으로부터 Cursor 생성
curs = conn.cursor()

# SQL문 실행
sql = "select review,star from BedroomReview where subcate_no='침대' and pd_no=0"
curs.execute(sql)

# 데이타 Fetch
rows = curs.fetchall()
# # print(rows)     # 전체 rows

filename = './training/bed0-bedreview.csv'  # bed 폴더랑 파일 이름 만 변경하면됨
file = open(filename, 'r', encoding='utf-8-sig')
df1 = pd.read_csv(file,usecols=[1,2])

df2= pd.DataFrame(rows)
df2 = pd.DataFrame(rows, columns=['review', 'star'])
df = pd.concat([df1,df2])
print(df)

#별점이 3이상이면 label 1 긍정 / 나머지는 label 0 부정
df['star'] = pd.to_numeric(df['star'])
df['label'] = np.select([df.star > 3], [1], default=0)
print(df['label'].values)

df['star'].nunique(), df['review'].nunique(), df['label'].nunique()
df.drop_duplicates(subset=['review'], inplace=True)  # reviews 열에서 중복인 내용이 있다면 중복 제거

text_list=df['review'].tolist()
label_list=df['label'].tolist()

#train / test data 나누기
print(df.isnull().values.any())
train_data, test_data,y_train,y_test = train_test_split(text_list,label_list, test_size=0.25, random_state=42)
print('훈련용 리뷰의 개수 :', len(train_data))
print('테스트용 리뷰의 개수 :', len(test_data))
print("train_data",train_data)

# 객체 복원
with open('pipe.dat', 'rb') as fp:
    pipe = pickle.load(fp)


    for i in range(len(test_data)):
        text = test_data[i]
        str=[text]
        # 예측 정확도
        r1 = np.max(pipe.predict_proba(str) * 100)
        # 예측 결과
        r2 = pipe.predict(str)[0]

        if r2 == 0:
            print(str)
            print('부정적 리뷰입니다.')
        elif r2== 1:
            print(str)
            print('긍정적 리뷰입니다.')
        print('정확도 : %.3f' % r1)

# Connection 닫기
conn.close()
