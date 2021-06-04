# -- coding: utf-8 --
from flask import Flask
#from flask_cors import CORS
import urllib.request

import pymysql
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import collections
import numpy as np
from flask import request
#from flask_cors import CORS
import pickle

HOST = 'database-1.cfi9ak8locdw.ap-northeast-2.rds.amazonaws.com'
PORT = 3306
USER = 'admin'
PASSWORD = 'dzbz2021'

app = Flask(__name__)
# CORS(app)
app.config['JSON_AS_ASCII'] = False


# MySQL Connection 연결
conn = pymysql.connect(host=HOST, port=PORT, user=USER, password=PASSWORD, db='boardback', charset='utf8')

# Connection 으로부터 Cursor 생성
curs1 = conn.cursor()
curs2 = conn.cursor()

# SQL문 실행
sql1 = "select user_id,pd_no,subcate_no,category_no from LikeList"
# sql1 = "select user_id, pd_no, l.subcate_no, l.category_no from LikeList l " \
#        "left join PurchaseList p " \
#        "on l.user_id=p.user_id and l.pd_no=p.pd_no and l.subcate_no=p.subcate_no and l.category_no=p.category_no"\
#        "union all" \
#        "select l.user_id, l.pd_no, l.subcate_no, l.category_no from LikeList l " \
#        "right join PurchaseList p"\
#        "on l.user_id=p.user_id and l.pd_no=p.pd_no and l.subcate_no=p.subcate_no and l.category_no=p.category_no"\
#        "where l.user_id is not null"

curs1.execute(sql1)
sql2 = "select user_id,pd_no,subcate_no,category_no from PurchaseList"
curs2.execute(sql2)

######################################## 데이터 전처리 시작 ###################################

# 데이타 Fetch
rows1 = curs1.fetchall()
#print(rows1)
#print(len(rows1))
rows2 = curs2.fetchall()
#print(rows2)
#print(len(rows2))

dfcol = ['user_id', 'pd_no', 'subcate_no', 'category_no']

# dataframe 만들기
df1 = pd.DataFrame(rows1, columns=dfcol)
df2 = pd.DataFrame(rows2, columns=dfcol)
df = pd.concat([df1, df2])

# user_id를 행번호로
df = df.set_index('user_id')

# 3개 칼럼 하나의 아이템이름으로 합치기
pdcol = ["pd_no", "subcate_no", "category_no"]
df['pd'] = df[pdcol].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

# db데이터 상품 배열
pddbarray = df['pd']
dbcol = []
for i in range(len(pddbarray)):
    dbcol.append(pddbarray[i].split('_'))

# 인기상품
counts = collections.Counter(pddbarray)
hot = counts.most_common(1)[0][0]

# 원래 df (중복상품 제거 전)
pddbarray_df = pd.DataFrame(pddbarray)

# user-item 데이터프레임생성
user_item_df = pddbarray_df.groupby(['user_id'])['pd'].apply(','.join).reset_index()

# 중복상품 제거 ->item 명으로 사용
pdarray = df['pd'].unique()
pdarray_df = pd.DataFrame(pdarray)

# user 명으로 사용
username = user_item_df['user_id']

# 중복상품 제거 한 후의 [pd_no,subcate_no,category_no]배열 생성 -> 나중에 상품 구분할 때 쓰일 예정
col = []
for i in range(len(pdarray)):
    col.append(pdarray[i].split('_'))

# 행번호 user_id로
user_item_df1 = user_item_df.set_index('user_id')

# user_item_df1와 pdarray 비교하여 값 겹치는 인덱스번호 모음 배열
idx = []
pdslice = []
for i in range(len(user_item_df1)):
    pdslice.append(user_item_df1.pd[i].split(","))

for i in range(len(pdslice)):
    line = []
    for j in range(len(pdslice[i])):
        for k in range(len(pdarray)):
            if pdslice[i][j] == pdarray[k]:
                line.append(k)
    idx.append(line)

# dataframe -> list로
userarray = username.tolist()

# pdarray로 데이터프레임 생성 -> mydf
mydf = pd.DataFrame(user_item_df1, columns=pdarray)

# 데이터 변경 쉽게하기 위하여 dataframe->list로 변환
mydf_list = mydf.values.tolist()

# idx의 배열의 인덱스는 mydf의 행, idx의 배열의 값은 그 mydf의 행의 열값 1로 변경
for i, v in enumerate(idx):
    for j, k in enumerate(v):
        mydf_list[i][k] = 1

# 다시 dataframe으로 변경
mydf1 = pd.DataFrame(mydf_list, columns=pdarray, index=userarray)
print("mydf1", mydf1)

# 아이템기반으로 해야하니 행 열 전환
mydf1 = mydf1.transpose()
print(mydf1)

# NaN -> 0으로 채워주기
mydf1 = mydf1.fillna(0)
print(mydf1)

########################### 아이템 기반 협업필터링 #########################

# 아이템 코사인 유사도 구하기
item_based = cosine_similarity(mydf1)
item_based = pd.DataFrame(data=item_based, index=pdarray, columns=pdarray)
print(item_based.head())



################# 한 상품이름에 대해 8개 추천 코사인유사도도 함께 출력 ###############################
@app.route("/rec/recommend", methods=['POST'])
def recommend():
    json = request.json
    pdNo = json['pdNo']
    subcateNo = json['subcateNo']
    categoryNo = json['categoryNo']
    pdname = str(pdNo)+"_"+subcateNo+"_"+categoryNo
    print(pdname)
    return dict(item_based[pdname].sort_values(ascending=False)[:8])

##################  user 데이터 없을 때 인기상품 1개 관련 상품 보여주기 ############################
@app.route("/rec/nodata")
def nouserdata():
    return dict(item_based[hot].sort_values(ascending=False)[:8])


@app.route("/rec/predict", methods=['POST'])
def reviewPrediction():
    json = request.json
    pdNo = json['pdNo']
    subcateNo = json['subcateNo']
    categoryNo = json['categoryNo']
    review = json['review']
    if categoryNo == "침실가구":
        filename = 'bedpipe.dat'
    elif categoryNo == "수납가구":
        filename = 'storagepipe.dat'
    elif categoryNo == "거실가구":
        filename = 'livingpipe.dat'
    else: return
    result = ""
    with open(filename, 'rb') as fp:
        pipe = pickle.load(fp)

        # #test_data가 새로 들어온 리뷰의 텍스트여야함
        # for i in range(len(test_data)):
        #     text = test_data[i]
        str=[review]
        #str = ["매트리스가 높네요ㅠ"]

        # 예측 정확도 ->
        r1 = np.max(pipe.predict_proba(str) * 100)
         # 예측 결과
        r2 = pipe.predict(str)[0]

        if r2 == 0:
            print(str)
            #print('정확도 : %.3f' % r1,'로 부정적 리뷰입니다.')
            acc = format(r1, '.3f')
            result = '정확도 : ' + acc + '로 부정적 리뷰입니다.'

            print(result)
        elif r2== 1:
            print(str)
            acc = format(r1, '.3f')
            result = '정확도 : ' + acc + '로 긍정적 리뷰입니다.'
        return result



if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1", port=5000)