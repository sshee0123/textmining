import pymysql
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

HOST = 'database-1.cfi9ak8locdw.ap-northeast-2.rds.amazonaws.com'
PORT = 3306
USER = 'admin'
PASSWORD = 'dzbz2021'

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
print(rows1)
print(len(rows1))
rows2 = curs2.fetchall()
print(rows2)
print(len(rows2))

dfcol=['user_id','pd_no','subcate_no','category_no']

#dataframe 만들기
df1 = pd.DataFrame(rows1,columns=dfcol)
df2 = pd.DataFrame(rows2,columns=dfcol)
df = pd.concat([df1,df2])

#user_id를 행번호로
df=df.set_index('user_id')

#3개 칼럼 하나의 아이템이름으로 합치기
pdcol=["pd_no","subcate_no","category_no"]
df['pd'] = df[pdcol].apply(lambda row: '_'.join(row.values.astype(str)),axis=1)

#db데이터 상품 배열
pddbarray=df['pd']
dbcol=[]
for i in range(len(pddbarray)):
    dbcol.append(pddbarray[i].split('_'))

#df
pddbarray_df=pd.DataFrame(pddbarray)
print(pddbarray_df)

user_item_df = pddbarray_df.groupby(['user_id'])['pd'].apply(','.join).reset_index()
print("user_item_df",user_item_df)

#중복상품 제거 ->item 명으로 사용
pdarray=df['pd'].unique()
pdarray_df = pd.DataFrame(pdarray)

#user 명으로 사용
username = user_item_df['user_id']
print(username)

#중복상품 제거 한 후의 [pd_no,subcate_no,category_no]배열 생성 -> 나중에 상품 구분할 때 쓰일 예정
col=[]
for i in range(len(pdarray)):
    col.append(pdarray[i].split('_'))
print(col)

user_item_df1=user_item_df.set_index('user_id')
print(user_item_df1)

#user_item_df1와 pdarray 비교하여 값 겹치는 인덱스번호 모음 배열
idx=[]
# for i in range(len(pddbarray)):
#     for j in range(len(pdarray)):
#         if pddbarray[i]==pdarray[j]:
#             idx.append(j)

pdslice=[]
for i in range(len(user_item_df1)):
    pdslice.append(user_item_df1.pd[i].split(","))

print("pdslice",pdslice)
print(len(pdslice))

for i in range(len(pdslice)):
    line = []
    for j in range(len(pdslice[i])):
        for k in range(len(pdarray)):
            if pdslice[i][j]==pdarray[k]:
                line.append(k)
    idx.append(line)

print(idx)
print(len(idx))

userarray = username.tolist()
#pdarray로 데이터프레임 생성 -> mydf
mydf = pd.DataFrame(user_item_df1,columns=pdarray)
print("mydf",mydf)

#데이터 변경 쉽게하기 위하여 dataframe->list로 변환
mydf_list = mydf.values.tolist()

#idx의 배열의 인덱스는 mydf의 행, idx의 배열의 값은 그 mydf의 행의 열값 1로 변경
for i,v in enumerate(idx):
    for j,k in enumerate(v):
        mydf_list[i][k]=1


#다시 dataframe으로 변경
mydf1 = pd.DataFrame(mydf_list,columns=pdarray,index=userarray)
print("mydf1",mydf1)



#아이템기반으로 해야하니 행 열 전환
mydf1 = mydf1.transpose()
print(mydf1)

#NaN -> 0으로 채워주기
mydf1 = mydf1.fillna(0)
print(mydf1)

########################### 아이템 기반 협업필터링 #########################

#아이템 코사인 유사도 구하기
item_based = cosine_similarity(mydf1)
print(item_based)

item_based = pd.DataFrame(data = item_based, index = pdarray, columns= pdarray)
print(item_based.head())

#user가 어떤 상품 구매하거나 좋아요 눌렀을 때 추천해주는 함수
# def get_item_based(pdname):
#     return item_based[pdname].sort_values(ascending=False)[:8]
#
# print(get_item_based("11_기타주방가구_주방가구"))

#한 상품이름에 대해 8개 추천 코사인유사도도 함께 출력
recommend=[]
for i in pdarray:
    recommend.append(item_based[i].sort_values(ascending=False)[:8])
print(recommend)