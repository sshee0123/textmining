import operator
import pickle
from hanspell import spell_checker
import nltk
import pymysql
from konlpy.tag import Okt
from konlpy.utils import pprint
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from collections import Counter
import csv

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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

#부정적 리뷰 크롤링한 파일 가져오기
filename = './training/bed0-bedreview.csv'
file = open(filename, 'r', encoding='utf-8-sig')
df1 = pd.read_csv(file,usecols=[1,2])

#db랑 부정적리뷰 합쳐서 하나의 df만들기
df2= pd.DataFrame(rows)
df2 = pd.DataFrame(rows, columns=['review', 'star'])
df = pd.concat([df1,df2])
print(df)

#별점이 3이상이면 label 1 표시 (긍정) / 나머지는 label 0 표시 (부정)
df['star'] = pd.to_numeric(df['star'])
df['label'] = np.select([df.star > 3], [1], default=0)
print(df['label'].values)

# reviews 열에서 중복인 내용이 있다면 중복 제거
df['star'].nunique(), df['review'].nunique(), df['label'].nunique()
df.drop_duplicates(subset=['review'], inplace=True)

#review, label -> 원하는 데이터로 train / test data 나누기
text_list=df['review'].tolist()
label_list=df['label'].tolist()
hanspell_sent =[]
#한글 맞춤법 검사기
for i in range(len(text_list)):
    spelled_sent = spell_checker.check(text_list[i])
    hanspell_sent.append(spelled_sent.checked)

train_data, test_data,y_train,y_test = train_test_split(hanspell_sent,label_list, test_size=0.25, random_state=42)
print('훈련용 리뷰의 개수 :', len(train_data))
print('테스트용 리뷰의 개수 :', len(test_data))

#train data 토큰화, 정규화
okt = Okt()
values1="".join(str(i) for i in train_data)
lists1=okt.pos(values1,norm=True,stem=True)

#test data 토큰화, 정규화
values2="".join(str(i) for i in test_data)
lists2=okt.pos(values2,norm=True,stem=True)

#/////////////////////////////////////////////////////////////////////#

#명사
train_nounlist1=[]
test_nounlist1=[]
#형용사
train_adjectivelist1=[]
test_adjectivelist1=[]

for i in range(len(lists1)):
    if lists1[i][1]=="Noun":
             train_nounlist1.append(lists1[i][0])
    elif lists1[i][1]=="Adjective":
            train_adjectivelist1.append(lists1[i][0])

for i in range(len(lists2)):
    if lists2[i][1]=="Noun":
             test_nounlist1.append(lists2[i][0])
    elif lists2[i][1]=="Adjective":
            test_adjectivelist1.append(lists2[i][0])


#불용어 제거 후의 명사 형용사 배열
train_nounlist2=[]
test_nounlist2=[]
train_adjectivelist2=[]
test_adjectivelist2=[]

#불용어 파일 불러오기 (불용어 계속 갱신 바람)
stopwordfile = open('stopwords-ko.txt', 'r', encoding='utf-8')
stopwords=[]
for line in stopwordfile.readlines():
    stopwords.append(line.rstrip())

stopwordfile.close()

#명사, 형용사에 불용어 제거
for w in train_nounlist1:
    if w not in stopwords:
        train_nounlist2.append(w)

for w in test_nounlist1:
    if w not in stopwords:
        test_nounlist2.append(w)

for w in train_adjectivelist1:
    if w not in stopwords:
        train_adjectivelist2.append(w)

for w in test_adjectivelist1:
    if w not in stopwords:
        test_adjectivelist2.append(w)

mylist=[]
for w in lists1:
    if w not in stopwords:
        mylist.append(w)

#///////////////////////////// 빈도수로 단어 벡터화 하여 보여주고, 단어사전 생성 //////////////////
#TF-IDF
nounvect1=CountVectorizer()
#nounvect1=nounvect1.fit_transform(train_nounlist2)
# print(nounvect1.fit_transform(train_nounlist2).toarray())
print(nounvect1.fit_transform(train_nounlist2).toarray())
print("훈련 명사해시태그",nounvect1.vocabulary_)
dict1=nounvect1.vocabulary_

#///////////////////////////csv파일로 저장하여 db삽입/////////////////////////////////
nounresult=[]
pd_no='0'
subcate_no='침대'
category_no='침실가구'
for index, (key,elem) in enumerate(dict1.items()):
    nounresult.append([index,key,elem,pd_no,subcate_no,category_no])

noundf=pd.DataFrame(nounresult)
print("명사df",noundf)
noundf.columns=['noun_id','noun_name','noun_frequency','pd_no','subcate_no','category_no']
filename1 = 'bed0-nounhashtag.csv'
noundf.to_csv(filename1,encoding='utf-8-sig',index = True)

# 명사 Top10 출력
noundict=dict(sorted(dict1.items(), key=operator.itemgetter(1),reverse=True)[:10])
print("명사 키워드 결과",noundict)

adjectvect1=CountVectorizer()
#adjectvect1=adjectvect1.fit_transform(train_adjectivelist2)
print(adjectvect1.fit_transform(train_adjectivelist2).toarray())
print("훈련 형용사해시태그",adjectvect1.vocabulary_)
dict2=adjectvect1.vocabulary_

# #csv파일로 저장하여 db삽입
adjectresult=[]
for index, (key,elem) in enumerate(dict2.items()):
    adjectresult.append([index,key,elem,pd_no,subcate_no,category_no])

adjectdf=pd.DataFrame(adjectresult)
adjectdf.columns=['adject_id','adject_name','adject_frequency','pd_no','subcate_no','category_no']
filename2 = 'bed0-adjecthashtag.csv'
adjectdf.to_csv(filename2,encoding='utf-8-sig',index =True)

# 형용사 Top10 출력
adjectdict=dict(sorted(dict2.items(), key=operator.itemgetter(1),reverse=True)[:10])
print("형용사 키워드 결과",adjectdict)

#///////////////////// 훈련 (LogisticRegression, pipeline사용 )/////////////////////////

#문장별 나오는 단어수 카운팅한 수치 + 형용사 벡터화 한 것 -> 훈련시키기
logistic = LogisticRegression(C=10.0,penalty='l2',random_state=0)
pipe = Pipeline([('vect',adjectvect1),('clf',logistic)])
#훈련
pipe.fit(train_data,y_train)
#예측
y_pred=pipe.predict(test_data)
print(accuracy_score(y_test,y_pred))

# 모델 저장
with open('bedpipe.dat','wb') as fp:
    pickle.dump(pipe,fp)
print('저장완료')

#////////////////////////////// 긍정 부정 리뷰 예측하기 //////////////////////
# 모델 객체 복원
with open('bedpipe.dat', 'rb') as fp:
    pipe = pickle.load(fp)

    #test데이터로 리뷰 긍정, 부정 예측하기
    for i in range(len(test_data)):
        text = test_data[i]
        str = [text]
        # 예측 정확도
        r1 = np.max(pipe.predict_proba(str) * 100)
        # 예측 결과
        r2 = pipe.predict(str)[0]

        if r2 == 0:
            print(str)
            print('부정적 리뷰입니다.')
        elif r2 == 1:
            print(str)
            print('긍정적 리뷰입니다.')
        print('정확도 : %.3f' % r1)

# Connection 닫기
conn.close()
