'''
1. 데이터 읽기
   pandas를 사용하여 Womens Clothing E-Commerce Reviews(수정).csv 데이터를 읽고 dataframe 형태로 저장
'''

>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns


# Womens Clothing E-Commerce Reviews(수정).csv 데이터를 pandas를 사용하여 dataframe 형태로 불러옵니다.
df_origin = pd.read_csv("D:\pandas_jny\data\Womens Clothing E-Commerce Reviews(수정).csv")


# 5개의 데이터 샘플을 출력
df_origin.head()
   Unnamed: 0  Unnamed: 0.1  ...  Department Name  Class Name
0           0             0  ...         Intimate   Intimates
1           1             1  ...          Dresses     Dresses
2           2             2  ...          Dresses     Dresses
3           3             3  ...          Bottoms       Pants
4           4             4  ...             Tops     Blouses

[5 rows x 12 columns]


# dataframe의 정보를 요약해서 출력
df_origin.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 23486 entries, 0 to 23485
Data columns (total 12 columns):
 #   Column                   Non-Null Count  Dtype 
---  ------                   --------------  ----- 
 0   Unnamed: 0               23486 non-null  int64 
 1   Unnamed: 0.1             23486 non-null  int64 
 2   Clothing ID              23486 non-null  int64 
 3   Age                      23486 non-null  int64 
 4   Title                    19676 non-null  object
 5   Review Text              22641 non-null  object
 6   Rating                   23486 non-null  int64 
 7   Recommended IND          23486 non-null  int64 
 8   Positive Feedback Count  23486 non-null  int64 
 9   Division Name            23472 non-null  object
 10  Department Name          23472 non-null  object
 11  Class Name               23472 non-null  object
dtypes: int64(7), object(5)
memory usage: 2.2+ MB
  

# 수치형 변수의 데이터 정보를 요약하여 출력
df_origin.describe()
         Unnamed: 0  Unnamed: 0.1  ...  Recommended IND  Positive Feedback Count
count  23486.000000  23486.000000  ...     23486.000000             23486.000000
mean   11742.500000  11742.500000  ...         0.177638                 2.535936
std     6779.968547   6779.968547  ...         0.382216                 5.702202
min        0.000000      0.000000  ...         0.000000                 0.000000
25%     5871.250000   5871.250000  ...         0.000000                 0.000000
50%    11742.500000  11742.500000  ...         0.000000                 1.000000
75%    17613.750000  17613.750000  ...         0.000000                 3.000000
max    23485.000000  23485.000000  ...         1.000000               122.000000

[8 rows x 7 columns]





'''
2. 데이터 정제
'''

# 결측값 처리 전에,   우선 의미 없는 변수인 'Unnamed: 0, Unnamed: 0.1'를 drop을 사용하여 삭제
df_clean = df_origin.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])


'''
2.1. 결측값 확인
   각 변수별로 결측값이 몇개가 있는지 확인
'''

df_clean.isnull().sum()       # 결측값 정보를 출력
Clothing ID                   0
Age                           0
Title                      3810
Review Text                 845
Rating                        0
Recommended IND               0
Positive Feedback Count       0
Division Name                14
Department Name              14
Class Name                   14
dtype: int64

   
# 아래 3개의 변수들의 결측값 정보를 알아보고 싶어서 그 데이터들을 출력
df_clean[df_clean['Division Name'].isnull()]
       Clothing ID  Age  ... Department Name Class Name
9444            72   25  ...             NaN        NaN
13767          492   23  ...             NaN        NaN
13768          492   49  ...             NaN        NaN
13787          492   48  ...             NaN        NaN
16216          152   36  ...             NaN        NaN
16221          152   37  ...             NaN        NaN
16223          152   39  ...             NaN        NaN
18626          184   34  ...             NaN        NaN
18671          184   54  ...             NaN        NaN
20088          772   50  ...             NaN        NaN
21532          665   43  ...             NaN        NaN
22997          136   47  ...             NaN        NaN
23006          136   33  ...             NaN        NaN
23011          136   36  ...             NaN        NaN

[14 rows x 10 columns]



'''
2.2. 결측값 처리
   Review Text 에 있는 데이터만을 머신러닝 입력으로 사용할 것이므로->   Review Text의 결측값이 있는 샘플을 삭제
'''

# 결측값이 아닌 부분을 골라내어 df_clasn에 저장
df_clean = df_clean[~df_clean['Review Text'].isnull()]


# 결측값 정보를 출력
df_clean.isnull().sum()
Clothing ID                   0
Age                           0
Title                      2966
Review Text                   0
Rating                        0
Recommended IND               0
Positive Feedback Count       0
Division Name                13
Department Name              13
Class Name                   13
dtype: int64
  
  
  


'''
3. 데이터 시각화
   시각화를 통해 각 변수 분포를 알아본다.
   일반적으로는 막대그래프를 그리는 방법으로 시각화를 수행하나,
   문자열로 이루어진 Title 데이터와 Review Text 데이터는 word cloud란 방식을 사용하여 시각화를 수행
'''

'''
3.1. Title word cloud
'''

>>> import nltk
>>> from nltk.corpus import stopwords
>>> from nltk import sent_tokenize, word_tokenize
>>> from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
>>> from collections import Counter
>>> from nltk.tokenize import RegexpTokenizer
>>> import re


# 'Title'의 결측값을 삭제합니다.
df_clean_title = df_clean[~df_clean['Title'].isnull()]

# findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져온다 (소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean_title['Title'].str.lower().str.cat(sep=' '))


nltk.download('stopwords')       # nltk에서 지원하는 'stopwords'를 다운


# 영어 'stopwords'를 가져온다
en_stops = set(stopwords.words('english'))

# tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장
filtered_sentence = [token for token in tokens if not token in en_stops]
filtered_sentence


# 출력 사이즈를 설정
plt.rcParams['figure.figsize'] = (16, 16)

# wordcloud를 저장
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(' '.join(filtered_sentence))

# wordcloud를 출력
plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.show()



'''
3.2. Review Text word cloud
'''

# findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져온다 (소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean['Review Text'].str.lower().str.cat(sep=' '))

# tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장
filtered_sentence = [token for token in tokens if not token in en_stops]
filtered_sentence


# 출력 사이즈를 설정
plt.rcParams['figure.figsize'] = (16, 16)

# wordcloud를 저장
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(filtered_sentence))

# wordcloud를 출력
plt.imshow(wordcloud,interpolation="bilinear")

plt.axis("off")

plt.show()



'''
3.3. Recommended IND 시각화
'''

# 분포를 막대 그래프를 사용하여 출력합니다.
df_clean['Recommended IND'].value_counts().plot(kind='bar')

# 분포를 도수분포표로 확인합니다.
df_clean['Recommended IND'].value_counts()
0    18540
1     4101
Name: Recommended IND, dtype: int64

      
    
    
    
'''
4. 데이터 전 처리
   - 상품 추천 여부 예측을 수행하기 위해서 주어진 이커머스 데이터에 대해서 분류 모델을 사용
   
   - 분류 모델의 필요한 입력 데이터를 준비 하기위해서 다음과 같은 전처리를 수행
1. Review Text 데이터 자연어 전 처리 - Tfidf 활용
2. 학습 데이터와 테스트 데이터로 나누기
'''

'''
4.1. 자연어 전 처리 - Tfidf
'''

>>> from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer을 불러온다 (stop_words 는 영어로 설정)
vectorizer = TfidfVectorizer(stop_words = 'english')

# 소문자화 'Review Text'데이터를 Tfidf로 변환
X = vectorizer.fit_transform(df_clean['Review Text'].str.lower())


# 변환된 X의 크기를 살펴봅니다.
X.shape
(22641, 13855)


# 예측해야 할 변수 'Recommended IND' 만을 선택하여 numpy 형태로 y에 저장
y = df_clean['Recommended IND']
y = y.to_numpy().ravel()      # 1 차원 벡터 형태로 출력하기 위해 ravel 사용

>>> vectorizer.get_feature_names()



'''
4.2. 학습, 테스트 데이터 분리
   머신러닝의 성능을 평가 하기 위해서는 전체 데이터를 학습에 사용하지 않고 학습용 데이터와 테스트용 데이터를 나누어 사용
'''

>>> from sklearn.model_selection import train_test_split

# sklearn에서 제공하는 train_test_split을 사용하여 손 쉽게 분리 할 수 있습니다.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





'''
5. 머신러닝 모델 학습
   전처리된 데이터를 바탕으로 분류 모델을 학습을 수행하고 학습 결과를 출력
   먼저 기본적인 분류 모델인 의사결정나무(Decision Tree)를 사용하여 학습을 수행
'''

'''
5.1. 기본 분류 모델 학습 - 의사결정나무
'''

>>> from sklearn.tree import DecisionTreeClassifier

# 의사결정나무 DecisionTreeClassifier class를 가져온다
model = DecisionTreeClassifier()

# fit 함수를 사용하여 데이터를 학습
model.fit(x_train, y_train)
DecisionTreeClassifier()


# score 함수를 사용하여 모델의 성능을 출력
>>> print(model.score(x_train, y_train))
1.0
>>> print(model.score(x_test, y_test))
0.8032678295429455



'''
5.2. 다양한 분류 모델 학습
   의사결정나무 모델 이외의 다양한 분류 알고리즘을 사용하고 그 성능을 비교
'''

>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.naive_bayes import MultinomialNB
>>> from sklearn.naive_bayes import BernoulliNB
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.svm import SVC
>>> import xgboost as xgb
>>> from xgboost.sklearn import XGBClassifier

>>> models = []
>>> models.append(('KNN', KNeighborsClassifier()))  # KNN 모델
>>> models.append(('NB-M', MultinomialNB()))  # 멀티노미얼 나이브 베이즈
>>> models.append(('NB-B', BernoulliNB()))  # 베르누이 나이브 베이즈 모델
>>> models.append(('RF', RandomForestClassifier()))  # 랜덤포레스트 모델
>>> models.append(('SVM', SVC(gamma='auto')))  # SVM 모델
>>> models.append(('XGB', XGBClassifier()))  # XGB 모델
>>> for name, model in models:
    model.fit(x_train, y_train)
    msg = "%s - train_score : %f, test score : %f" % (name, model.score(x_train, y_train), model.score(x_test, y_test))
    print(msg)
    
KNeighborsClassifier()
KNN - train_score : 0.885987, test score : 0.831972
MultinomialNB()
NB-M - train_score : 0.836020, test score : 0.814749
BernoulliNB()
NB-B - train_score : 0.907078, test score : 0.866416
RandomForestClassifier()
RF - train_score : 1.000000, test score : 0.842129
SVC(gamma='auto')
SVM - train_score : 0.821996, test score : 0.806359
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=100, n_jobs=12,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
XGB - train_score : 0.947659, test score : 0.872378


      
# xgb 모델에서 변수 중요도를 출력
>>> max_num_features = 20
>>> ax = xgb.plot_importance(models[-1][1], height = 1, grid = True, importance_type = 'gain', show_values = False, max_num_features = max_num_features)
>>> ytick = ax.get_yticklabels()
>>> word_importance = []
>>> for i in range(max_num_features):
    word_importance.append(vectorizer.get_feature_names()[int(ytick[i].get_text().split('f')[1])])

>>> ax.set_yticklabels(word_importance)
[Text(0, 0, 'shame'), Text(0, 1, 'fits'), Text(0, 2, 'strange'), Text(0, 3, 'looked'), Text(0, 4, 'perfectly'), Text(0, 5, 'bad'), Text(0, 6, 'returned'), Text(0, 7, 'awful'), Text(0, 8, 'huge'), Text(0, 9, 'poor'), Text(0, 10, 'unfortunately'), Text(0, 11, 'compliments'), Text(0, 12, 'weird'), Text(0, 13, 'disappointing'), Text(0, 14, 'comfortable'), Text(0, 15, 'disappointed'), Text(0, 16, 'perfect'), Text(0, 17, 'returning'), Text(0, 18, 'unflattering'), Text(0, 19, 'cheap')]



>>> plt.rcParams['figure.figsize'] = (10, 15)

>>> plt.xlabel('The F-Score for each features')
Text(0.5, 0, 'The F-Score for each features')

>>> plt.ylabel('Importances')
Text(0, 0.5, 'Importances')

>>> plt.show()



'''
Q. 위 학습된 XGBClassifier 모델에서 feature importance가 30번째인 토큰을 문자열 형태로 구하세요
'''

# plot_importance() 함수의 인자 값을 조절

>>> max_num_features = 30
>>> ax = xgb.plot_importance(models[-1][1], height = 1, grid = True, importance_type = 'gain', show_values = False, max_num_features = max_num_features)
>>> ytick = ax.get_yticklabels()
>>> word_importance = []
>>> for i in range(max_num_features):
    word_importance.append(vectorizer.get_feature_names()[int(ytick[i].get_text().split('f')[1])])

>>> ax.set_yticklabels(word_importance)
[Text(0, 0, 'feminine'), Text(0, 1, 'lay'), Text(0, 2, 'dressed'), Text(0, 3, 'jeans'), Text(0, 4, 'casual'), Text(0, 5, 'odd'), Text(0, 6, 'hopes'), Text(0, 7, 'excited'), Text(0, 8, 'terrible'), Text(0, 9, 'wanted'), Text(0, 10, 'shame'), Text(0, 11, 'fits'), Text(0, 12, 'strange'), Text(0, 13, 'looked'), Text(0, 14, 'perfectly'), Text(0, 15, 'bad'), Text(0, 16, 'returned'), Text(0, 17, 'awful'), Text(0, 18, 'huge'), Text(0, 19, 'poor'), Text(0, 20, 'unfortunately'), Text(0, 21, 'compliments'), Text(0, 22, 'weird'), Text(0, 23, 'disappointing'), Text(0, 24, 'comfortable'), Text(0, 25, 'disappointed'), Text(0, 26, 'perfect'), Text(0, 27, 'returning'), Text(0, 28, 'unflattering'), Text(0, 29, 'cheap')]

>>> plt.rcParams['figure.figsize'] = (10, 15)

>>> plt.xlabel('The F-Score for each features')
Text(0.5, 0, 'The F-Score for each features')

>>> plt.ylabel('Importances')
Text(0, 0.5, 'Importances')

>>> plt.show()

>>> word_importance
['feminine', 'lay', 'dressed', 'jeans', 'casual', 'odd', 'hopes', 'excited', 'terrible', 'wanted', 'shame', 'fits', 'strange', 'looked', 'perfectly', 'bad', 'returned', 'awful', 'huge', 'poor', 'unfortunately', 'compliments', 'weird', 'disappointing', 'comfortable', 'disappointed', 'perfect', 'returning', 'unflattering', 'cheap']





'''
6. 평가 및 예측
   recall 방식을 포함한 또 다른 대표적인 평가 방법에 대해서 알아보고 주어진 데이터에 대해서 예측하는 것을 수행
'''

'''
accuracy의 경우 얼마나 정확히 예측했는가를 정량적으로 나타낸다.
Accuracy    =   Numberofcorrectpredictions  /  Totalnumberofpredictions 

현재 데이터는 추천을 한다(0) 는 데이터가 추천을 하지 않는다(1) 데이터에 비해 월등히 많은 상황
이런 경우, 추천 한다(0)만을 정확히 예측해도 높은 accuracy 값을 가질 수 있다.

   그러므로  이번 실습에서는 또 다른 성능 지표인 recall 값 또한 살펴봐야 한다.
   recall 방식은 추천을 하지 않는다(1) 대비 추천을 한다(0)의 비율을 나타내기에 accuracy에서 놓칠 수 있는 결과 해석을 보충
'''


'''
6.1. Confusion Matrix
   기존 score에서 볼 수 있었던 결과는 accuracy 기반의 결과
   confusion matrix를 출력하여 각 class 별로 예측한 결과에 대해서 자세히 알아본다
'''

>>> from sklearn.metrics import confusion_matrix

# 의사결정나무 모델에 confusion matrix를 사용하기 위하여 테스트 데이터의 예측값을 저장
model_predition = model.predict(x_test)

# sklearn에서 제공하는 confusion_matrix를 사용
cm = confusion_matrix(y_test, model_predition)

# 출력 파트 - seaborn의 heatmap을 사용
plt.rcParams['figure.figsize'] = (5, 5)

>>> sns.set(style = 'dark', font_scale = 1.4)
>>> ax = sns.heatmap(cm, annot=True)

>>> plt.xlabel('Real Data')
Text(0.5, 17.249999999999993, 'Real Data')

>>> plt.ylabel('Prediction')
Text(24.75, 0.5, 'Prediction')

>>> plt.show()

>>> cm
array([[3522,  130],
       [ 448,  429]], dtype=int64)




'''
Q. XGBClassifier 모델에서 평가용 데이터(x_test, y_test)의 confusion matrix는?
'''

# XGBClassifier의 x_test에 대한 예측값을 구하고 confusion_matrix() 를 사용하면 confusion matrix를 구할 수 있다

# 의사결정나무 모델에 confusion matrix를 사용하기 위하여 테스트 데이터의 예측값을 저장
model_predition_xgb = models[-1][1].predict(x_test)

# sklearn에서 제공하는 confusion_matrix를 사용
cm_xgb = confusion_matrix(y_test, model_predition_xgb)



'''
6.2. Precision & Recall
'''

>>> from sklearn.metrics import recall_score
>>> from sklearn.metrics import precision_score


# sklearn에서 제공하는 recall_score, precision_score를 사용하여 recall과 precision 결과물을 출력
>>> print("Recall score: {}".format(recall_score(y_test, model_predition)))
Recall score: 0.48916761687571264
   
>>> print("Precision score: {}".format(precision_score(y_test, model_predition)))
Precision score: 0.7674418604651163
  


'''
6.3. 테스트 데이터의 예측값 출력
'''

# 0번부터 4번까지 5개를 출력
for i in range(5): 
    # 의사결정나무 모델을 사용
    prediction = model.predict(x_test[i])
    print("{} 번째 테스트 데이터 문장: \n{}".format(i, df_clean['Review Text'][i]))
    print("{} 번째 테스트 데이터의 예측 결과: {}, 실제 데이터: {}\n".format(i, prediction[0], y_test[i]))
    
0 번째 테스트 데이터 문장: 
Absolutely wonderful - silky and sexy and comfortable
0 번째 테스트 데이터의 예측 결과: 0, 실제 데이터: 0

1 번째 테스트 데이터 문장: 
Love this dress!  it's sooo pretty.  i happened to find it in a store, and i'm glad i did bc i never would have ordered it online bc it's petite.  i bought a petite and am 5'8".  i love the length on me- hits just a little below the knee.  would definitely be a true midi on someone who is truly petite.
1 번째 테스트 데이터의 예측 결과: 0, 실제 데이터: 0

2 번째 테스트 데이터 문장: 
I had such high hopes for this dress and really wanted it to work for me. i initially ordered the petite small (my usual size) but i found this to be outrageously small. so small in fact that i could not zip it up! i reordered it in petite medium, which was just ok. overall, the top half was comfortable and fit nicely, but the bottom half had a very tight under layer and several somewhat cheap (net) over layers. imo, a major design flaw was the net over layer sewn directly into the zipper - it c
2 번째 테스트 데이터의 예측 결과: 0, 실제 데이터: 1

3 번째 테스트 데이터 문장: 
I love, love, love this jumpsuit. it's fun, flirty, and fabulous! every time i wear it, i get nothing but great compliments!
3 번째 테스트 데이터의 예측 결과: 0, 실제 데이터: 0

4 번째 테스트 데이터 문장: 
This shirt is very flattering to all due to the adjustable front tie. it is the perfect length to wear with leggings and it is sleeveless so it pairs well with any cardigan. love this shirt!!!
4 번째 테스트 데이터의 예측 결과: 0, 실제 데이터: 0


