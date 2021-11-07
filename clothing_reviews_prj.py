>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns
Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    import seaborn as sns
ModuleNotFoundError: No module named 'seaborn'
>>> import seaborn as sns
>>> # Womens Clothing E-Commerce Reviews(수정).csv 데이터를 pandas를 사용하여 dataframe 형태로 불러옵니다.
df_origin = pd.read_csv("D:\pandas_jny\data\Womens Clothing E-Commerce Reviews(수정).csv")
>>> 
>>> 
>>> # 5개의 데이터 샘플을 출력합니다.
df_origin.head()
   Unnamed: 0  Unnamed: 0.1  ...  Department Name  Class Name
0           0             0  ...         Intimate   Intimates
1           1             1  ...          Dresses     Dresses
2           2             2  ...          Dresses     Dresses
3           3             3  ...          Bottoms       Pants
4           4             4  ...             Tops     Blouses

[5 rows x 12 columns]
>>> # dataframe의 정보를 요약해서 출력합니다.
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
>>> # 수치형 변수의 데이터 정보를 요약하여 출력합니다.
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
>>> 
>>> 
>>> # 결측값을 처리하기 전에 우선 의미 없는 변수인 'Unnamed: 0, Unnamed: 0.1'를 drop을 사용하여 삭제합니다.
df_clean = df_origin.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])
>>> # 결측값 정보를 출력합니다.
df_clean.isnull().sum()
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
>>> 
>>> # 아래 3개의 변수들의 결측값 정보를 알아보고 싶어서 그 데이터들을 출력합니다.
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
>>> # 결측값이 아닌 부분을 골라내어 df_clasn에 저장합니다.
df_clean = df_clean[~df_clean['Review Text'].isnull()]
>>> # 결측값 정보를 출력합니다.
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
>>> import nltk
Traceback (most recent call last):
  File "<pyshell#21>", line 1, in <module>
    import nltk
ModuleNotFoundError: No module named 'nltk'
>>> import nltk
>>> from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
SyntaxError: multiple statements found while compiling a single statement
>>> from nltk.corpus import stopwords
>>> from nltk import sent_tokenize, word_tokenize
>>> from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
Traceback (most recent call last):
  File "<pyshell#26>", line 1, in <module>
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
ModuleNotFoundError: No module named 'wordcloud'
>>> from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
Traceback (most recent call last):
  File "<pyshell#27>", line 1, in <module>
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
ModuleNotFoundError: No module named 'wordcloud'
>>> from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
>>> from collections import Counter
>>> from nltk.tokenize import RegexpTokenizer
>>> import re
>>> # 'Title'의 결측값을 삭제합니다.
df_clean_title = df_clean[~df_clean['Title'].isnull()]

# findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져옵니다.(소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean_title['Title'].str.lower().str.cat(sep=' '))
>>> # nltk에서 지원하는 'stopwords'를 다운받습니다.
nltk.download('stopwords')
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\i\AppData\Roaming\nltk_data...
[nltk_data]   Unzipping corpora\stopwords.zip.
True
>>> # 영어 'stopwords'를 가져옵니다.
en_stops = set(stopwords.words('english'))

# tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장합니다.
filtered_sentence = [token for token in tokens if not token in en_stops]
filtered_sentence
>>> filtered_sentence
Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    filtered_sentence
NameError: name 'filtered_sentence' is not defined
>>> # 출력 사이즈를 설정합니다.
plt.rcParams['figure.figsize'] = (16, 16)

# wordcloud를 저장합니다.
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(' '.join(filtered_sentence))

# wordcloud를 출력합니다.
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis("off")
plt.show()
>>> plt.show()
>>> # 'Title'의 결측값을 삭제합니다.
df_clean_title = df_clean[~df_clean['Title'].isnull()]
>>> # findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져옵니다.(소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean_title['Title'].str.lower().str.cat(sep=' '))
>>> # 영어 'stopwords'를 가져옵니다.
en_stops = set(stopwords.words('english'))
>>> # tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장합니다.
filtered_sentence = [token for token in tokens if not token in en_stops]
>>> filtered_sentence

>>> # 출력 사이즈를 설정합니다.
plt.rcParams['figure.figsize'] = (16, 16)
>>> # wordcloud를 저장합니다.
wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(' '.join(filtered_sentence))
>>> # wordcloud를 출력합니다.
plt.imshow(wordcloud,interpolation="bilinear")
<matplotlib.image.AxesImage object at 0x000001D586AD5E20>
>>> plt.axis("off")
(-0.5, 399.5, 199.5, -0.5)
>>> plt.show()
>>> # findall 함수를 사용하여 띄어 쓰기 단위로 글자만을 가져옵니다.(소문자로 변환도 수행)
tokens = re.findall("[\w']+", df_clean['Review Text'].str.lower().str.cat(sep=' '))
>>> # tokens에서 'stopwords'에 해당되지 않는 단어를 골라내어 filtered_sentence에 저장합니다.
filtered_sentence = [token for token in tokens if not token in en_stops]
>>> filtered_sentence

>>> # 출력 사이즈를 설정합니다.
plt.rcParams['figure.figsize'] = (16, 16)
>>> # wordcloud를 저장합니다.
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(filtered_sentence))
>>> # wordcloud를 출력합니다.
plt.imshow(wordcloud,interpolation="bilinear")
<matplotlib.image.AxesImage object at 0x000001D593A6EF70>
>>> plt.axis("off")
(-0.5, 399.5, 199.5, -0.5)
>>> plt.show()
>>> # 분포를 막대 그래프를 사용하여 출력합니다.
df_clean['Recommended IND'].value_counts().plot(kind='bar')
<AxesSubplot:>
>>> # 분포를 도수분포표로 확인합니다.
df_clean['Recommended IND'].value_counts()
0    18540
1     4101
Name: Recommended IND, dtype: int64
>>> from sklearn.feature_extraction.text import TfidfVectorizer
Traceback (most recent call last):
  File "<pyshell#58>", line 1, in <module>
    from sklearn.feature_extraction.text import TfidfVectorizer
ModuleNotFoundError: No module named 'sklearn'
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> # TfidfVectorizer을 불러옵니다. (stop_words 는 영어로 설정)
vectorizer = TfidfVectorizer(stop_words = 'english')

>>> # 소문자화 'Review Text'데이터를 Tfidf로 변환합니다.
X = vectorizer.fit_transform(df_clean['Review Text'].str.lower())
>>> # 변환된 X의 크기를 살펴봅니다.
X.shape
(22641, 13855)
>>> # 예측해야 할 변수 'Recommended IND' 만을 선택하여 numpy 형태로 y에 저장합니다.
y = df_clean['Recommended IND']
>>> y = y.to_numpy().ravel() # 1 차원 벡터 형태로 출력하기 위해 ravel 사용
>>> vectorizer.get_feature_names()

Warning (from warnings module):
  File "C:\Users\i\AppData\Local\Programs\Python\Python39\lib\site-packages\sklearn\utils\deprecation.py", line 87
    warnings.warn(msg, category=FutureWarning)
FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.

>>> from sklearn.model_selection import train_test_split
>>> # sklearn에서 제공하는 train_test_split을 사용하여 손 쉽게 분리 할 수 있습니다.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
>>> from sklearn.tree import DecisionTreeClassifier
>>> # 의사결정나무 DecisionTreeClassifier class를 가져 옵니다.
model = DecisionTreeClassifier()
>>> # fit 함수를 사용하여 데이터를 학습합니다.
model.fit(x_train, y_train)
DecisionTreeClassifier()
>>> # score 함수를 사용하여 모델의 성능을 출력합니다.
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
SyntaxError: multiple statements found while compiling a single statement
>>> # score 함수를 사용하여 모델의 성능을 출력합니다.
print(model.score(x_train, y_train))
1.0
>>> print(model.score(x_test, y_test))
0.8032678295429455
>>> 
>>> 
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from sklearn.naive_bayes import MultinomialNB
>>> from sklearn.naive_bayes import BernoulliNB
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.svm import SVC
>>> import xgboost as xgb
Traceback (most recent call last):
  File "<pyshell#81>", line 1, in <module>
    import xgboost as xgb
ModuleNotFoundError: No module named 'xgboost'
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

Warning (from warnings module):
  File "C:\Users\i\AppData\Local\Programs\Python\Python39\lib\site-packages\xgboost\sklearn.py", line 1224
    warnings.warn(label_encoder_deprecation_msg, UserWarning)
UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
[20:09:52] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.5.0/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
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


>>> # xgb 모델에서 변수 중요도를 출력합니다.
>>> max_num_features = 20
>>> ax = xgb.plot_importance(models[-1][1], height = 1, grid = True, importance_type = 'gain', show_values = False, max_num_features = max_num_features)
>>> ytick = ax.get_yticklabels()
>>> word_importance = []
>>> for i in range(max_num_features):
    word_importance.append(vectorizer.get_feature_names()[int(ytick[i].get_text().split('f')[1])])

    

Warning (from warnings module):
  File "C:\Users\i\AppData\Local\Programs\Python\Python39\lib\site-packages\sklearn\utils\deprecation.py", line 87
    warnings.warn(msg, category=FutureWarning)
FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
>>> ax.set_yticklabels(word_importance)
[Text(0, 0, 'shame'), Text(0, 1, 'fits'), Text(0, 2, 'strange'), Text(0, 3, 'looked'), Text(0, 4, 'perfectly'), Text(0, 5, 'bad'), Text(0, 6, 'returned'), Text(0, 7, 'awful'), Text(0, 8, 'huge'), Text(0, 9, 'poor'), Text(0, 10, 'unfortunately'), Text(0, 11, 'compliments'), Text(0, 12, 'weird'), Text(0, 13, 'disappointing'), Text(0, 14, 'comfortable'), Text(0, 15, 'disappointed'), Text(0, 16, 'perfect'), Text(0, 17, 'returning'), Text(0, 18, 'unflattering'), Text(0, 19, 'cheap')]
>>> plt.rcParams['figure.figsize'] = (10, 15)
>>> plt.xlabel('The F-Score for each features')
Text(0.5, 0, 'The F-Score for each features')
>>> plt.ylabel('Importances')
Text(0, 0.5, 'Importances')
>>> plt.show()
>>> 
>>> 
>>> Q. 위 학습된 XGBClassifier 모델에서 feature importance가 30번째인 토큰을 문자열 형태로 구하세요.
SyntaxError: invalid syntax
>>> #  Q. 위 학습된 XGBClassifier 모델에서 feature importance가 30번째인 토큰을 문자열 형태로 구하세요.
>>> # plot_importance() 함수의 인자 값을 조절하면 쉽게 구할 수 있습니다.
>>> 
>>> max_num_features = 30
ax = xgb.plot_importance(models[-1][1], height = 1, grid = True, importance_type = 'gain', show_values = False, max_num_features = max_num_features)
ytick = ax.get_yticklabels()
word_importance = []
for i in range(max_num_features):
    word_importance.append(vectorizer.get_feature_names()[int(ytick[i].get_text().split('f')[1])])

ax.set_yticklabels(word_importance)

plt.rcParams['figure.figsize'] = (10, 15)
plt.xlabel('The F-Score for each features')
plt.ylabel('Importances')
plt.show()
SyntaxError: multiple statements found while compiling a single statement
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
>>> # str형으로 문자열만 저장합니다. 예시: quiz_1 = 'cheap'
quiz_1 = 'feminine'
>>> 
>>> 
>>> from sklearn.metrics import confusion_matrix
>>> # 의사결정나무 모델에 confusion matrix를 사용하기 위하여 테스트 데이터의 예측값을 저장합니다.
model_predition = model.predict(x_test)
>>> # sklearn에서 제공하는 confusion_matrix를 사용합니다.
cm = confusion_matrix(y_test, model_predition)
>>> # 출력 파트 - seaborn의 heatmap을 사용
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
>>> # Q. XGBClassifier 모델에서 평가용 데이터(x_test, y_test)의 confusion matrix를 구하세요.
>>> # XGBClassifier의 x_test에 대한 예측값을 구하고 confusion_matrix() 를 사용하면 confusion matrix를 구할 수 있습니다.
>>> # 의사결정나무 모델에 confusion matrix를 사용하기 위하여 테스트 데이터의 예측값을 저장합니다.
model_predition_xgb = models[-1][1].predict(x_test)
>>> # sklearn에서 제공하는 confusion_matrix를 사용합니다.
cm_xgb = confusion_matrix(y_test, model_predition_xgb)
>>> # confusion_matrix() 결과값을 저장합니다. 
quiz_2 = cm_xgb
>>> quiz_2
array([[3522,  130],
       [ 448,  429]], dtype=int64)
>>> 
>>> 
>>> 
>>> 
>>> from sklearn.metrics import recall_score
>>> from sklearn.metrics import precision_score
>>> 
# sklearn에서 제공하는 recall_score, precision_score를 사용하여 recall과 precision 결과물을 출력합니다.
>>> print("Recall score: {}".format(recall_score(y_test, model_predition)))
Recall score: 0.48916761687571264
>>> print("Precision score: {}".format(precision_score(y_test, model_predition)))
Precision score: 0.7674418604651163
>>> # 0번부터 4번까지 5개를 출력해보겠습니다.
for i in range(5): 
    
    # 의사결정나무 모델을 사용하였습니다.
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

>>> 
