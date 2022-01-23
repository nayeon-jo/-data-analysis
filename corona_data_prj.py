# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 21:36:43 2021

@author: jony
"""
"""
1. 데이터 읽기
    필요한 패키지 설치 및 import한 후 pandas를 사용하여 <데이터를 읽고 어떠한 데이터가 저장되어 있는지 확인>
"""

"""
1.1. 데이터 불러오기
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# pd.read_csv를 통하여 dataframe 형태로 읽어온다
corona_all=pd.read_csv("D:\pandas_jny\data\서울시 코로나19 확진자 현황.csv")
corona_all.head()          # 상위 5개 데이터를 출력
corona_all.info()          # dataframe 정보를 출력





"""
2. 데이터 정제
    결측값(missing data), 이상치(outlier)를 처리하는 데이터 정제 과정
"""

"""
2.1. 비어있는(데이터가 존재하지 않는) column 지우기
"""

corona_del_col = corona_all.drop(columns = ['국적','환자정보','조치사항'])
corona_del_col.info()          # 정제 처리된 dataframe 정보를 출력





"""
3. 데이터 시각화
    corona_del_col 데이터를 바탕으로 각 column의 변수별로 어떠한 데이터 분포를 하고 있는지 시각화를 통하여 알아본다
"""

"""
3.1. 확진일 데이터 전처리하기
    월별, 일별 분석을 위해서는 문자열 형식의 데이터를 나누어 숫자 형 데이터로 변환해야한다
"""

#corona_del_col['확진일']
# `확진일` 데이터를 나누어     month, day column에 int64 형태로 저장  


# dataframe에 추가하기 전, 임시로 데이터를 저장해 둘 list를 선언
month = []
day = []
for data in corona_del_col['확진일']:
    # split 함수를 사용하여 월, 일을 나누어 list에 저장
    month.append(data.split('.')[0])
    day.append(data.split('.')[1])
    
    
# corona_del_col에 `month`, `day` column을 생성하며 동시에 list에 임시 저장된 데이터를 입력
corona_del_col['month'] = month
corona_del_col['day'] = day

corona_del_col['month'].astype('int64')
corona_del_col['day'].astype('int64')



"""
3.2. 월별 확진자 수 출력
    나누어진 month의 데이터를 바탕으로 달별 확진자 수를 막대그래프로 출력
"""

# 그래프에서 x축의 순서를 정리하기 위하여 order list를 생성
order = []
for i in range(1,11):
    order.append(str(i))
order


# 그래프의 사이즈를 조절
plt.figure(figsize=(10,5))

# seaborn의 countplot 함수를 사용하여 출력
sns.set(style="darkgrid")
ax = sns.countplot(x="month", data=corona_del_col, palette="Set3", order = order)


# series의 plot 함수를 사용한 출력 방법도 있다
# corona_del_col['month'].value_counts().plot(kind='bar')


# value_counts()는 각 데이터를 세어서 내림차순으로 정리하는 함수입니다.
corona_del_col['month'].value_counts()



"""
3.3. 8월달 일별 확진자 수 출력
    확진자 수가 가장 많았던 8월에 확진자 수가 어떻게 늘었는지 일별 확진자 수를 막대그래프로 출력
"""

# 그래프에서 x축의 순서를 정리하기 위하여 order list를 생성
order2 = []
for i in range(1,32):
    order2.append(str(i))
order2
    
    
# seaborn의 countplot 함수를 사용하여 출력
plt.figure(figsize=(20,10))
#sns.set(style="darkgrid")
ax = sns.countplot(x="day", data=corona_del_col[corona_del_col['month'] == '8'], palette="rocket_r", order = order2)


#8월 평균 일별 확진자 수는?
corona_del_col[corona_del_col['month'] == '8']['day'].count()/31
corona_del_col[corona_del_col['month'] == '8']['day'].value_counts().mean()



"""
3.4. 지역별 확진자 수 출력
"""

corona_del_col['지역']
#지역 데이터는 oo구 형태의 문자열 데이터


#지역별로 확진자가 얼마나 있는지 막대그래프로 출력
import matplotlib.font_manager as fm

font_dirs = ['/usr/share/fonts/truetype/nanum', ]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)
    
    
plt.figure(figsize=(20,10))
# 한글 출력을 위해서 폰트 옵션을 설정
sns.set(font="NanumBarunGothic", 
       rc={"axes.unicode_minus":False},
       style='darkgrid')
ax = sns.countplot(x="지역", data=corona_del_col, palette="Set2")


"""
지역 이상치 데이터 처리
    - 종랑구라는 잘못된 데이터와 한국이라는 지역과는 맞지 않는 데이터가 있다
    - 종랑구 -> 중랑구, 한국 -> 기타로 데이터를 변경
"""

# replace 함수를 사용하여 해당 데이터를 변경
# 이상치가 처리된 데이터이기에 새로운 Dataframe으로 저장
corona_out_region = corona_del_col.replace({'종랑구':'중랑구', '한국':'기타'})

# 이상치가 처리된 데이터를 다시 출력
plt.figure(figsize=(20,10))
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
ax = sns.countplot(x="지역", data=corona_out_region, palette="Set2")



'''
3.5. 8월달 지역별 확진자 수 출력
    감염자가 많았던 8월에는 지역별로 확진자가 어떻게 분포되어 있는지 막대그래프로 출력
'''

# 논리연산을 이용한 조건을 다음과 같이 사용하면 해당 조건에 맞는 데이터를 출력할 수 있다
corona_out_region[corona_del_col['month'] == '8']

# 그래프를 출력
plt.figure(figsize=(20,10))
sns.set(font="NanumBarunGothic", 
        rc={"axes.unicode_minus":False},
        style='darkgrid')
ax = sns.countplot(x="지역", data=corona_out_region[corona_del_col['month'] == '8'], palette="Set2")



'''
3.6. 월별 관악구 확진자 수 출력
    확진자가 가장 많았던 관악구 내의 확진자 수가 월별로 어떻게 증가했는지 그 분포를 막대그래프로 출력
'''

# 해당 column을 지정하여 series 형태로 출력
corona_out_region['month'][corona_out_region['지역'] == '관악구']

# 그래프를 출력
plt.figure(figsize=(10,5))
sns.set(style="darkgrid")
ax = sns.countplot(x="month", data=corona_out_region[corona_out_region['지역'] == '관악구'], palette="Set2", order = order)



'''
3.7. 서울 지역에서 확진자를 지도에 출력
'''

# 지도 출력을 위한 라이브러리 folium을 import
import folium

# Map 함수를 사용하여 지도를 출력
map_osm = folium.Map(location=[37.529622, 126.984307], zoom_start=10)
map_osm


#지역마다 지도에 정보를 출력하기 위해서는 각 지역의 좌표정보가 필요
# 서울시 행정구역 시군 정보 데이터를 불러와 사용

# CRS에 저장
CRS=pd.read_csv("D:\pandas_jny\data\서울시 행정구역 시군구 정보 (좌표계_ WGS1984).csv")
# Dataframe을 출력
CRS


CRS[CRS['시군구명_한글'] == '중구']     #저장된 데이터에서 지역명이 중구인 데이터를 뽑는다


# corona_out_region의 지역에는 'oo구' 이외로 `타시도`, `기타`에 해당되는 데이터가 존재 
# 위 데이터에 해당되는 위도, 경도를 찾을 수 없기에 삭제하여 corona_seoul로 저장
corona_seoul = corona_out_region.drop(corona_out_region[corona_out_region['지역'] == '타시도'].index)
corona_seoul = corona_seoul.drop(corona_out_region[corona_out_region['지역'] == '기타'].index)

# 서울 중심지 중구를 가운데 좌표로 잡아 지도를 출력
map_osm = folium.Map(location=[37.557945, 126.99419], zoom_start=11)


#for 문을 사용하여 지역마다 확진자를 원형 마커를 사용하여 지도에 출력

# 지역 정보를 set 함수를 사용하여 25개 고유의 지역을 뽑아낸다
for region in set(corona_seoul['지역']):

    # 해당 지역의 데이터 개수를 count에 저장
    count = len(corona_seoul[corona_seoul['지역'] == region])
    # 해당 지역의 데이터를 CRS에서 뽑아냄
    CRS_region = CRS[CRS['시군구명_한글'] == region]

    # CircleMarker를 사용하여 지역마다 원형마커를 생성
    marker = folium.CircleMarker([CRS_region['위도'], CRS_region['경도']], # 위치
                                  radius=count/10 + 10,                 # 범위
                                  color='#3186cc',            # 선 색상
                                  fill_color='#3186cc',       # 면 색상
                                  popup=' '.join((region, str(count), '명'))) # 팝업 설정
    
    # 생성한 원형마커를 지도에 추가
    marker.add_to(map_osm)

map_osm


# 6월에 확진자가 가장 많이 나온 지역은?
top=corona_out_region[corona_del_col['month'] == '6']['지역'].value_counts()
top.index[0]







