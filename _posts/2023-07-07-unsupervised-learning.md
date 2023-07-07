---
title: 비지도학습
category:
  - AI
tags:
  - [Aiffel, ML]
math: true
date: 2023-07-07
---

## 비지도학습(Unsupervised Learning)이란?

- 학습 데이터로 정답(lable)이 없는 데이터가 주어지는 학습 방식
- 주어지는 데이터가 어떻게 구성되어 있는지 스스로 알아냄
- 군집화(클러스터링, clustering), 차원 축소(dimensionality reduction), 생성 모델(generative model) 등을 모두 포괄한다





## 클러스터링(1) K-means

- 군집화: 명백한 분류 기준이 없는 상황에서 데이터를 분석하여 가까운(유사한) 것들끼리 묶어주는 것
- K-means 알고리즘
  - k 값이 주어질 때 데이터들을 k개의 클러스터로 묶는 알고리즘
  - 대표적인 클러스터링 기법 중 하나



### K-means 알고리즘의 적용

- 유클리드 거리(Euclidian distance)
  - L2 Distance라고도 부른다
- K-mean 알고리즘의 순서
  1. 원하는 클러스터의 수(K)를 결정한다
  2. K개의 중심점(centroid)를 무작위 선정한다. 이들은 각각의 클러스터를 대표한다.
  3. 나머지 점들은 중심점들과의 유클리드 거리를 계산한 후, 가장 가까운 거리를 가지는 중심점의 클러스터에 속하게 한다.
  4. K개 클러스터의 중심점을 재조정한다. 특정 클러스터에 속하는 모든 점들의 평균값이 해당 클러스터 다음 iteration의 중심점이 된다.<sup>[[1]](#footnote_1)</sup> 
  5. 조정된 중심점과 다른 모든 점들 사이의 유클리드 거리를 계산한 후, 가장 가까운 클러스터에 재배정한다.
  6. 4번과 5번 과정을 반복 수행한다. 반복 횟수는 사용자가 적절히 조정한다. 특정 iteration 이상이 되면 수렴하게 된다.
- K-means 알고리즘이 잘 동작하지 않는 경우
  - 군집의 개수(K값)을 알거나 예측하기 어려운 경우
  - 유클리드 거리를 기반으로 하기 때문에 데이터의 분포에 따라 군집화가 성공적으로 수행되지 않는 경우가 있음
    - 원형으로 분포된 데이터
    - 초승달 모양의 데이터
    - 대각선 모양의 데이터

- 사용법

  ```python
  from sklearn.cluster import KMeans # 모듈 import
  
  kmeans_cluster = KMeans(n_clusters=K) # 클러스터 수가 K개인 k-means 알고리즘 적용
  kmeans_cluster.fit(data) # data에 대하여 K-means iteration 수행
  ```





## 클러스터링(2) DBSCAN

- DBSCAN(Density Based Spatial Clustering of Applications with Noise)

  - 밀도(density) 기반의 군집 알고리즘
  - 군집의 개수를 사전에 지정하지 않아도 됨
  - 밀도가 높은 클러스터를 군집화하는 방법이기 때문에 불특정한 형태의 군집도 찾을 수 있음
  - [DBSCAN 시각화](http://primo.ai/index.php?title=Density-Based_Spatial_Clustering_of_Applications_with_Noise_(DBSCAN))
  - 단점
    - 데이터 수가 많을수록 알고리즘 수행 시간이 급격하게 늘어난다
    - 데이터 분포에 맞는 epsilon과 minPts의 지정이 필요하다

- DBSCAN의 변수와 용어

  - epsilon: 클러스터의 반경
  - minPts: 클러스터를 이루는 개체의 최솟값
  - core point: 반경 epsilon 내에 minPts 개 이상의 점이 존재하는 중심점
  - border point: 군집의 중심이 되지는 못하지만, 군집에 속하는 점
  - noise point: 군집에 포함되지 못하는 점

- DBSCAN 알고리즘의 순서

  1. 임의의 점 p를 설정하고, p 주변 epsilon에 포함된 점의 개수를 센다.
  2. 해당 epsilon에 minPts개 이상의 점이 포함되어 있으면 점 p를 core point로 간주하고 하나의 클러스터로 묶는다
  3. 해당 원에 minPts개 미만의 점이 포함되어 있다면 pass한다
  4. 모든 점에 대해 1번부터 3번까지의 과정을 반복한다. 새로운 core point가 된 점 p'가 기존 클러스터에 속한다면 두 개의 클러스터를 하나의 클러스터로 묶는다.
  5. 모든 점에 대해 클러스터링 과정을 끝내고 클러스터에 속하지 못하는 점이 있다면 noise point로 간주한다. 특정 클러스터에 속하지만 core point가 아닌 점들은 border point라고 한다.

- 사용법

  ```python
  from sklean.cluster import DBSCAN # 모듈 import
  
  epsilon, minPts = 0.2, 3 # 과정에서 사용할 epsilon, minPts값 설정
  circle_dbscan = DBSCAN(eps=epsilon, min_samples=minPts) #DBSCAN setting
  circle_dbscan.fit(data) # data에 대하여 DBSCAN 수행
  ```





## 차원 축소(1) PCA

- 주성분분석(PCA)
  - 데이터 분포의 주성분을 찾아주는 방법
    - 주성분: 데이터의 분산이 가장 큰 방향벡터
  - 데이터들의 분산을 최대로 보존하면서 서로 직교하는 기저(basis)<sup>[[2]](#footnote_2)</sup>들을 찾아 고차원 공간을 저차원 공간으로 사영(projection)
    - 가장 분산이 긴 축을 첫 기저로 잡고 그 기저에 직교하는 축 중 가장 분산이 큰 값을 다음 기저로 잡는다
    - 이렇게 찾은 가장 중요한 기저를 주성분방향, pc축이라고 한다
  - 기존 feature를 선택하는 것이 아닌 선형 결합하는 방식
  - 주로 선형적인 데이터의 분포를 가지고 있을 때 정보가 잘 보존됨





## 차원 축소(2) T-SNE

- T-SNE(T-Stochastic Neighbor Embedding)
  - 시각화에 많이 쓰임
  - 데이터 간 상대적 거리를 보존해 시각화했을 때 숫자들 사이의 경계가 뚜렷함
  - 정보 손실량에 주목하지 않아 저차원 축이 아무런 물리적 의미가 없기 때문에 **시각화**에만 유리하다





## 정리 및 요약

|             |                             장점                             |                             단점                             |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **K-means** | - 군집의 수(K)가 주어졌을 때 빠른 시간에<br>유클리드 거리 기반으로 군집화 수행<br>- 알고리즘이 단순하며 이해하기 쉬움 | - 초기 중심점이 어떻게 주어지느냐에<br /> 따라 결과값이 달라짐<br/>-전체 거리 평균값에 영향을 주어<br/>outlier에 민감 |
| **DBSCAN**  | - 밀도 기반의 군집화 알고리즘으로<br>outlier에 강건<br>- K-means와 달리 초기 중심점 및<br>K값을 설정할 필요가 없음 | - 데이터의 수가 많아질수록<br/>K-means 알고리즘에 비해<br/>오랜 시간이 소요됨<br/>-  epsilon, minPts 등 초기에<br />설정해 주어야 하는 변수 존재 |
|   **PCA**   | - 데이터 분포의 분산을 최대한<br/>유지한 채로 feature의 차원을<br/>줄이는 차원 축소 알고리즘<br/>- 상관관계가 적은 feature를 최대한<br/>배제하고 분산이 최대가 되는<br/>서로 직교하는 기저들을 기준으로<br/>데이터를 나타내기 때문에 raw data<br/>를 사용하는 것보다 정확하고<br/>간결하게 데이터 표현 | 단순히 변환된 축이 최대분산<br />방향과 정렬되도록 좌표 회전을<br />수행하는 것이기 때문에<br />최대분산 방향이 feature의 구분을<br />좋게 한다는 보장이 없음 |
|  **T-SNE**  | - 차원축소 전후 데이터의<br/>상대적 거리를 유지하므로<br/>시각화 결과가 우수 | 차원 축소 과정에서<br />좌표축의 물리적 의미를 무시하므로<br />시각화 이외의 다른<br />분석용도로는 사용하기 어려움 |





<a name="footnote_1">[1]</a>: 이때의 중심점은 실제로 존재하는 데이터가 아니어도 상관없다.

<a name="footnote_2">[2]</a>: 새로운 좌표계 역할을 할 수 있는 벡터의 모음. 우리가 사용하는 좌표값은 기저의 선형결합이다. 예를 들면 (1,1)이라는 좌표는 x방향 1, y방향 1의 선형결합이다.