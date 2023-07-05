---
title: 배열(array)과 표(Table)
Excerpt: "numpy와 pandas로 data 표현하기"
categories:
  - Python
tags:
  - [python, Aiffel, numpy, pandas]
Date: 2023-06-27
Last_modified_at: 2023-06-27
---

# Data를 어떻게 표현하면 좋을까? 배열(array)와 표(table)

## list와 array

- ### List

  - 여러 유형의 데이터 타입을 포함할 수 있다.
  - negative indexing을 지원한다.
  - 대괄호 안에 값을 넣어 사용하며, List의 내용은 파이썬 내장 함수를 사용하여 쉽게 병합하고 복사할 수 있다.

- ### array

  - 동일한 데이터 타입을 포함하는 벡터이다.
  - 배열의 크기는 고정되어 있다.
  - List에 비해 삽입 및 삭제 비용이 높지만 인덱싱이 빠르다.

- ### List와 array의 차이점

  |         목록          |      List      |   array   |
  | :-------------------: | :------------: | :-------: |
  | 다른 데이터 타입 구성 |      가능      |  불가능   |
  |  명시적인 모듈 선언   |     불필요     |   필요    |
  |       산술 연산       | 직접 처리 불가 | 직접 처리 |
  |  적합한 항목 시퀀스   |  짧은 데이터   | 긴 데이터 |
  |        유연성         |      높음      |   낮음    |
  |      명시적 루프      |     불필요     |   필요    |
  |      메모리 크기      |       큼       |   작음    |

- ### 중앙값 코드 구현하기

  ```python
  def median(nums):
    num.sort() #정렬
    size = len(nums)
    p = size // 2
    if size % 2 == 0: #size가 짝수이면
      pr = p
      pl = p-1
      mid = float((nums[pl]+nums[pr])/2)
    else:
      mid = nums[p]
    return mid
  
  print('X: ',X)
  median(X)
  ```

- ### 표준편차 코드 구현하기

  ```python
  def std_dev(nums, avg): #avg는 평균 함수로 구한 평균
    texp = 0.0
    for i in range(len(nums)):
      texp = texp + (nums[i] - avg)**2
    return (texp/len(nums)) ** 0.5
  
  std_dev(X, avg)
  ```

  

> 위와 같은 방법으로 연산하는 함수를 구현할 수 있다. 하지만 **numpy**를 사용하면,
>
> - 빠르고 메모리를 효율적으로 사용하며  다양한 연산을 지원하는 다차원 배열 ndarray 데이터 타입 지원
> - 반복문을 작성할 필요 없이 전체 데이터 배열에 대해 빠른 연산을 제공하는 표준 수학 함수 제공
> - 배열 데이터 디스크에 저장하거나 불러오기 가능
> - 선형대수, 난수발생기, 푸리에 변환 가능
>
> 이런 기능을 편리하게 사용할 수 있다.



## NumPy

- ### install

  ```python
  pip install numpy
  ```

  

- ### import

  ```python
  import numpy as np
  ```

  

- ### ndarray 만들기

  ```python
  A = np.arange(5)
  B = np.array([0,1,2,3,4])
  
  #둘의 결과는 동일하다.
  ```

  

- ### 크기

  ```python
  ndarray.size # 행렬의 모양
  ndarray.ndim # 행렬의 축 개수
  ndarray.shape # 행렬 내 원소의 개수 
  ```

  

- ### reshape()

  ```python
  np.arange(10).reshape(2,5)
  ```

  <span style="color:red"><u>모양과 원소의 개수를 맞춰주어야 한다.</u></span>

  *예) 원소 10개의 ndarray는 (3,3)으로 reshape할 수 없다.*

  

- ### type

  ```python
  A = np.array([0,1,2,3,4,5])
  print(A)
  print(A.dtype) #int64
  print(type(A)) #<class 'numpy.ndarray'>
  
  B = np.array([0,1,2,3,'4',5])
  print(B)
  print(B.dtype) #<U21
  print(type(B)) #<class 'numpy.ndarray'>
  
  C = np.array([0,1,2,3,[4,5],6])
  print(D)
  print(D.dtype) #object
  print(type(D)) #<class 'numpy.ndarray'>
  ```

  - numpy: numpy.array.dtype
    - 동일한 데이터 타입
    - NumPy ndarray **원소**의 데이터 타입 반환
  - Python: type()
    - 행렬 A의 자료형 반환
  - ndarray 안에 list가 들어갔을 때 dtype이 object를 반환함: warning 문구를 통해 경고

    

- ### 특수 행렬

  - 단위행렬

    ```python
    np.eye(3)
    ```

    \[6]:

    ```result
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    ```

  - 0 행렬

    ```python
    np.zeros([2,3])
    ```

    \[7]:

    ```result
    array([[0., 0., 0.],
           [0., 0., 0.]])
    ```

  - 1 행렬

    ```python
    np.ones([3,3])
    ```

    [8]:

    ```result
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])
    ```

​			

- ### [브로드캐스트](https://numpy.org/devdocs/user/basics.broadcasting.html)

  ndarray와 상수 또는 서로 크기가 다른 ndarray끼리 **산술연산**이 가능한 기능

  ```python
  A = np.arange(9).reshape(3,3)
  ```

  [9]:

  ```result
  array([[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]])
  ```

  

  ```python
  A*2
  ```

  [10]:

  ```result
  array([[ 0,  2,  4],
         [ 6,  8, 10],
         [12, 14, 16]])
  ```

  

  ```py
  A+2
  ```

  [11]:

  ```result
  array([[ 2,  3,  4],
         [ 5,  6,  7],
         [ 8,  9, 10]])
  ```

  

  ```python
  A = np.arange(9).reshape(3,3)
  B = np.array([1, 2, 3])
  print("A:", A)
  print("B:", B)
  print("\nA+B:", A+B)
  ```

  ```result
  A: [[0 1 2]
   [3 4 5]
   [6 7 8]]
  B: [1 2 3]
  
  A+B: [[ 1  3  5]
   [ 4  6  8]
   [ 7  9 11]]
  ```

  

  ```python
  A = np.arange(9).reshape(3,3)
  D = np.array([1, 2])
  print("A:", A)
  print("D:", D)
  print("\nA+D:", A+D)
  ```

  ```result
  A: [[0 1 2]
   [3 4 5]
   [6 7 8]]
  D: [1 2]
  ---------------------------------------------------------------------------
  ValueError                                Traceback (most recent call last)
  /tmp/ipykernel_14/3123796488.py in <module>
        4 print("A:", A)
        5 print("D:", D)
  ----> 6 print("\nA+D:", A+D)
  
  ValueError: operands could not be broadcast together with shapes (3,3) (2,) 
  ```

​		(3,3)에 (1,3), (3,3)에 (3,1)은 연산이 가능하지만 **(3,3)에 (1,2)**는 불가능하다.



- ### random

  ```python
  np.random.random() # 0에서 1 사이의 실수형 난수 생성
  np.random.randint(a,b) # a에서 b-1 사이의 정수형 난수 생성
  np.random.choice([list]) # 리스트에 주어진 값 중 하나를 랜덤하게 골라줌
  
  np.random.permutation(a) # 0에서 a-1까지의 리스트의 요소가 무작위로 섞인 배열 생성
  np.random.permutation([list]) # 리스트의 요소가 무작위로 섞인 배열 생성
  
  np.random.normal(loc=m, scale=s, size=n) # 평균 m, 표준편차 s인 정규분포를 따르는 변수 n개 추출
  np.random.uniform(low=d, high=u, size=n) # 최소값 d, 최대값 u인 균등분포를 따르는 변수 n개 추출
  ```

  

- ### 전치행렬

  - arr.T

    행렬의 **행**과 **열**을 맞바꾼다.

    ```python
    A = np.arange(24).reshape(2,3,4)
    print("A: ", A)
    print("A의 전치행렬: ", A.T)
    print("A의 전치행렬의 shape: ", A.T.shape)
    ```

    ```result
    A: [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]
    A의 전치행렬: [[[ 0 12]
      [ 4 16]
      [ 8 20]]
    
     [[ 1 13]
      [ 5 17]
      [ 9 21]]
    
     [[ 2 14]
      [ 6 18]
      [10 22]]
    
     [[ 3 15]
      [ 7 19]
      [11 23]]]
    A의 전치행렬의 shape: (4, 3, 2)
    ```

  - np.transpose

    **축**을 기준으로 행렬의 행과 열을 맞바꾼다.

    ```python
    B = np.transpose(A, (2,0,1))
    print("A: ", A)
    print("B: ", B)
    pirnt("B.shape: ", B.shape)
    ```

    ```result
    A: [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]
    B: [[[ 0  4  8]
      [12 16 20]]
    
     [[ 1  5  9]
      [13 17 21]]
    
     [[ 2  6 10]
      [14 18 22]]
    
     [[ 3  7 11]
      [15 19 23]]]
    B.shape: (4, 2, 3)
    ```



- ### [데이터의 행렬 변환](http://jalammar.github.io/visual-numpy/)

  - 소리 데이터: **1차원 array**로 표현한다. CD음원파일의 경우, 44.1kHz의 샘플링 레이트로 -32767~32768의 정수 값을 갖는다.
  - 흑백 이미지: 이미지 사이즈의 **2차원 ndarray**로 나타내고, 각 원소는 픽셀별로 명도를 0~255의 숫자로 환산하여 표시한다. 0은 검정, 255는 흰색이다.
  - 컬러 이미지: 이미지 사이즈의 **3차원 ndarray**로 나타낸다. 세로X가로X3 형태이며, 3은 RGB 3색을 의미한다.
  - 자연어: 임베딩(Embedding)이라는 과정을 거쳐 ndarray로 표현될 수 있다. 예를 들어 71,290개의 단어가 있는 문장으로 이루어진 데이터셋이 있을 때, 이를 단어별로 나누고 0~71,289로 넘버링한다. 이를 **토큰화 과정**이라고 한다. 이 토큰을 50차원의 [word2vec embedding](https://lovit.github.io/nlp/representation/2018/03/26/word_doc_embedding/)을 통해 [batch_size, sequence_length, embedding_size]의 ndarray로 표현할 수 있다.

  

- ### 이미지의 행렬 변환

  - 픽셀과 이미지

    - 이미지는 수많은 픽셀들로 구성되어 있다.
    - 각각의 픽셀은 RGB 3개 요소의 튜플로 색상이 표시된다.
    - 흑백의 경우에는 Gray 스케일로 나타낸다.
    - Color는 투명도를 나타내는 A를 포함해 RGBA 4개로 표시하기도 한다.
    - 이미지의 좌표는 보통 왼쪽 위를 (0,0)으로 표시한다.

  - 이미지와 관련된 파이썬 라이브러리

    - matplotlib
    - PIL

  - 간단한 이미지 조작

    ```python
    from PIL import Image, ImageColor
    img = Image.open(path) # 이미지 파일 open, dtype: PIL.JpegImagePlugin.JpegImageFile
    W, H = img.size #이미지 가로*세로 사이즈 튜플 값으로 반환
    print((W,H))
    print(img.format) # 이미지 파일 타입
    print(img.mode) # 이미지 색상 정보
    img.crop((30,30,100,100)) # 인자로 가로와 세로의 시작점,종료점 총 4개를 받아 자른다.
    img.save(filename) # 저장
    img_arr = np.array(img) # 이미지를 행렬로 변환
    Image.open(path).convert('L') # convert('L')로 흑백 처리
    ImageColor.getcolor('RED', 'RGB') # 색이 rgb값으로 어떻게 표현되는지 반환
    ```

  

## Pandas

### Pandas의 특징

- NumPy 기반
- 축의 이름에 따라 데이터를 정렬할 수 있음
- 다양한 방식의 indexing
- 시계열 데이터와 비시계열 데이터를 함께 다룰 수 있음
- 누락 데이터 처리 가능
- 데이터를 합치고 관계 연산을 수행할 수 있음



### Series

- 일련의 객체를 담을 수 있는, 1차원 배열과 비슷한 자료 구조

- Series.values

  ```python
  import pandas as pd
  ser = pd.Series(['a','b','c',3])
  ser.values
  ```

  [3]:

  ```result
  array(['a', 'b', 'c', 3], dtype=object)
  ```

- Series.index

  ```python
  ser.index
  ```

  [4]:

  ```result
  RangeIndex(start=0, stop=4, step=1)
  ```

- Series 인덱스 설정

  ```python
  ser2 = pd.Series(['a', 'b', 'c', 3], index=['i','j','k','h']) # Series의 인자로 넣기
  ser2.index = ['i', 'j', 'k', 'h'] # 할당 연산자
  ```

  [5]:

  ```result
  i    a
  j    b
  k    c
  h    3
  dtype: object
  ```

- NumPy와 Series의 차이점

  |      차이점      |            NumPy             |              Series               |
  | :--------------: | :--------------------------: | :-------------------------------: |
  |      인덱싱      | 0으로 시작하는 정수형 인덱스 |            사용자 지정            |
  |   데이터 타입    |       단일 데이터 타입       | 각 요소마다 다른 데이터 타입 가능 |
  | 누락 데이터 처리 |    누락 데이터 표현 안 함    |     누락 데이터 NaN으로 표시      |

  

### DataFrame

- 표(table)과 같은 자료구조

- Series/DataFrame 변환

  ```python
  pd.Series(dataframe) # DataFrame to Series
  pd.DataFrame(series) # Series to DataFrame
  ```

- Pandas 통계 관련 메서드

  ```python
  df.count() # NA 제외한 수 반환
  df.describe() # 요약 통계 계산
  df.argmin() # 최소값을 가지고 있는 값 반환
  df.argmax() # 최댓값을 가지고 있는 값 반환
  df.idxmin() # 최소값을 가지고 있는 인덱스 반환
  df.idxmax() # 최대값을 가지고 있는 인덱스 반환
  df.consum() # 누적합 계산
  df.pct_change() # 퍼센트 변화율 계산
  ```

  
