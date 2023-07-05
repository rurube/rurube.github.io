---
title: Regularization
category:
  - AI
tags:
  - [Aiffel, DL]
date: 2023-07-04
last_modified_at: 2023-07-04
math: true
---



## Regularization과 Normalization

- Regularization
  - **정칙화**
  - 과적합을 해결할 수 있는 방법 중 하나
  - 모델에 제약 조건을 걸어 모델의 train loss를 증가시키고, 그에 따라 validation loss나 test loss를 감소시킨다
  - L1 Regularization, L2 Regularization, Dropout, Batch normalization
- Normalization
  - **정규화**
  - 데이터 전처리 과정
  - z-score, minmax scaler



## L1 Regularization(Lasso)

​	L1 regularization은 다음과 같이 정의된다.
​    
$\hat{\beta}^{lasso}:=argmin_{\beta}{1\over2N}\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^Dx_{ij}\beta_j)^2+\lambda\sum_{j=1}^D\left|\beta_j\right|$

여기서 linear regression과 다른 부분은 아래와 같다.

$\lambda\sum_{j=1}^D\left|\beta_j\right|$
​	

이 부분이 L1 norm에 해당하는 부분인데, L2 regularization과의 차이가 나타나는 중요한 부분이다.

> Lp norm?
>
> norm은 벡터나 행렬, 함수 등의 거리를 나타내는 것으로 Lp norm의 정의는 다음과 같다. [Norm(mathematics)](https://en.wikipedia.org/wiki/Norm_(mathematics))
> $\lVert x \rVert_p:=(\sum_{i=1}^n\lvert x_i\rvert ^P)^{1/p}$

p =1인 경우 L1 norm은 다음과 같이 나타낼 수 있는데,

$\lVert x \rVert_1=\sum_{i=1}^n\lvert x_i\rvert$

이는 (2)의 식과 일치한다. 때문에 p=1이므로 L1 regularization이라고 부르는 것이다.



## L2 Regularization(Ridge)

​	L2 regularization은 아래와 같이 정의된다.   
​    
$\hat{\beta}^{ridge}:=argmin_{\beta}{1\over 2N}\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^Dx_{ij}\beta_j)^2+\lambda\sum_{j=1}^D\beta_j^2$

L2 regularization의 핵심 내용은 다음 부분이다.

$\lambda\sum_{j=1}^D\beta_j^2$

그리고 (6)의 식은 p=2일 때의 L2 norm의 형태와 일치한다. L2 regularization의 이름도 이러한 특징에서 왔다.



## L1/L2 Regularization의 차이점

![l1andl2](https://d3s0tskafalll9.cloudfront.net/media/images/F-46-1.max-800x600.png)

|                                  |        L1 regularization         |         L2 regularization         |
| :------------------------------: | :------------------------------: | :-------------------------------: |
|            제약 조건             |           마름모 형태            |              원 형태              |
| 정답 파라미터 집합과 만나는 지점 |           대체로 축 위           |           대체로 축 밖            |
|           정칙화 결과            | 일부 coefficient 값이 0으로 변함 | coefficient값이 0으로 변하지 않음 |
|               특징               |     차원 축소와 비슷한 역할      |         수렴 속도가 빠름          |

## Lp norm

### Vector norm

​	L1/L2 regularization의 norm은 벡터에서 정의된 norm으로 다음과 같다.
​    
$\lVert x\rVert_p:=(\sum_{i=1}^nx_i^p)^{1/p}$

아래 코드를 이용해 p의 값과 x의 형태를 바꾸어가며 실험해보자.

```python
import numpy as np

def playnorm(x, p):
  norm_x = np.linalg.norm(x, ord=p)
	making_norm = (sum(x**p))**(1/p)
	print(f'result of numpy package norm function: {norm_x:.5f}')
	print(f'result of making norm: {making_norm:.5f}')
```

```python
x = np.array([1,10,1,1,1])
p = 5
playnorm(x,p)
```

```result
result of numpy package norm function : 10.00008
result of making norm : 10.00008 
```

```python
x = np.array([1,10,1,1,7])
p = 9
playnorm(x,p)
```

```result
result of numpy package norm function : 10.04405
result of making norm : 10.04405 
```

​	p가 무한대인 infinity norm의 경우는 x에서 가장 큰 숫자를 출력한다.
​    
$\lVert x\rVert_\infty:=max(x)$

```python
x = np.array([1,10,1,1,7])

norm_x = np.linalg.norm(x, ord=np.inf)
print(f'result of infinite norm : {norm_x:.5f}')
```

```result
result of infinite norm : 10.00000 
```



### Matrix norm

행렬의 norm의 경우에는 주로 p가 1이거나 무한대일 때를 많이 접하게 된다. A가 mxn 행렬이라고 할 때,

$\lVert A\rVert_1=\max_{1\leq j\leq n}\sum_{i=1}^m\lvert a_{ij}\rvert$

$\lVert A\rVert_{\infty}=\max_{1\leq i\leq m}\sum_{j=1}^n\lvert a_{ij}\rvert$

p가 1일 때는 column의 합이 가장 큰 값이 출력되고, p가 무한대일 때는 row의 합이 가장 큰 값이 출력된다.

```python
A = np.array([[1,2,3], [1,2,3], [4,6,8]])

one_norm_A = np.linalg.norm(A, ord=1)
print('result one norm of A :', one_norm_A)

inf_norm_A = np.linalg.norm(A, ord=np.inf)
print('result inf norm of A :', inf_norm_A)
```

```result
result one norm of A : 14.0
result inf norm of A : 18.0
```



## Dropout

[Dropout: A simple Way to Prevent Neural Networks from Overfitting(2014)](https://jmlr.org/papers/v15/srivastava14a.html)

- 확률적으로 랜덤하게 몇 가지의 뉴런만 선택하여 정보 전달
- 과적합을 막는 regularization layer중 하나
- 확률을 너무 높이면 값이 잘 전달되지 않아 학습이 잘 되지 않고 너무 낮추면 fully connected layer와 같이 동작하면서 과적합이 생길 수 있다.
- [Keras Dropout](https://keras.io/api/layers/regularization_layers/dropout/)



## Batch Normalization

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift(2015)](https://arxiv.org/pdf/1502.03167.pdf)

#### 기존의 Mini-batch Gradient Descent

- 데이터셋 전체를 본 다음 업데이트하는 Batch Gradient Descent와 데이터 하나마다 업데이트하는 Stochastic Gradient Descent의 절충안
- 데이터셋을 여러 개의 mini-batch로 쪼갠 다음 하나의 batch를 처리할 때마다 가중치를 업데이트한다.
- 학습 속도가 빠르고 안정성이 높다
- 딥러닝 모델 안에서 데이터가 처리되면서 mini-batch들 사이에 **데이터 분포의 차이**가 생길 수 있다
- 데이터 분포의 차이가 생기면?
  - gradient값의 차이가 생긴다
  - gradient vanishing이나 gradient explode가 생길 수 있다

#### Batch Normalization

- 각 mini-batch마다 평균과 분산을 계산하여 정규화를 수행한다.

- scale and shift 변환을 적용하여 각 batch들이 비슷한 데이터 분포를 가지도록 한다.

- mini-batch의 평균($\mu_B$)과 분산($\sigma^2_B$)을 구해서 입력 데이터를 정규화하고, 이 값에 scale($\gamma$)과 shift($\beta$)를 추가한 것이다. 결국 입력 데이터($x_i$)는 batch normalization을 거쳐 $y_i(=\gamma \hat{x_i}+\beta)$이 된다. 수식은 아래와 같다.

  **Input**: Values of x over a mini-batch: $B={x_1,\dots ,x_m}$ Parameters to be learned: $\gamma , \beta$

  **Output**: ${y_i=BN_{\gamma , \beta}(x_i)}$

  1. **mini-batch mean:** $\mu_B \gets {1 \over m}\sum_{i=1}^mx_i$
  2. **mini-batch variance**: $\sigma_B^2\gets {1\over m}\sum_{i=1}^m(x_i-\mu_B)$
  3. **normalize:** $\hat{x_i}\gets \frac{x_i-\mu_B}{sqrt{\sigma^2_B + \epsilon}}$
  4. **scale and shift:** $y_i\gets \gamma \hat{x_i} + \beta \equiv BN_{\gamma ,\beta}(x_i)$

  - 중간에 $\epsilon$은 분산이 0일 경우 나눗셈 오류를 방지하기 위한 것이다.

  - $\gamma$와 $\beta$값은 학습 파라미터로 모델 학습이 진행되면서 가중치와 함께 업데이트된다.

    

    
