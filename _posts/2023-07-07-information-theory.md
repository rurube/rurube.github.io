---
title: 정보이론
category:
  - AI
tags:
  - [Aiffel]
math: true
date: 2023-07-06
---

## Information Content

- **정보 이론**(information content)이란 추상적인 '정보'라는 개념을 정량화하고 정보의 저장과 통신을 연구하는 분야이다.

- 정보를 정량적으로 표현하기 위해 필요한 세 가지 조건

  - 일어날 가능성이 높은 사건은 정보량이 낮고, 반드시 일어나는 사건에는 정보가 없는 것이나 마찬가지이다
  - 일어날 가능성이 낮은 사건은 정보량이 높다
  - 두 개의 독립적인 사건이 있을 때, 전체 정보량은 각각의 정보량을 더한 것과 같다

- 정보량은 한 가지 사건에 대한 값이다

- 사건 x가 일어날 확률을 P(X=x)라고 할 때, 사건의 정보량 I(x)는 다음과 같이 정의된다.
  
  
  $$
  I(x)=-\log_bP(x)
  $$

​	

## Entropy

- 엔트로피(Entropy): 특정 확률분포를 따르는 사건들의 정보량 기댓값



### For Discrete Random Variables

​	이산 확률 변수 X가 $x_1,x_2,\ldots ,x_n$ 중 하나의 값을 가진다고 가정할 때 엔트로피는 각각의 경우의 수가 가지는 정보량에 확률을 곱한 후 그 값을 모두 더한 값이다.


$$
H(X)=\mathbb{E}_{X\sim P}[I(x)]=-\sum_{i=1}^np_i\log p_i\qquad (p_i:=P(X=x_i))
$$

- 확률 변수가 가질 수 있는 값의 가짓수가 다양할수록 엔트로피 값은 증가한다.
- 확률 변수가 가질 수 있는 값의 가짓수가 같을 때, 사건들의 확률이 균등할수록 엔트로피 값은 증가한다



### For Continuous Random Variables

- X가 이산 확률 변수일 경우
  - 정보량에 확률을 각각 곱해서 모두 더한 값으로 정의된다
- X가 연속 확률 변수인 경우
  - 적분의 형태로 정의한다
  - 미분 엔트로피(differential entropy)라고 부르기도 한다



## Kullback Leibler Divergence

- 결정 모델(discriminative model)
  - 데이터의 실제 분포를 모델링하지 않음
  - 결정 경계(decision boundary)만을 학습
- 생성 모델(generative model)
  - 데이터의 실제 분포를 간접적으로 모델링
  - 학습 시 두 확률 분포의 차이를 나타내는 지표 필요
  - 대표적으로 **쿨백-라이블러 발산**(Kullback-Leibler divergence, KL divergence)

<br>

데이터가 따르는 실제 확률 분포를 $P(x)$, 모델이 나타내는 확률 분포를 $Q(x)$라고 하자. 두 확률 분포의 KL divergence는 다음과 같다.


$$
D_{KL}(P\Vert Q)=\mathbb{E}_{X\sim P}[-\log Q(x)]-\mathbb{E}_{X\sim P}[-\log P(x)]=\sum P(x)log\left( \frac{P(x)}{Q(x)} \right)
$$


연속 확률 변수의 경우에는 아래와 같다.


$$
D_{KL}(P\Vert Q)=\int P(x)\log \left( \frac{P(x)}{Q(x)} \right)dx
$$




1. $D_{KL}(P\Vert Q)\geq 0$

2. $D_{KL}(P\Vert Q)=0 if and only if P=Q$

3. $non-symmertric:D_{KL}(P\Vert Q)\neq D_{KL}(Q\Vert P)$

위의 세 가지는 KL divergence의 대표적인 특성이다. 





​	머신러닝 문제에서는 두 확률 분포의 차이를 줄여야 하므로 $D_{KL}(P\Vert Q)$를 최소화하는 방향으로 모델을 학습시킨다.


$$
D_{KL}(P\Vert Q)=\sum_{x\in X}P(x)\log \left( \frac{P(x)}{Q(x)} \right)=\left(-\sum P(x)\log Q(x) \right)-\left(-\sum P(x)\log P(x) \right)
$$


P(x)는 데이터의 실제 분포이기 때문에 조절할 수 없는 값이다. 우리가 바꿀 수 있는 부분은 Q(x)에 대한 부분이기 때문에 KL 발산을 최소화하는 것은 곧 Q(x)부분을 최소화하는 것과 동일하다. 이것이 P(x)에 대한 Q(x)의 **교차 엔트로피(cross entropy)**이다.



### Cross Entropy

 	위의 식에서 교차 엔트로피는 Q(x)가 포함된 부분으로,


$$
H(P,Q)=H(P)+D_{KL}(P\Vert Q)
$$


로 나타낼 수 있다. 위 식에서 KL 발산을 최소화하는 것과 교차 엔트로피를 최소화하는 것이 수학적으로 같음을 확인할 수 있다.



## Cross Entropy Loss

- 손실 함수(loss function): 모델이 나타내는 확률 분포와 데이터가 따르는 실제 확률 분포 사이의 차이를 나타내는 함수

  - 파라미터에 의해 결정됨
  - 예) 최소제곱법, cross entropy

- Cross entropy

  
  $$
  H(P,Q)=-\mathbb{E}_{X\sim P}[\log Q(x)]=-\sum P(x)\log Q(x)
  $$
  

  ​	분류 문제에서 데이터의 라벨은 one-hot encoding으로 표현된다. 입력 데이터의 특성값이 모델을 통과하면 출력 레이어의 소프트맥스 함수를 지나 각각 클래스에 속할 확률이 계산되는데, 이 확률 값들이 모델이 추정한 확률 분포 Q(x)를 구성한다. 3개의 클래스 $c_1, c_2, c_3$이 존재하는 분류 문제에서 출력값을 다음과 같이 가정했을 때,

  
  $$
  softmax(input)=\begin{bmatrix}0.2\\0.7\\0.1\end{bmatrix}
  $$
  

  이 결과는 다음 식과 같다.


$$
\begin{matrix}Q(X=c_1)=0.2\\Q(X=c_2)=0.7\\Q(X=c_3)=0.1 \end{matrix}
$$


​		데이터가 실제로 2번 클래스에 속한다고 하자. 데이터의 실제 확률 분포는 [0, 1, 0]이 된다.


$$
\begin{matrix}P(X=c_1)=0\\P(X=c_2)=1\\P(X=c_3)=0 \end{matrix}
$$


​		cross entropy를 사용하면 P(x)와 Q(x)의 차이를 다음과 같이 계산할 수 있다.


$$
\begin{aligned}H(P,Q)&=-\sum P(x)\log Q(x)\\&=-(0\cdot\log0.2+\log0.7+0\cdot\log0.1)\\&=-\log0.7\approx0.357 \end{aligned}
$$


### Cross Entropy와 Likelihood의 관계

​	모델의 파라미터를 $\theta$로 놓으면 모델이 표현하는 확률 분포는 $Q(y\vert X,\theta)$로, 데이터의 실제 분포는 $P(y\vert X)$로 나타낼 수 있다. 그런데 $Q(y\vert X,\theta)$는 데이터셋과 파라미터가 주어졌을 때의 예측값의 분포이므로 모델의 likelihood와 동일하다.


$$
\begin{aligned}H(P,Q)&=-\sum P(y\vert X)\log Q(y\vert X,\theta)\\&=\sum P(y\vert X)(-\log Q(y\vert X,\theta)) \end{aligned}
$$


우리가 변경할 수 있는 값은 log로 묶인 부분이기 때문에, cross entropy를 최소화하는 파라미터를 구하는 것은 곧 negative log likelihood를 최소화하는 파라미터를 구하는 것과 같다.



## Dicision Tree와 Entropy

​	의사결정 트리는 가지고 있는 기준을 세워 데이터를 나누었을 때 나누기 전보다 **엔트로피가 감소하는지**를 따진다(분류 후 엔트로피가 감소되었다는 것은 데이터 집합 내의 클래스 종류가 줄어들었다는 것, 즉 비슷한 것끼리 묶였다는 것으로 판단할 수 있다). 그리고 만약 엔트로피가 감소했다면 그만큼 모델 내부에 **정보 이득**을 얻었다고 본다.

​	만약 모델 내에 많은 분기를 만든다면 어떨까? 엔트로피의 관점에서는 분기가 많아질수록 큰 정보 이득을 얻을 수 있다. 그러나 분기가 많아질수록 데이터의 노이즈나 오류에 민감해지고 overfitting될 확률이 높아진다. 모델의 일반화 성능이 떨어지고 test data에만 지나치게 fitting되는 것이다. 그렇기 때문에 **적절한 복잡도**를 선정하는 것이 중요하다.
