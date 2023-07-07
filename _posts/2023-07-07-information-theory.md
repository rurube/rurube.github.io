---
title: 정보이론
category:
  - AI
tags:
  - [Aiffel]
math: true
date: 2023-07-06
published: false
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
  - 

