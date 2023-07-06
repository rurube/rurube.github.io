---
title: 확률 및 확률분포
category:
  - AI
tags:
  - [Aiffel, math]
math: true
date: 2023-07-05
Last_modified_at: 2023-07-05
published: false
---

## 시행과 표본 공간 & 사건

### 시행과 표본 공간

- 실험(experimnets): 시도(trial)나 관찰(observation)의 과정
- 시행(random experiment)
  - **같은 조건** 하에서 반복 가능할 것
  - 결과를 알 수 **없다**
  - 일어날 **가능성이 있는** 경우의 수를 사전에 알 수 있다
- 표본 공간(sample space): 시행에서 일어날 가능성이 있는 모든 결과를 원소로 하는 집합
  - 이산(discrete) 표본 공간: 표본 공간을 구성하는 집합의 원소가 **이산형 데이터**
  - 연속(continuous) 표본 공간: 표본 공간을 구성하는 집합의 원소가 **연속형 데이터**



### 사건

- 사건(event): 표본 공간에서 특정한 규칙에 의해 정해진 부분 집합



## 확률

### 공리적 정의(수학적 정의)

표본 공간 $\Omega$에서 사건 $A$가 발생할 확률을 $P(A)$라고 하며 이는 다음을 만족한다.

- **공리 1** $0\le P(A)$(확률은 항상 0보다 크거나 같다)

- **공리 2** $P(\Omega)=1$(표본 공간 전체의 확률의 합은 1이다)

- **공리 3** 하나의 표본 공간에서 정의된 n개의 서로 배타적인 사건 $A_1, A_2, \ldots , A_n$에 대해서 다음 식이 성립한다.

  $P(A_1\cup A_2 \cup \cdots A_n)=P(A_1)+P(A_2)+\cdots +P(A_n)$

시행의 모든 시도에서 A 또는 B 사건이 발생했을 때 다른 하나의 사건이 발생하지 않을 경우 '사건 A와 B가 **서로 배타적(mutually exclusive)**이다.'혹은 '사건 A와 B는 서로 **배반사건**이다.' 라고 한다.



### 상대 도수 정의(통계적 정의)

시행이 n번 수행됐을 때 사건 A가 $n_A$번 발생했다면, 사건 A의 확률 $P(A)$는 다음과 같이 정의된다.

$P(A)=\displaystyle\lim_{n\to \infty}(\frac{n_A}{n})$

이때 비율 $n_A/n$을 사건 A의 **상대 도수(relative frequency)**라고 부른다. 상대 도수 정의는 비용이 비싸 여러번 시행을 못하거나 limit값이 존재하지 않는 경우도 있어 한계가 있다.



### 고전적 정의

고전적 정의에서 사건 A의 확률 $P(A)$는 다음과 같이 표현된다.

$P(A)=\frac{N_A}{N}$

이 확률은 **선험적(a priori)**으로 실험 없이 결정된다. 예로 동전 던지기 실험 같은 경우, 우리는 이미 동전의 모든 가능한 경우의 수 N이 2임을 알고 있으므로 앞면이 나올 확률이 1/2임을 알 수 있다.



## 집합 이론 기초

​	집합(set)은 원소들(elements)의 모음이다. 사건(event)은 일반적으로 집합이기 때문에 집합의 연산은 사건에도 동일하게 적용된다. 

- $x\in A$: x는 A에 포함된다.
- $x \notin A$: x는 A에 포함되지 않는다.
- $B\subset A$: B는 A의 부분집합이다.
- $\Omega$: 전체집합
- $\varnothing$: 공집합



### 집합의 연산들

- 같음(equality)

  $A=B\leftrightarrow A\subset B\, and\, B\subset A$

- 교집합(intersection)

  $A\cap B=\\{k\vert k\in A \, and\, k\in B\\}$

- 합집합(union)

  $A\cup B = \\{k\vert k\in \Omega \, and\, k \notin A\\}$

- 여집합(complementary set)

- $$
  a
  $$

  
