---
title: Likelihood
category:
  - AI
tags:
  - [Aiffel, Math]
math: true
date: 2023-07-06
---

## 확률 변수로서의 모델 파라미터

​	$y=f(x)=ax+b\,\,\,\,\,\,\,\,\,\,\,\,\,\, a,b\in \mathbb{R}$

​	간단하게 일차함수 모델을 예시로 들어보자. 위 식에서 실수 a, b는 $f$라는 함수로 표현되는 모델의 형태를 결정하는 파라미터이다. 따라서 a, b 값을 바꾸면 모델이 변형된다.

> $\mathbb{R}^2$공간 안의 모든 점 (a, b)은 일차함수 f에 대응된다.

이때 (a, b)가 위치하는 $\mathbb{R}^2$공간을 **파라미터 공간(parameter space)**라고 부른다.

![모델 공간 매핑](https://d3s0tskafalll9.cloudfront.net/media/images/math02_1-1-edited.max-800x600.png) 

![모델 공간 매핑+확률분포](https://d3s0tskafalll9.cloudfront.net/media/images/math02_1-2.max-800x600.png)

> 위 그림에서 파라미터 공간에 주어진 확률 분포는 평균이 (1, 0)인 정규분포이므로 $y=ax+b$에서 a와 b의 값이 각각 1과 0에 가까울 확률, 즉 $y=x$에 가까울 확률이 크다고 본다.



## posterior와 prior, likelihood 사이의 관계

### 베이지안 머신러닝 모델

- 데이터를 통해 파라미터 공간의 확률 분포 학습
- 모델 파라미터를
  - 고정된 값이 아닌 불확실성(uncertainty)을 가진 확률 변수로 봄
  - 데이터를 관찰하면서 업데이트되는 값으로 봄



### 사전 확률, 가능도, 사후 확률(prior, likelihood, posterior)

​	데이터의 집합 X, 데이터가 따르는 확률 분포를 p(X)라고 할 때 $y=ax+b=\theta^{^{\top}} \mathbf{x}$는 p(X)를 가장 잘 나타내는 일차함수 모델이다.

<center>$\theta = \begin{bmatrix}a\\\\b\end{bmatrix}, \,\,\,\,\, \mathbf{x}=\begin{bmatrix}x\\\\1\end{bmatrix}$</center><br>

​	데이터를 관찰하기 전 파라미터 공간에 주어진 확률 분포 $p(\theta)$ 를 **prior**(prior probability, **사전 확률**)이라고 한다. prior는 일반적인 정규분포가 될 수도 있고, 데이터의 특성이 반영된 특정 확률 분포가 될 수도 있다.

​	prior 분포를 고정시키면, 주어진 파라미터 분포에 대해서 준비한 데이터가 얼마나 '그럴듯한지' 계산할 수 있다. 이것을 나타내는 값이 **likelihood**(가능도, 우도)이다. 파라미터의 분포 $p(\theta)$가 정해졌을 때 데이터 x가 관찰될 확률로, 식으로 나타내면 다음과 같다.

<center>$p(X=x\vert \theta)$</center><br>

​	likelihood가 $\theta$에 의해 결정되는 함수임을 강조하기 위해 가능도 함수를 $\mathcal{L}(\theta \vert x)$로 표현하기도 한다. likelihood가 높다는 것은 곧 지정한 파라미터 조건에서 데이터가 관찰될 확률이 높다는 것이고, 데이터의 분포를 모델이 잘 표현하고 있다는 것이다. 이렇게 데이터들의 likelihood값을 최대화하는 방향으로 모델을 학습시키는 방법을 **최대 가능도 추정**(maximum likelihood estimation, **MLE**)라고 한다.

​	반대로, 데이터 집합 X가 주어졌을 때 파라미터 $\theta$의 분포 $p(\theta \vert X)$를 '데이터를 관찰한 후 계산되는 확률'이라는 뜻에서 **posterior**(posterior probability, **사후 확률**)라고 부른다. 이 값이 결국 우리가 필요한 값이지만, 데이터 포인트의 개수는 유한하기 때문에 데이터가 따르는 확률 분포 p(X)는 정확하게 알 수 없다. 그래서 posterior를 직접 계산해서 최적의 $\theta$를 찾는 것이 아니라, prior와 likelihood에 관한 식으로 변형한 다음 그 식을 최대화하는 파라미터 $\theta$를 찾는다.

​	이렇게 posterior를 최대화하는 방향으로 모델을 학습시키는 방법을 **최대 사후 확률 추정**(maximum a posterior estimation, **MAP**)이라고 한다.



### posterior와 prior, likelihood의 관계

​	확률의 곱셈 정리에 의해 확률 변수 X와 $\theta$의 joint probability $p(X, \theta)$는 다음과 같이 나타낼 수 있다.

<center>$p(X, \theta)=p(\theta \vert X)p(X)=p(X\vert \theta)p(\theta)$</center><br>

양변을 p(X)로 나누면 베이즈 정리(Bayes' theorem)과 같은 식이 나온다.

<center>$p(\theta \vert X)=\frac{p(X\vert \theta)p(\theta)}{p(X)}$</center><br>

<center>$\left( posterior=\frac{likelihood \times prior}{evidence},posterior \propto likelihood \times prior\right) $ </center><br>



## likelihood와 머신러닝

![content img](https://d3s0tskafalll9.cloudfront.net/media/original_images/math02_2-1-edited.png)

​	머신러닝 모델은 한정된 파라미터로 데이터의 실제 분포를 근사하는 역할을 하기 때문에 입력 데이터로부터 예측한 출력 데이터(prediction)과 데이터의 실제 값(label)사이의 오차는 생길 수밖에 없다. 우리는 데이터에 노이즈가 섞여 있기 때문에 이런 오차가 발생한다고 해석한다.

​	노이즈 분포를 평균이 0이고 표준편차가 $\sigma$인 정규분포로 가정한다면, 출력값의 분포는 평균이 $\theta ^{\top} x_n$이고 표준편차가 $\sigma$인 정규분포가 된다.

<center>$p(y_n\vert \theta , x_n)=\mathcal{N}(y_n\vert \theta ^{\top} x_n , \sigma^2)=\frac{1}{\sqrt{2\pi \sigma^2}}\exp(-\frac{(y_n - \theta ^{\top} x_n)^2}{2\sigma^2})$</center>

![content img](https://d3s0tskafalll9.cloudfront.net/media/images/math02_2-2_Na6BA2w.max-800x600.png)

​	xy 평면 위에 모델에 해당하는 빨간색 직선이 있다. 출력값의 분포를 나타내기 위해 p(y)좌표축을 추가한다. 입력 데이터가 $x_n$일 때 모델의 예측값은 $\theta ^{\top} x_n$이고, 출력값의 분포 $p(y_n\vert , x_n)$은 노란색으로 표시한 정규분포 그래프와 같다.



### likelihood가 왜 중요한가?

​	데이터 포인트가 모델 함수에서 멀어질수록 데이터의 likelihood는 기하급수적으로 감소한다. 즉, 머신러닝의 목표는 곧 데이터 포인트들의 likelihood 값을 크게 하는 모델을 찾는 것이다. 데이터의 likelihood 값을 최대화하는 모델 파라미터를 찾는 방법이 **최대 가능도 추론**(maximum likelihood estimation, MLE)이다.



## MLE: 최대 가능도 추론

### 데이터셋 전체의 likelihood

​	모델 파라미터 $\theta$가 주어졌을 때, 데이터 포인트 $(x_n, y_n)$의 likelihood는 다음과 같다.

<center>$p(y_n\vert \theta , x_n)=\mathcal{N}(y_n\vert \theta ^{\top} x_n , \sigma^2)=\frac{1}{\sqrt{2\pi \sigma^2}}\exp(-\frac{(y_n - \theta ^{\top} x_n)^2}{2\sigma^2})$</center><br>

좋은 머신러닝 모델은 하나의 데이터에 대해서만 likelihood가 큰 모델이 아니라 모든 데이터 포인트의 likelihood 값을 크게 만드는 모델이다. 따라서 데이터셋 전체의 likelihood를 구할 필요가 있다.

​	데이터 포인트가 서로 독립이고(independent) 같은 확률 분포를 따른다고(identically distributed) 가정하자(이 조건을 줄여서 i.i.d.라고 부른다). 데이터 포인트들이 서로 독립이므로, 데이터셋 전체의 likelihood는 데이터 포인트 각각의 likelihood를 모두 곱한 값과 같다.

<center>$p(Y\vert \theta ,X)=\prod_{n}p(y_n\vert \theta ,x_n)$</center><br>

​	MLE를 실제로 적용할 때는 likelihood 대신 log likelihood를 최대화하는 파라미터를 구한다. 데이터셋의 likelihood가 데이터 포인트 각각의 likelihood를 곱한 형태인데, 로그를 씌우면 미분 계산이 편리해지기 때문이다. 또한 로그 함수는 단조 증가(monotonically increasing)<sup>[[1]](#monotonically)</sup>하므로 likelihood를 최대화하는 파라미터와 log likelihood를 최대화하는 파라미터 값이 같아 학습 결과에 영향을 주지 않는다.<sup>[[2]](#footnote_2)</sup>

​	likelihood 계산식에 로그를 씌우면 아래와 같은 결과가 나온다.

$$\begin{aligned}

\log_p(Y\vert \theta ,X)&=\log \prod_{n}p(y_n\vert \theta ,x_n)\\\

&=\sum_{n}\log p(y_n\vert \theta , x_n)\\\

&=\sum_{n}\log \left( \frac{1}{\sqrt{2\pi \sigma^2}}\exp \left( -\frac{(y_n-\theta^{\top} x_n)^2}{2\sigma^2}\right) \right)\\\

&=\sum_{n}\log \frac{1}{\sqrt{2\pi\sigma^2}}+\sum_{n}\log\exp\left( -\frac{(y_n-\theta^{\top} x_n)^2}{2\sigma^2}\right)\\\

&=\sum_{n}\log\frac{1}{\sqrt{2\pi\sigma^2}}+\sum_{n}\left( -\frac{(y_n-\theta^{\top} x_n)^2}{2\sigma^2} \right)\\\

&=constant+\frac{1}{2\sigma^2}\sum_{n}(-(y_n-\theta^{\top} x_n)^2)

\end{aligned}$$

likelihood를 최대화하는 파라미터를 $\theta_{ML}$(ML: maximum likelihood)라고 하면,

$$\begin{aligned}

\theta_{ML}&=\arg\max_{\theta}\log p(Y\vert\theta ,X)\\\

&= \arg\max_{\theta}\left( \frac{1}{2\sigma^2}\sum_{n}(-(y_n-\theta^{^{\top}} x_n)^2)+constant \right)

\end{aligned}$$

$\theta$와 관계없는 부분을 빼고 식을 정리하면 다음과 같다.<sup>[[3]](#footnote_3)</sup><sup>[[4]](#footnote_4)</sup> 

$$
\theta_{ML}=\arg\min_{\theta}\frac{1}{2\sigma^2}\sum_{n}(y_n-\theta^{\top} x_n)^2
$$

최소화해야 할 식을 $\theta$에 대한 함수 $\mathcal{L}(\theta)$로 놓으면,

$$
\mathcal{L}(\theta)=\frac{1}{2\sigma^2}\sum_{n}(y_n-\theta^{\top}x_n)^2
$$

$\mathcal{L}(\theta)$를 최소화하는 $\theta$의 값은 $\mathcal{L}(\theta)$를 $\theta$에 대해 미분한 식을 0으로 만드는 $\theta$의 값과 같다. 일반적으로는 도함수의 부호 변화나 local minimum 여부 등도 따져야 하지만, $\mathcal{L}(\theta)$는 $\theta$에 대한 이차식이므로 유일한 최솟값을 가진다. 따라서 도함수를 0으로 만드는 값을 찾는 것으로도 충분하다.

$$\begin{aligned}

\frac{d\mathcal{L}}{d\theta}&=\frac{d}{d\theta}\left( \frac{1}{2\sigma^2}\sum_{n}(y_n-\theta^{\top}x_n)^2 \right) = \frac{d}{d\theta}\left( \frac{1}{2\sigma^2}\sum_{n}(y_n-x_n^{\top}\theta)^2 \right)\\\

&=\frac{1}{2\sigma^2}\frac{d}{d\theta}(y-X\theta)^{\top}(y-X\theta)\\\

&=\frac{1}{2\sigma^2}\frac{d}{d\theta}(y^{\top}y-y^{\top}X\theta-\theta^{\top}X^{\top}y+\theta^{\top}X^{\top}X\theta)\\\

&=\frac{1}{2\sigma^2}\frac{d}{d\theta}(y^{\top}y-2y^{\top}X\theta +\theta^{\top}X^{\top}X\theta )\\\

&=\frac{1}{\sigma^2}\left( -y^{\top}X+\theta^{\top}X^{\top}X \right)

\end{aligned}$$

데이터셋 $X=[x_1x_2\ldots x_n]^{\top}\in \mathbb{R}^{n\times d}$(d차원 데이터 $x_n=[x_{n1}\ldots x_{nd}]^{\top}$), 라벨 $y=[y_1y_2\ldots y_n]^{\top}$일 때, 최적 파라미터 $\theta{ML}$은 다음과 같다.

$$
\theta_{ML}=(X^{\top}X)^{-1}X^{\top}y
$$





## MAP: 최대 사후 확률 추정

### prior 분포의 등장

​	MLE로 구하는 최적해는 관측된 데이터 값에만 의존한다. 그래서 계산이 비교적 간단하다는 장점이 있지만, 관측된 데이터에 노이즈가 많이 섞여있는 경우나 이상치 데이터가 존재하는 경우에는 모델의 안정성이 떨어진다는 단점도 있다. 

​	머신러닝 모델의 최적 파라미터를 찾는 방법에는 likelihood를 이용한 MLE와 아래에 다룰 최대 사후 확률 추정(maximum a posteriori estimation, MAP)가 있다. MAP는 데이터셋이 주어졌을 때 파라미터의 분포 $p(\theta\vert X)$에서 확률 값을 최대화하는 파라미터 $\theta$를 찾는다. 다시 말해 주어진 데이터에서 채택될 확률이 가장 높은 파라미터를 구한다.

​	지도 학습의 경우 posterior는 $p(\theta\vert X, Y)$로 나타낼 수 있다. 이 식을 prior $p(\theta)$와 likelihood$p(Y\vert\theta, X)$에 관한 식으로 변형하면 다음과 같다.

$$\begin{aligned}

p(\theta\vert X, Y)&=\frac{p(\theta ,X,Y)}{p(X,Y)}\\\\

&=\frac{p(X,Y\vert\theta)p(\theta)}{p(X,Y)}\\\

&=\frac{p(Y\vert\theta ,X)p(\theta)}{p(Y\vert X)}

\end{aligned}$$

​	prior 분포 $p(\theta)$는 관찰된 데이터가 없을 때 파라미터 공간에 주어진 확률 분포이다. $p(\theta =\theta_0)$값을 크게 만드는 파라미터 $\theta_0$을 '그럴듯한'파라미터로 생각하는 것이다. $p(\theta)$를 평균이 (0,0)이고 공분산 $\sum=\alpha^2\boldsymbol{I}$인 정규분포라고 하자. ($\sum=\alpha^2\boldsymbol{I}$이므로 $\lvert\sum\rvert =\alpha^4 , \sum^{-1}=(1/\alpha^2)\boldsymbol{I}$)

$$\begin{aligned}

p(\theta)&=\mathcal{N}(\mathbf{0}, \alpha^2\boldsymbol{I})\\\

&=\frac{1}{\sqrt{(2\pi)^2\lvert\sum\rvert}}\exp(-\frac{1}{2}(\theta-\mathbf{0})^{\top}\sum^{-1}(\theta-\mathbf{0})) \\\

&= \frac{1}{2\pi\alpha^2}\exp\left( -\frac{1}{2\alpha^2}\theta^{\top}\theta \right)

\end{aligned}$$

​	MLE에서 negativel log likelihood를 최소화한 것처럼, MAP에서도 negative log posterior를 최소화하는 파라미터를 구한다.

$$\begin{aligned}

\arg\min_{\theta}(-\log p(\theta\vert X, Y)) &= \arg\min_{\theta}\left( -\log\frac{p(Y\vert\theta ,X)p(\theta)}{p(Y\vert X)} \right) \\\

&= \arg\min_{\theta}(-\log p(Y\vert\theta ,X)-\log p(\theta) +\log p(Y\vert X))

\end{aligned}$$

$\theta$에 의한 식이 아닌 부분을 제외하면

$$\arg\min_{\theta}(-\log p(Y\vert\theta ,X)-\log p(\theta))$$

negative log posterior 함수를 편의상 $\mathcal{P}(\theta)$라고 하면,

$$\begin{aligned}

\mathcal{P}(\theta)&=-\log p(Y\vert\theta ,X)-\log p(\theta) \\\

&= -\left( constant+\frac{1}{2\sigma^2}\sum_{n}(-(y_n-\theta^{\top}x_n)^2) \right)-\left( constant-\frac{1}{2\alpha^2}\theta^{\top}\theta \right) \\\

&= \frac{1}{2\sigma^2}\sum_{n}(y_n-\theta^{\top}x_n)^2+\frac{1}{2\alpha^2}\theta^{\top}\theta+constant

\end{aligned}$$

MAP에서 $\mathcal{P}(\theta)$를 최소화하는 $\theta$를 찾는 방법은 MLE에서와 동일하다. 같은 방식으로 계산하면 MAP의 최적 파라미터 $\theta_{MAP}$은 다음과 같다.

$$
\theta_{MAP}=\left( X^{\top}X+\frac{\sigma^2}{\alpha^2}\boldsymbol{I} \right)^{-1}X^{\top}y
$$

MLE와 다른 점은 $(\sigma^2/\alpha^2\boldsymbol{I})$항이 더해졌다는 것이다.



### MAP as L2 regularization

​	$\theta_{MAP}$의 $(\sigma^2/\alpha^2\boldsymbol{I})$항을 정리하면 최소제곱법의 정규화항과 같은 형태가 된다. 최소제곱법에서는 손실 함수에 파라미터의 크기에 관한 식을 더해 파라미터가 큰 값으로 튀는 것을 막고 과적합을 예방했다면, MAP에서는 파라미터 분포를 평균이 (0,0)인 정규분포로 놓아서 파라미터들이 각각 0에 가까운 값으로 학습되도록 제약 조건을 걸어준다. 파라미터 분포의 종류가 달라진다면 식은 달라지겠지만, 파라미터 값에 특정한 제약 조건을 준다는 점에서 효과는 같다.<br><br><br><br>



<a name="monotonically">[1]</a>: 수학에서 단조 함수는 주어진 순서를 보존하는 함수이다. 기하학적으로, 실수 단조 증가 함수의 그래프는 왼쪽에서 오른쪽으로 줄곧 상승한다. [위키백과](https://ko.wikipedia.org/wiki/%EB%8B%A8%EC%A1%B0%ED%95%A8%EC%88%98)

<a name="footnote_2">[2]</a>: 구현 측면에서, likelihood 값은 0에 가까운 수로 많이 계산되는데 이 수들을 곱하다 보면 CPU가 계산할 수 있는 범위를 넘어서는 언더플로우가 발생할 가능성이 있다. likelihood값에 로그를 씌우면 이런 문제를 예방할 수 있다.

<a name="footnote_3">[3]</a>: 손실 함수를 최소화하는 관점에서, log likelihood를 최대화하는 대신 negative log likelihood $(-\log p(Y\vert\theta ,X))$를 최소화하는 식으로 나타내기도 한다.

<a name="footnote_4">[4]</a>: 최소제곱법의 식과 같은 형태인데, 선형 모델에서의 최소제곱법은 노이즈의 분포가 $\mathcal{N}(0,\sigma^2)$라고 가정할 때 negative log likelihood를 최소화하는 파라미터를 찾는 것과 본질적으로 같다.