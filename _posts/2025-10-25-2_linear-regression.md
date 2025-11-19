---
layout: post
title:  "[CSED515] 2. Linear Regression"

categories:
  - Lecture
  - CSED515
tags:
  - [CSED515, ML, Lecture]

toc: true
toc_sticky: true

date: 2025-10-21 22:18:00 +0900
last_modified_at: 2025-10-25 14:34:00 +0900
---

## 0. Intro  

Regression이 무엇인가. input $x$에 대한 output $y$의 dependence를 모델링하는 것을 말한다. 모든 regression은 다음과 같이 일반화할 수 있다.  

$$
y = f(x) + \epsilon
$$

이때, $y$ 는 output, $x$는 input, $\epsilon$은 noise 또는 관측되지 않은 factor를 의미한다.  

Regression의 learning goal은 MSE로, 예측값과 실제 output 사이의 오차를 제곱한 값의 평균이다.  

$$
\begin{align}
\mathcal{E}(f) &= \mathbb{E}\left[ |y - f(x)|^{2} \right] \\
            &= \int \int |y - f(x)|^{2} p(x,y) dx dy
\end{align}
$$

위의 식을 최소화하는 $f(x)$를 찾는 것이 learning goal이다. 이 식을 $f(x)$ 에 대해 편미분하여 derivative가 0이 되는 $f(x)$ 값을 찾아본다.  

$$
\begin{align}
\frac{\partial \mathcal{E}(f)}{\partial f(x)} &= \int \int - 2(y - f(x))p(x, y) dx dy \\
                                           &= \int \int - 2yp(x,y) + f(x)p(x,y) dx dy \\
                                           &= \int \int -2y p(x) p(y | x) dx dy + 2y p(x) p(y | x) dx dy = 0
\end{align}
$$

$$
\begin{align}
& \int \int y p(x) p(y | x) dx dy = \int \int f(x) p(x) p(y | x) dx dy \\
& \int p(x) \left( \int y p(y|x) dy \right) dx = \int f(x) p(x) \left( \int p(y|x) dy \right) dx \\
& \int \mathbb{E}\left[ y | x \right] p(x) dx = \int f(x) p(x) dx \\
& \int (\mathbb{E}\left[ y | x \right] - f(x)) p(x) dx = 0
\end{align}
$$

이때, $p(x) \geq 0 \forall x \in \mathbb{R}$ 이므로, 이 적분 식이 0이 되기 위해서는 $\mathbb{E}\left[ y | x \right] = f(x)$ 이 성립해야 한다.  

참고로, 이 풀이에 오류가 있다고 생각하는데, (마지막 근거가 부족하다. 꼭 저 두 term이 같지 않더라도 적분값을 0으로 만들 여지는 충분하기 때문). 다만 이 식이 모든 데이터 분포 $p(x)$ 에 대해 성립해야 하고, 
$ f(x) = \mathbb{E}\left[ y|x \right] $
는 확실히 이 식의 답이 되므로, 일단 이렇게 일반화를 할 수 있긴 하다. 다만, 더 정확하게 풀기 위해서는 다음과 같이 식을 시작하면 된다.  

$$
\begin{align}
\int \left(\int |y - f(x)|^{2} p(y|x) dy \right) p(x) dx
\end{align}
$$

이때, $p(x) \geq 0 \quad \forall x \in \mathbb{R}$ 이므로, 이 식을 최소화하기 위해서는 안쪽에 있는 적분값을 항상 최소로 만드는 f(x)를 찾으면 된다. 즉, 목표가 다음과 같이 바뀌는 것이다.  

$$
\begin{align}
\min_{f} \int |y - f(x)|^{2} p(y|x) dy
\end{align}
$$

이 식을 위와 비슷한 방식으로 편미분한 뒤 정리하면 $f(x) = \mathbb{E}\left[ y|x \right]$ 가 된다.  

이제부터 간단한 모델의 regression에 대해 상세히 알아보도록 하자.  

## 1. Ordinary Linear Regression  

가장 기본적인 Linear Regression Model을 떠올리라면, 선형적인 모델을 떠올릴 수 있을 것이다.  

Input $x \in \mathbb{R}^{d}$, output $y \in \mathbb{R}$ 에 대하여,

$$
\begin{align}
y = \theta^{T}x = x^{T}\theta
\end{align}
$$

데이터가 N개 있는 dataset 
$\mathcal{D} = \left\\{ (x_1, y_1), \ldots, (x_N, y_N) \right\\}$
에 대해, 다음과 같이 Notation을 적을 수도 있다.  

$$
\begin{align}
y = X\theta
\end{align}
$$

이때, $y \in \mathbb{R}^{N}$, $X \in \mathbb{R}^{N \times d}$ 가 된다.  

이 모델은 너무 간단해보인다. 왜냐하면, input에 대해 선형적인 모델만을 모델링하기 때문이다. 하지만, linear regression에 basis function을 도입한다면 이런 단점이 상쇄될 수 있다. **basis function**은 element를 원래 space에서 다른 space로 매핑하는 function을 의미한다. 예를 들어, 다음과 같은 basis function이 있다고 가정해보자.  

$$
\begin{align}
\phi(x) = \begin{bmatrix} 1 \\ x \\ x^{2} \\ x^{3} \end{bmatrix}
\end{align}
$$

이 basis function을 linear model에 적용하면 다음과 같아진다.  

$$
\begin{align}
\theta^{T} \phi(x) = w_{1} + w_{2}x + w_{3}x^{2} + w_{4}x^{3}
\end{align}
$$

이는 polynomial한 regression model이다. 이와 같이 basis function을 잘 선정하면 non-linear relation을 capture할 수 있다.  
basis function은 기본적으로, 데이터에서 useful information을 잡을 때 사용한다. 정해진 function이 아니므로, 데이터셋에 따라 사용자가 수동으로 설정해야 한다는 번거로움이 있지만, 일단 잘 설정한다면, linear model만으로도 충분히 좋은 regression을 할 수 있다.  

이렇게 basis function을 적용한 Linear Regression을 다음과 같이 나타낼 수 있다.  

<div class="callout">
  <div class="callout-header">Linear Regression</div>
  <div class="callout-body" markdown="1">

Linear Regression은 basis function $\phi_{\ell}$ 과의 linear combination이다.

$$
f(x) = \sum_{\ell = 1}^{L} w_{\ell}\phi_{\ell}(x) = w^{T}\phi(x)
$$

이때, $w$는 weight vector이다.

  </div>
</div>
<br>

참고로, 위의 linear model은 $x$ 에 대해서는 nonlinear하겠지만, $w$에 대해서는 linear하긴 하다. (그래서 linear regression이라고 할 수 있는 건가..)  

![Fig 1. Polynomial Regression](/assets/img/CSED515/chap2/2-1.png){: width="500"}  

$\ell$ 로 polymonial의 차수를 조절한다. 이때, 차수가 큰 것이 표현력이 좋아서 항상 좋다고 생각할 수 있는데, 위의 figure에서 볼 수 있듯이, 실제 data의 분포보다 더 복잡한 함수로 over-fitting되는 문제가 발생할 수 있다.  

한편, basis function을 사용하는 Linear Regression 모델도 learning objective로 MSE를 사용할 수 있다. Training data 
$\left\{ x_{n} \in \mathbb{R}^{D}, y_{n} \in \mathbb{R} \right\}_{n \in [N]}$
에 대해, 아래의 식을 최소로 만드는 weight vector 
$ w = \left[ w_{1}, \ldots, w_{L} \right]^{T}$
를 찾는 것이 목표가 된다.  

$$
\mathcal{E}_{LS}(w) = \frac{1}{2} \sum_{n=1}^{N} (y_{n} - w^{T}\phi(x_{n}))^{2} = \frac{1}{2} |y - \Phi w|^{w}
$$  

이떄, $y = [y_{1}, \ldots, y_{N}]^{T}$, $\Phi = \begin{bmatrix} \phi_{1}(x_{1}) & \phi_{2}(x_{1}) & \cdots & \phi_{L}(x_{1}) \\ \phi_{1}(x_{2}) & \phi_{2}(x_{2}) & \cdots & \phi_{L}(x_{2}) \\ \vdots & \vdots & & \vdots \\ \phi_{1}(x_{N}) & \phi_{2}(x_{N}) & \cdots & \phi_{L}(x_{N}) \end{bmatrix} \in \mathbb{R}^{N \times L}$  

이 식을 $w$ 에 대해 편미분하여 0이 되는 $w$를 찾으면 다음과 같다.  

$$
\begin{align}
\frac{\partial}{\partial w} \mathcal{E}_{LS}(w) &= \frac{\partial}{\partial w} \frac{1}{2} |y - \Phi w|^{2} \\
                                             &= - \Phi^{T}(y - \Phi w) = 0
\end{align}
$$

이 식을 정리하면,  

$$
\begin{align}
&\Phi^{T}y = \Phi^{T}\Phi w \\
&w_{LS} = (\Phi^{T}\Phi)^{-1}\Phi^{T}y = \Phi^{†}y
\end{align}
$$

참고로 $\Phi^{†}$ 은 $\Phi$ 의 Moore-Penrose pseudo-inverse로, $\Phi^{T}\Phi$ 가 inverible하면 그대로 계산하면 되고, 만약 invertible하지 않더라도, 여전히 존재하는 행렬이다. 이렇게 closed-form solution이 존재하지만, 사실 gradient descent method를 사용해도 괜찮다.  

![Fig 2. Guassian Noise](/assets/img/CSED515/chap2/2-2.png){: width="500"} 

이 Least Square method는 놀랍게도 Maximum Likelihood Estimation으로 해석한 해와 같다. Output $y$ 가 $f(x_{n}) = w^{T}\phi(x_{n})$ 과 Gaussian noise $\epsilon \sim \mathcal{N}\left( 0, \sigma^{2} \right)$ 의 합으로 나타난다고 하자. 즉,  

$$
\begin{align}
y = w^{T}\phi(x) + \epsilon
\end{align}
$$

이라 하자. (위의 figure는 Gaussian noise를 적용한 linear model을 나타낸 것이다.) 이때, 이 식의 log-likelihood $\mathcal{L}$ 은 다음과 같이 나타난다.  

$$
\begin{align}
\mathcal{L} \propto \log p(y | \Phi, w) &= \sum_{n=1}^{N} \log p(y_{n} | \phi(x_{n}), w) \\
                                        &= -\frac{N}{2} \log \sigma^{2} - \frac{N}{2} \log 2\pi - \frac{1}{\sigma^{2}} \underbrace{\left( \frac{1}{2} \sum_{n=1}^{N} (y_{n} - w^{T}\phi(x_{n}))^{2} \right)}_{\mathcal{E}_{LS}}
\end{align}
$$

MLE는 likelihood를 최대로 만드는 것으로, 이 식을 최대로 만들기 위해서는 
$\mathcal{E}\_{LS}$ 부분을 minimize해야 한다. 즉, additive Gaussian noise 가정 하에서, $w\_{LS} = w\_{MLE}$ 가 성립한다.  

<br><br>

## 2. Ridge Regression - Overfitting and Regularization  

위에서 잠깐 언급했듯, parameter의 개수가 많아지면, 데이터의 분포 패턴을 학습하는 것이 아닌, 데이터 자체를 memorizing하는 경향이 생기는데, 이를 overfitting 이라고 한다. Over-fitting이 나타난 것을 어떻게 알 수 있는가 하면, training loss는 작은데, test loss가 크게 차이가 난다면, over-fitting이 발생한 것이다. 이를 해결하기 위해, loss에 새로운 term (regularization)을 추가하는 방안을 떠올릴 수 있다. 일반적으로 overfitting은 parameter의 magnitude가 클 때 발생하므로, 파라미터의 magnitude 자체를 loss에 추가한다면 overfitting을 어느 정도 방지할 수 있다. 이 강의자료에서는 두 개의 Regularizer를 살펴보는데, 그 중 하나가 Ridge다.  

<div class="callout">
  <div class="callout-header">Ridge Regression</div>
  <div class="callout-body" markdown="1">

Ridge regression은 다음을 최소화한다.  

$$
\mathcal{E} = \frac{1}{2} |y - \Phi w|^{2} + \frac{\lambda}{2} |w|^{2}
$$

이때 $\lambda$ 는 ridge parameter다.  

  </div>
</div>
<br>

이 loss를 최소화하는 $w$ 는 derivative를 0으로 만드는 $w$ 를 구하여 찾을 수 있다. 구체적으로 다음과 같다.  

$$
\begin{align}
\frac{\partial \mathcal{E}}{\partial w} = \Phi^{T} (y - \Phi w) + \lambda w = 0
\end{align}
$$

이 식을 정리하면,  

$$
\begin{align}
&w_{ridge}(\Phi^{T}\Phi + \lambda I) = \Phi^{T} y \\
&w_{ridge} = (\Phi^{T}\Phi + \lambda I)^{-1}\Phi^{T} y
\end{align}
$$

이 식은 zero-mean Gaussian prior 를 적용한 linear model의 MAP와 동일한 form이다. $w$ 의 분포가 다음과 같을 때,  

$$
\begin{align}
p(w) = \mathcal{N} \left( w | 0, \Sigma \right)
\end{align}
$$

이때, $y = \Phi w + \epsilon$, $\epsilon \sim \mathcal{N} \left( y | \Phi w, \sigma^{2} I \right)$ 이므로,  

$$
\begin{align}
p (y | \Phi , w) = \mathcal{N} (y | \Phi w, \sigma^{2}I) 
\end{align}
$$

가 되며, 이때, posterior는 다음과 같으므로,  

$$
\begin{align}
p(w | y, \Phi) = \frac{p(y | \Phi, w) p(w)}{p(y | Phi)}
\end{align}
$$

이를 최대로 만드는 $w$ 를 찾는다. 먼저, 식에 log를 취해주도록 하자.  

$$
\begin{align}
\log p(w | y, \Phi) &= \log p(y | \Phi, w) + \log p(w) - \log p(y | \Phi)
\end{align}
$$

이 식을 최소로 만드는 $w$ 를 찾기 전에, $w$ 와 관련된 term을 제외한 다른 term은 전부 지워주도록 하자.  

$$
\begin{align}
\log p(y | \Phi, w) = -\frac{N}{2} \log{2\pi} - \frac{1}{2} \log{|\sigma^{2}I|} - \frac{1}{2}(y - \Phi w)^{T} (\sigma^{2}I)^{-1}(y - \Phi w) \\
                    \propto - (y - \Phi w)^{T}(\sigma^{2}I)^{-1}(y - \Phi w)
\end{align}
$$

$$
\begin{align}
\log p(w) = -\frac{D}{2} \log{2\pi} - \frac{1}{2} \log{|\Sigma|} - \frac{1}{2} w^{T}\Sigma^{-1}w \\
          \propto - w^{T}\Sigma^{-1}w
\end{align}
$$

$$
\begin{align}
\log p(w | y, \Phi) \propto - (y - \Phi w)^{T}(\sigma^{2}I)^{-1}(y - \Phi w) - w^{T}\Sigma^{-1}w
\end{align}
$$

MAP 는 이 Posterior를 최대로 만드는 것이므로, 주어진 식을 최대로 만드는 $w$ 를 찾기 위해, $w$ 로 편미분을 하여 0이 되는 $w$를 찾는다.  

$$
\begin{align}
\frac{\partial \log p(w | y, \Phi)}{\partial w} &\propto \Phi^{T}(\sigma^{2}I)^{-1}(y - \Phi w) - \Sigma^{-1} w \\
                                                &= \frac{1}{\sigma^{2}}\Phi^{T}(y - \Phi w) - \Sigma^{-1} w \\
                                                &= \frac{1}{\sigma^{2}}\Phi^{T}y - \frac{1}{\sigma^{2}}\Phi^{T}\Phi w - \Sigma^{-1}w \\
                                                &= -(\frac{1}{\sigma^{2}}\Phi^{T} + \Sigma^{-1})w + \frac{1}{\sigma^{2}}\Phi^{T}y \\
                                                &= -(\Phi^{T} + \sigma^{2}\Sigma^{-1})w + \Phi^{T}y = 0
\end{align}
$$

$$
\begin{align}
(\Phi^{T} + \sigma^{2}\Sigma^{-1})w = \Phi^{T}y \\
w_{MAP} = (\Phi^{T} + \sigma^{2}\Sigma^{-1})^{-1}\Phi^{T}y
\end{align}
$$

이를 위에서 구한 ridge regression과 비교하면,  

$$
\begin{align}
w_{ridge} = (\Phi^{T}\Phi + \lambda I)^{-1}\Phi^{T} y \\
w_{MAP} = (\Phi^{T} + \sigma^{2}\Sigma^{-1})^{-1}\Phi^{T}y
\end{align}
$$

$\Sigma = \frac{\sigma^{2}}{\lambda}I$ 이면, MAP가 ridge regression과 동일해짐을 알 수 있다.  

<br><br>

## 3. Lasso Regression  

Lasso Regression은 Ridge Regression과 다른 regularizer를 사용한다.  

<div class="callout">
  <div class="callout-header">Lasso Regression</div>
  <div class="callout-body" markdown="1">

Lasso regression은 다음을 최소화한다.  

$$
\mathcal{E}(w) = \frac{1}{2} |y - \Phi w|^{2} + \lambda |w|_{1}
$$

이때 $\lambda$ 는 trade-off를 control한다.  

  </div>
</div>
<br>

regularizer가 $\ell_{1}$ norm이 되었다. 이와 같이 $\ell_{1}$ norm을 사용하면 objective가 not differentiable해지는 단점이 존재한다. 하지만 $\ell_{1}$ norm을 사용할 떄의 장점도 존재한다. 바로, 일종의 feature selector로 작용할 수 있다는 점이다.  

![Fig 3. Lasso vs Ridge](/assets/img/CSED515/chap2/2-3.png){: width="500"} 

위의 Figure에서 볼 수 있듯, L1 norm을 사용하게 되면, regularization으로 인한 boundary가 각지게 된다.  이렇게 되면, optimal point가 L2 norm에 비해 feature axis에 형성될 확률이 높아진다. 위의 예시를 보면, L2의 optimal point는 $w_1$ 과 $w_2$ 를 모두 0이 아닌 값으로 만드는 반면, L1의 optimal point는 $w_1$ 의 값을 0으로 만들고, $w_2$ 의 값만 0이 아니게 된다. 즉, L1 norm은 regression에서 비교적 의미 없는 feature (여기에서는 $w_1$)를 필터링해주는 역할을 수행할 수 있다.  

<br><br>

## 4. Non-linear Regression and Gradient Method  

Learning problem은 optimization problem으로 정의될 수 있다. 구체적으로, sum loss function을 최소화하는 model을 찾는 optimization problem으로 정의될 수 있다.  

$$
\begin{align}
\argmin_{\theta} L(\theta) = \argmin_{\theta} \sum_{i=1}^{N} \ell \left( y_{i}, f(x_{i}, \theta) \right)
\end{align}
$$

예를 들어, training dataset에 대해 $\theta$ 로 나타낸 learning objective가 다음과 같은 모델을 생각해보자.  

$$
\begin{align}
L(\theta) = \sum_{i=1}^{N}\ell(y_{i}, f(x_{i}, \theta)) = \theta^{4} + 7\theta^{3} + 5\theta^{2} - 17\theta + 3
\end{align}
$$

이 learning rate를 최소로 만드는 parameter $\theta$ 를 찾기 위해, 다음과 같이 gradient가 0이 되는 지점을 찾을 수 있다.  

$$
\begin{align}
\frac{d L(\theta)}{d\theta} = 0
\end{align}
$$

하지만 이런 방법에는 문제가 있으니, 바로 closed-form expression이 존재하지 않는 경우가 많다는 것이다. 그래서 생각해낸 방법이 바로 Gradient Descent와 같은 numarical methods다. gradient의 정의를 알아볼텐데, 현재 우리는 loss function에 대해 알아보고 있으므로, $f: \mathbb{R}^{D} \rightarrow \mathbb{R}$ 인 real-value function에 대한 gradient를 알아본다.  

$$
\begin{align}
\nabla f(x) = \left[ \frac{\partial f(x)}{\partial x_{i}} \right] = \begin{bmatrix} \frac{\partial f(x)}{\partial x_{1}} \\ \vdots \\ \frac{\partial f(x)}{\partial x_{D}} \end{bmatrix}
\end{align}
$$

참고로, derivative와 gradient의 다른 점은 차원이다. gradient는 column vector, derivative는 row vector다. (서로 transpose 관계)  

이 gradient를 사용하여 어떻게 solution을 찾을 수 있을까. 이미 대부분이 알듯이 gradient descent라는 방법을 사용한다. objective function $L : \mathbb{R}^{n} \rightarrow \mathbb{R}$ 이 parameter $\theta$ 에 대한 function일 때, 어떤 initial point $\theta_{0}$ 부터 시작하여, 다음과 같은 gradient path를 따라 이동하는 방법이다.  

$$
\begin{align}
\theta_{1} = \theta_{0} - \gamma \nabla L(\theta_{0})  
\end{align}
$$

이때, $\gamma \geq 0$ 은 step-size로, hyperparameter다.  

gradient descent update rule은 위의 과정을 계속해서 반복하는 것으로, 식으로 정리하면 다음과 같다.  

$$
\begin{align}
\theta_{t+1} = \theta_{t} - \gamma_{t} \nabla L(\theta_{t})  
\end{align}
$$

![Fig 4. GD](/assets/img/CSED515/chap2/2-4.png){: width="500"} 

위의 figure는 GD를 도식화한 것이다. gradient가 양수가 되는 point에서는 왼쪽으로, gradient가 음수가 되는 point에서는 오른쪽으로 이동하게 되므로, 아래로 볼록(convex)한 곡선을 따라 point가 이동하게 된다.  

이를 Linear Regression 모델에 적용해보자. MSE objective는 다음과 같았다.  

$$
\begin{align}
L(w) = \frac{1}{2} \sum_{n=1}^{N}\left( \phi_{n}^{T}w - y_{n} \right)^{2} = \frac{1}{2} |\Phi w - y|^{2}_{2}
\end{align}
$$

이 식의 $w = w_{t}$ 에서의 gradient는 다음과 같다.  

$$
\begin{align}
g_{t} = \sum_{n=1}^{N} \left( w_{t}^{T}\phi_{n} - y_{n} \right) \phi_{n}
\end{align}
$$

따라서 GD update는 다음과 같다.  

$$
\begin{align}
w_{t+1} = w_{t} - \gamma_{t}g_{t} = w_{t} - \gamma_{t}\sum_{n=1}^{N}\left( w_{t}^{T}\phi_{n} - y_{n} \right) \phi_{n}
\end{align}
$$

다만 GD는 그 자체로 한계점이 있다. 바로 step size의 영향을 매우 크게 받는다는 점이다.  

![Fig 5. GD example 1](/assets/img/CSED515/chap2/2-5.png){: width="500"} 

이 figure가 바로 이런 한계를 나타낸다. figure에 서술된 initial point와 step size를 사용하면 GD가 minimum에 수렴하지만, 처음 initial point부터 convergence point까지 가는 경로가 매우 비효율적이며, 매우 느리게 수렴하게 된다. GD를 사용하여 빠르게 converge 하게 만들기 위해서는 적절한 step size를 설정할 필요가 있다.  

![Fig 6. GD stepsize](/assets/img/CSED515/chap2/2-6.png){: width="500"} 

step size가 작은 경우, 위의 figure의 왼쪽과 같이 convergence point까지 매우 느리게 수렴하게 된다. 한편 step size를 크게 설정하게 되면, 오른쪽과 같이 convergence point에 수렴하지 못하고 발산할 수 있다.  

우리는 위의 figure로부터 알 수 있는 사실을 바탕으로 step size 문제를 해결하는 간단한 heuristic을 생각해볼 수 있다.  

- gradient step 이후 objective function value가 상승하는 경우 -> undo 후 step-size 를 줄인다.  
- gradient step 이후 objective function value가 줄어든 경우 -> step-size를 늘려본다.  

하지만 objective function이 복잡한 경우, 이런 전략은 매우 오래 걸리며 비효율적일 수 있다. 이에 step size와 관련한 문제를 완화해줄 수 있는 몇몇 technique이 개발되어왔다. 그 중 하나가 Momentum이다.  

### Momentum  

![Fig 7. Momentum](/assets/img/CSED515/chap2/2-7.jpg){: width="500"} 

Momentum은 이전의 gradient step을 기억하여 다음 step에 일부 적용하는 방법이다. 식으로 나타내면 다음과 같다.  

$$
\begin{align}
\theta_{i+1} = \theta_{i} - \gamma_{i}\nabla f(\theta_{i}) + \alpha \Delta \theta_{i} \\
\Delta \theta_{i} = \theta_{i} - \theta_{i-1} = \alpha \Delta \theta_{i-1} - \gamma_{i-1} \nabla f(\theta_{i-1})
\end{align}
$$

이때, $\alpha$ 는 0과 1 사이의 값을 가지는 hyperparameter로, 이전의 update를 얼마나 반영할 지 결정한다. momentum을 적용하면, gradient update가 비교적 더 smooth해진다. 당연한 것이, 이전의 update vector를 일부 반영하므로, 방향이 덜 꺾이게 되는 것이다. 위의 figure를 보면, momentum을 적용하지 않을 때보다 적용할 때 convergence point 쪽으로 이동하는 것을 볼 수 있다. 또한, 데이터셋에 따라 update를 가속화하기도 한다.  

### SGD  

어떤 dataset에 대해, batch gradient $L(\theta)$ 는 다음과 같이 모든 data point에 대한 gradient의 합으로 정의된다.  
$$
\begin{align}
L(\theta) = \sum_{i=1}^{N} L_{i}(\theta) = \sum_{i=1}^{N} \ell \left( y_{i}, f_{\theta}(x_{i}) \right) \\
\frac{d L(\theta)}{d\theta} = \sum_{i=1}^{N}\frac{d L_{i}(\theta)}{d\theta}
\end{align}
$$

만약 dataset이 너무 크다면, batch gradient를 계산하는 데 매우 많은 시간이 들 것이다. 따라서 우리는 일부의 데이터(mini-batch)만으로 gradient를 approximate하는 시도를 해볼 수 있다. 그런데, mini-batch로 구한 gradient가 실제 batch gradient와 얼마나 비슷할 줄 알고 이런 시도를 하는 것일까.  

$L(\theta)$ 를 expected risk, 
$\hat{L}\_{\mathcal{D}}(\theta)$ 를 batch에 대한 empirical risk, 
$\hat{L}\_{\mathcal{B}}(\theta)$ 를 mini-batch에 대한 empirical risk라 하자. 이때, 임의의 데이터셋 
$\mathcal{L}$ 에 대한 empirical risk를 다음과 같이 정의하자. 
$\hat{L}\_{\mathcal{S}}(\theta) := \frac{1}{|\mathcal{S}|} \sum\_{i=1}^{|\mathcal{S}|} \ell \left( y\_{i}, f\_{\theta}(x\_{i}) \right)$  

이때, $\hat{L}_{\mathcal{B}}(\theta)$ 는 $L(\theta)$ 에 대한 consistent estimator가 된다. 즉, 다음이 성립한다.  

$$
\begin{align}
\frac{dL(\theta)}{d\theta} = \mathbb{E}\left[ \frac{d \hat{L}_{\mathcal{B}} (\theta)}{d\theta} \right]
\end{align}
$$

이에 대한 증명은 다음과 같이 할 수 있다.  

$$
\begin{align}
\frac{d L(\theta)}{d\theta} &= \mathbb{E} \left[ \frac{d L_{i}(\theta)}{d\theta} \right] \quad (\because \text{def of expected risk, Leibniz rule}) \\
                            &= \mathbb{E} \left[ \frac{1}{|\mathcal{B}|} \sum_{i: x_{i} \in \mathcal{B}} \frac{d L_{i}(\theta)}{d\theta} \right] \quad (\because \text{Linearity of expectation, I.I.D assumption}) \\
                            &= \mathbb{E} \left[ \frac{d \hat{L}_{\mathcal{B}} (\theta)}{d\theta} \right] \quad (\because \text{Def of mini-batch loss, derivative of sum, linearity of differentiation})  
\end{align}
$$

참고로 당연히 같은 이유로 $\hat{L}_{\mathcal{D}}(\theta)$ 또한 consistent estimator다. 그렇다면 SGD의 장단점은 무엇이 있을까?  

장점으로 강의 노트에서 세 가지를 언급한다. 먼저, memory efficient하다. 또, expectation 관점에서, true gradient와 stochastic gradient가 동일하며, 마지막으로 일반적으로 batch gradient descent보다 stochastic gradient descent가 더 좋은 성능을 보인다는 점이 있다.  
한편, mini-batch size에 따라 variance가 높아질 수 있다는 단점이 있다.  
