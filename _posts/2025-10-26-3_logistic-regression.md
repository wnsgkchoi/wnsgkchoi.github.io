---
layout: post
title:  "[CSED515] 3. Logistic Regression"

categories:
  - Machine_Learning
tags:
  - [CSED515, ML, Lecture]

toc: true
toc_sticky: true

date: 2025-10-26 22:18:00 +0900
last_modified_at: 2025-10-26 23:34:00 +0900
---

## Model Class: Logistic Regression Models  

### Why Logistic Regression Models?  

Linear regression 모델의 목표는 MSE를 최소화하는 것이다. 하지만 이는 너무 단순한 나머지, boundary에 있는 example에 대해 매우 좋지 않은 예측 성능을 보여준다. linear model은 분류에 성공했더라도, 예측값과 실제 데이터 값의 차이가 크다면 큰 penalty를 주어 모델을 왜곡시킬 수 있다. 이는 linear model의 출력 값이 continuous하며, 음의 무한대부터 양의 무한대까지 범위가 매우 넓기 때문에 발생한다.  
반면 logistic regression의 출력은 0과 1 사이의 probability로 나타나며, 이는 event의 likelihood를 예측하기에 적절하다.  

### Properties of Logistic Function  

logistic function $\sigma(t)$ 는 다음과 같다.  

$$
\sigma(t) = \frac{1}{1 + e^{-t}}  
$$

이 함수는 다음과 같은 특징을 가진다.  

1. $\sigma(t) \rightarrow 0$ as $t \rightarrow -\infty$  
2. $\sigma(t) \rightarrow 1$ as $t \rightarrow \infty$  
3. $0 < \sigma(t) < 1$  
4. $\sigma(-t) = 1 - \sigma(t)$ &nbsp; (symmetry property)  
5. $\frac{d}{dt} \sigma(t) = \sigma(t)\sigma(-t) = \sigma(t)\left( 1 - \sigma(t) \right)$
6. $\sigma(t) = \frac{e^{t}}{e^{t} + 1} \rightarrow \int{\sigma(t) dt} = \int{\frac{1}{u} du} = \ln{u} = \ln{(1+e^x)}$

### Logistic Regression Model in Binary Classification  

<div class="callout">
  <div class="callout-header">Definition (a logistic regression model)</div>
  <div class="callout-body" markdown="1">

$$
f(x) = \sigma (wx + b) = \sigma(w^{T}x),
$$

where $w, x \in \mathbb{R}^{D}, b \in \mathbb{R}, w = \left[ w^{T}, b^{T} \right]^{T}, \text{and} x = \left[ x^{T}, 1 \right]^{T}$

  </div>
</div>
<br>

이 classifier는 input $x$를 다음 기준에 따라 분류한다.  

$$
h(x) = \begin{cases}
        +1 & \text{if } \sigma (w^{T}x) \geq 0.5 \\
        -1 & \text{otherwise}
    \end{cases}
    = \begin{cases}
        +1 & \text{if } w^{T}x \geq 0 \\
        -1 & \text{otherwise}
    \end{cases}
$$

이는 위에서도 간단히 언급했듯, 확률적으로 해석할 수 있다.  

$$
\hat{p}(y = 1 | x, w) = f(x) = \frac{1}{1 + exp(-w^{T}x)} \text{ and } \hat{p}(y = -1 | x, w) = 1 - f(x) = \frac{1}{1 + exp(w^{T}x)}
$$

간단히 다음과 같이 표현할 수도 있다.  

$$
\hat{p} \left( y | x, w \right) = \frac{1}{1 + exp(-yw^{T}x)}
$$

이 classifier가 어떤 경계를 형성하는지 다음 figure를 보자.  

![fig 1. decision boundary of logistic regression](/assets/img/CSED515/chap3/image.png){: width="500"}

결국 $w^T x + b$의 부호에 따라 classification되므로, 경계는 $w^T x$가 b이 되는 부분이 될 것이고, 이는 곧 vector $w$와 orthogonal한 vector의 집합에 b를 더한 값이 되는 vector의 집합인 hyperplane 될 것이다. 말을 어렵게 했는데, 사실 그냥 특수해($x_0, y_0, z_0$)를 찾은 뒤에, 해당 점을 지나고, w와 orthogonal인 벡터의 hyperplane을 그리면 된다. 실제로 위의 figure를 보면, vector $w$와 수직한 hyperplane이 경계임을 볼 수 있다.  

그렇다면, Logistic은 항상 Linear model보다 좋은 모델일까? 당연하지만, Logistic도 만능은 아니며, 장단이 존재한다.  
Logistic regression model은 output이 구간 [0, 1] 내에 있다는 것은 장점이 되지만, linear model과 다르게 closed form solution이 없어 한 번에 계산하지 못한다는 단점도 있다. 다만, iterative method를 통해 충분히 좋은 solution을 구할 수 있어 이 단점은 어느 정도 상쇄된다.  

### Learning objective of Logistic Regression  

<div class="callout">
  <div class="callout-header">Objective for classification</div>
  <div class="callout-body" markdown="1">

Maximize the likelihood of a dataset $Z$ given a parameterized probabilistic model $p(\cdot | w)$, i.e.,

$$
\hat{w} = \argmax_{w} p(Z | w)
$$

  </div>
</div>
<br>

꽤 MAP 같은 objective다. 아무튼, 이 objective를 어떻게 model할까? 먼저 식을 전개해보자.  

$$
\begin{align}
p(Z | w) &= \prod_{n=1}^{N} P(x_n, y_n | w) \quad (\because \text{i.i.d. assumption}) \\
         &= \prod_{n=1}^{N}p(y_n | x_n, w) p(x_n | w) \quad (\because \text{chain rule}) \\
         &= \prod_{n=1}^{N} p(y_n | x_n, w)p(x_n) \quad (\because x_n \text{and } w \text{ are independent})
\end{align}
$$

따라서, 위의 식과 결합하면 objective는 다음과 같다.  

$$
\hat{w} = \argmax_{w} \prod_{n=1}^{N} p(y_n | x_n, w)
$$

위의 식에서 $p(x_n)$ 이 사라진 이유는, 당연하게도 $w$ 와 관련이 없어 argmax할 때 상수로 취급되기 때문이다.  

이를 쉽게 해결하기 위해, negative Log-likelihood minimization으로 문제를 바꾼다.  

<div class="callout">
  <div class="callout-header">Negative Log-likelihood Minimization</div>
  <div class="callout-body" markdown="1">

Minimize the negative log-likelihood of a dataset $Z$ by modeling $p(y_n | x_n, w)$ via a logistic regression model, i.e.,  

$$
\hat{w} = \argmin_{w} - \sum_{n=1}^{N} y_n \ln {\sigma (w^{T}x_n)} + (1 - y_n) \ln {(1-\sigma(w^{T}x_n))}
$$

  </div>
</div>
<br>

왜 이런 결과가 나올까? 먼저 위에서 정리했던 식에 -log를 취한 것을 생각해보면,

$$
- \ln p(Z | w)= - \ln{\prod_{n=1}^{N}{p(y_n | x_n, w)}}
$$

이때, 
$p(y_n | x_n, w) = \frac{1}{1 + exp(-y_nw^{T}x_n)} = \sigma{y_nw^{T}x_n}$
이므로, 식을 아래와 같이 변형할 수 있다.  

$$
- \ln{\prod_{n=1}^{N}{\sigma(y_nw^{T}x_n)}}
$$

만약 $y_n = 1$ 인 경우, 해당 데이터에 대한 식은 $\sigma{w^{T}x_n}$ 이 되고, $y_n = -1$ 인 경우, 위의 식은 $\sigma{-w^{T}x_n}$ 이 됨을 위에서 살펴보았다. 따라서, 이 식은 모든 데이터 $N$ 개에 대하여 $y_n = 1$ 인 경우 $\sigma{w^{T}x_n}$ 을, $y_n = -1$ 인 경우 $\sigma{-w^{T}x_n}$ 을 곱하는 것임을 알 수 있다. 이때, 편의를 위해 $\mathcal{Y} = \{0, 1\}$ 이라 하자. 그러면 위의 식을 아래와 같이 적을 수 있다.  

$$
- \ln{\prod_{n=1}^{N}{\sigma{w^{T}x_n} ^{y_n} \sigma{-w^{T}x_n}^{(1-y_n)}}}
$$

그리고, $\sigma(-x) = 1 - \sigma(x)$ 임을 사용하면 위의 식을 다음과 같이 정리할 수 있다.  

$$
- \ln{\prod_{n=1}^{N}{\sigma(w^{T}x_n)^{y_n}(1 - \sigma(w^{T}x_n))^{(1-y_n)}}}
$$

이제 log를 풀어 다음과 같이 바꾸면 식이 완벽히 정리가 된다.  

$$
- \sum_{n=1}^{N}{\left[ y_n \ln \sigma(w^{T}x_n) + (1 - y_n)\ln(1-\sigma(w^{T}x_n)) \right]}
$$

이때, 이 식을 보고 무언가 떠오를 수 있다. 바로, cross-entropy다. cross-entropy의 식은 다음과 같았다.  

$$
- \sum_{y \in \mathcal{Y}}{p(y) \ln q(y)}
$$

$p(y)$ 를 true label(즉 위에서 $y_n$), $q(y)$ 를 $\sigma(w^{T}x)$ 라 하면, 위의 objective(Negative log-likelihood를 최소화하는 $w$ 찾기)가 average cross entropy를 minimize하는 것으로 바뀔 수도 있다는 것을 알 수 있다.  

그런데, 하나 근본적인 의문이 발생한다. 분명히 우리가 linear보다 logistic을 사용하는 이유로, classification에 더 적합함을 들었는데, 과연 cross entropy를 minimize하는 것이 classification error를 minimize하는 것과 동일할까?  

i.i.d 조건을 만족하는 samples $((x_n, y_n))_{n \in [ N ]} \sim \mathcal{D}^N$ 이 주어졌을 때, 다음 식이 성립한다.  

$$
\mathbb{E}_{(x, y) \sim D} \mathbb{1}(h(x) \neq y) \approx \frac{1}{N} \sum_{n=1}^{N}\mathbb{1}(h(x_n) \neq y_n)
\leq \frac{1}{N} \sum_{n=1}^{N}C l_{ce}(x_n, y_n, f)
$$

where $l_{ce}(x, y, f) = - \left( y\ln{f(x)} + (1-y)\ln(1-f(x)) \right)$  
결국 다음과 같은 관계가 성립한다는 뜻이다.  

$$
\mathbb{1}(h(x) \neq y) \leq C l_{ce}(x, y, f)
$$

좌변은 zero-one loss로, classificaton은 궁극적으로 이 값을 최소화하기 위해 노력한다. 우변은 cross-entropy loss에 상수 $C$ 를 곱한 것으로, 결국 cross-entropy loss를 줄이면, zero-one loss의 upper bound가 줄어들어 zero-one loss가 줄어들 수 있다는 것이 수식으로 증명된다.  

## Algorithm: Gradient-based Methods  

지금까지 regression의 objective에 대해 알아보았다. 이제, 지금까지 배운 loss 함수를 minimization하는 방법에 대해 알아보자. 사실, objective function이 수학적으로 계산 가능한 solution(closed-form solution)이라면, 굳이 알고리즘이 필요하지 않을 수 있다. 하지만, 아쉽게도 logistic regression의 loss는 그러한 solution이 존재하지 않는다. 따라서, 학습 알고리즘을 필요로 한다.  

### Gradient Descent Method: A First-order Method  

가장 먼저 살펴볼 알고리즘은 GD다. (수업시간 중 교수님께서 G-Dragon으로 농담을 하셨다.) 이 알고리즘은 비교적 간단하다.  

<div class="callout">
  <div class="callout-header">Gradient Descent Method</div>
  <div class="callout-body" markdown="1">

learning rate $\gamma$, learning objective $\mathcal{L}$ 에 대하여, GD method는 parameter를 다음과 같이 update한다.  

$$
w^{new} \leftarrow w^{old} - \gamma \bigtriangledown_{w^{old}}\mathcal{L}(w)
$$

  </div>
</div>
<br>

그렇다면, 이 GD를 위에서 배운 Logistic Regression Model에 적용하면 update 식은 어떻게 될까?  
먼저, 이전에 살펴본 logistic regression model의 loss는 다음과 같았다.  

$$
- \sum_{n=1}^{N}{\left[ y_n \ln \sigma(w^{T}x_n) + (1 - y_n)\ln(1-\sigma(w^{T}x_n)) \right]}
$$

이 식을 w에 대해 미분해보도록 하자.  

$$
\begin{align}
\frac{\partial \mathcal{L}(w)}{\partial w} &= -\frac{\partial}{\partial w} \sum_{n=1}^{N}{\left[ y_n \ln \sigma(w^{T}x_n) + (1 - y_n)\ln(1-\sigma(w^{T}x_n)) \right]} \\
                                           &= -\sum_{n=1}^{N} \left[ y_n \frac{\sigma (w^T x_n) (1 - \sigma(w^T x_n)) x_n }{\sigma (w^T x_n)} + 
                                           (1-y_n) \frac{\sigma(-w^T x_n) (1 - \sigma(-w^T x_n)) (-x_n)}{\sigma(-w^T x_n) } \right] \\
                                           &= -\sum_{n=1}^{N} \left[ y_n (1 - \sigma(w^T x_n))x_n + (1-y_n)\sigma(w^T x_n)(-x_n) \right] \\ 
                                           &= -\sum_{n=1}^{N} \left[ y_n x_n - \sigma(w^{T}x_n) x_n \right] \\
                                           &= -\sum_{n=1}^{N} \left[ (y_n - \sigma(w^{T}x_n)) x_n \right]
\end{align}
$$

이 결과를 update 식에 그대로 붙여 넣으면 아래와 같다.  

$$
w^{new} \leftarrow w^{old} + \gamma \sum_{n=1}^{N}\left[ y_n - \sigma \left( (w^{old})^T x_n \right) \right]x_n
$$

### Newton's Method: A Second-order Method  

GD는 미분을 한 번만 한 gradient만을 사용했다면, Newton's Method는 여기에 이계도함수(hessian)까지 추가로 사용하는 방법이다.  
먼저, 확실한 이해를 위해 gradient와 hessian의 수학적 표기를 알아보자.  

$$
\bigtriangledown f(x) = \begin{bmatrix} 
                            \frac{\partial f(x)}{\partial x_1} \\
                            \frac{\partial f(x)}{\partial x_2} \\
                            \vdots \\
                            \frac{\partial f(x)}{\partial x_n}
                        \end{bmatrix}
$$

gradient는 function value의 변화값을 의미한다. 참고로, derivative는 gradient와 매우 비슷하지만, 딱 하나, vector shape만이 다르다. gradient를 transpose하면 derivative at $x$ 가 된다.  

Hessian Matrix는 다음과 같다.  

$$
H = \bigtriangledown ^{2} f(x) = \left[ \frac{\partial^{2} f(x)}{\partial x_i \partial x_j} \right] 
  = \begin{bmatrix} 
        \frac{\partial^2 f(x)}{\partial x_1 ^2} & \frac{\partial^{2} f(x)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^{2} f(x)}{\partial x_1 \partial x_D} \\ 
        \vdots &  & & \vdots \\
        \frac{\partial^{2} f(x)}{\partial x_D \partial x_1} & \frac{\partial^{2} f(x)}{\partial x_D \partial x_2} & \cdots & \frac{\partial^{2} f(x)}{\partial x_D \partial x_D}
    \end{bmatrix}
$$

즉, Hessian은 function value를 parameter에 대해 2번 미분한 것과 동일하며, 이는 곧, gradient의 미분과 같음을 의미한다. 또, 정의에 의해 Hessian은 반드시 symmetic하다.  

Taylor Series에 대해서도 다루는데, 아마 이 내용은 고등학생 때 자주 다루었을 것이므로 과감히 정리를 생략한다.  

그래서, 왜 Hessian을 사용할까? Hessian의 도입 배경을 먼저 살펴보자.  

![fig 2. why hessian?](/assets/img/CSED515/chap3/image-1.png){: width="500"}

위의 그래프에서, $f(x)$ 가 objective function이다. $f_{quad} (x)$ 는 $f(x)$ 에 대한, $x_k$ 에서의 approximation이다. (taylor 급수에서 2차까지만 사용.) 위의 그림에서 볼 수 있듯이, Newton's method로 근사시킨 그래프의 minimum이나 maximum이, 실제 그래프와 비교할 때, 그럴 듯한 값이 된다. 즉, 꽤나 비슷한 값을 내놓으므로, Newton's method를 사용해도 큰 문제는 없다는 뜻이다. 그리고, 이 슬라이드에서 명시적으로 언급하고 있지는 않지만, Hessian을 사용하게 되면, gradient가 더 빠르게 하강하는 지역을 예측할 수 있어, 더 빠르게 수렴하는 방향으로 이동할 수 있다는 장점 또한 존재하며, 복잡한 수식을 2차함수로 치환하여 계산하는 만큼, 비교적 계산이 용이해질 수 있다.  

이제 도입 동기를 살펴보았으니, 실제로 어떤 방식으로 parameter를 update하는 지 살펴보자.  

<div class="callout">
  <div class="callout-header">Newton's Method</div>
  <div class="callout-body" markdown="1">

$$
w^{new} \leftarrow w^{old} - \left[ \bigtriangledown^2 \mathcal{L} (w^{old}) \right]^{-1} \bigtriangledown \mathcal{L}(w^{old})
$$

  </div>
</div>
<br>

GD와 다른 점을 찾자면, learning rate를 사용하지 않는다는 점을 찾을 수 있다. 위의 식을 Hessian의 inverse를 learning rate처럼 사용한다고 해석할 수도 있다.  
하지만, Newton's Method는 굉장히 큰 단점이 있는데, 바로 computational cost다. Hessian 계산에는 $O(N D^2)$이 소모되고, Inverse of Hessian 계산에는 $O(D^3)$ 이 소모된다.  

참고로, $L(w)$ 가 "strongly" convex 하다면 (예를 들어, L2 regularized convex function), Hessian은 반드시 positive definite하여 inverse가 항상 가능하다.  

#### proof  

위의 식을 증명해보자.  

먼저 $\mathcal{L}(w)$ 를 $w = w^{(k)}$ 에서 2차 근사시켜보자. 그러면, 다음과 같은 근사 식을 얻을 수 있다.  

$$
\mathcal{L}(w) \approx \mathcal{L}(w^{(k)}) + \left[ \bigtriangledown \mathcal{L} (w^{(k)}) \right]^{T} \left(w - w^{(k)} \right) + \frac{1}{2} \left(w - w^{(k)} \right)^{T} \bigtriangledown^{2}\mathcal{L}(w^{(k)}) \left(w - w^{(k)} \right) = \tilde{\mathcal{L}}(w)
$$

이 함수를 미분하여 0이 되는 지점을 찾는다.  

$$
\bigtriangledown \tilde{\mathcal{L}}(w) = \bigtriangledown \mathcal{L}(w^{(k)}) + \bigtriangledown^2 \mathcal{L}(w^{(k)}) (w - w^{(k)}) = 0
$$

위의 식을 정리하면, 다음과 같은 결과를 얻을 수 있다.  

$$
w = w^{(k)} - \left[ \bigtriangledown^2 \mathcal{L}(w^{(k)}) \right]^{-1} \bigtriangledown \mathcal{L}(w^{(k)})
$$

#### Newton's Method for Logistic Regression  

logistic regression의 loss function에 Newton's method를 적용해보자.  

$$
\mathcal{J}(w) = - \sum_{n=1}^{N}\left[y_n \log{\hat{y}_n} + (1 - y_n) \log (1 - \hat{y}_n)\right]
$$

이 식의 Gradient는 다음과 같다.  

$$
\bigtriangledown \mathcal{L}(w) = -\sum_{n=1}^{N} (y_n - \hat{y}_n)x_n^{T}
$$

이 식의 hessian은 다음과 같다.  

$$
\bigtriangledown^2 \mathcal{L}(w) = \bigtriangledown \left [ \bigtriangledown \mathcal{L}(w) \right]
                                  = \frac{\partial}{\partial w} \left[ - \sum_{n=1}^{N} (y_n - \hat{y}_n)x_n^T \right] 
                                  = \sum_{n=1}^{N} \hat{y}_n (1- \hat{y}_n)x_n x_n^T
$$

수식적으로는 어느 정도 납득이 된다. 이를 어떻게 기하학적으로 이해할 수 있을까? 우리는 chapter 2에서 function의 contour plot이 elongated elliptical (긴 타원형)일 때, gradient descent에 어떤 문제가 발생하는지 관찰했었다. 해당 plot을 다시 가져오면 다음과 같다.  

![Fig 3. Contour plot of elongated elliptical function](/assets/img/CSED515/chap3/3-3.png){: width="500"}

step size가 작으면 왼쪽의 plot과 같이 wide plat한 부분에서 gradient의 값이 작아 update가 매우 느려지게 된다. 반면 step size가 큰 경우, gradient descent의 업데이트가 zig-zag가 된다. 이는 contour plot에서 narrow steep한 부분의 gradient의 반대 방향이 이 steep한 부분을 가로지르는 방향이 되며 동시에 큰 gradient 값을 가지기 때문이다. 이를 Newton's method를 통해 완화할 수 있다.  

Hessian matrix $H$ 은 위에서 살펴보았듯이 gradient가 얼마나 빠르게 증가하는지 또는 감소하는지를 나타낸다. 이를 Inverse 했으므로, gradient가 빠르게 변화하는 부분(narrow steep)은 Hessian의 역행렬이 작다. 반대로 gradient가 느리게 변하는 부분(wide flat)은 Hessian의 역행렬이 큰 값을 가지게 된다. 이를 gradient와 곱하게 되면, narrow steep 했던 부분의 gradient가 작아지고, wide flat 했던 부분의 gradient가 커지면서 elongated elliptical 했던 contour line이 circular line 모양에 가깝게 변하게 된다. 따라서 gradient update가 더 optimal point 방향으로 빠르게 수렴하게 된다.  

#### Logistic Regression for Multiclass Classification  

logistic regression을 무려 multiclass classification에 적용할 수 있다는 사실. input과 output을 softmax transformation으로 매핑한다.  

$$
p(y = k|x) = \frac{exp(w_k^Tx)}{\sum_{j=1}^{K} exp(w_j^Tx)}
$$

이 식의 negative log-likelihood를 계산하면 다음과 같다.  

$$
\mathcal{L}(w) = - \sum_{n=1}^{N}\sum_{k=1}^{K} y_{kn} \log{p(y_n = k | x_n)}
$$

where $y_{kn} = \mathbb{1}[y_n = k]$