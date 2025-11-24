---
layout: post
title:  "[CSED515] 4. Neural Network"

categories:
  - Machine_Learning
tags:
  - [CSED515, ML, Lecture]

toc: true
toc_sticky: true

date: 2025-10-27 22:18:00 +0900
last_modified_at: 2025-10-27 23:34:00 +0900
---

# Chap 4. Neural Network  

## Perceptron: A Neural Network for a Single Neuron  

### Definition  

Perceptron은 single-layer, single neuron neural network를 말하며, thresholding activation function을 가지고 있다. 아래와 같이 나타낼 수 있다.  

$$
y = \text{sign}(w^Tx + b)
$$

일반적으로, online training이며, mistake-driven 방식을 취한다. 이는 misclassified 되었을 때에만 update를 진행함을 의미한다. perceptron에 대한 학습은 data가 linearly separable하다면 convergence가 보장된다.  

### Mistakes  

Perceptron의 목적은 mistake를 최소로 만드는 것이다. Mistake는 다음과 같이 정의한다.  

<div class="callout">
  <div class="callout-header">Definition: Mistakes</div>
  <div class="callout-body" markdown="1">

We say that $w$ makes a mistake if  

$$
y \neq \text{sign}(w^T x + b)
$$

  </div>
</div>
<br>

Perceptron의 예측과 실제 label이 다른 것을 mistake라고 칭한다. perceptron은 data가 linearly separable하다는 가정 하에, 다음 조건을 만족하도록 학습된다.  

$$
y_n w^T x_n > 0 \; \forall n
$$

### Learning Objective

위에서 정의한 Mistake를 기반으로 Learning objective를 더 엄밀하게 기술해보자.  

<div class="callout">
  <div class="callout-header">Learning Objective (Perceptron)</div>
  <div class="callout-body" markdown="1">

The mistake criterion leads to the following objective function to be minimized:  

$$
\mathcal{J}(w) = -\sum_{n \in \mathcal{M}(w)} {y_n w^T x_n}
$$

where $\mathcal{M}(w) = \{n : y_nw^Tx_n < 0 \}$ is the set of **misclassified** samples under $w$ .  

  </div>
</div>
<br>

참고로 bias term b는 편의를 위해 없앤 것이다. Objective에 대해 굳이 풀어서 설명하자면, 잘못 분류된 샘플에 대한 모델의 예측값을 모두 더한 값을 최소화하는 것이다. $y_n$ 이 곱해지면서 자연스럽게 $y_nw^Tx_n$ 은 음수가 되므로, 모두 더한 값에 -를 붙여 양수로 만든다. 이 값을 최소화하는 것이 목적이다.  

Perceptron을 online learning 하는 과정은 다음과 같이 요약할 수 있다.  

1. training sample $(x_n, y_n)$ 을 random sampling  
2. 잘못 분류된 sample에 대해 $w_{t+1} \leftarrow w_{t} + \alpha y_nx_n$ update 식을 따라 parameter 업데이트  
3. Converge할 때까지 위의 과정을 반복  

![Update_fig](/assets/img/CSED515/chap4/4-0.jpg){: width="1500"}

그림으로 설명하자면 위와 같다. $\alpha = 1$ 로 가정할 때 위처럼 misclassified된 데이터의 반대 벡터와 w를 합성하여 새로운 classify line을 생성한다.  

### Convergence  

위의 알고리즘을 따를 때, perceptron은 과연 convergence를 보장할까? update step이 매우 길어지는 것은 아닐까? 이에 대해 알아보도록 하자.  
먼저, 지금까지 가정했던 Linearly Separable한 상황을 똑같이 가정한다. 그렇다면, 주어진 데이터셋을 완벽히 separation하는 
$w^{\*}$ 
와 
$b^{\*}$ 
가 존재한다. 우선 편의를 위해 bias term $b$ 는 제외하고 설명한 뒤, 나중에 bias term을 추가할 예정이다.  

알고리즘이 convergence하지 않는다고 가정하자. 그렇다면 알고리즘은 무한히 업데이트를 진행하게 된다.  
이때, 최적의 파라미터 $w^{\*}$ 와 현재 업데이트하는 파라미터 $w$ 의 내적은 다음과 같다.  

$$
\begin{align}
w_{t+1}w^{*} &= (w_{t} + y_{n}x_{n})w^{*} \\
          &= w_{t}w^{*} + y_{n}(x_{n}w^{*})
\end{align}
$$

이때, $w^{\*}$ 는 최적 파라미터이므로, $x_{n}w^{\*}$ 는 항상 올바른 prediction을 해야 하므로 $y_{n}$ 과 부호가 동일하다. 따라서, $y_{n}(x_{n}w^{\*})$ 는 항상 양수이다. 따라서, $w_{t+1}w^{\*}$ 는 $w_{t}w^{\*}$ 보다 항상 크다. 이때, 매 스텝마다 증가하게 되므로, 총 K step만큼 update가 진행되었다고 할 때, $w_{t}w^{\*}$ 는 $\mathcal{O}\left( K \right)$ 에 비례하여 증가한다.  

한편, 다음 식을 생각해보자.  

$$
\begin{align}
|w_{k+1}|^{2} &= (w_{k} + \alpha y_{n}x_{n})^{2} \\
              &= (w_{k})^{2} + 2\alpha y_{n}(w_{k}x_{n}) + \alpha^{2} x_{n}^{2}
\end{align}
$$

이때, update는 misclassify 에만 발생하므로, $y_{n}(w_{k}x_{n})$ 는 반드시 음수가 된다. 따라서,  

$$
\begin{align}
|w_{k+1}|^{2} < (w_{k})^{2} + \alpha^{2} x_{n}^{2}
\end{align}
$$

이때, dataset은 유한하므로, 
$|x_{n}|^{2}$ 
의 최댓값이 존재한다. 이를 
$R^{2}$ 
이라 해보자. 그러면, 
$|w_{0}|^{2} = c$
라 할 때, 위의 식에 따라, K step 이후 
$|w_{k+1}|^{2}$ 
의 최댓값은 다음과 같다.  

$$
|w_{K}|^{2} <= c + K R^{2}
$$

그렇다면, 
$|w_{k}|$ 
는 step에 대해 
$O(\sqrt{K})$ 
에 비례하여 증가하게 된다.  

다음 cosine similarity에 이 두 가지 결론을 적용해보자.  

$$
\cos(\theta_{k}) = \frac{w_{k}w^{*}}{|w_{k}| |w^{*}|}
$$

이 식의 분자는 $\mathcal{O} (k)$, 분모는 $\mathcal{O} (\sqrt{k})$ 를 따르므로,$k \rightarrow \infty$ 일 때, 식이 $\mathcal{O}(\sqrt{K})$ 로 증가해야 한다. 하지만, cosine의 최댓값은 1임이 명백히 알려져 있다. 따라서 이는 모순이다. 즉, 귀류법에 의해 가정인 '알고리즘이 무한히 업데이트를 진행하게 된다.' 가 거짓이 되므로, 알고리즘이 수렴함을 증명할 수 있다.  

만약 bias term을 추가하고 싶다면, $\tilde{w} = \begin{bmatrix} w \\ b \end{bmatrix}$, $\tilde{x} = \begin{bmatrix} x \\ 1 \end{bmatrix}$ 로 놓으면 된다. 그러면,  

$$
\tilde{w}^{T}\tilde{x} = (w^{T}x + b \cdot 1) = w^{T}x + b
$$

가 되어 일반화된 linear classification 식이 되며, 이 $\tilde{w}$ 와 $\tilde{x}$ 를 사용하여 위와 같은 논리로 convergence를 증명할 수 있다.  

### Limitation  

다만, perceptron은 오직 linearly separable한 데이터에만 적용 가능하므로 한계가 너무 명확하다.  

![XOR](/assets/img/CSED515/chap4/4-1.png){: width="500"}

예를 들어 위와 같은 데이터에 대해서는 Perceptron이 classification을 완벽히 수행할 수 없다.  
<br>

## Multilayer Perceptron (MLP): A Neural Network for Several Layers of Neurons  

### Example: XOR Problem  

![decision boundary](/assets/img/CSED515/chap4/4-2.jpg){: width="1000"}

이런 문제를 단순히 Perceptron을 여러 개 stacking하여 해결할 수 있다. 위에서 본 XOR 문제에 대해, 다음 두 perceptron을 생각해보자.  

$$
h_1(x) = \mathbb{1}\left[ x_1 + x_2 - 0.5 > 0 \right] \\
h_2(x) = \mathbb{1}\left[ x_1 + x_2 - 1.5 > 0 \right]
$$

이 각각의 perceptron은 다음과 같이 Boolean operator로 생각할 수 있다.  

$$
h_1(x) = \mathbb{1}\left[ x_1 + x_2 - 0.5 > 0 \right] \Rightarrow x_1 \text{ OR } x_2 \\
h_2(x) = \mathbb{1}\left[ x_1 + x_2 - 1.5 > 0 \right] \Rightarrow x_1 \text{ AND } x_2  
$$

이제 두 perceptron을 다음과 같이 결합한다.  

$$
\begin{align}
y(h) = \mathbb{h_1 - h_2 - 0.5 > 0} &= \mathbb{1}\left[h_1 + (1-h_2) - 1.5 > 0 \right] \\
                                    &\Rightarrow h_1 \text{ AND } (\text{ NOT } h_2) = x_1 \text{ XOR } x_2
\end{align}
$$

$y(h)$ 의 경우, 위의 그림에서 $h1$과 $h2$ 사이의 공간이 positive, 나머지 공간은 negative가 되도록 한다.  

조금 다르게 해석할 수도 있는데, $h_{1}$ 과 $h_{2}$ 는 일종의 transformation을 수행한다고 볼 수 있다. 즉, space 자체를 바꾸는 것으로 이해할 수 있다.  

![Fig. Transformation](/assets/img/CSED515/chap4/4-11.jpg){: width="1000"}

이와 같이, $h_{1}, h_{2}$ 로 구성된 space로 data point를 옮기는 것으로 이해할 수도 있다. 위의 figure를 보면 알 수 있듯이 기존의 space에서는 lineary non-separable 했던 data point들이, $h_{1}-h_{2}$ space에서는 linearly separable하게 됨을 알 수 있다.  

### Perceptron as Logical Operator  

위에서 살펴볼 수 있듯이, perceptron을 일종의 logical operator처럼 취급할 수 있고, 하나의 perceptron은 AND, OR 연산자 역할을 수행할 수 있다. 또한 NOT 연산 또한 수행할 수 있다. 이때, AND와 OR 그리고 NOT을 결합하면 모든 logical 연산을 수행할 수 있다. 즉, perceptron을 여러 개 사용함으로써 모든 logical operator를 구현할 수 있다. 이를 Universal Approximation Theorem이라 한다. 사실 CSED273에서 배운 것처럼 생각하면 된다.  

이에 대해 직관적으로 생각해보자. input space가 $\{0, 1 \}$ 또는 $\{-1, 1 \}$ 이고 input dimension이 $D$ 라고 하자. 그러면 가능한 모든 input expression의 개수는 $2^{D}$ 다. 그렇다면, 총 $2^{D}$ 개의 perceptron을 사용하여 각각이 하나의 특정 input을 나타내도록 한다고 해보자. 그 뒤에 마지막 layer가 positive label에 대해 OR 연산을 한다고 하면, 가능한 모든 input을 처리할 수 있게 된다. SOP(Sum of Product)로 생각하면 된다.  

### Example 2  

![example 2](/assets/img/CSED515/chap4/4-3.png){: width="500"}

위의 example을 보자. 이 example 또한 linearly separable하지 않다. 이 데이터를 다음과 같이 나누어보자.  

![example 2 four boundaries](/assets/img/CSED515/chap4/4-4.png){: width="1000"}

위의 figure처럼 네 영역으로 나눈 뒤, 결합하여 데이터를 올바르게 classification할 수 있다.  

### Toward Deep Learning  

Perceptron을 쌓아 Neural Network를 만들었다고 가정하자. 그렇다면 이 Neural Network는 어떻게 학습시켜야 할까? Depp Learning은 미분 가능한 함수들을 조합하여 function approximation을 하는데, 지금까지 배운 perceptron은 아쉽게도 미분이 의미가 없다. 왜냐하면, 미분이 불가능한 지점을 제외한 모든 부분에서 gradient가 0이기 때문이다. 이 문제를 해결해야, perceptron을 deep learning에 적용할 수 있다. 어떻게 해결할까?  

![Activation Functions](/assets/img/CSED515/chap4/4-5.png){: width="500"}

가장 널리 사용되고 있는 방법은 activation 함수를 사용하는 것이다. 위의 figure는 매우 많은 activation function 중에서 대표적인 네 가지를 나타낸 것이다. 살펴보면, 미분이 가능하도록 만든 것을 알 수 있을 것이다.  

### Multi-Layer Perceptron  

<div class="callout">
  <div class="callout-header">Definition: Fully Connected Layer</div>
  <div class="callout-body" markdown="1">

A layer where $D$ input features and $H$ output units are connected "with" weight $W$ parameters, bias $b$ parameters, and activation functions i.e.,  

$$
\phi (Wx + b)
$$

is a fully connected layer, where $x \in \mathbb{R}^{D}, W \in \mathbb{R}^{H \times D}, b \in \mathbb{R}^{H}$  

  </div>
</div>
<br>

MLP를 살펴보기 전에, 먼저 fully connected layer를 이해해야 한다. 간단하게, 모든 input이 모든 output과 각각 연결된 layer를 fully connected layer라고 한다.  

<div class="callout">
  <div class="callout-header">Definition: MLP</div>
  <div class="callout-body" markdown="1">

A multilayer perceptron (MLP) is a stack of fully-connected layers, i.e.,  

$$
h^{(1)} = f^{(1)}(x) = \phi(W_1x + b_1) \\
h^{(2)} = f^{(2)}(h^{(1)}) = \phi(W_2 h^{(1)} + b_1) \\
\vdots \\
y = f^{(L)}\left(h^{(L-1)} \right)
$$

or simply  

$$
y = f^{(L)} \circ \cdots \circ f^{(1)}(x).
$$

  </div>
</div>
<br>

위에서 살펴볼 수 있듯이, MLP는 fully-connected layer를 쌓은 것을 의미한다. 덕분에, MLP는 Modularity를 제공하여, MLP 위에 MLP를 쌓는 등 매우 자유롭게 MLP를 조작할 수 있다.  

<div class="callout">
  <div class="callout-header">Learning Objective: MLP</div>
  <div class="callout-body" markdown="1">

$$
\sum_{n=1}^{N} \ell (y_n, \hat{f}_n)  
$$

  </div>
</div>
<br>

objective는 보기에는 매우 간단하다. 단순하게, 최종적으로 나온 예측값과 실제 라벨간의 차이를 줄이는 것이 목적이다. 문제는, MLP는 너무 복잡해서, parameter를 train하기 어려워보인다는 것이다. 물론, 이미 정답을 알고 있겠지만, 이는 back propagation을 통해 해결한다.  

![BackPropagation](/assets/img/CSED515/chap4/4-6.png){: width="500"}

Local Gradient는 현재 perceptron 내에서의 gradient를 말한다. 현재 perceptron의 output의 input에 대한 gradient가 된다.  
Upstream gradient는 위의 perceptron에서 내려오는 gradient다. Loss의 현재 perceptron의 output에 대한 gradient다.  
downstream gradient는 local gradient와 upstream gradient를 곱한 것을 말한다. 이는 하위 layer에 그대로 전달되어 그대로 upstream gradient가 된다.  

![example of backpropagation](/assets/img/CSED515/chap4/4-7.png){: width="1500"}

다음 예시에 대해 backpropagation을 수행해보도록 하자. 먼저, backpropagation을 수행하기 위해서는 forward pass를 한 번 거쳐야 한다.  

![forward pass](/assets/img/CSED515/chap4/4-8.jpg){: width="1500"}

다음과 같이 forward pass를 했다면, 이제 backpropagation을 할 수 있다.  

![Backpropgation](/assets/img/CSED515/chap4/4-9.jpg){: width="1500"}

그런데, 이 과정을 간단하게 만들 수 있다. 위의 식에서, sigmoid의 역할을 하는 perceptron이 있는데, 이를 하나로 합하는 것이다.  

![Sigmoid](/assets/img/CSED515/chap4/4-10.png){: width="1500"}

이렇게 합하면, sigmoid의 gradient는 이미 잘 알려져있으므로 더 간단하게 layer를 나타낼 수 있게 된다.  
이런 방식으로, 아무리 neural network가 깊더라도 gradient를 구할 수 있다.  

## Loss Functions for Neural Networks  

### Regression  

Regression task에는 MSE를 사용할 수 있다.  

$$
\frac{1}{2N} \sum_{n=1}^{N} |y_{n} - \hat{f}_{n} |^{2}
$$

Classification task에는 Cross Entropy loss를 사용할 수 있다.  

$$
\hat{f}_{k} = \frac{\exp\left( w_{k}^{T}h \right)}{\sum_{j=1}^{k} \exp\left( w_{j}^{T}h \right)}
$$

where 
$\hat{f} = \left[ \hat{f}\_{1}, \ldots, \hat{f}\_{K} \right]^{T}$

Cross Entropy loss에 대해 더 자세히(?) 알기 위해 KL Divergence를 보도록 하자. KL Divergence를 다음과 같이 정의한다.  

<div class="callout">
  <div class="callout-header">KL Divergence</div>
  <div class="callout-body" markdown="1">

Kullback-Leibler divergence measures a distance between two distributions as follows:  

$$
KL(p, q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

  </div>
</div>
<br>

discrete distribution을 가정한 식이다. ($\sum$ 사용.) 이때, KL divergence는 asymmetric하다. 즉, $KL(p,q) \neq KL(q,p)$ 다. 일반적으로, $p$ 에 true distribution, $q$ 에 true distribution과 비교할 distribution을 배치한다.  

한편, Cross Entropy는 다음과 같이 정의할 수 있다.  

<div class="callout">
  <div class="callout-header">Cross Entropy</div>
  <div class="callout-body" markdown="1">

Cross entropy between two probability distributions $p$ and $q$ (over the same underlying set of events) measures a distance between two distributions as follows:  

$$
H(p, q) = \mathbb{E}_{p}\left[ -\log q \right] = - \sum_{x} p(x) \log q(x)
$$

  </div>
</div>
<br>

왜 Cross entropy를 사용할까? 다시 말해, KL divergence와 비교할 때 cross entropy가 가지는 장점은 무엇일까?  

먼저, $H(p,q) = H(p) + KL(p,q)$ 가 성립한다. 이때, $H(p) = H(p,p)$ 이다. 이 식에서, $H(p)$ 는 우리가 최소화하려는 예측 분포인 $q$ 와는 관계가 없는 true distribution이므로, 최소화할 때 무시할 수 있다. 즉, $H(p,q)$ 를 최소화하는 것은 곧 $KL(p,q)$ 를 최소화하는 것과 동일하다는 것이다. 이때, KL divergence보다 Cross Entropy가 더 계산하기 쉬우므로 CE loss를 사용한다.  
또, Cross Entropy는 확률적으로 해석 가능하다. 정확히, CE를 최소화하는 것은 MLE와 동일하다. Likelihood를 최대화하는 것과 Cross Entropy를 최소화하는 것이 동일하므로, CE는 꽤 괜찮은 objective가 된다.  
마지막으로, 위에서 본 KL divergence는 divide by zero 상황이 나올 수 있지만, CE는 divide by zero가 없다. 이는 일반화나 구현의 관점에서 볼 때 큰 이점이 될 수 있다.  

CE를 classification에 사용한다고 가정하자. True distribution인 $p$ 로 올바른 class만 1이고 다른 class는 0인 one-hot vector를 사용할 수 있다. 그러면 CE 식에 의해, 올바른 class에 대한 모델의 예측 negative-log likelihood가 된다. 왜냐하면, 식이 다음과 같이 전개되기 때문이다.  

$$
p(x_{n}) \log q(x_{n}) = \begin{cases} 
                    \log q(x_{n}) & \text{if } p(x_{n}) = 1 \text{(correct class)} \\
                    0 & \text{if } p(x_{n}) = 0 \text{(wrong class)}
                 \end{cases}
$$