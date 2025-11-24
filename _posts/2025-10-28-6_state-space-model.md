---
layout: post
title:  "[CSED515] 6. State Space Model"

categories:
  - Machine_Learning
tags:
  - [CSED515, ML, Lecture]

toc: true
toc_sticky: true

date: 2025-10-28 02:00:00 +0900
last_modified_at: 2025-10-28 02:00:00 +0900
---

# State Space Model  

voice와 같은 sequential data를 처리하는 방식에 대해 다루는 듯.  

## Markov Models  

기존의 sequential data를 분석하는 모델은 다음과 같은 modeling을 사용했다.  

$$
p(x_{1}, \ldots, x_{N}) = \prod_{n=1}^{N} p(x_{n} | x_{1}, \ldots, x_{n-1})
$$

이 모델은 i.i.d.를 가정하고 modeling (independent를 가정하고 단순히 각 확률을 곱)하는 것보다 sequential data 처리에서 더 좋은 성능을 보인다. 하지만, 문제는 parameter 수가 감당이 되지 않는다는 점이다. 조건부 확률을 모두 저장하기 위해 사용할 parameter 수를 생각해보자. variable이 discrete하다면, table 형식으로 저장한다고 생각할 때, variable이 가질 수 있는 값의 개수가 variable의 개수를 지수로 가지는 꼴이 되어 parameter 수가 기하급수적으로 증가하게 된다.  

### First-order Markov Chain  

![Fig 1. 1st-order Markov chain](/assets/img/CSED515/chap6/6-1.png){: width="500"}

한편 Markov 모델은 현재 상태에는 오직 바로 이전의 상태만 영향을 미친다는 가정(**1st-order Markov Model**)을 한다. 이를 수식으로 나타내면 다음과 같다.  

$$
p(x_{1}, \ldots, x_{N}) = p(x_{1}) \prod_{n=2}^{N} p(x_{n} | x_{n-1})
$$

이 모델은 homogeneous Markov chain이라고도 부르는데, 그 이유는 전이 확률 
$p(x\_n | x\_{n-1})$ 
이 시간에 따라 변하지 않고 항상 동일하다고 가정하기 때문이다. 한편 이 모델은 지나치게 단순화한 모델일 수 있다. 예를 들어, 이 모델이 
$ x\_{t} = x\_{t-1} + x\_{t-2}$ 
와 같이 단순히 이전 timestep만이 영향을 주는 것이 아닌 더 이전의 timestep의 데이터에도 영향을 받는 경우를 잘 모델링할 수 있을까? 당연히 그렇지 않다. 이를 해결하기 위해 order를 늘릴 수 있다.  

### Second-order Markov Chain  

![Fig 2. 2nd-order Markov Chain](/assets/img/CSED515/chap6/6-2.png){: width="500"}

Second-order Markov Chain은 현재 state 이전의 두 state를 고려한다. 이를 식으로 나타내면 다음과 같다.  

$$
p(x_1, \ldots, x_N) = p(x_1) p(x_2 | x_1) \prod_{n=3}^{N} p(x_n | x_{n-1}, x_{n-2})
$$

이렇게 하면 더 풍부한 표현력을 가지게 되지만, 그 반대급부로 parameter 수가 늘어나게 된다. 이런 trade-off를 latent variable을 추가하여 해결할 수 있다.  

### Markov Chain of Latent Variables  

![Fig 3. Markov Chain with Latent Variables](/assets/img/CSED515/chap6/6-3.png){: width="500"}

위와 같이 눈에 보이지 않는 latent variable을 추가하여 markov chain을 만든 것을 **State Space Model** 이라고 말한다. 이러한 modeling을 수식으로 표현하면 다음과 같다.  

$$
p(x_{1}, \ldots, x_{N}, z_{1}, \ldots, z_{N}) = p(z_{1})\left[ \prod_{n=2}^{N} p(z_{n} | z_{n-1}) \right] \left[ \prod_{n=1}^{N} p(x_{n} | z_{n}) \right]
$$

이런 모델 중 latent variable이 discrete한 경우를 Hidden Markov Model (HMM), continuous한 경우를 dynamic system이라 부른다.  

## Hidden Markov Models  

hidden markov model의 modeling은 크게 두 가지 components로 나눌 수 있다.  

transition probability: 
$ p(z\_{n} | z\_{n-1})$  
emission probability: 
$ p(x\_{n} | z\_{n})$  

### Transition Probability  

![Fig 4. Transition Probability](/assets/img/CSED515/chap6/6-4.png){: width="500"}

먼저, Transition Probability에 대해 살펴보자. latent variable이 discrete하므로, latent variable이 가질 수 있는 state의 개수를 K라 하자. 이제, latent variable을 일종의 one-hot vector로 표현할 수 있다.  

$$
z_{n} \in \left\{ 0, 1 \right\}^{K}
$$

이떄, 특정 latent variable($z_{n-1}$)에서, 다음 latent variable($z_{n}$)로 넘어가는 transition probability는 아래와 같이 표현할 수 있다.  

$$
A_{j, k} := p(z_{n, k} = 1 | z_{n-1, j} = 1)
$$

where $0 \leq A_{j, k} \leq 1$ with $\sum_{k} A_{j, k} = 1$  

notation을 보면, latent variable이 가질 수 있는 state에 일종의 label을 붙여주고, 이전 state를 i, 다음 state를 j라 할 때, i에서 j로 transition될 확률을 $A_{i, j}$ 로 표현하는 것을 알 수 있다.  
이때, $A$ 는 총 $K(K-1)$ 개의 independent variable을 가지게 될 것이다. 이는 homogeneous 가정이 HMM에도 그대로 적용되기 때문으로 variable의 개수는 parameter의 개수에 영향을 주지 않으며, 전이 전 state의 개수 K, 전이 후 state의 개수 K개에서, 전이 후 는 저장하지 않아도 1 - sum으로 계산 가능하므로state 중 하나, 총 parameter의 개수는 K(K-1)이 된다.  

이를 더 수학적으로 표현해보자.  

<div class="callout">
  <div class="callout-header">Transition Probability</div>
  <div class="callout-body" markdown="1">

$$
p(z_{n} | z_{n-1} ) := p(z_{n} | z_{n-1}, A) := \prod_{k=1}^{K} \prod_{j=1}^{K} A_{j, k}^{z_{n-1}, j}z_{n, k}
$$

where 
$p(z\_{1}) := p(z\_{1} | \pi) = \prod\_{k=1}^{K} \pi\_{k}^{z\_{1}, k}$  

  </div>
</div>
<br>  

이 식을 이해하기 위해 먼저 $A$ 의 지수부분을 살펴보자. $z_{n-1, j}z_{n,k}$ 에서, $z_{n-1, j}$ 는 오직 n-1번째 latent variable이 j state일 때에만 1이고 다른 경우는 0이다. 마찬가지로, $z_{n, k}$ 도 n번째 latent variable이 k state일 때에만 1이고, 다른 경우는 0이다. 따라서, $z_{n-1, j}z_{n,k}$ 는 n-1 번째 latent variable이 j state, n 번째 latent variable이 k state일 때에만 1이고, 다른 경우 0이 된다. 이를 $A$ 의 지수에 놓음으로써, 일종의 switch처럼 작동하게 된다. n-1 번째 latent variable이 j state, n 번째 latent variable이 k state일 때에는 $A^{1} = A$ 가 되어 실제 transition probability가 곱해지게 되고, 그렇지 않은 경우, $A^{0} = 1$ 이 되어, 곱셈에 영향을 주지 않게 된다. 즉, 주어진 식은 실제 latent variable의 state에 따른 transition probability의 곱을 의미한다.  
$p(z_{1})$ 또한 비슷하게 이해할 수 있다. $p(z_{1})$ 의 latent variable $z_{1}$ 은 이전의 latent variable이 없어 transition probability 식의 영향을 받지 못하며, initial state probability인 $\pi$ 에 의해 결정된다. 이때, $k$가 $z_{1}$ 의 실제 state라면, $z_{1, k}$ 값은 1이 되어 해당 $k$ state에 대한 initial state probability는 곱셈에 정상적으로 들어가게 되고, $k$ 가 $z_{1}$ 의 실제 state가 아니라면, $z_{1, k}$ 가 0이 되어, 해당 state에 대한 initial state probability가 몇이든, 0제곱이 되어 1이 된다. 1은 곱셈에서 항등원이므로, 결과에 영향을 주지 않게 되어, 이 식은 결국 $z_{1}$ 의 실제 state에 대한 initial state probability 자체가 된다.  

### Emission Probability  

emission probability는 다음과 같이 나타낼 수 있다.  

$$
p( x_{n} | z_{n} ) := p( x_{n} | z_{n}, \phi) = \prod_{k=1}^{K} p(x_{n} | \phi_{k})^{z_{n, k}}
$$

일단 식을 보면, $x_{n}$ 은 $z_{n}$ 의 영향도 받지만, 추가적인 parameter $\phi$ 의 영향도 받는다. 마지막 term의 지수 부분에 $z_{n, k}$ 를 놓아 switch 역할을 하도록 했다. n번째 latent variable의 실제 state와 일치하는 state일 때에만 1이 되어 확률에 곱해지고, 그렇지 않은 경우는 0이 되어 확률을 1로 만들어버려 곱셈에 영향을 주지 않도록 만든다. 결국, latent variable $z_{n}$ 에서 variable $x_n$ 으로 emission될 확률은 오직 parameter $\phi$ 의 element 중, $z_{n}$ 의 실제 state에 해당하는 element의 영향만 받게 된다.  

이를 modeling하는 방법은 여러가지가 있는데, $x_{n}$ 이 discrete하면 table을 사용하면 되며, continuous 한 경우 Gaussian이나 probabilistic classifier를 사용할 수 있다. 예를 들어, 
$p(x\_{n} | z\_{n}) = \frac{p(x\_{n}) p(z\_{n} | x\_{n})}{p(z\_{n})}$, 
$p(z\_{n} | x\_{n}) := \text{Softmax}(f(x\_{n}))$ 
을 사용할 수 있다.  

### The Viterbi Algorithm  

이미 학습된 파라미터 $(A, \pi, \phi)$ 가 있다고 가정하자. 이때, 가장 가능성 높은 latent sequence는 어떻게 찾을 수 있을까? 가장 간단한 방법은 확률을 최대로 만드는 latent variable을 구하는 것이다. 아래와 같이 확률을 최대로 만드는 latent variables 쌍을 찾는 것을 **inference rule**이라 한다.  

![Fig 5. net](/assets/img/CSED515/chap6/6-5.png){: width="500"}

$$
\max_{z_{1}, \ldots, z_{N}} \ln p(x_{1}, \ldots, x_{N}, z_{1}, \ldots, z_{N})
$$

다만, 이 방법은 $K^{N}$ 개의 경로를 모두 탐색해야 하므로 매우 비효율적이다. 이때, 경로가 겹치는 부분이 존재하므로 dynamic programming을 사용하여 이를 해결하는 것을 생각해볼 수 있다.  

$$
\omega(z_{n}) = \begin{cases}
                \ln p(x_{n}| z_{n}) + \max_{z_{n-1}} \left( \ln (z_{n} | z_{n-1}) + \omega (z_{n-1}) \right) & \text{if } n \geq 2 \\
                \ln p(x_{1}| z_{1}) + \ln (z_{1}) & \text{if } n = 1
                \end{cases}
$$

이때, $\omega(z_{n}) := \max_{z_{1}, \ldots, z_{n-1}} \ln p(x_{1}, \ldots, x_{N}, z_{1}, \ldots, z_{N})$ 이다.  

즉, $\omega(z_{n})$ 은 현재 시점 $n$ 에서 latent variable의 state가 $z_{n}$ 일 때, 시점 1부터 n까지의 observed data $(x_{1}, \ldots, x_{n})$ 을 설명하는 가장 확률이 높은 latent sequence의 log-likelihood다. 이때, 우리는 latent variable이 가질 수 있는 상태를 $K$ 개로 가정하므로, $\omega(z_{n})$ 는 $K$ 개의 값을 가진 vector가 된다.  
그리고, 이 식은 위와 같이 재귀적으로 나타낼 수 있는데, initial step은 매우 직관적으로, latent variable이 $z_{1}$ 일 확률과 $z_{1}$ 일 때, $x_{1}$ 로 emission될 확률을 곱한 것에 log를 취한 것이다. recursive step은 먼저 $\ln (z_{n} | z_{n-1}) + \omega(z_{n-1})$ 의 경우, 시점 1부터, n-1번째 시점까지 $z_{n-1}$ 에 도달하는 최적의 경로에 대한 확률과 해당 state $z_{n-1}$ 에서 $z_{n}$ 으로 가는 확률을 더한 것이다. 이를 maximize하는 $z_{n-1}$ 을 찾으므로, 결국 이 식은 $z_{n}$ 에 도달하는 최적의 경로의 확률을 찾는 식이 되며, 이 확률과 $z_{n}$ 에서 $x_{n}$ 으로 emission될 확률을 더한다.  

한편, 식의 정의에 의해, inference rule은 다음과 같이 나타낼 수 있게 된다.  

$$
\max_{z_{1}, \ldots, z_{N}} \ln p(x_{1}, \ldots, x_{N}, z_{1}, \ldots, z_{N}) = \max_{z_{N}} \max_{z_{1}, \ldots, z_{N-1}} \ln p(x_{1}, \ldots, x_{N}, z_{1}, \ldots, z_{N}) = \max_{z_{N}}\omega(z_{N})
$$

이때, $z_{N}$ 은 discrete하므로, $\omega (z_{n})$ 은 K-dimensional vector다. 식에서, $\omega(z_{n})$ 을 알면, $z_{N}$ 에 대해 max를 하는 것은 간단하기 때문에, $\omega(z_{n})$ 을 dynamic programming (즉, memoization)하여 계산을 더 빠르게 할 수 있다.  

위의 식의 유도 과정을 살펴보도록 하자.  

$$
\begin{aligned}
&\max_{z_{1}, \ldots, z_{N}} \ln p(x_{1}, \ldots, x_{N}, z_{1}, \ldots, z_{N}) \\
&= \max_{z_{1}, \ldots, z_{N}} \ln \left( p(z_{1}) \left[ \prod_{n=2}^{N} p (z_{n} | z_{n-1}) \right] \left[ \prod_{n=1}^{N}p(x_{n} | z_{n}) \right]\right) \quad (\because \text{dependency of Markov structure}) \\
&= \max_{z_{1}, \ldots, z_{N}} \left( \ln p(z_{1}) + \sum_{n=2}^{N} \ln p(z_{n} | z_{n-1}) + \sum_{n=1}^{N} \ln p(x_{n} | z_{n}) \right) \\
&= \max_{z_{N}} \underbrace{\max_{z_{1}, \ldots, z_{N-1}} \left( \ln p(z_{1}) + \sum_{n=2}^{N} \ln p(z_{n} | z_{n-1}) + \sum_{n=1}^{N} \ln p(x_{n} | z_{n}) \right)}_{\omega(z_{N})} \quad (\because \text{basic property of max function: Associative Law})\\
&= \max_{z_{N}} \left( \ln p(x_{N} | z_{N}) + \max_{z_{N-1}} \left( \ln (z_{N} | z_{N-1}) + \underbrace{\max_{z_{1}, \ldots, z_{N-2}} \left( \ln p(z_{1}) + \sum_{n=2}^{N-1} \ln p(z_{n} | z_{n-1}) + \sum_{n=1}^{N-1} \ln p(x_{n} | z_{n}) \right)}_{\omega(z_{N-1})} \right) \right) \\
&= \max_{z_{N}} \left( \ln p(x_{N} | z_{N}) + \max_{z_{N-1}} \left( \ln (z_{N} | z_{N-1}) + \omega(z_{N-1}) \right) \right)
\end{aligned}
$$

### Learning Objective  

Hidden Markov Model의 learning objective는 다음과 같이 나타낼 수 있다.  

<div class="callout">
  <div class="callout-header">Learning Objective</div>
  <div class="callout-body" markdown="1">

Let $\theta = \left( \pi, A, \phi \right)$, then we find $\theta$ that minimizes the negative log-likelihood of data, i.e.,  

$$
\begin{aligned}
& \min_{\theta} \sum_{s=1}^{S} -\ln p(X^{(s)} | \theta) = \min_{\theta} \sum_{s=1}^{S} - \ln \sum_{Z} p (X^{(s)}, Z | \theta) \quad \text{ for unsupervised learning, } \\
& \text{where } X^{(s)} = \left( x_{1}^{(s)}, \ldots, x_{N_{s}}^{(s)} \right)
\end{aligned}
$$

or

$$
\begin{aligned}
& \min_{\theta} \sum_{s=1}^{S} - \ln p(X^{(s)}, Z^{(s)} | \theta) \quad \text{ for supervised learning, } \\
& \text{where } X^{(s)} = \left( x_{1}^{(s)}, \ldots, x_{N_{s}}^{(s)} \right) \text{ and } Z^{(s)} = \left( z_{1}^{(s)}, \ldots, z_{N_{s}}^{(s)} \right)
\end{aligned}
$$

  </div>
</div>
<br>

Unsupervised Learning의 경우, Expectation-Maximization (EM) algorithm을 사용하게 되는데, 아직 배운 내용이 아니라서 자세한 사항은 skip한다. supervised learning의 경우, 간단하게 counting을 하여 학습을 진행하게 되는데, observation sequence와 latent sequence 모두 가지고 있으므로, 해당 sequence를 따를 때의 probability를 최대로 만드는 parameter를 찾는 것이 목표가 된다. 이와 관련하여 더 자세히 살펴보도록 하자.  

<div class="callout">
  <div class="callout-header">Supervised Learning Objective</div>
  <div class="callout-body" markdown="1">

$\min_{\theta} \sum_{s=1}^{S} - \ln p\left( X^{(s)}, Z^{(s)} | \theta \right) = $  

$$
\min_{\theta} - \sum_{s=1}^{S} \ln p(z_{1}^{(s)} | \pi) - \sum_{s=1}^{S} \sum_{n=2}^{N_{s}} \ln p(z_{n}^{(s)} | z_{n-1}^{(s)}, A) - \sum_{s=1}^{S} \sum_{n=1}^{N_{s}} \ln p \left( x_{n}^{(s)} | z_{n}^{(s)}, \phi \right)
$$

where $\theta = \left( \pi, A, \phi \right)$, $X^{(s)} = \left( x_{1}^{(s)}, \ldots, x_{N_{s}}^{(s)} \right)$, and $Z^{(s)} = \left( z_{1}^{(s)}, \ldots, z_{N_{s}}^{(s)} \right)$  

  </div>
</div>
<br>  

식을 직관적으로 이해해보면, 모든 observe sequence에 대해, initial state probability $\pi$ 를 따를 때, initial state가 $z_{1}^(s)$ 일 확률과 Transition probability $A$ 를 따를 때, $z_{n-1}$ 애서 $z_{n}$ 으로 전이될 확률과 emission probability $\phi$ 를 따를 때, $z_{n}$ 에서 $x_{n}$ 으로 emit될 확률의 곱을 log 취한 것으로 이해할 수 있다.  

## Dynamical Systems  

Dynamical System은 continuous latent variable을 사용하는 state space model이다. 위에서 다룬 Hidden Markov Model(HMM)과 비교할 때, latent variable이 HMM은 discrete하고 Dynamical System은 continuous하다는 것이 다르다. 이 모델은 두 가지의 component를 가진다.  

- $p(z_{n} | z_{n-1})$: 
a transition probability  
- $p(x_{n} | z_{n})$: 
an emission probability  

값이 continuous하기 때문에, 일반적으로, 주어진 observation에 따른 가장 확률이 높은 state sequence를 찾는 것은 쉽지 않다. 일단, 이 강의노트에서는 transition과 emission function이 linear Gaussian model이라고 가정한다.  

### A Linear Gaussian Models  

Linear Gaussian Transition Model은 아래와 같이 정의된다.  

<div class="callout">
  <div class="callout-header">Linear Gaussian Transition Model</div>
  <div class="callout-body" markdown="1">

$$
p(z_{n} | z_{n-1}) = \mathcal{N}(z_{n}| Az_{n-1}, \Gamma) \quad \text{and} \quad p(z_{1}) = \mathcal{N}(z_{1}| \mu_{0}, V_{0})
$$

Equivalently,  

$$
z_{n} = Az_{n-1} + w_{n} \quad \text{and}\quad z_{1} = \mu_{0} + u,
$$

where 
$w\_{n} \sim \mathcal{N}(w|0,\Gamma)$ 
and 
$u \sim \mathcal{N}(u | 0, V\_{0})$

  </div>
</div>
<br>

<div class="callout">
  <div class="callout-header">Linear Gaussian Emission Model</div>
  <div class="callout-body" markdown="1">

$$
p(x_{n} | z_{n}) = \mathcal{N}(x_{n} | Cz_{n}, \Sigma)
$$

or  

$$
x_{n} = Cz_{n} + v_{n} \quad \text{where}\quad v_{n} \sim \mathcal{N}(v | 0, \Sigma)
$$

  </div>
</div>
<br>

### Inference: Kalman Filtering  

observed sequence가 주어졌을 때, 확률이 최대인 $z_{n}$ 을 찾는 것이 목표다. HMM에서는 가장 가능성 높은 경로를 찾는 것이 목표였으나, 이번에는 가장 가능성이 높은 현재 상태를 찾는 것이 목표다.  
이를 식으로 나타내면,  

$$
\max_{z_{n}} p(z_{n} | x_{1}, \ldots, x_{n})
$$

이다. Gaussian Assumption 덕분에, posterior distribution 또한 Gaussian이 된다. 따라서, Gaussian의 parameter를 찾으면 된다.  

$$
p(z_{n} | x_{1}, \ldots, x_{n}) \sim \underbrace{p (x_{n} | z_{n}) \overbrace{\int p(z_{n} | z_{n-1}) p(z_{n-1}| x_{1}, \ldots, x_{n-1}) dz_{n-1}}^{\text{predict}}}_{\text{update}}
$$

이 식은 다음과 같이 유도할 수 있다.  

$$
\begin{aligned}
p(z_{n} | x_{1}, \ldots, x_{n}) &= \frac{p(x_{n} | z_{n}, x_{1}, \ldots, x_{n-1}) p(z_{n} | x_{1}, \ldots, x_{n-1})}{p(x_{n} | x_{1}, \ldots, x_{n-1})} \quad (\because \text{Bayes' theorem}) \\
                                &\propto p(x_{n} | z_{n}, x_{1}, \ldots, x_{n-1}) p(z_{n} | x_{1}, \ldots, x_{n-1}) \\
                                &= p(x_{n} | z_{n}) p(z_{n} | x_{1}, \ldots, x_{n-1}) \quad (\because \text{Assumption of SSM (Emission)}) \\
                                &= p(x_{n} | z_{n}) \int p(z_{n}, z_{n-1} | x_{1}, \ldots, x_{n-1})dz_{n-1} \quad (\because \text{sum rule}) \\
                                &= p(x_{n} | z_{n}) \int p(z_{n} | z_{n-1}, x_{1}, \ldots, x_{n-1}) p(z_{n-1} | x_{1}, \ldots, x_{n-1}) \quad (\because \text{Chain Rule, Marginalization}) \\
                                &= p(x_{n} | z_{n}) \int p(z_{n} | z_{n-1}) p(z_{n-1}| x_{1}, \ldots, x_{n-1}) dz_{n-1} \quad (\because \text{Assumption of SSM (Transition)})
\end{aligned}
$$

이때, Gaussian distribution의 property에 의해 식을 다음과 같이 간단하게 변형할 수 있다.  

$$
\begin{aligned}
& p(z_{n} | x_{1}, \ldots, x_{n}) = \mathcal{N} (z_{n} | \mu_{n}, V_{n}) \\
& p(x_{n} | z_{n}) = \mathcal{N}(x_{n} | Cz_{n}, \Sigma) \\
& p(z_{n} | z_{n-1}) = \mathcal{N}(z_{n} | Az_{n-1}, \Gamma) \\
& p(z_{n-1} | x_{1}, \ldots, x_{n-1}) = \mathcal{N}(z_{n-1} | \mu_{n-1}, V_{n-1})
\end{aligned}
$$

$$
\begin{aligned}
& p(z_{n} | z_{n-1}) p(z_{n-1}| x_{1}, \ldots, x_{n-1}) dz_{n-1} = \mathcal{N}(z_{n} | A\mu_{n-1}, P_{n-1}) \\
& p(x_{n} | z_{n}) \int p(z_{n} | z_{n-1}) p(z_{n-1}| x_{1}, \ldots, x_{n-1}) dz_{n-1} = \mathcal{N}(z_{n} | A\mu_{n-1} + K_{n}(x_{n} - CA\mu_{n-1}), (I-K_{n}C)P_{n-1})
\end{aligned}
$$

where $P_{n-1} = AV_{n-1}A^{T} + \Gamma$ and $K_{n} = P_{n-1}C^{T}(CP_{n-1}C^{T} + \Sigma)^{-1}$  

이와 같이 Recursive relation과 Gaussian dist. assumption을 사용하여 비교적 쉽게 목표 함수를 구할 수 있다.  

### Learning Objective  

HMM의 learning objective와 비슷하다.  

<div class="callout">
  <div class="callout-header">Learning Objective</div>
  <div class="callout-body" markdown="1">

Let $\theta = \left( A, \Gamma, C, \Sigma, \mu_{0}, V_{0} \right)$, then we find $\theta$ that minimizes the negative log-likelihood of data, i.e.,  

$$
\begin{aligned}
& \min_{\theta} \sum_{s=1}^{S} -\ln p(X^{(s)} | \theta) = \min_{\theta} \sum_{s=1}^{S} - \ln \int_{Z} p (X^{(s)}, Z | \theta) \quad \text{ for unsupervised learning, } \\
& \text{where } X^{(s)} = \left( x_{1}^{(s)}, \ldots, x_{N_{s}}^{(s)} \right)
\end{aligned}
$$

or

$$
\begin{aligned}
& \min_{\theta} \sum_{s=1}^{S} - \ln p(X^{(s)}, Z^{(s)} | \theta) \quad \text{ for supervised learning, } \\
& \text{where } X^{(s)} = \left( x_{1}^{(s)}, \ldots, x_{N_{s}}^{(s)} \right) \text{ and } Z^{(s)} = \left( z_{1}^{(s)}, \ldots, z_{N_{s}}^{(s)} \right)
\end{aligned}
$$

  </div>
</div>
<br>

Unsupervised learning objective의 경우, HMM과 똑같이 EM algorithm으로 풀어야 한다. 이에 관해서는 이 강의노트에서 다루지 않는다.  
Supervised learning objective의 경우, simple Gaussian fitting으로 풀 수 있다. 이 강의노트는 이것도 다루지 않는다.