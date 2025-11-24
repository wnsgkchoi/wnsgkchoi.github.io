---
layout: post
title:  "[CSED515] 7. Online Learning"

categories:
  - Machine_Learning
tags:
  - [CSED515, ML, Lecture]

toc: true
toc_sticky: true

date: 2025-10-28 04:00:00 +0900
last_modified_at: 2025-10-28 04:00:00 +0900
---

# Online Learning  

## 0. Intro  

Expert의 Advice로 classification을 학습하는 경우를 생각해보자. 다음과 같은 Protocol을 생각할 수 있다.  

<div class="callout">
  <div class="callout-header">Protocol</div>
  <div class="callout-body" markdown="1">

for $t = 1, \ldots, T$ do <br>
&nbsp;&nbsp;&nbsp;&nbsp; Each expert observes $x_{t}$ from which predicts $\hat{y}\_{t,k} = \hat{p}\_{t,k}(x_{t})$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Learner receives an expert advice $\hat{y}\_{t, k}$ for $k = 1, \ldots, K$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Learner outputs a prediction $\hat{y}\_{t}$ from $\hat{y}\_{y, 1}, \ldots, \hat{y}\_{t, K}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Learner receives a true label $y_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Learner suffers loss $\ell (\hat{y}\_{t}, y\_{t})$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Learner updates model parameters <br>
end for  

  </div>
</div>
<br>

Online Learning은 다음과 같은 상황에서 강점을 가진다.  

- observation이 sequential한 경우  
- previous update를 keep하면 좋은 경우  

즉, i.i.d. 가정이 없는, sequential한 데이터에 대해 online learning은 강점을 가진다. 특히, sequential data의 본질에 의해 online learning은 distribution shift를 handle할 수 있다.  
예를 들어, Batch Learning은 데이터셋에 대해 한 번에 update를 진행하므로, 새로운 데이터가 들어오게 되면 처음부터 학습을 해야 한다. 반면, Online Learning은 데이터가 새로 들어와도 괜찮기 때문에, 다시 학습할 필요가 없다.  

## 1. Regret  

Regret을 정의하기 위해, regret을 구성하는 term들에 대해 먼저 이해해야 한다. 그 첫 번째가 바로 cumulative loss다.  

<div class="callout">
  <div class="callout-header">Cumulative Loss</div>
  <div class="callout-body" markdown="1">

Suppose that a learner chooses predictions $\hat{y}\_{t} := h_{t}(x\_{t})$ over time. For some loss function $\ell : \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}\_{\geq 0}$, a cumulative loss is  

$$
\sum_{t=1}^{T} \ell(\hat{y}_{t}, y_{t}),
$$

where $T$ is the number of rounds (a.k.a. time horizon), $\hat{y}\_{t}$ is a learner's prediction, and $y\_{t}$ is a true label.  

  </div>
</div>
<br>  

Expected loss를 쓰지 않는 이유는 I.I.D assumption을 사용하지 않기 때문이다. Expected loss는 다음과 같이 정의할 수 있다.  

$$
\begin{aligned}
\mathbb{E}\left[ \ell(h(x), y) \right] = \int \ell(h(x), y) p(x, y) dx dy  
\end{aligned}
$$

식에 $p(x, y)$가 있음을 알 수 있는데, 이는 고정된 distribution에서 데이터가 생성된다는 i.i.d. 가정에 의해 나온 것이다. 만약 distribution이 고정되지 않고 가변적이라면, 정의에 의해 Expected loss를 사용하는 것이 불가능하다. 해당 데이터에 의한 loss의 가중치를 정할 수 없기 때문이다. 혹자는 간단하게 monte-carlo로 쌓아올리면 된다고 생각할 수 있다. 사실 cumulative loss를 T로 나눈 loss의 평균을 사용해도, 결국 cumulative loss와 T 배수 관계에 있으므로 본질적으로 같다고 생각할 수 있다. 하지만, 향후 살펴볼 Goodness Metric (좋음을 평가하는 metric)을 적용하기에 cumulative loss가 더 적합하다. 일단 이렇게 이해하고 넘어가도록 하자.  

<div class="callout">
  <div class="callout-header">Best Cumulative Loss</div>
  <div class="callout-body" markdown="1">

Suppose that a learner chooses candidate predictors from a hypothesis set $\mathcal{H}$. For some loss function $\ell : \mathcal{Y} \times \mathcal{Y} \rightarrow \mathbb{R}_{\geq 0}$, a best cumulative loss is  

$$
\min_{h \in \mathcal{H}} \sum_{t=1}^{T} \ell \left( h(x_{t}, y_{t}) \right),
$$

where $T$ is the number of rounds (a.k.a. time horizon) and $y_{t}$ is a true label.  

  </div>
</div>
<br>

식을 살펴보면, 이름이 매우 직관적으로 지어졌음을 알 수 있는데, hypothesis space에 있는 모든 hypothesis 중에서 가장 좋은 cumulative loss를 가지는 hypothesis의 cumulative loss를 의미한다. 이제 이 값을 일종의 benchmark로 사용한다.  

이 두 가지 term을 사용하여 Regret을 정의하게 된다.  

<div class="callout">
  <div class="callout-header">Definition of Regret</div>
  <div class="callout-body" markdown="1">

$$
Reg_{t} := \sum_{t=1}^{T} \ell(h_{t}(x_{t}), y_{t}) - \min_{h \in \mathcal{H}} \sum_{t=1}^{T} \ell(h(x_{t}), y_{t})
$$

  </div>
</div>
<br>

직관적으로, Regret은 현재 사용하고 있는 hypothesis의 loss가 가장 좋은 hypothesis의 loss보다 얼마나 더 큰지 나타내는 것으로 이해할 수 있다. 이 loss는 i.i.d. assumption을 필요로 하지 않는다. 이때, 다음과 같은 상황을 가정해보자.  

> 데이터를 적대적으로 생성하는 환경에서는 어떤 일이 일어날까?

### Learning Objective: Regret Minimization  

다음과 같은 환경을 생각해보자.  

<div class="callout">
  <div class="callout-header">Adversarial Environment</div>
  <div class="callout-body" markdown="1">

This is a learning environment where an adversary can arbitrarily generate $(x\_{t}, y\_{t})$ possibly depending on the learners' previous predictions, i.e., $\hat{y}\_{1}, \ldots, \hat{y}\_{t-1}$  

  </div>
</div>
<br>

'arbitrarily'에서도 알 수 있듯, 환경은 i.i.d. assumption을 따르지 않으며, 임의의 데이터를 생성한다. 이때, 최악의 시나리오로 learner에게 완전히 적대적인 데이터를 생성해낼 수도 있다. Adversarial Environment는 learner의 이전까지의 prediction을 바탕으로 데이터 $(x\_{t}, y\_{t})$ 를 생성한다. learner는 이 데이터 중 $x\_{t}$ 를 보고 prediction $\hat{y}\_{t}$ 를 제출한다. Environment는 이후 정답 $y\_{t}$ 를 공개한다. 이를 바탕으로 learner는 학습을 진행한다. 이때, learner의 previous prediction을 사용하고, current prediction을 사용하지 않는 이유는, adversarial environment의 일종의 cheating을 막기 위함이다. 사용자의 현재 예측을 본 후 정답 $y\_{t}$ 을 생성하게 된다면, 사용자의 현재 prediction과 완전히 반대되는 $y\_{t}$ 를 '고의적으로' 생성하는 환경이 존재할 수 있다. 이런 경우, 모델은 계속해서 틀린 prediction을 하게 되고, 결과적으로 학습이 불가능해질 것이다.  

지금까지 살펴본 것을 사용한 online learning의 objective는 다음과 같다.  

<div class="callout">
  <div class="callout-header">Learning Objective: Regret Minimization</div>
  <div class="callout-body" markdown="1">

The main goal of online learning is to find a learner that minimizes $Reg\_{T}$ under an adversarial environment.  

  </div>
</div>
<br>

batch learning처럼, cumulative loss를 줄이는 것만 고려해도 된다. 다만, online learning은 batch learning보다 theoretical aspect에 집중한다. 더 구체적으로, $Reg\_{T} = \mathcal{O}(\sqrt{T})$ 인 learner를 찾는 것이 목표다. 이 목표를 sub-linear라고 한다. 왜 sub-linear가 목표가 되었을까. 직관적인 예시를 먼저 본 뒤에, 수학적인 측면을 살펴보도록 하자.  

### A Negative Result  

직관적인 예시를 살펴보기에 앞서, Regret에 대한 모든 증명은 다음 상황을 가정한다.  

- {-1, 1}을 예측하는 binary classification task.  
- =zero-one loss 사용 (i.e., $\mathbb{1}(\hat{y}\_{t} \neq y\_{t})$)  
- learner는 deterministic

이런 상황에서 다음 lemma가 성립한다.  

<div class="callout">
  <div class="callout-header">Lemma 1</div>
  <div class="callout-body" markdown="1">

For any deterministic learner, there exists an $\mathcal{H}$ and the sequence of labeled examples such that  

$$
Reg_{T} \geq \frac{T}{2}
$$

  </div>
</div>
<br>

왜 항상 $\frac{T}{2}$ 보다 큰 Regret을 가지는 hypothesis가 존재할까. 항상 1을 출력하는 hypothesis $h_{-1}$ 과 항상 -1을 출력하는 hypothesis $h_{+1}$ 를 가정하자. 그렇다면, 어떤 데이터를 adversarial environment가 생성하든, 둘 중 하나는 $\frac{T}{2}$ 보다 작을 수밖에 없다. 즉, 최적의 hypothesis의 cumulative loss는 $\frac{T}{2}$ 보다 작을 수밖에 없고, 항상 틀리는 hypothesis의 cumulative loss는 $T$ 가 되어, Regret이 $\frac{T}{2}$ 보다 크게 된다. 이를 수학적으로 **굳이** 서술하자면 다음과 같다.  

$$
\begin{aligned}
\ell(h_{-1}(x_{t}), y_{t}) + \ell(h_{+1}(x_{t}), y_{t}) = 1 &\Rightarrow \sum_{t=1}^{T} \ell(h_{-1}(x_{t}), y_{t}) + \sum_{t=1}^{T} \ell(h_{+1}(x_{t}), y_{t}) = T \\
                                                            &\Rightarrow \sum_{t=1}^{T} \ell(h_{-1}(x_{t}), y_{t}) \leq \frac{T}{2} \quad or \quad \sum_{t=1}^{T} \ell(h_{+1}(x_{t}), y_{t}) \leq \frac{T}{2} \\
                                                            &\Rightarrow Reg_{T} := \underbrace{\sum_{t=1}^{T} \ell(\hat{y}_{t}, y_{t})}_{=T} - \underbrace{\min_{h \in \mathcal{H}}\sum_{t=1}^{T}\ell(h(x_{t}), y_{t})}_{\leq \frac{T}{2}} \geq \frac{T}{2}
\end{aligned}
$$

이를 n-classes classfication으로 확장할 수 있으며, 이때는 regret이 $\frac{1}{n}$ 보다 큰 hypothesis가 항상 있음을 보일 수 있다. 이처럼, 학습이 불가능한 상태의 hypothesis의 regret이 linear function을 따르게 된다. 이는 곧, 데이터의 개수가 무한히 많아질 때, hypothesis의 cumulative loss가 best cumulative loss와 차이의 평균이 0이 아님을 의미한다. 즉,  

$$
\lim_{t \rightarrow \infty} \frac{Reg_{T}}{T} = C \quad (C\text{는 0이 아닌 상수})
$$

이 된다. 하지만, regret이 sub-linear인 hypothesis를 가정하면,  

$$
\lim_{t \rightarrow \infty} \frac{Reg_{T}}{T} = \frac{\mathcal{O}(\sqrt{T})}{T} = 0
$$

이 되어, 평균 regret이 0이 된다. 이런 learner를 찾는 것을 목표로 하게 된다.  

## Exponential Weighting  

online learning에서 여러 expert의 advice를 따를 때, learner는 간혹 하나의 또는 다수의 best expert를 골라 이것의 prediction을 반복하려 할 수 있다. 또는, 전문가에 따른 가중치를 두고 싶을 수 있다. 문제는 expert의 성능이 시간이 지남에 따라 달라질 수 있다는 점이다. 이것을 고려하여 각각의 expert에게 가중치를 부여하는 알고리즘을 살펴본다.  

###  Weighted Majority Algorithm  

<div class="callout">
  <div class="callout-header">Weighted Majority Algorithm</div>
  <div class="callout-body" markdown="1">

$w_{1} \leftarrow \left( 1, \ldots, 1 \right)$  
for $t = 1, \ldots, T$ do <br>
&nbsp;&nbsp;&nbsp;&nbsp; Observe $x_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Predict $\hat{y_{t}} \mathbb{1} \left( \sum_{i: h_{i}(x_{t}) = 1} w_{t}(i) \geq \sum_{i: h_{i}(x_{t}) = 0} w_{t}(i) \right)$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Observe $y_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; if $\hat{y}\_{t} \neq y\_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update $w\_{t+1} (i) \leftarrow \beta w\_{t}(i)$ for $h\_{i}(x\_{t}) \neq y\_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update $w\_{t+1} (i) \leftarrow w\_{t}(i)$ for $h\_{i}(x\_{t}) = y\_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; end if <br>
end for  

  </div>
</div>
<br>

이 알고리즘은 매우 직관적이고 좋아 보인다. 1을 예측한 expert의 가중치의 합과 0을 예측한 expert의 가중치의 합을 비교하여, 더 큰 쪽으로 예측한 뒤, 예측이 틀린 경우, 틀린 예측을 한 expert set의 가중치를 줄이는 업데이트를 진행한다. 하지만 이 알고리즘에는 단점이 두 가지 존재한다. 먼저, 알고리즘이 deterministic하다. 항상 모든 expert가 업데이트에 관여한다는 뜻이다. 이는 adversarial environment에서 큰 단점이 된다. 왜냐하면, adversarial한 environment가 learner의 답을 예측할 수 있게 되기 때문이다. 다른 하나의 단점은 classification에만 한정된 algorithm이라는 것이다.  

### "Randomized" Weighted Majority Algorithm  

위의 단점 중 deterministic함을 해결하기 위해, randomization을 추가한 알고리즘이다.  

<div class="callout">
  <div class="callout-header">Randomized Weighted Majority Algorithm</div>
  <div class="callout-body" markdown="1">

$w_{1} \leftarrow \left( 1, \ldots, 1 \right)$  
for $t = 1, \ldots, T$ do <br>
&nbsp;&nbsp;&nbsp;&nbsp; Observe $x_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Predict $\hat{y_{t}} = h_{i}(x_{t})$, where $i \sim p_{t} := \frac{w_{t}}{\sum_{i}w_{t}(i)}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Observe $y_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Update $w_{t+1} (i) \leftarrow \beta w_{t}(i)$ for $h_{i}(x_{t}) \neq y_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Update $w_{t+1} (i) \leftarrow w_{t}(i)$ for $h_{i}(x_{t}) = y_{t}$ <br>
end for  

  </div>
</div>
<br>

이제 알고리즘이 randomized되어 adversarial environment에서 사용될 수 있다. 하지만 여전히 classification에 한정된다는 문제점은 남아 있다.  
이때, 이 알고리즘의 update rule은 다음과 같이 표현할 수 있다.  

$$
w_{t+1}(i) \leftarrow (1-\eta)^{\ell \left( h_{i}(x_{t}),y_{t} \right)},
$$

where $\beta := 1- \eta$ and $\ell \left( h_{i}(x_{t}),y_{t} \right) := \mathbb{1} \left( h_{i}(x_{t}) \neq y_{t} \right) w_{t}(i)$.  

이때, 이 update에서, 다음과 같은 근사식을 생각해볼 수 있다.  

$$
(1-\eta)^{\ell \left( h_{i}(x_{t}),y_{t} \right)} \approx \exp(-\eta\ell\left( h_{i}(x_{t}),y_{t} \right)) \quad \text{for small } \eta.
$$

이 exponential 함수를 사용하게 되면, zero-one loss 대신 사용할 loss가 생기므로, classification에 국한되지 않는 알고리즘을 만들 수 있다. 아래 figuyre는 위의 근사식이 원래 함수와 얼마나 비슷한지 보여준다.  

![Fig 1. Approx](/assets/img/CSED515/chap7/7-1.png){: width="500"}

### Exponential Weighting  

위의 근사식을 사용하여 update를 하는 algorithm을 exponential weighting이라 한다. 구체적으로 아래와 같다.  

<div class="callout">
  <div class="callout-header">Exponential Weighting</div>
  <div class="callout-body" markdown="1">

$w_{1} \leftarrow \left( 1, \ldots, 1 \right)$  
for $t = 1, \ldots, T$ do <br>
&nbsp;&nbsp;&nbsp;&nbsp; Observe $x_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Predict $\hat{y_{t}} = h_{i}(x_{t})$, where $i \sim p_{t} := \frac{w_{t}}{\sum_{i}w_{t}(i)}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Observe $y_{t}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; Update $w_{t+1} (i) \propto w_{t}(i)exp(-\eta \ell (h_{i}(x_{t}), y_{t}))$ for $i \in \left\\{ 1, \ldots, |\mathcal{H}| \right\\}$ <br>
end for  

  </div>
</div>
<br>

이 알고리즘을 사용할 때, 다음과 같은 regret bound가 형성된다고 한다.  

<div class="callout">
  <div class="callout-header">Theorem</div>
  <div class="callout-body" markdown="1">

For any loss function $\ell$ with the range of $\left[0, 1 \right]$, we have  

$$
\mathbb{E} Reg_{T} := \sum_{t=1}^{T} \mathbb{E}_{i \sim p_{t}} \ell \left( h_{i}(x_{t}), y_{t} \right) - \min_{h \in \mathcal{H}} \sum_{t=1}^{T} \ell \left( h(x_{t}), y_{t} \right) \leq \sqrt{T \ln |\mathcal{H}|}  
$$

if 
$\eta = \sqrt{\frac{8\ln\|\mathcal{H}\|}{T}}$ 
under an adversarial environment  

  </div>
</div>
<br>

이때, 
$\frac{\mathbb{E}Reg_{T}}{T} = \sqrt{\frac{\ln \|\mathcal{H}\|}{T}}$ 
이므로, learnable하다. 다만, expert의 수가 유한하다고 가정하는데, 이는 실제 사용에서 한계점이 될 수 있다.