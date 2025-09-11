---
layout: post
title:  "[CSED515] 1. Learning Theory"

categories:
  - Lecture
tags:
  - [CSED515, Lecture]

toc: true
toc_sticky: true

date: 2025-09-08 02:15:41 +0900
last_modified_at: 2025-09-08 02:15:41 +0900
---

# Introduction: Learning Theory  

```text

```

Learning Theory는 machine이 data로부터 학습을 할 수 있는 조건에 대해 탐구하는 이론이다.  
일반적으로 Learning theory를 Statistical learning theory와 Online learning theory로 나눌 수 있다.  

이번 introduction에서는 statistical learning theory에 대해서만 다룬다.  

**Four Key Ingradient of Learning Theory**  
Statistical learning theory의 목표를 간단히 나타내면 아래와 같다.  
$$
\begin{align}

\text{find} \quad & f \\
\text{subj. to} \quad \, &f \in \mathcal{F} \\
& \mathbb{E}_{(x,y)~D} \; l(x, y, f) \leq \epsilon

\end{align}
$$

아래와 같이 표현할 수도 있다.
$$
\min_{f \in \mathcal{F}} \mathbb{E}_{(x,y)~D} \quad l(x, y, f)
$$

이 수식을 바탕으로 생각할 때, Learning Theory에서 주목해야 할 네 가지 key ingredients가 있다.  

- A distribution $D$  
- Hypothesis space $\mathcal{F}$  
- A loss function $l$  
- A learning algorithm  










































## p.17   

$A(S)$ 는 앞 슬라이드에서 다루는 hypothesis $h$를 일반화한 것으로 생각하면 될 듯.  
$S^{*}$ 는 오토마타 및 형식언어에서 배운 * notation이라고 생각하면 될 듯.  

## p.20  

$m$: num of data..?  

$$L(A(S)) \leq \frac{1}{m} (\log |H| + \log \frac{1}{\delta}) \leq \epsilon$$

## p.24  
agnostic-PAC learnable 조건에는 PAC learnable과 비교할 때 다음 term 하나가 추가된다.  
$\min_{h \in \mathcal{H}} L(h)$  

대충 가장 좋은 loss를 가지는 것과 비교할 때, 매우 작은 차이가 나는 algorithm을 찾는 느낌으로 생각하면 될 듯.  
다만, $\min_{h \in \mathcal{H}} L(h)$ 를 찾는 과정 자체가 복잡하기 때문에, 단순하게, $L(A(S))$ 를 최소화하는 방향으로 좋은 알고리즘을 찾는 경우가 많다고 한다.  

## p.26  

generalization error = $L(h) - \hat{L}(h)$  
- expected error - empirical error  

## p.28  

empirical error와 expected error간 차이가 최대 $\sqrt{\frac{\log_{e}{|H|}+ \log_e \frac{1}{\delta}}{2n}}$ 난다는 것이 증명되었다.  
따라서 error를 줄이기 위해서는, n을 증가시키거나(데이터 샘플 수 증가), hypothesis space size를 줄이거나, delta를 늘린다. 다만, |H|를 줄이면, empirical error가 증가할 수 있으므로, 이 trade-off를 반드시 고려해야 한다. (p.30의 내용과 연관)  

## p.29  
algorithm A가 upper bound를 최소화한다면, expected error 또한 최소화된다.  
이런 알고리즘 중 하나가 ERM으로, ERM은 단순히 empirical error를 최소화하는 알고리즘이다.  
여기에 regularization term을 추가하면, 더 일반적인 알고리즘인 regularized ERM이 완성된다.  
