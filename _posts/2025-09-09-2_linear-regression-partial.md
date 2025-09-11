---
layout: post
title:  "[CSED515] 2. Linear Regression"

categories:
  - Lecture
tags:
  - [CSED515, Lecture]

toc: true
toc_sticky: true

date: 2025-09-09 02:15:41 +0900
last_modified_at: 2025-09-09 02:15:41 +0900
---

# Linear Regression  

```text
ㅁㄴㅇㄻㄴㅇㄹ
```

## p.3  

best $f$를 구하기 위해, MSE 식을 $f$에 대해 편미분하여 gradient가 0이 되는 부분을 찾는다. 먼저, $p(x,y) = p(y|x)p(x)$ 를 사용하여 MSE 식을 다음과 같이 변형해보자.  

$$
\int \int |y - f(x)|^{2} p(x,y)dxdy = \int \left[ \int |y - f(x)|^{2} p(y|x) dy \right] p(x) dx  
$$

이 식이 최소가 되도록 하려면, 모든 x에 대해, 대괄호 안의 식이 최소가 되어야 한다. 따라서, 우리의 목표는 대괄호 안의 식이 최소가 되도록 하는 $f$를 찾는 것이 된다. 대괄호 안의 식을 $f$로 편미분하자.  

$$
\int -2(y-f(x)) p(y|x) dy = 0  
$$

식을 간단히 정리하면 다음과 같다.  

$$
\int (y-f(x)) p(y|x) dy = 0  
$$

이제 적분 기호를 분배한 뒤, 이항해보자.  

$$
\begin{align}
\int y p(y|x) dy &= \int f(x) p(y | x) dy  \\
                 &= f(x) \int p(y | x) dy  \\
                 &= f(x)
\end{align}
$$

따라서, $f(x) = \int y p(y|x) = \mathbb{E} [y | x]$ 가 되며, 이는 주어진 x에 대한 y값의 평균을 의미한다.  

## Feature Functions  

linear regression은 매우 간단한 나머지, 실제 세상의 데이터 처리를 못한다고 생각할 수 있다. 하지만 input data를 feature function과 결합하여 input space를 바꾸는 것만으로도 polynomial하게 regression을 할 수 있다.  
예를 들어, 다음과 같이 feature function을 정의한다고 해보자.  
$$
\phi (x) =
   \begin{bmatrix}
        1 \\
        x \\
        x^2 \\
        x^3
   \end{bmatrix}
$$

이 feature function을 $x$ 대신 linear regression 식에 대입하게 되면  
$$
y = \theta ^T \phi (x) = w_1 + w_2x + w_3x^2 + w_4x^3
$$

이 되어, linear model이 non-linear relation을 학습할 수 있게 된다.  

feature function (basis function)은 꼭 polynomial할 필요는 없다.  
$\phi _l (x) = x^{l-1}$ 처럼 polynomial 한 basis도 있고, $\phi _l (x) = \exp(-\frac{||x - \mu _l||^2}{2\sigma^2})$ 처럼 gaussian basis도 있을 수 있으며, 구간마다 다른 polynomial이 적용되도록 하는 basis도 있다. 이 외에도 수많은 basis가 있다.  

## Learning objective of Least Squares (LS) Method  

feature function과 결합된 linear regression 문제의 최적해를 찾아보자. 먼저 problem setting부터 해야 하는데, 이제부터 loss function으로 MSE를 사용한다. linear regression은 $y = \theta ^T \phi (x)$ 로 나타낼 수 있으므로, MSE는 다음과 같아진다.  

$$
\frac{1}{2} \sum_{n=1}^{N} \left( y_n - w^T \phi (x_n) \right) ^2 = \frac{1}{2} ||y - \Phi w||^2
$$

왜 $\frac{1}{2}$ 을 표시하냐면, 계산의 편의도 있고, gaussian noise가 포함된 linear regression 문제에 대한 Maximum likelihood estimator를 구하는 과정에서 비슷한 식이 나오는데, 여기에 $\frac{1}{2}$ 이 포함되기 때문이기도 할 것이다. 그냥 이렇게만 알아두도록 하자. (바로 아래 슬라이드에서 소개하고 있긴 하다.)  

이제 위의 식을 w에 대해 편미분하여 gradient가 0이 되도록 하는 w의 값을 찾으면, 그 w가 MSE를 최소로 만들 것이다.  

$$
\begin{align}
\frac{\partial}{\partial w} \frac{1}{2} ||y - \Phi w||^2 &= \frac{\partial}{\partial w} \frac{1}{2} (y - \Phi w)^T(y - \Phi w) \\
                                                               &= \frac{\partial}{\partial w} \frac{1}{2} (y^Ty - 2y^T\Phi w + w^T\Phi^T \Phi w)  
                                                               &= -y^T \Phi + w^T \Phi^T \Phi
                                                               &= 0
\end{align}
$$

이 식을 정리하면 다음과 같은 결과를 얻을 수 있다.  
$$
w^T \Phi^T \Phi = y^T \Phi \\
w^T = y^T \Phi (\Phi^T \Phi)^{-1} \\
w = (\Phi^T \Phi)^{-1} \Phi^T y
$$

참고로 moore-penrose pseudoinverse는 항상 존재하며, SVD(특이값 분해)를 통해 구한다.  
다만, 슬라이드에 있는 $(A^T A)^{-1} A^T$는 항상 존재하는 것이 아니며, 오직 행렬 $A$의 rank가 full일 때에만 $A†$ 와 일치한다.  
