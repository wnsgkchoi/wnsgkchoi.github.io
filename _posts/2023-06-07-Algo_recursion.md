---
title: "[Algorithm] 1. Recursion"
excerpt: "recursion concept, algorithm, and code"

categories:
    - Algorithm

tags:
    - [algorithm, ]

toc: true
toc_sticky: true

date: 2023-06-07
last_modified_at: 2023-06-12

---
## 1. Reductions
Reduction은 주어진 문제를 다른 문제로 치환하는 방법이다.  

임의의 문제 A와 B를 가정하자. A를 풀어야 하고 B는 이미 해결 알고리즘을 알고 있다. 이때 A를 B문제로 바꾸어 풀 수 있다면 B 해결 알고리즘으로 A를 풀 수 있을 것이다.

프로그래밍의 관점에서 이 과정을 생각해보자.  
풀어야 하는 문제를 A라 하고, 이를 B로 reduction한다고 가정하자.
> 1단계: 입력 I를 B에 대한 입력 f(I)로 바꾼다.  
> 2단계: 바꾼 입력 f(I)로 문제 B를 푼다.  
> 3단계: B의 해답을 A에 대한 답으로 바꾼다.  

Reduction 알고리즘의 run time은 입력과 출력을 변환하는 시간 T(etc)와 B를 푸는 시간 T(B)의 합이다.

## 2. Recursion
Recursion 알고리즘은 reduction의 일종으로 자신을 반복하는 방법이다.

예를 들어 n!을 계산하는 알고리즘에 대해 생각해보자. n!을 출력하는 함수를 f(n)이라고 하면, f(n) = f(n-1) * n으로 생각할 수 있다.  
f(1) = 1(base case)이므로, n!을 계산할 때, 더 작은 입력인 n-1을 사용하여 문제를 풀 수 있다.