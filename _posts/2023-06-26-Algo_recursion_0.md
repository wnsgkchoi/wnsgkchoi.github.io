---
title: "[Recursion] 0. Intro"
excerpt: "recursion concept"

categories:
    - recursion

tags:
    - [algorithm, recursion]

toc: true
toc_sticky: true

date: 2023-06-26
last_modified_at: 2023-06-26

---
> 본 게시물은 포스텍 안희갑 교수님의 CSED331 알고리즘 강의 자료를 바탕으로 만들어졌습니다.


## 0. Intro
이번 포스트부터 한동안 recursion 알고리즘에 대해 적을 것이다.  
이번 포스트에서는 recursion이 무엇인지 간략히 살펴볼 것이다. recursion을 사용하는 문제는 다음 포스트부터 차례대로 적을 것이다.  
Recursion 알고리즘을 알기 전에, 먼저 reduction에 대해 알아보고 이를 토대로 recursion 알고리즘을 살펴보자.

## 1. Reductions
Reduction은 주어진 문제를 다른 문제로 치환하는 방법이다.  

임의의 문제 A와 B를 가정하자. A를 풀어야 하고 B는 이미 해결 알고리즘을 알고 있다. 이때 A를 B문제로 바꾸어 풀 수 있다면 B 해결 알고리즘으로 A를 풀 수 있을 것이다.

프로그래밍의 관점에서 이 과정을 생각해보자.  
풀어야 하는 문제를 A라 하고, 이를 B로 reduction한다고 가정하자.
![image](https://github.com/wnsgkchoi/wnsgkchoi.github.io/assets/135838609/6bdc17b6-59f6-405d-912c-375d6b8df611)
> 1단계: 입력 I를 B에 대한 입력 f(I)로 바꾼다.  
> 2단계: 바꾼 입력 f(I)로 문제 B를 푼다.  
> 3단계: B의 해답을 A에 대한 답으로 바꾼다.  

예를 들어, 원소 n개를 가지고 있는 수열 A에 대하여 k번째로 작은 수를 얻는 문제를 보자.  
만약 수열 A를 오름차순으로 정렬하는 sorting 알고리즘이 있다면 sorting 알고리즘을 수행하고 A[k-1]를 반환하여 문제를 해결할 수 있다.  
이와 같이 selection 문제를 sorting 문제로 reduction하여 문제를 해결할 수 있다.

Reduction 알고리즘의 run time은 입력과 출력을 변환하는 시간 T(etc)와 B를 푸는 시간 T(B)의 합이다.

## 2. Recursion
Recursion 알고리즘은 reduction의 일종으로 자신을 반복하는 방법이다.  
조금 더 자세히 말하면 주어진 문제를 input을 간단히 하여 자기 자신으로 reduction하는 것이다. (만약 input을 간단히 하지 않고 계속 자신으로 reduction하면 알고리즘이 종료되지 않을 것이다.)

예를 들어 n!을 계산하는 알고리즘에 대해 생각해보자.  
![image](https://github.com/wnsgkchoi/wnsgkchoi.github.io/assets/135838609/ba0dffb7-5bf7-44dc-9673-8eb33a178e1f)  
n!을 출력하는 함수를 f(n)이라고 하면, f(n) = f(n-1) * n으로 생각할 수 있다. f(1) = 1(base case)이므로, n!을 계산할 때, 더 작은 입력인 n-1을 사용하여 문제를 풀 수 있다.
즉, f(n)을 f(n-1)로 reduction할 수 있고, 이는 자기 자신에 대한 reduction이므로 recursion이라고 할 수 있다.

<details>
<summary> C++ Code </summary>
<div markdown="1">

```c++
#include <iostream>
using namespace std;

int factorial(int n) {
  if (n==1) return 1;
  return factorial(n-1) * n;
}

int main() {
  int n;
  cin >> n;   //input
  cout << factorial(n) << endl; //output
  return 0;
}
```
</div>
</details>

## 3. Summary
이번 포스트에서는 recursion의 정의와 팩토리얼 계산을 recursion으로 해결하는 방법을 살펴 보았다.
다음 포스트에서는 recursion으로 풀 수 있는 여러 알고리즘을 포스팅하려 한다.