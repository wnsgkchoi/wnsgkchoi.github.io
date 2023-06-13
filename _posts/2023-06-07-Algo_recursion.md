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
> 본 게시물은 포스텍 안희갑 교수님의 CSED331 알고리즘 강의 자료를 바탕으로 만들어졌습니다.

## 1. Reductions
Reduction은 주어진 문제를 다른 문제로 치환하는 방법이다.  

임의의 문제 A와 B를 가정하자. A를 풀어야 하고 B는 이미 해결 알고리즘을 알고 있다. 이때 A를 B문제로 바꾸어 풀 수 있다면 B 해결 알고리즘으로 A를 풀 수 있을 것이다.

프로그래밍의 관점에서 이 과정을 생각해보자.  
풀어야 하는 문제를 A라 하고, 이를 B로 reduction한다고 가정하자.
![image](https://github.com/wnsgkchoi/wnsgkchoi.github.io/assets/135838609/6bdc17b6-59f6-405d-912c-375d6b8df611)
> 1단계: 입력 I를 B에 대한 입력 f(I)로 바꾼다.  
> 2단계: 바꾼 입력 f(I)로 문제 B를 푼다.  
> 3단계: B의 해답을 A에 대한 답으로 바꾼다.  

Reduction 알고리즘의 run time은 입력과 출력을 변환하는 시간 T(etc)와 B를 푸는 시간 T(B)의 합이다.

## 2. Recursion
Recursion 알고리즘은 reduction의 일종으로 자신을 반복하는 방법이다. 

### 1) Factorial
예를 들어 n!을 계산하는 알고리즘에 대해 생각해보자.  
![image](https://github.com/wnsgkchoi/wnsgkchoi.github.io/assets/135838609/ba0dffb7-5bf7-44dc-9673-8eb33a178e1f)  
n!을 출력하는 함수를 f(n)이라고 하면, f(n) = f(n-1) * n으로 생각할 수 있다. f(1) = 1(base case)이므로, n!을 계산할 때, 더 작은 입력인 n-1을 사용하여 문제를 풀 수 있다.

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

### 2) Tower of Hanoi
3개의 기둥이 주어지고 n개의 반지름의 크기가 다른 원판이 주어진다. 모든 원판을 A 기둥에서 C 기둥으로 옮겨야 하는 문제이다.
이때 크기가 작은 원판은 크기가 큰 원판보다 항상 위에 있어야 한다.  
![image](https://github.com/wnsgkchoi/wnsgkchoi.github.io/assets/135838609/43f8c689-3974-4f96-a0e2-1f5d4ad49b8c)
하노이 탑 문제도 recursion으로 풀 수 있다. 간단히 설명하자면 아래와 같다.  
> 1. 가장 큰 원판을 제외한 n-1개의 원판을 A기둥에서 B로 옮긴다.(n-1개의 하노이 문제)
> 2. 가장 큰 원판을 C로 옮긴다.
> 3. B에 있는 n-1개의 원판을 C로 옮긴다. (n-1개의 하노이 문제)  

위와 같이 문제를 풀면, 알고리즘의 시간복잡도는 O(2^n)이 된다.

### 3) Binary Search
포스텍 재학생이라면 CSED101 수업에서 이미 binary search가 어떤 알고리즘인지 배웠을 것이다.  
이 글을 읽는 사람 중에서 Search 문제가 무엇인지 모르는 사람을 위해 아래에 설명을 적어두었다.  

<details>
<summary> Search Algorithm </summary>
<div markdown="1">
Search 문제는 주어진 수열에서 원하는 원소를 찾아내는 문제를 말한다.<br>
일반적으로 주어진 수열이 오름차순 또는 내림차순으로 정렬된 상태를 가정한다.<br>
예를 들어 수열 [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]과 입력 34가 주어졌다고 하자. 34는 주어진 수열에 존재하지 않으므로 해당 알고리즘은 false를 출력할 것이다.<br>
가장 기본적인 search 알고리즘은 주어진 수열을 처음부터 끝까지 탐색하는 것이다.  해당 알고리즘의 시간복잡도는 O(n)이다.
</div>
</details>
<br>
주어진 수열이 정렬되어 있다면, 주어진 수열의 특정 원소와 입력값의 대소를 비교하여 탐색 범위를 좁힐 수 있다.  
예를 들어 수열 [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]과 입력 34가 주어졌다고 하자. 수열의 원소 21과 입력값 34를 비교하면 21 < 34 이므로, 34를 찾기 위해 21 오른쪽의 원소만 탐색하면 된다는 것을 알 수 있다.  
이처럼 주어진 수열의 중앙에 있는 원소와 입력값을 비교하여 수열의 크기를 절반으로 줄이는 재귀함수를 호출하면 O(log n)에 search를 수행할 수 있다.

### 4) Merge Sort
