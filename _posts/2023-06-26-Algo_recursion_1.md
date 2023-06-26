---
title: "[Recursion] 1. Tower of Hanoi (하노이탑)"
excerpt: "Tower of Hanoi recursion algorithm"

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
이번 포스트에서는 Hanoi Tower에 대해 다루려고 한다.

## 1. What is Tower of Hanoi?
![image](https://github.com/wnsgkchoi/wnsgkchoi.github.io/assets/135838609/43f8c689-3974-4f96-a0e2-1f5d4ad49b8c)
위의 그림과 같이 3개의 기둥이 주어지고 n개의 반지름의 크기가 다른 원판이 주어진다. 우리가 하고자 하는 것은, 모든 원판을 A 기둥에서 C 기둥으로 옮기는데, 원판 이동 횟수를 최소로 하는 것이다. 이때 다음 조건을 항상 지켜야 한다.
1. 한 번에 하나의 원판만을 옮길 수 있다.  
2. 각 기둥의 맨 위에 있는 원판만을 옮길 수 있다.  
3. 크기가 작은 원판은 크기가 큰 원판보다 항상 위에 있어야 한다.  

아직 하노이탑 문제가 무엇인지 이해가 되지 않는다면 아래 사이트에서 직접 문제를 풀어보며 이해해보자.
https://www.novelgames.com/ko/tower/

## 2. Algorithm of Tower of Hanoi
그렇다면 하노이 탑의 최소 이동 횟수는 어떻게 구할 수 있을까?
이를 위해 문제 hanoi(n)을 n개의 원판이 주어졌을 때 최소 이동 횟수로 정의하자.
hanoi(n)은 hanoi(n-1)로 쪼갤 수 있다. 간단히 설명하자면 아래와 같다.  
> 1. 가장 큰 원판을 제외한 n-1개의 원판을 start 기둥에서 auxilary 기둥으로 옮긴다.(n-1개의 하노이 문제)
> 2. 가장 큰 원판을 destination 기둥으로 옮긴다.
> 3. auxilary 기둥에 있는 n-1개의 원판을 destination 기둥으로 옮긴다. (n-1개의 하노이 문제)  

예를 들어 hanoi(3)을 푼다고 생각해보자.
![image](https://github.com/wnsgkchoi/wnsgkchoi.github.io/assets/135838609/d141de0b-a871-4c3b-a031-a422dbb4e244)  
위의 그림을 보며 위의 알고리즘을 이해해보자.  

- hanoi(3)이 호출된다. 이때, 문제 정의에 의해 start = A, auxilary = B, destination = C이다.  
  - 가장 큰 원판을 제외하고 2개의 원판을 B로 옮겨야 하므로, hanoi(2)를 start = A, destination = B, auxilary = C로 하여 hanoi(2)를 호출한다.  
    - hanoi(1)을 start = A, destination = C, auxilary = B로 호출한다. hanoi(1)은 base case이므로 1번 원판이 C로 갈 것이다. (그림에서 2번 단계)
    - 가장 큰 원판인 2번 원판을 destination인 B로 옮긴다. (그림에서 3번)
    - auxilary에 있는 n-1개의 기둥을 destination으로 옮기기 위해 hanoi(1)을 start = C, destination = B, auxilary = A로 호출한다. hanoi(1)은 base case이므로 1번 원판이 B로 이동하게 된다. (그림에서 4번)
  - 가장 큰 원판인 3번 원판을 destination인 C로 옮긴다. (그림에서 5번)
  - auxilary에 있는 n-1개의 기둥을 destination으로 옮기기 위해 hanoi(2)를 start = B, auxilary = A, destination = C로 호출한다.
    - hanoi(1)을 start = B, destination = A, auxilary = C로 호출한다. hanoi(1)은 base case이므로 1번 원판이 A로 갈 것이다. (그림에서 6번)
    - 가장 큰 원판인 2번 원판을 destination인 C로 옮긴다. (그림에서 7번)
    - auxilary에 있는 n-1개의 기둥을 destination으로 옮기기 위해 hanoi(1)을 start = A, destination = C, auxilary = B로 호출한다. hanoi(1)은 base case이므로 1번 원판이 C로 이동하게 된다. (그림에서 8번)


## 3. Time complexity of Algorithm
위에 주어진 알고리즘의 시간복잡도는 어떨까?  
원판의 수가 n개일 때 이동 횟수를 T(n)이라 하자. 그러면, 알고리즘에 의해 T(n) = T(n-1) + 1 + T(n-1) = 2T(n-1) + 1이다. 이때 T(1) = 1이므로, 식을 정리하면 T(n) = 2^n - 1 이 된다.  
따라서 이 알고리즘의 시간복잡도는 O(2^n)이다.

## 4. C++ code of Tower of Hanoi
[백준 1914번](https://www.acmicpc.net/problem/1914)  
이제 하노이탑 알고리즘의 C++ 코드를 짜보자.  
코드는 아래에 숨겨진 항목으로 놔두었다. 혼자 코드를 짜보고 감이 잡히지 않는다면 참고하면 좋을 것이다.

<details>
<summary> C++ Code </summary>
<div markdown="1">

```c++
#include <iostream>
#include <cmath>

using namespace std;

void hanoi(int n, int start, int aux, int dest) {
    if (n == 1) {
        printf("%d %d\n", start, dest);
        return;
    }
    hanoi(n-1, start, dest, aux);
    printf("%d %d\n", start, dest);
    hanoi(n-1, aux, start, dest);
}

int main() {
    int n;
    cin >> n;
    
    string ans = to_string(pow(2, n));
    int x = ans.find('.');
    ans = ans.substr(0,x);
    ans[ans.length() - 1] -= 1;

    cout << ans << endl;
    if (n <= 20) {
        hanoi(n, 1, 2, 3);
    }
    return 0;
}
```
</div>
</details>

## 5. Summary
이번 포스트에서는 하노이탑 문제와 이 문제에 대한 재귀 알고리즘에 대해 살펴보았다.
다음 포스트에서는 recursion을 사용한 Search 알고리즘에 대해 살펴볼 예정이다.