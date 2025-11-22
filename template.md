## 로컬 서버 구동  

```bash
bundle exec jekyll serve
```

## 포스트 제목  

---
layout: post
title:  "[CSED515] 2. Linear Regression"

categories:
  - Lecture
  - CSED515
tags:
  - [CSED515, ML, Lecture]

toc: true
toc_sticky: true

date: 2025-10-21 22:18:00 +0900
last_modified_at: 2025-10-25 14:34:00 +0900
---

## in-box (theorem이나 lemma 등에 사용)

<div class="callout">
  <div class="callout-header">Ridge Regression</div>
  <div class="callout-body" markdown="1">

Ridge regression은 다음을 최소화한다.  

$$
\mathcal{E} = \frac{1}{2} |y - \Phi w|^{2} + \frac{\lambda}{2} |w|^{2}
$$

이때 $\lambda$ 는 ridge parameter다.  

  </div>
</div>
<br>

## 이미지 첨부 예시 탬플릿  

![Fig 1. Polynomial Regression](/assets/img/CSED515/chap2/2-1.png){: width="500"}  

## 