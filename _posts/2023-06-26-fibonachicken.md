---
title: FibonaChicken
Excerpt: "피보나치 수열을 이용한 피보나치킨 코드 구현하기"
categories:
  - Python
tags:
  - [python]
Toc: true
Toc_sticky: true
Date: 2023-06-26
Last_modified_at: 2023-06-26

---

# 피보나치킨?

​	[피보나치킨](https://your-calculator.com/life/food/fibonacci-chicken)은 피보나치 수열을 이용하여 인원 수에 따른 치킨 마리 수의 **황금비율**을 알려 주는 사이트이다. 

> 피보나치 수열은 첫째 및 둘째 항이 1이며 그 뒤의 모든 항이 바로 앞 두 항의 합으로 이루어진 무한수열이다. <cite>[출처](https://ko.wikipedia.org/wiki/%ED%94%BC%EB%B3%B4%EB%82%98%EC%B9%98_%EC%88%98)</cite>

​	피보나치 수열의 n번째 숫자만큼의 사람이 있을 때 가장 이상적인 치킨 마리수는 n-1번째 피보나치 숫자라고 한다. 피보나치킨 사이트는 이 계산을 해주는 서비스를 제공한다. 오늘은 Python의 재귀함수(recursive function)을 이용해 이 프로그램을 구현해볼 것이다.

↓ **같이 보기** ↓

<details>
  <summary>제켄도르프 정리</summary>
  <div markdown="1">
    제켄도르프 정리는 '모든 자연수는 연속하지 않는 피보나치의 합으로 표현할 수 있고, 그 합의 표현은 유일하다'는 정리이다. 몇 개의 자연수로 예를 들면,
    1 = 1
    2 = 2
    3 = 3
    4 = 3 + 1
    5 = 5
    6 = 5 + 1
    7 = 5 + 2
    8 = 8
    9 = 8 + 1
    ...
    이렇게 모든 자연수에서 성립한다.
  </div>
</details>

### 구현하기

- **N**을 입력받았을 때, fibonacci(k) = N인 k값을 찾고 fibonacci(k-1)을 구한다.
- 피보나치 수가 아닌 자연수에 대해서는 **제켄도르프 분해**를 적용한다.
- *예) N=6일 경우, 5+1로 나눈다.*

```python
memory = {1:1, 2:1}
  
def fibonacci(n):
    if n in memory:
        number = memory[n]
    else:
        number = fibonacci(n-1) + fibonacci(n-2)
        memory[n] = number
    return number

def fibonachicken(n):
    k = 1
    while n > fibonacci(k):
        k += 1
  
    for k, person in memory.items():
        if person == n:
            if n == 1:
                chicken = memory[1]
            else:
                chicken = memory[k-1]
            print(f'필요한 치킨은 {chicken}마리!')
            return chicken
    chicken = zec(n)
    print(f'필요한 치킨은 {chicken}마리!')


def zec(n):
    chicken = 0
    person = 0
    max_k = len(memory) - 1
    person = memory[max_k]
    rest = n - person
    chicken = memory[max_k-1]
    for k, i in reversed(memory.items()):
        if i <= rest:
            person += i
            chicken += memory[k-1]
            rest -= i
            if rest == 0: break
    return chicken

in_put = int(input('인원수를 입력하세요.: '))
fibonachicken(in_put)
```

