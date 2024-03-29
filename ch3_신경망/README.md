# Chapter 3 - 신경망

신경망과 퍼셉트론
- `활성화 함수`는 퍼셉트론의 임계치와 비슷한 역할을 함
  - 신호를 모두 더해서 활성화 함수의 입력값으로 사용
  - 특정 연산을 거쳐 출력값 계산

## 활성화 함수
- Sigmoid
- ReLU
- 계단 함수 등..

> 오늘날에는 주로 ReLU를 많이 사용한다고 함

## 신경망에서의 행렬 곱

```
// w는 생략
          (y1)
(x1)
          (y2)
(x2)
          (y3)
```

- 가중치는 입력(2) * 출력(3) 크기의 행렬
- 위 신경망을 행렬로 나타내면 아래와 같다.
  - X * W = Y
  - X(2) * W(2*3) = Y(3)

행렬곱을 할 때 형상을 주의해야 함 (X와 W의 대응하는 차원의 원소 수가 같아야 함)
