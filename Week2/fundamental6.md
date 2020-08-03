
# Fundamental 6. Data 어떻게 표현하면 좋을까?

> 학습 목표

- 데이터를 배열로 저장하는 것에 대해 이해하고 list, numpy의 사용법을 학습합니다.
- 구조화된 데이터를 저장하는 것을 이해하고, dictionary와 pandas 사용법을 학습합니다. 
- 이미지 데이터를 numpy 배열로 저장하는 것을 이해하고, 그 사용법을 학습합니다.
- 학습한 자료구조를 활용해서 통계 데이터를 어떻게 계산하는지 학습합니다.

## Numpy 소개

> Numpy ; Numerical Python

고성능 과학계산 컴퓨팅과 데이터 분석에 필요한 파이썬 패키지입니다.

Numpy의 몇가지 장점을 소개하자면,

- 빠르고 메모리를 효율적으로 사용하여 벡터의 산술연산과 브로드캐스팅 연산을 지원하는 다차원 배열 ndarray 데이터 타입을 지원한다.
- 반복문을 작성할 필요 없이 전체 데이터 배열에 대해 빠른 연산을 제공하는 다양한 표준 수학 함수를 제공한다.
- 배열 데이터를 디스크에 쓰거나 읽을 수 있다. (즉, 파일로 저장)
- 선형대수, 난수발생기, 푸리에 변환 가능, C/C++ 포트란으로 쓰여진 코드를 통ㅇ합한다.

## 따라하며 Numpy를 배워보자!

### 1. ndarray 만들기

- arange()
- array([])

```python
import numpy as np

# 아래 A와 B는 결과적으로 같은 ndarray 객체를 생성합니다.
A = np.arange(5)
B = np.array([0, 1, 2, 3, 4]) # 파이썬 리스트를 numpy ndarray로 변환
```

### 2. 크기 (size, shape, ndim)

- ndarray.size
- ndarray.shape
- ndarray.ndim
- reshape()

`size`, `shape`, `ndim`는 각각 행렬 내 원소의 개수, 행렬의 모양, 행렬의 축(axis)의 개수를 의미합니다. `reshape()` 메소드는 행렬의 모양을 바꿔줍니다. 모양을 바꾸기 전후 행렬의 총 원소 개수 (size)가 맞아야 한다.

아래와 같이 사용할 수 있다.

```python
A = np.array(10).reshape(2, 5) # 2x5 2차원 행렬로 만듦
print(A.size)
print(A.shape)
print(A.ndim)
```

만약 원소의 개수가 10개인데 3x3 행렬로 바꾸려고 하는 경우에는 에러가 난다.



### 3. Type

Numpy 라이브러리 내부의 자료형들은 파이썬 내장함수와 동일하다. 그런데 살짝 헷갈리는 기능이 있는데, 바로 내장함수 `type()`와 `dtype()`다.

- Numpy : numpy.array.dtype
- 파이썬 : type()

```python
A = np.arange(6).reshape(2, 3)
print(A)
print(A.dtype)
print(type(A))
```

출력

```
int64
<class 'numpy.ndarray'>
```

Numpy의 메소드인 `dtype`는 Numpy의 ndarray의 '원소' 데이터 타입을 반환한다. 반면 파이썬 내장함수인 `type(A)`를 사용하면 행렬 A의 자료형을 반환!


### 4. 특수 행렬

Numpy는 수학적으로 의미가 있는 행렬들을 함수로 제공하고 있습니다.

- 단위행렬
- 0행렬
- 1행렬

아래와 같이 사용할 수 있습니다.

```python
# 단위행렬
np.eye(3)

# 0 행렬
np.zeros([2, 3])

# 1 행렬
np.ones([3, 3])
```

### 5. 브로드캐스트

Numpy의 강력한 연산 기능 중 하나!

![img](https://numpy.org/devdocs/_images/theory.broadcast_2.gif)

ndarray와 상수, 또는 서로 다른 ndarray끼리 산술연산이 가능한 기능을 말합니다.

아래 코드를 돌려봐서 어떻게 다른지 한번 확인해보세요.

```python
print([1, 2] + [3, 4])
print([1, 2] + 3)

import numpy as np
print(np.array([1, 2]) + np.array([3, 4]))
print(np.array([1, 2]) + 3)
```

### 6. 슬라이스와 인덱싱

Numpy도 파이썬 내장 리스트와 마찬가지로 슬라이스와 인덱싱 연산을 제공한다. 


### 7. Random

Numpy에서 다양한 의사 난수를 지원한다.

- np.random.randint()
- np.random.choice()
- np.random.permuataion()
- np.random.normal()
- np.random.uniform()

```python
# 0에서 1 사이의 실수형 난수 하나 생성
print(np.random.random())

# 0~9 사이 1개 정수형 난수 생성
print(np.random.randint(0, 10))

# 리스트에 주어진 값 중 하나를 랜덤하게 선택
print(np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]))


# 랜덤하게 리스트를 섞는다
print(np.random.permutation(10))
print(np.random.permutation([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))


# 이것은 정규분포를 따릅니다.
print(np.random.normal(loc=0, scale=1, size=5))    # 평균(loc), 표준편차(scale), 추출개수(size)를 조절해 보세요.

# 이것은 균등분포를 따릅니다. 
print(np.random.uniform(low=-1, high=1, size=5))  # 최소(low), 최대(high), 추출개수(size)를 조절해 보세요.

```

### 8. 전치행렬

행렬의 행과 열을 맞바꾸기, 행렬의 축을 서로 바꾸기 등에 사용도는 기능입니다.

- arr.T
- np.transpose

```python
A = np.arange(24).reshape(2,3,4)
print(A)               # A는 (2,3,4)의 shape를 가진 행렬입니다. 
print(A.T)            # 이것은 A의 전치행렬입니다. 
print(A.T.shape) # A의 전치행렬은 (4,3,2)의 shape를 가진 행렬입니다.
```
