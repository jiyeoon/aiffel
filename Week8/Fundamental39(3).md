# Spark의 데이터 처리 : RDD(Resilient Distributed Dataset)

이전 게시글 
- [빅데이터 연대기](https://butter-shower.tistory.com/163)
- [빅데이터 양대산맥, Hadoop Ecosystem과 Spark Ecosystem](https://butter-shower.tistory.com/164)

하둡과 스파크는 매우 큰 시스템입니다. 하둡은 주 언어가 Java 기반이고 스파크는 주 언어가 Scala 기반입니다. 그리고 기본적으로 클러스터 환경에서 동작하는 프로그램이기 때문에 AWS, Azure 혹은 GCP 등의 클라우드 환경에서 주로 사용합니다.

이번 포스팅에서는 Spark의 동작 원리에 대해서 살펴보도록 하겠습니다.

아래 논문은 마태 자하리아의 논문에서 소개된 RDD와 Spark에 대한 소개글입니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-10.max-800x600.png)

> We have implemented RDDs in a system called Spark, which we evaluate through a variety of user applications and benchmarks.

즉, 스파크는 RDD(Resilient Distributed Dataset)을 구현하기 위한 프로그램입니다. RDD를 스파크라는 프로그램을 통해 실행시킴으로써 메모리 기반의 대량의 데이터 연산이 가능하게 되었고, 이는 하둡보다 100배는 빠른 연산을 가능하게 해주었습니다.

따라서 이번 노드에서는 스파크 기본 개념으로 RDD와 스파크가 어떻게 실행되는지(Spark Execution)에 대해서 알아봅시다.



## RDD 등장 배경

우선 하둡 기반의 데이터 처리 방법과 스파크 기반의 데이터 처리 방법을 비교해봅시다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-11.max-800x600.png)

하둡은 파일을 디스크에 저장한 뒤 그걸 불러와 연산(주로 맵리듀스 연산)을 하고 다시 디스크에 저장하면서 파일 처리 작업을 수행합니다. 모든 연산마다 디스크에 파일을 읽고 불러오니 디스크에서 파일을 읽고 쓰는데 시간이 너무 오래걸립니다. 즉, **I/O 바운드가 하둡의 주요 병목 현상입니다.**

이것을 해결하기 위해 스파크는 하드디스크에서 파일을 읽어온 뒤 연산 단계에는 데이터를 메모리에 저장하자는 아이디어를 생각해냈습니다. 그랬더니 속도가 매우 빨라졌습니다. 그런데 메모리는 태생이 *비휘발성*입니다. 뭔가 매모리에 적재하기 좋은 새로운 형태의 *추상화 작업*(abstraction)이 필요합니다. 그렇게 고안된 것이 바로 **RDD(Resilient Distributed Dataset)**입니다. 번역하자면 탄력적 분산 데이터셋입니다.

> 정리하자면, RDD는 스파크에서 사용하는 기본 추상 개념으로, 클러스터의 머신(노드)의 여러 메모리에 분산하여 저장할 수 있는 데이터의 집합을 의미합니다.


## RDD의 특징

메모리에 저장된 데이터가 중간에 데이터가 유실되면 어떻게 될까요? (결함이 생기면) 스파크는 새로운 방법을 고안합니다. 메모리의 데이터를 읽기 전용(Read only)로 만듭니다. 그리고 데이터를 만드는 방법을 기록하고 있다가 데이터가 유실되면 다시 만드는 방법을 사용합니다. 이를 데이터 만드는 방법, 즉 계보(Lineage, 리니지(!!))를 저장합니다.

RDD의 특징을 요약하자면 아래와 같습니다.

- In-Memory
- Fault Tolerance
- Immuatble (read-only)
- Partition

각 파티션은 RDD 전체 데이터중 일부를 나타냅니다. 스파크는 데이터를 여러대 머신에 분할하여 저장하며, chunk, 혹은 파티션으로 분할되어 저장합니다. 파티션을 RDD 데이터의 부분을 표현하는 단위 정도로 이해하면 좋습니다.


---

# RDD의 생성과 동작

## RDD 생성 (Creation)

RDD를 만드는 방법에는 두가지가 있습니다.

- 내부에서 만들어진 데이터 집합을 **병렬화**하는 방법 : `parallelize()` 함수 사용
- 외부의 파일을 로드하는 방법 : `.textFile()` 함수 사용


## RDD 동장 (Operation)

RDD 동작은 크게 Transformations과 Actions 2가지 입니다.

- Transformations
- Actions

각각의 동작들에 해당하는 함수들은 아래와 같습니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-12.max-800x600.png)

RDD는 Immutable(불변)하다고 했습니다. 따라서 연산 수행에 있어 기존의 방식과는 다르게 수행됩니다. Transformations는 RDD에게 변형 방법(연산 로직, 계보, lineage)을 알려주고 새로운 RDD를 만듭니다. 그러나 실제 연산의 수행은 Actions를 통해 행해집니다.

이를 도식화하면 아래와 같습니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-13.rdd.max-800x600.png)

Transformation을 통해 새로운 RDD를 만듭니다. actions은 결과값을 보여주고 저장하는 역할을 하며 실제 Transformations 연산을 지시하기도 합니다.

![Img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-14.max-800x600.png)

다음 그림은 RDD의 생성, Transformations 동작, Actions 동작을 도식화 한 그림입니다.

![Img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-15.max-800x600.png)

`sc.textFile()`을 통해 RDD를 생성합니다.

그러나 이 작업은 실제로는 RDD의 Lineage(계보)를 만드는데 지나지 않습니다. 실제 객체는 생성되지 않았습니다. 그리고 transformations 함수 중 하나인 `filter()`를 만듭니다. 이 역시 lineage를 만드는 일에 지나지 않습니다.

실제 RDD가 생성되는 시점은 **Actions**의 함수인 `counts()`를 실행할 때 입니다.

이런식으로 결과값이 필요할 때 가지 계산을 늦추다가 정말 필요한 시기에 계산을 수행하는 방법을 **느긋한 계산법(Lazy Evaluation)**이라고 합니다. 느긋한 계산법은 아래 링크에서 한번 다시 확인해보세요.

- [위키백과 - 느긋한 계산법](https://ko.wikipedia.org/wiki/%EB%8A%90%EA%B8%8B%ED%95%9C_%EA%B3%84%EC%82%B0%EB%B2%95)
