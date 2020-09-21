# 빅데이터 양대산맥, Hadoop Ecosystem과 Spark Ecosystem

2003년부터 빅데이터 처리를 위한 새로운 시스템이 생겨났고, 가장 기본이 되는 기술은 하둡과 스파크입니다. 빅데이터 기술의 양대산맥을 이루고 있는 하둡과 Spark를 하나씩 알아봅시다.

## Hadoop Ecosystem

하둡은 맵리듀스 HDFS가 나온 이후에도 각각의 컴포넌트들이 추가되며 매우 큰 SW를 이루었습니다. 이를 **Hadoop Ecosystem**이라고 합니다. 아래 그림은 하둡의 에코시스템을 도식화한 그림입니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-6.max-800x600.png)

각각의 컴포넌트들을 역할에 따라 재정렬하면 아래와 같습니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-7.max-800x600.png)

위 그림에 나타난 주요 컴포넌트들의 역할을 소개하면 아래와 같습니다.

- **데이터 수집 (Data Ingestion)**
    - 스쿱 (Sqoop) : RDBMS와 하둡 사이의 데이터를 이동시켜줍니다.
    - 플럼 (Flume) : 분산 환경에서 대량의 로그 데이터를 효과적으로 수집하여 합친 후 다른 곳으로 전송합니다.
- **데이터 처리 (Data Processing)**
    - 하둡 분산파일시스템(HDFS) : 하둡의 분산 처리 파일 시스템
    - 맵리듀스 (Mapreduce) : Java 기반의 맵리듀스 프로그래밍 모델
    - 얀 (Yarn) : 하둡 클러스터의 자원(Resource)를 관리
    - 스파크 (Spark) : In-memory 기반의 클러스터 컴퓨팅 데이터 처리. 스파크 안에도 스파크 코어, 스파크 SQ, Milib, GraphX와 같은 컴포넌트가 있습니다.
- **데이터 분석 (Data Analysis)**
    - 피크 (Pig) : 맵리듀스로 실행하기 어려운 데이터 관련 작업, filter, join, query와 같은 작업을 실행합니다.
    - 임팔라 (Impala) : 고성능의 SQL 엔진
    - 하이브 (Hive) : 임팔라와 유사한 SQL 관련 기능을 제공합니다.
- **데이터 검색 (Data Exploration)**
    - 클라우데라 서치 (Cloudera Search) : real-time으로 데이터에 검색이 가능합니다.
    - 휴 (Hue) : 웹 인터페이스 제공
- **기타**
    - 우지 (Oozie) : 워클로우 관리, Job 스케줄러
    - Hbase : NoSQL 기반으로 HDFS에 의해 처리된 데이터를 저장합니다.
    - 제플린 (Zeppelin) : 데이터 시각화
    - SparkMLib, 머하웃(Mahout) : 머신러닝 관련 라이브러리


--- 

## Spark Ecosystem

이번엔 스파크에 대해서도 한번 알아봅시다. 앞서 하둡 에코시스템에서 스파크는 In-memeory 기반의 클러스터 컴퓨팅 데이터 처리 프로그램이라고 했는데요, 그렇게 보면 스파크가 완전히 독립적인 생태계를 이루고 있는 것이 아니라 하둡 기반의 빅데이터 생태계를 이루는 주요 컴포넌트로 어울려 존재하고 있다고 할 수 있습니다.

하지만 스파크 안에서도 Spark SQL, Spark Streaming, MiLib과 같은 라이브러리가 있으며, 스파크 관점에서 빅데이터 생태계를 아래 그림과 같이 재구성해볼 수 있습니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-8.max-800x600.png)


#### 프로그래밍 언어 : Scala, Java, Python, R, SQL

스파크가 지원하는 프로그래밍 언어로는 Scala, Java, Python, R이 있습니다. 위 그림에는 없지만 SQL 역시 지원하고 있습니다.

#### 스파크 라이브러리

각각의 라이브러리는 다음과 같은 역할을 합니다.

- Spark SQL : SQL 관련 작업
- Streaming : Streaming 데이터 처리
- MLlib : Machine Learning 관련 라이브러리
- GraphX : Graph Processing

자원관리(주로 클러스터 관리)는 하둡의 Yarn을 사용하기도 하고 Mesos를 사용하거나 Spark Spark를 사용합니다.

데이터 저장은 Local FS(File System)이나 하둡의 HDFS를 이용하거나 AWS의 S3 인스턴스를 이용하기도 합니다. 그리고 기존의 RDBMS나 NoSQL을 사용하는 경우도 있습니다. 하둡의 HDFS같이 스파크의 전용 분산 데이터 저장 시스템을 별도로 가지고 있지 않다는 점에서 스파크의 에코 시스템이 가지는 유연한 확장성이 강조된 설계 사상을 확인할 수 있습니다. 


## Hadoop과 Spark 비교

빅데이터의 큰 시스템인 하둡과 스파크에 대해서 알아보았습니다.

끝으로 두 시스템의 차이점을 한번 확인해보세요!

<iframe width="560" height="315" src="https://www.youtube.com/embed/xDpvyu0w0C8" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

