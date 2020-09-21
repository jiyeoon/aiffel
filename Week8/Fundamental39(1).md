# 빅데이터 연대기

빅데이터 기술은 2000년대 들어 급속도로 발전해왔습니다. 시간순으로 나열하면 GFS(Google File System)와 맵리듀스 기술이 공개되고 Hadoop이 나오고 그 뒤 Spark가 발표되었습니다. 빅데이터 기술 자체가 굉장히 크기 때문에 API 형태로 필요한 기술만 쓰는 경우도 물론 많이 있지만 보통 Hadoop Ecosystem이라고 부르는 Hadoop 기반의 거대한 빅데이터 플랫폼 위에 Spark가 적용되는 것이 일반적입니다.

그리고 Hadoop은 기본적으로 자바 기반이며, Spark는 Java와 Scala가 기본 언어입니다. 그러나 일부 기능에 한에 파이썬으로도 조작이 가능하도록 API 형태로 제공하기도 합니다. 

이번 포스팅에서는 시간순으로 빅데이터 기술이 어떻게 발전되어 왔는지, 그리고 Hadoop Ecosystem과 Spark란 무엇인지에 대해 알아보도록 하겠습니다.

## 빅데이터 연대기

### 2003년 : GFS (Google File System)

다들 친숙하실 구글은 언제나 새로운 데이터에 대한 니즈와 그 변화의 바람을 먼저 읽고 준비한 회사라고 할 수 있습니다. 초기부터 데이터에 대한 고민을 정말 많이 하였고 새로운 데이터 시대에 맞는 파일 시스템과 프로그래밍 모델을 각각 발표하는데요, 이는 현재 빅데이터의 근간이 되는 기술입니다.

우선 제일 처음으로 발표한 2003년 빅데이터용 새로운 파일시스템인 GFS(Google File System)을 발표합니다. 시스템 고장을 효과적으로 처리할 수 있게 복제가 굉장히 용이하고 분산처리에 적합한 파일 시스템입니다.

아래는 GFS에 관한 논문입니다. 한번 참고해보세요.

- [The Google File System](https://static.googleusercontent.com/media/research.google.com/ko//archive/gfs-sosp2003.pdf)

### 2004년 : Mapreduce on Simplified Data Processing on Large Clusters

뒤이어 바로 1년 뒤인 2004년 구글에서는 분산 처리 환경에 맞는 프로그래밍 모델인 맵리듀스를 발표합니다. GFS와 같은 분산 처리 파일 시스템에 적용하기 쉬운 프로그래밍 모델입니다. 데이터 관련 작업을 맵(map) 함수와 리듀스(reduce)함수의 두가지 작업으로 나누어 처리하는 것으로, 맵함수에서는 키-값 쌍을 처리해 중간의 키-값 쌍을 생성하고 리듀스 함수는 동일한 키와 연관된 모든 중간 값들을 병합하는 함수였습니다.

### 2004년 - 2005년 : NDFS Project

아파치(Apache) 재단의 더그 커팅(Doug Cutting)과 마이크 카파렐라(Mike Cafarella)가 중심이 되어 검색 엔진의 효과적인 분산 처리를 위해 NDFS(Nutch Distributed File System)이란 프로젝트를 시작합니다. Nutch(너치)는 엘라스틱 서치라고 하는 검색 엔진의 전신 소프트웨어입니다.

### 2006년 - 2007년 : Apache Hadoop

그리고 더그 커팅이 중심이 되어 구글이 발표한 맵리듀스와 GFS 개념을 더 보완하여 빅데이터용 오픈소스 프로젝트를 시작하는데요, 하둡이 어느정도 성과를 보이자 아파치 재단에서는 이 프로젝트를 가장 우선순위가 높은 프로젝트로 공식 프로젝트를 발표하고, 2006년 하둡 1.0이 만들어집니다. 하둡 1.0의 핵심 기술은 HDFS(Hadoop Distributed File System)과 하둡 MapReduce 2가지입니다. 현재 빅데이터의 근간이 되는 기술들입니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-2.max-800x600.png)

### 2007년 - 2008년 : 폭발적인 성장

그리고 이무렵 더그 커팅은 야후(yahoo)에서 근무하게 됩니다. 야후에서 그 확장성을 인정받은 뒤, 세계의 유수한 기업 Facebook, LinkedIn, Twitter 등의 회사에서 하둡을 사용하며 그 인기가 증가합니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-3.max-800x600.png)

### 2009년 - 2013년 : Apache Spark

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-4.max-800x600.png)

하둡에도 한가지 단점이 있는데요, 바로 **하드 디스크에서 파일을 처리한다는 점**입니다. 물론 하드디스크가 값도 싸고 보존에도 용이하다는 장점이 있지만 간혹 고속으로 데이터를 처리할 때, 메모리에 올려서 처리해야할 때도 있습ㄴ디ㅏ.이러한 니즈가 점점 생겨나면서 메모리 기반(In-memory)의 데이터 처리 방법에 대해서 고민하게 됩니다. 

UC 버클리의 마태자하리아(matei zaharia)는 이 점을 개선하기 위한 프로젝트를 시작합니다. 하둡과 똑같은 맵리듀스 개념을 사용하지만 데이터 처리 방법과 Task 정리 방법을 개선하여 **Rdd(Resilient Distributed Dataset)를 이용한 스파크**란 프로그램을 발표합니다. RDD를 한국어로 번역하자면 *탄력적 분산 데이터셋*이라고 합니다. 이는 스파크의 기본이 되는 데이터셋입니다. 본 프로젝트는 2009년부터 시작하여 2012년 마테 자하리아의 박사과정 논문으로 공식 발표됩니다. 

아래는 그 논문 링크입니다. 한번 읽어보세요.

- [Resilient Distributed Datasets: A Fault-Tolerant Abstraction forIn-Memory Cluster Computing](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf)
- [Spark: Cluster Computing with Working Sets](https://www.usenix.org/legacy/event/hotcloud10/tech/full_papers/Zaharia.pdf)


그리고 2012년부터 이 프로젝트는 아파치 재단으로 넘어가 아파치 재단의 최상위 프로젝트로 선정되어 2013년 아파치 스파크0.7이 발표됩니다.


### 2014년 - 2020년 : Databricks와 Apache Spark

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/F-39-5.max-800x600.png)

데이터브릭스(databricks)란 회사를 들어보신 적 있으신가요? 214년도에 설립된 스타트업 회사로 스파크를 만든 마태 자하리아가 설립한 회사입니다. 스파크는 기본적으로 클러스터 위에서 동작하는 프로그램입니다. (물론 로컬에서도 동작합니다.) 데이터 브릭스는 스파크 관련 데이터 분석 및 클러스터 환경을 제공해주는 회사입니다. 이 회사에서 상업용으로 만들거나 스파크 관련 클라우드 환경까지 제공해주고 있습니다. 동시에 스파크는 아파치 재단에서 오픈소스로도 개발을 진행하고 있으며, 현재(2020년) 아파치 스파크는 3.0.0이 공개되었습니다.

아래 동영상은 마태 자하리아의 spark 소개 영상입니다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/p8FGC49N-zM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


---

빅데이터 기술은 21세기 들어 급격하게 발전되었습니다. 

끝으로 Hadoop을 만든 더그 커딩 관련 재밌는 글이 있어 첨부합니다. 한번 읽어보세요!

- [빅데이터 시대를 열다, 하둡을 창시한 더그 커팅](https://brunch.co.kr/@hvnpoet/98)