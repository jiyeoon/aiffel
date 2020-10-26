# 물체를 분리하자! 세그멘테이션 살펴보기

이번 시간에는 세그멘테이션을 살펴볼 것입니다. **세그멘테이션(segmentation)**은 <u>픽셀 수준</u>에서 이미지의 각 부분이 어떤 의미를 갖는 영역인지를 분리해내는 방법입니다. 세그멘테이션은 이미지 분할 기술입니다. 세그멘테이션에 대하여 하나하나 살펴봅시다!


## 세그멘테이션의 종류

이미지 내에서 영역을 분리하는 접근 방식은 크게 두가지 방식이 있습니다. 바로 **인스턴스 세그멘테이션(Instance segmentation)**과 **시멘틱 세그멘테이션(sementic segmentation)**입니다. 영역을 분리한다는 관점에서 비슷하지만, 딥러닝에서는 두 방법은 차이가 있습니다.

아래 그림을 보면 둘의 차이를 바로 이해하실 수 있을 것입니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/original_images/semantic_vs_instance.png)

위 그림과 같이, sementic segmentation에서는 양 3마리를 하나의 '양'이라는 클래스로 묶지만, instance segmentation에서는 양 한마리, 양 두마리, 세마리로 구분합니다. 몇마리의 양이 각각 어느 위치에 있는지 알 수 있습니다. 

### 1) 시맨틱 세그멘테이션 (Sementic Segmentation)

- [SegNet 데모 사이트](https://mi.eng.cam.ac.uk/projects/segnet/#demo)

위 사이트에서는 이미지를 12종류의 라벨로 구분했습니다. (하늘, 빌딩, 나무 등등) 예를 들자면 인물사진 모드라면 사람의 영역과 배경 클래스 2가지 라벨로 구분할 수 있습니다.

### 2) 인스턴스 세그멘테이션 (Instance Segmentation)

인스턴스 세그멘테이션은 같은 클래스 내라도 각 개체들을 분리하여 세그멘테이션을 수행합니다. 

이러한 방식 중 가장 대표적인 것이 **Mask R-CNN**입니다. 2017년에 발표된 Mask R-CNN은 2-stage object detection의 가장 대표적은 Faster R-CNN을 계씅한 것으로서, Faster R-CNN의 아이디어인 Rol(Region of Interest) Pooling Layer(RolPool) 개념을 개선하여 정확한 segmentation에 유리하게 한 **RolAlign**, 그리고 **클래스별 마스크 분리**라는 단순한 두가지 아이디어를 통해 클래스별 object detection과 semantic segmentation을 사실상 하나의 task로 엮어낸 것으로 평가받는 중요한 모델입니다.



---

## 주요 세그멘테이션 모델 (1) - FCN

대표적인 세그멘테이션 방법들에 대해서 살펴보겠습니다. 가장 먼저, *Fully Convolutional Networks for Semantic Segmenatation* 논문의 **FCN(Fully Convolutional Network)**부터 살펴봅시다.

주요 참고 자료

- [Fullly Convolutional Networks for Semantic Segmentation - 허다운](https://www.youtube.com/watch?v=_52dopGu3Cw&feature=youtu.be&ab_channel=%EB%94%A5%EB%9F%AC%EB%8B%9D%EB%85%BC%EB%AC%B8%EC%9D%BD%EA%B8%B0%EB%AA%A8%EC%9E%84)
- [FCN 논문 리뷰 - Fully Convolutional Networks for Semenatic Segmentation](https://medium.com/@msmapark2/fcn-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-fully-convolutional-networks-for-semantic-segmentation-81f016d76204)
- 원본 논문 : [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/fcn.max-800x600.png)

FCN은 이미지넷 첼린지 (ImageNet Challenge)에서 좋은 성적을 거두었던 AlexNet, VGG-16등의 모델을 세그멘테이션에 맞게 변형한 모델입니다. 기본적인 VGG 모델은 이미지의 특성을 추출하기 위해 네트워크 뒷단에 Fully Connected Layer를 붙이지만, FCN에서는 세그멘테이션을 위해 네트워크 뒷단에 FC레이어 대신 CNN을 붙여줍니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/fcn_2.max-800x600.png)

CNN을 붙여주면 위치정보를 그대로 유지시킬 수 있게 됩니다. 위치의 특성을 유지하면서 이미지 분류를 하기 위해 마지막 CNN에서는 1X1의 커널 크기(kernel size)와 클래스의 갯수만큼의 채널을 갖습니다. 이렇게 CNN을 거치면 클래스 히트맵을 얻을 수 있습니다.

하지만 히트맵의 크기는 일반적으로 원본 이미지보다는 작습니다. CNN과 Pooling 레이어를 거치면서 크기가 줄었기 때문인데요, 이를 키워주는 방법을 **upsampling**이라고 합니다. Upsampling에는 여러가지 방법이 있습니다. 그 중 FCN은 **Deconvolution**과 **Interpolation**방식을 활용합니다. Deconvolution은 컨볼루션 연산을 거꾸로 해준 것이고, Interpolation은 보간법으로 주어진 값들을 통해 추정해야하는 픽셀 값을 추정하는 방법입니다.

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/fcn_3.max-800x600.png)

Upsampling만 하면 원하는 세그멘테이션 맵을 얻을 수 있습니다. 그것이 바로 FCN-32s의 경우입니다.

하지만 논문에서는 더 나은 성능을 위해서 한 가지 방법을 더해줍니다. 위 그림에서 확인할 수 있는 **Skip Architecture**라는 방법입니다. 논문에서는 FCN-32s, FCN-16s, FCN-8s로 결과를 구분해 설명합니다. FCN-16s는 앞쪽 블록에서 얻은 예측 결과맵과, 2배로 upsampling한 맵을 더한 후, 한 번에 16배로 upsampling을 해주어 얻습니다. 여기서 한번 더 앞쪽 블록을 사용하면 FCN-8s를 얻을 수 있습니다. 

|FCN-16s|FCN-8s|
|:---:|:---:|
|![img](https://miro.medium.com/max/700/1*-1hOIxlnFn3qd7n5JEgzFg.png)|![img](https://miro.medium.com/max/700/1*1r-KVNqt9V7JiDT-zyOEAQ.png)|

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/fcn_4.max-800x600.png)


<br/>

## 주요 세그멘테이션 모델 (2) - U-Net

*주요 참고 자료*

- [딥러닝논문읽기모임의 U-Net: Convolutional Networks for Biomedical Image Segmentation](https://www.youtube.com/watch?v=evPZI9B2LvQ&ab_channel=%EB%94%A5%EB%9F%AC%EB%8B%9D%EB%85%BC%EB%AC%B8%EC%9D%BD%EA%B8%B0%EB%AA%A8%EC%9E%84)
- [U-Net 논문 리뷰 — U-Net: Convolutional Networks for Biomedical Image Segmentation](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a)
- 원본 논문: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/u-net.max-800x600.png)

U-Net은 네트워크 구조가 U자 형태를 띄고 있습니다. FCN에서 upsampling을 통해 특성맵을 키운 것을 입력값과 *대칭적*으로 만들어준 것인데요, 특이한 점은 U-Net이 세그멘테이션 뿐만 아니라 여러가지 이미지 태스크에서 사용되는 유명한 네트워크가 되었지만, 본래 의학 관련 논문으로 시작되었다는 점입니다. 논문 제목도 *Biomedical Image Segmentation*이라고 표현했죠. 


### 전체 구조

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FciXwVs%2FbtqxyMoIeIu%2FBw1HTfQ5xIxPjKcJkgKPf0%2Fimg.png)

논문에서는 네트워크 구조를 최측의 **Contracting Path**와 우측의 **Expansive Path**로 구분합니다.

- Contracting Path : 일반적인 Convloution network와 유사한 구조. 
    - 각 블록은 2개의 3x3 convloution 계층 + ReLU + 2x2 커널을 2 stride로 + 다시 2배 늘려줌
- Expansive Path : 세밀한 Localization 구성. 높은 차원의 채널을 갖는 Upsampling
    - 각 블록에 2x2 up-convolution이 붙어 채널이 절반씩 줄고 특성맵의 크기는 늘어난다.
    - 3x3 convloution이 두개씩 사용
    - 두 path에서 크기가 같은 블록의 출력과 입력은 skip connection처럼 연결해 low-level의 feature를 활용할 수 있도록 함.
    - 마지막에는 1x1 convolution으로 원하는 시멘틱 세그멘테이션을 얻을 수 있다.
    
결과적으로 입력엔ㄴ 572x572 크기의 이미지가 들어가고, 출력으로 388x388 크기의 두가지 클래스를 가진 세그멘테이션 맵(segmentation map)이 나옵니다.

마지막 세그멘테이션 맵의 크기가 입력 이미지의 크기와 다른 것은 resize를 통해 해결할 수 있습니다. 


### 타일 기법

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/unet.max-800x600.png)

U-Net과 FCN과의 차이점을 생각해봅시다. 구조부터 한눈에 파악이 되지만, 하나 더 꼽자면 얻을 수 있는 세그멘테이션의 해상도도 이에 해당합니다. FCN은 입력 이미지의 크기를 조정하여 세그멘테이션 맵을 얻어내었습니다. 반면 U-Net은 타일(tile) 방식을 사용하여 어느정도 서로 겹치는 구간으로 타일을 나누어 네트워크를 추론, 큰 이미지에서도 높은 해상도의 세그멘테이션 맵을 얻을 수 있도록 했습니다. 

위 그림에서는 파란 영역의 이미지를 입력하면 노란 영역의 segmentation 결과를 얻는 것을 보여줍니다. 다음 타일에 대한 세그멘테이션을 얻기 위해서는 이전 입력의 일부분이 포함되어야합니다. 이러한 이우로 *Overlap Title 전략*이라고 합니다.


### 데이터 불균형 해결

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/unet_2.max-800x600.png)

세포를 검출해내기 위해서는 세포들의 영역 뿐만 아니라 경계 또한 예측해야 합니다. 이때 픽셀 단위로 라벨을 매긴다고 생각하면, 데이터셋에 세포나 배경보다는 절대적으로 세포간 경계의 면적이 작을 것입니다. 이러한 클 데이터 양의 불균형을 해결해주기 위해서 분포를 고려한 **weight map**을 학습 때 사용했다고 합니다.

여기서 weight map의 weight를 신경망의 학습 파라미터를 가리키는 weight와는 다릅니다. 여기서 말하는 weight는 **손실 함수(loss)에 적용되는 가중치**를 의미합니다. 의료 영상에서 세포 내부나 배경보다는 상대적으로 면적이 작은 세포 경계를 명확하게 추론해내는 것이 더욱 중요하기 때문에, 세포 경계의 손실에 더 많은 패너티를 부과하는 방식입니다. 


## 주요 세그멘테이션 모델 (3) - DeepLab 계열

*주요 참고 자료*

- [Lunit 기술블로그의 DeepLab V3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/)
- [hyunjulie님의 2편: 두 접근의 접점, DeepLab V3+](https://medium.com/hyunjulie/2%ED%8E%B8-%EB%91%90-%EC%A0%91%EA%B7%BC%EC%9D%98-%EC%A0%91%EC%A0%90-deeplab-v3-ef7316d4209d)
- [Taeoh Kim님의 PR-045: DeepLab: Semantic Image Segmentation](https://www.youtube.com/watch?v=JiC78rUF4iI&ab_channel=TaeohKim)
- 원본 논문: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

DeepLabV3+라는 이름에서 볼 수 있듯이 이전의 많은 버전을 거쳐 개선을 이뤄온 네트워크입니다. 처음 DeepLab 모델이 제안된 뒤 이 모델을 개선하기 위해 Atrous Convolution와 Spatial Pyramid Pooling 등 많은 방법들이 제안되어 왔습니다. <br/>
DeepLabV3+의 전체 구조를 본 뒤 Dilated Convlution이라고도 불리는 Atrous Convolution과 Spatial Pyramid Pooling을 살펴보도록 하겠습니다.

### 전체 구조

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/deeplab_v3.max-800x600.png)

위 그림이 DeepLabV3+인데요, U-Net은 구조가 직관적으로 보였지만 DeepLabV3+는 다소 복잡해보입니다. U-Net에서의 Contracting Path와 Expansive Path의 역할을 하는 것이 여기서는 위 그림의 인코더, 디코더 입니다.


인코더는 이미지에서 필요한 정보를 특성으로 추출해내는 모듈이고, 디코더는 추출한 특성을 이용해 원하는 정보를 예측하는 모듈입니다. 3x3 convloution을 사용했던 U-Net과 달리 DeepLabV3+는 Atrous Convolution을 사용하고 있습니다. 그리고 이로 Atrous Convolution을 여러 크기에 다양하게 적용한 것이 ASPP(Atrous Spatial Pyramid Pooling)입니다. DeepLabV3+는 AS{{가 있는 블록을 통해 특성을 추출하고 디코더에서 Upsampling을 통해 세그멘테이션 마스크를 얻고 있습니다.

### Atrous Convolution

![img](https://aiffelstaticprd.blob.core.windows.net/media/original_images/atrous_conv_2.gif)

> 띄엄 띄엄 컨볼루션

위 그림에서 우측의 Atrous Convolution은 좌측의 일반적인 컨볼루션과 달리 더 넓은 영역을 보도록 해주기 위한 방법으로 커널이 일정 간격으로 떨어져 있습니다. 이를 통해 컨볼루션 레이어를 너무 깊게 쌓지 않아도 넓은 영역의 정보를 커버할 수 있게 됩니다.

### Spatial Pyramid Pooling

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/GC-5-L-SPP.max-800x600.png)

Spatial Pyramid Pooling은 여러가지 스케일로 convolution과 pooling을 하고 나온 다양한 특성을 연결(concatenate)해줍니다. 이를 통해서 멀티 스케일로 특성을 추출하는 것을 병렬로 수행하는 효과를 얻을 수 있습니다. 여기서 컨볼루션을 Atrous Convolution으로 바꾸어 적용한 것은 Atrous Spatial Pyramid Pooling이라고 합니다. 이러한 아키텍처는 입력 이미지의 크기에 관계 없이 동일한 구조를 활용할 수 있다는 장점이 있습니다. 그러므로 제각기 다양한 크기와 비율을 가진 Rol 영역에 대해 적용하기에 유리합니다.

- 참고 : [갈아먹는 Object Detection [2] Spatial Pyramid Pooling Network](https://yeomko.tistory.com/14)




---

## 세그멘테이션 결과를 평가하는 방법

### 1) 픽셀별 정확도

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/error_metric.max-800x600.jpg)

Pixel Accuracy는 쉽게 말해 픽셀에 따른 정확도를 의미합니다. 각 픽셀별 분류 정확도를 세그멘테이션 모델 평가 기준으로 삼는 것입니다. 




### 2) 마스크 IoU (Mask Intersection-over-Union)

물체 검출 모델을 평가할 때는 정답 라벨(ground truth)와 예측 결과 바운딩 박스 (prediction bounding box) 사이의 IoU(Intersection over Union)를 사용합니다. 마스크도 일종의 영역임을 생각했을 때, 세그멘테이션 문제에서는 정답인 영역과 예측한 범역의 IoU를 계산할 수 있을 것입니다.

![img](https://t1.daumcdn.net/cfile/tistory/993477505D14A25016)

마스크 IoU를 클래스별로 계산하면 한 이미지에서 여러 클래스에 대한 IoU 점수를 얻을 수 있습니다. 이를 평균하면 전체적인 시멘틱 세그멘테이션 성능을 가늠할 수 있습니다.



---

## Upsampling의 다양한 방법


지금까지 Segmentation에 대한 여러가지 종류와 접근 방식에 대해서 알아보았습니다. Segmentation에서는 마스킹 단계에 Upsampling이 중요하게 사용되었다는 것을 알 수 있었습니다. Convolution layer와 다양한 pooling 등으로 Feature의 크기를 줄여왔는데, 반대로 키우는 방법에는 어떤 방법들이 있을까요?

### 1) Nearest Neighbor

![img](https://aiffelstaticprd.blob.core.windows.net/media/original_images/upsampling1.png)

Nearest upsampling은 이름 그대로 scale을 키운 위치에서 원본에서 가장 가까운 값을 그대로 적용하는 방법입니다. 

### 2) Bilinear Interpolation

![img](https://aiffelstaticprd.blob.core.windows.net/media/original_images/bi_interpolation.png)

Biliner Interpolation은 두 축에 대해서 선형보간법을 통해 필요한 값을 메우는 방식입니다.

우리가 2x2 matrix를 4x4로 upampling할 때 위의 이미지처럼 빈 값을 채워야 합니다. 이때 선형보간법을 사용하는 것인데요, 축을 두 방향으로 활용하기 때문에 Bilinear interpolation이라고 표현합니다.

위 그림에서 두 가지 interpolation을 적용한 것을 순서대로 확인할 수 있습니다. $R_{1}$이 $Q_{11}$과 $Q_{21}$를 x축 방향으로 Interpolation 한 결과입니다. $R_2$는 $Q_{12}$와 $Q_{22}$의 y축 방향의 interpolation 결과입니다. 그리고 $R_1$과 $R_2$를 interpolation 하면 새로운 위치의 P 값을 추정할 수 있습니다.

### 3) Transposed Covolution

![img](https://aiffelstaticprd.blob.core.windows.net/media/images/transposed_conv.max-800x600.jpg)

Transposed Convolution은 학습할 수 있는 파라미터를 가졌습니다. 거꾸로 학습된 파라미터로 입력된 벡터를 통해 더 넓은 영역의 값을 추정해냅니다.

- 참고 : [Up-sampling with Transposed Convolution 번역](https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/)




















