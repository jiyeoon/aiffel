# 여러가지 Optimizer

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd1gcAq%2FbtqInMynwWE%2FKQYLFGKKtfD2OUskjaouuK%2Fimg.png)

위 사진이 이번 포스팅의 내용을 잘 요약한 한 장의 사진입니다. :)

인공지능 모델을 학습시킬 때 사용할 수 있는 방법은 단순히 경사하강법 말고도 다양한 방법이 있습니다. 하나하나 살펴보도록 합시다.


## 1\. 경사하강법 (GD, Gradient Descent)

뉴럴 네트워크의 weight들을 모아놓은 벡터를 $w$라고 했을 때, 뉴럴 네트워크에서 내놓은 결과값과 실제 결과값 사이의 차이를 정의하는 손실함수 $C(w)$의 값을 최소화하기 위해 gradient의 반대 방향으로 일정 크기만큼 이동해내는 것을 반복하여 손실 함수의 값을 최소화 하는 $w$ 값을 찾는 알고리즘을 **경사하강법(GD)**이라고 합니다. 가장 기본적인 방법이에요.

이때, 한 iteration에서의 변화식은 아래와 같습니다.

```math
$$w\_{t+1}=w\_t−\\eta\\nabla\_{w\_t} C(w\_t)$$
```

$\\eta$가 미리 정해진 걸음의 크기(step size)로서, 위에서 언급한 하이퍼파라미터인 _학습률_(learning rate)입니다. 보통 0.01 ~ 0.001 정도의 크기를 사용하며, 학습 초기에는 좀 더 높게 잡아주었다가 뒤로 가면서 미세 조절을 위해 작게 잡아주는 방식을 많이 택합니다.

## 2\. 확률적 경사하강법 (SGD, Stochastic Gradient Descent)

GD에서는 한번 step을 내딛을 때 전체 훈련 데이터에 대해 손실함수를 계산해야 하므로 너무 많은 계산량이 필요해서 속도가 매우 느립니다. 이를 방지하기 위해 확률적 경사하강법을 사용합니다.

이 방법은 손실함수를 계한할 때 전체 데이터 대신 _일부 조그마한 데이터의 모음_, 즉 **mini batch**를 사용합니다. GD보다 다소 부정확할 수 있지만, 훨씬 계산 속도가 빠르기 때문에 같은 시간에 더 많은 step을 갈 수 있으며 여러번 반복할 경우 보통 batch의 결과와 유사한 결과로 수렴합니다. 또한 SGD를 사용할 경우 GD에 비해 local minima에 빠지지 않고 더 좋은 방향으로 수렴할 가능성도 있습니다.

하나 유의할 점은 빠르다곤 했지만 어디까지나 GD에 비해 빠르다는 것이지, 밑에서 소개할 다른 방법들에 비하면 여전히 느립니다.

보통 뉴럴 넷에서 훈련을 할 때에는 이 GSD를 사용합니다. 즉, baseline이에요. 그러나 단순한 SGD를 이용하여 네트워크를 학습시키는 것에는 한계가 있습니다. 그래서 여러 문제점들을 개선시킨 다양한 옵티마이저들을 살펴봅시다.

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd1gcAq%2FbtqInMynwWE%2FKQYLFGKKtfD2OUskjaouuK%2Fimg.png)

위에서 보여드렸던 사진을 다시 한번 가져와보았습니다. 여기서 파란색 화살표로 연결된 것들은 Momentum 계열로, 속도를 최대한 빠르게 하는 것에 중점을 두고있고, 빨간색 화살표로 연결된 것을은 Adaptive 계열로, 방향을 최대한 일직선으로 하는데에 중점을 두고 있습니다.

---

# 모멘텀 (Momentum)

모멘텀 방식은 말 그대로 SGD를 통해 이동하는 과정에서 일종의 **관성**을 주는 것입니다. 현재 gradient를 통해 이동하는 방향과는 별개로, 과거에 이동했던 방식을 기억하면서 그 방향으로 일정 정도를 추가적으로 이동하는 방식입니다.

수식으로 표현하면 아래와 같은 식이 나옵니다. 여기서 $v\_t$를 time step $t$에서의 이동 벡터를 나타내는 겁니다.

$$v\_{t+1} = m v\_t + \\eta \\nabla\_{w\_t} C(w\_t)$$

$$w\_t=w\_{t-1}−v\_t$$

이때, $m$은 얼마나 momentum을 줄 것인지에 대한 _관성항_ 값으로, 보통 0.9 정도의 값을 사용합니다. 식을 살펴보면 과거에 얼마나 이동했는지에 대한 이동 항 $v\_t$에 관성항 값을 곱하고 gradient 벡터에 step size $\\eta$를 곱해서 더해줍니다.

이 방법의 대표적인 장점은 관성 효과로 인해 양방향, 음방향 순차적으로 일어나는 지그재그 현상이 줄어든다는 것입니다. 진동을 하더라도 중앙으로 가는 방향에 힘을 얻기 때문에 SGD에 비해 상대적으로 빠르게 이동할 수 있습니다.


![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbdd5fd%2FbtqIqmlSMpr%2FlDAGviIKWmrQfAUWAkWKp1%2Fimg.png)


## 아다그래드 (Adagrad)

Adagrad는 변수들을 업데이트 할 때 _각각의 변수마다 step size를 다르게 설정_해서 이동하는 방식입니다. 즉, _지금까지 많이 변화하지 않은 변수들은 step size를 크게 하고, 지금까지 많이 변화했던 변수들은 step size를 작게 하자_는 겁니다.

자주 등장하거나 변화를 많이 한 변수들의 경우 optimum에 가까이 있을 확률이 높기 때문에 작은 크기로 이동하면서 세밀한 값을 조정하고, 적게 변화한 변수들은 optimum 값에 도달하기 위해서는 많이 이동해야할 확률이 높기 때문에 먼저 빠르게 Loss 값을 줄이는 방향으로 이동하려는 방식입니다.

특히, word2vec이나 GloVe같이 word representation을 학습시킬 경우 단어의 등장 확률에 따라 variable의 사용 비율이 확연하게 차이나기 때문에 Adagrad와 같은 학습 방식을 사용하면 훨씬 더 좋은 성능을 거둘 수 있습니다.

Adagrad의 한 스텝을 수식화하여 나타내면 아래와 같습니다.

$$G\_{t+1} = G\_t + (\\nabla\_{w\_t} C(w\_t))^2$$

$$w\_{t+1} = w\_t - \\frac{\\eta}{\\sqrt{G\_t+\\epsilon}} \\cdot \\nabla\_{w\_t} C(w\_t)$$

여기에서 $G\_{t+t}$을 업데이트 하는 식에서 제곱은 element-wise 제곱을 의미하며, $w\_{t+1}$을 업데이트 하는 식에서도 ⋅ 은 element-wise한 연산을 의미합니다.

weight가 $k$개라고 할때, $G\_t$는 $k$차원 벡터임에 주의하세요. $G\_t$는 time step $t$까지 각 변수가 이동한 gradient의 sum or squares를 저장하구요.

$w\_{t+1}$을 업데이트하는 식에서 기존 step size $\\eta$$에 $G\_t$의 루트값을 반비례한 크기로 이동을 진행하여, 지금까지 많이 변화한 변수일수록 적게 이동하고 적게 변화한 변수일수록 많이 이동하는 것이 핵심 아이디어입니다. 이때 $\\epsilon$은 0으로 나누는 것을 방지하기 위해 $10^{−4} \\sim 10^{−8}$ 정도의 작은 값입니다.

Adagrad를 사용하면 학습을 진행하면서 굳이 step size decay 등을 신경써주지 않아도 된다는 장점이 있습니다. 보통 Adagrad에서 step size로는 0.01정도를 사용한 뒤, 그 이후로는 바꾸지 않습니다.

반면, Adagrad에는 학습을 계속 진행하면 step size가 너무 줄어든다는 문제점이 있습니다. $G$에는 계속 제곱한 값을 넣어주기 때문에 G의 값들은 계속해서 증가합니다. 그래서 학습이 오래 진행될 경우 step size가 너무 작아져서 결국 거의 움직이지 않게 됩니다.

이를 보완하여 나온 알고리즘이 아래 RMSprop입니다.

## RMSProp

위에서 말했듯이 RMSProp은 딥러닝의 대가 제프리 힌튼(Jeoffrey Hinton)이 제안한 방법으로서, Adagrad의 단점을 해결하기 위해 나온 방법입니다.

Adagrad의 식에서 gradient의 제곱값을 더해나가면서 구한 $G\_t$ 부분을 합이 아니라 _지수평균(Exponential Average)_ 으로 바꾸어서 대체한 방법입니다.

이렇게 대체할 경우 Adagrad처럼 $`G\_t`$가 무한정 커지지는 않으면서 최근 변화량의 변수간 상대적인 크기 차이는 유지할 수 있습니다. 식으로 표현하면 아래와 같습니다.

$$G\_{t+1} = \\gamma G\_t + (1-\\gamma)(\\nabla\_{w\_t} C(w\_t))^2$$

$$w\_{t+1} = w\_t - \\frac{\\eta}{\\sqrt{G\_t+\\epsilon}} \\cdot \\nabla\_{w\_t} C(w\_t)$$

## 아담 (Adam)

Adam은 RMSProp과 Momentum 방식을 합친 것 같은 알고리즘입니다. 이 방식에서는 Momentum 방식과 유사하게 지금까지 계산해온 기울기의 지수평균을 저장하며, RMSProp과 유사하게 기울기의 제곱값의 지수평균을 저장합니다.

$$m\_{t+1} = \\beta\_1 m\_t + (1-\\beta\_1)\\nabla\_{w\_t} C(w\_t)$$

$$v\_{t+1} = \\beta\_2 v\_t + (1-\\beta\_2)(\\nabla\_{w\_t} C(w\_t))^2$$

다만, Adam에서는 $m$과 $v$가 처음에 0으로 초기화되어있기 때문에 학습의 초반부에서는 $m\_t$, $v\_t$가 0에 가깝게 bias 되어잇을 것이라고 판단하여 이를 unbiased 하게 만들어주는 작업을 거칩니다.

보통 $\\beta\_1$로는 0.9, $\\beta\_2$로는 0.999, $\\epsilon$으로는 $10^{−8}$정도의 값을 사용합니다.

---

> References..

-   [그래디언트 디센트 @ ratsgo's blog](https://ratsgo.github.io/deep%20learning/2017/09/25/gradient/)
-   [Gradient Descent Optimization Algorithms 정리](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html)


> 수식이 깨지네요. 제 블로그에서 다시 한번 확인해보세요 : <https://butter-shower.tistory.com/156>