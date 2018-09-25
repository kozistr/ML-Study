1. 요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는? <br>
A. 
    1. sigmoid 는 relu 보다 cost 가 비싸다 (exp(x) 연산 때문에)
    2. |x| > 10 에서 saturated 되서 gradient 가 거의 0이 되어 버린다. (vanishing gradient)
    3. relu 는 non-linear function 이다.
    4. sigmoid 는 non-zero-centered 다.
    5. sigmoid 를 여러 번 사용 하게 되어보 미분 값이 (1-x)(x) 형태라 매 사용 시 gradient 가 25% 로 줄기 때문에
    또 gradient 가 vanishing 되는 문제가 있다.

+ Non-Linearity라는 말의 의미와 그 필요성은?
A.
    1. 비 선형성이라는 말이고 network 의 표현력을 높혀준다.
    
+ ReLU로 어떻게 곡선 함수를 근사하나?
A.
    1. target 이 convex function 일 때, 각 구간을 세세하게 미분하면 linear 하게 볼 수 있는데,
    그 부분을 ReLU 에 approximate 한다.
    
+ ReLU의 문제점은?
A.
    1. 여전히 non-zero-centered 하다.
    2. x < 0 부분에서는 gradient 가 0 이다.
    
+ Bias는 왜 있는걸까?
A.
    1. bias-variance trade-off 란 말이 있는데, 가정을 잘못 하게 되었을 때 발생하는 오차를 줄이기 위해 있는 것이다.
    2. 어찌 보면 bias 도 모델 표현력에 포함 된다고 할 수 있을 것 같다.

2. Gradient Descent에 대해서 쉽게 설명한다면?
A.
    1. 
   

+ 왜 꼭 Gradient를 써야 할까?
+ 그 그래프에서 가로축과 세로축 각각은 무엇인가?
+ 실제 상황에서는 그 그래프가 어떻게 그려질까?
+ GD 중에 때때로 Loss가 증가하는 이유는?
+ 중학생이 이해할 수 있게 더 쉽게 설명 한다면?
+ Back Propagation에 대해서 쉽게 설명 한다면?

3. Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?

+ GD가 Local Minima 문제를 피하는 방법은?
+ 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?

4. CNN에 대해서 아는대로 얘기하라

+ CNN이 MLP보다 좋은 이유는?
+ 어떤 CNN의 파라메터 개수를 계산해 본다면?
+ 주어진 CNN과 똑같은 MLP를 만들 수 있나?
+ 풀링시에 만약 Max를 사용한다면 그 이유는?
+ 시퀀스 데이터에 CNN을 적용하는 것이 가능할까?

5. Word2Vec의 원리는?

+ 그 그림에서 왼쪽 파라메터들을 임베딩으로 쓰는 이유는?
+ 그 그림에서 오른쪽 파라메터들의 의미는 무엇일까?
+ 남자와 여자가 가까울까? 남자와 자동차가 가까울까?
+ 번역을 Unsupervised로 할 수 있을까?

6. Auto Encoder에 대해서 아는대로 얘기하라

+ MNIST AE를 TF나 Keras등으로 만든다면 몇줄일까?
+ MNIST에 대해서 임베딩 차원을 1로 해도 학습이 될까?
+ 임베딩 차원을 늘렸을 때의 장단점은?
+ AE 학습시 항상 Loss를 0으로 만들수 있을까?
+ VAE는 무엇인가?

7. Training 세트와 Test 세트를 분리하는 이유는?

+ Validation 세트가 따로 있는 이유는?
+ Test 세트가 오염되었다는 말의 뜻은?
+ Regularization이란 무엇인가?

8. Batch Normalization의 효과는?

+ Dropout의 효과는?
+ BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?
+ GAN에서 Generator 쪽에도 BN을 적용해도 될까?

9. SGD, RMSprop, Adam에 대해서 아는대로 설명한다면?

+ SGD에서 Stochastic의 의미는?
+ 미니배치를 작게 할때의 장단점은?
+ 모멘텀의 수식을 적어 본다면?

10. 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?

+ 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?
+ Back Propagation은 몇줄인가?
+ CNN으로 바꾼다면 얼마나 추가될까?

11. 간단한 MNIST 분류기를 TF나 Keras 등으로 작성하는데 몇시간이 필요한가?

+ CNN이 아닌 MLP로 해도 잘 될까?
+ 마지막 레이어 부분에 대해서 설명 한다면?
+ 학습은 BCE loss로 하되 상황을 MSE loss로 보고 싶다면?
+ 만약 한글 (인쇄물) OCR을 만든다면 데이터 수집은 어떻게 할 수 있을까?

12. 간단한 MNIST DCGAN을 작성한다면 TF 등으로 몇줄 정도 될까?

+ GAN의 Loss를 적어보면?
+ D를 학습할때 G의 Weight을 고정해야 한다. 방법은?
+ 학습이 잘 안될때 시도해 볼 수 있는 방법들은?

13. 딥러닝할 때 GPU를 쓰면 좋은 이유는?

+ 학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는?
+ GPU를 두개 다 쓰고 싶다. 방법은?
+ 학습시 필요한 GPU 메모리는 어떻게 계산하는가?

14. TF 또는 Keras 등을 사용할 때 디버깅 노하우는?

15. Collaborative Filtering에 대해 설명한다면?

16. AutoML이 뭐하는 걸까?

딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는?

왜 갑자기 딥러닝이 부흥했을까요?

마지막으로 읽은 논문은 무엇인가요? 설명해주세요

Cost Function과 Activation Function은 무엇인가요?

Tensorflow, Keras, PyTorch, Caffe, Mxnet 중 선호하는 프레임워크와 그 이유는 무엇인가요?

Data Normalization은 무엇이고 왜 필요한가요?

알고있는 Activation Function에 대해 알려주세요. (Sigmoid, ReLU, LeakyReLU, Tanh 등)

오버피팅일 경우 어떻게 대처해야 할까요?

하이퍼 파라미터는 무엇인가요?

Weight Initialization 방법에 대해 말해주세요. 그리고 무엇을 많이 사용하나요?

볼츠만 머신은 무엇인가요?

요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는?

Non-Linearity라는 말의 의미와 그 필요성은?

ReLU로 어떻게 곡선 함수를 근사하나?

ReLU의 문제점은?

Bias는 왜 있는걸까?

Gradient Descent에 대해서 쉽게 설명한다면?

왜 꼭 Gradient를 써야 할까? 그 그래프에서 가로축과 세로축 각각은 무엇인가? 실제 상황에서는 그 그래프가 어떻게 그려질까?

GD 중에 때때로 Loss가 증가하는 이유는?

중학생이 이해할 수 있게 더 쉽게 설명 한다면?

Back Propagation에 대해서 쉽게 설명 한다면?

Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?

GD가 Local Minima 문제를 피하는 방법은?

찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?

Training 세트와 Test 세트를 분리하는 이유는?

Validation 세트가 따로 있는 이유는?

Test 세트가 오염되었다는 말의 뜻은?

Regularization이란 무엇인가?

Batch Normalization의 효과는?

Dropout의 효과는?

BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?

GAN에서 Generator 쪽에도 BN을 적용해도 될까?

SGD, RMSprop, Adam에 대해서 아는대로 설명한다면?

SGD에서 Stochastic의 의미는?

미니배치를 작게 할때의 장단점은?

모멘텀의 수식을 적어 본다면?

간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?

어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?

Back Propagation은 몇줄인가?

CNN으로 바꾼다면 얼마나 추가될까?

간단한 MNIST 분류기를 TF, Keras, PyTorch 등으로 작성하는데 몇시간이 필요한가?

CNN이 아닌 MLP로 해도 잘 될까?

마지막 레이어 부분에 대해서 설명 한다면?

학습은 BCE loss로 하되 상황을 MSE loss로 보고 싶다면?

만약 한글 (인쇄물) OCR을 만든다면 데이터 수집은 어떻게 할 수 있을까?

딥러닝할 때 GPU를 쓰면 좋은 이유는?

학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는?

GPU를 두개 다 쓰고 싶다. 방법은?

학습시 필요한 GPU 메모리는 어떻게 계산하는가?

TF, Keras, PyTorch 등을 사용할 때 디버깅 노하우는?

뉴럴넷의 가장 큰 단점은 무엇인가? 이를 위해 나온 One-Shot Learning은 무엇인가?
