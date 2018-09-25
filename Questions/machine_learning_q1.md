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
    1. 한국어로는 경사 하강법. 가장 저점을 찾아 오차를 줄여주는 것.

+ 왜 꼭 Gradient를 써야 할까?
A.
    1. gradient 를 통해 계산 된 loss 를 줄이기 위해 그 차이만큼 weight 를 adjust 하기 위해.
    
+ 그 그래프에서 가로축과 세로축 각각은 무엇인가?
A.
    1. (2d plot 에 그렸다면) 가로축은 gradient 세로축은 loss
    
+ 실제 상황에서는 그 그래프가 어떻게 그려질까?
A.
    1. weight-loss-vector 를 가지는 3d plot 으로 그려질 것이다.
    
+ GD 중에 때때로 Loss가 증가하는 이유는?
A.
    1. 현재 있는 위치가 local minina 여서 주위에서 다른 곳으로 가다가 loss 가 올라가거나
    2. step size 가 커서 그냥 건너 뛰다 더 작은 loss 값에 도달하지 못해서
    
+ 중학생이 이해할 수 있게 더 쉽게 설명 한다면?
A.
    1. 가장 차이가 작은 곳을 찾아줘서 좋은 성능을 뽑아 내 주는 것

+ Back Propagation에 대해서 쉽게 설명 한다면?
A.
    1. 위에서 저점인 곳을 찾은 만큼 결과를 업데이트 해 주는 작업.
    
3. Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?
A.
    1. local minima 에 빠져도 그 곳이 reasonable 한 local minima 라 performance 가 나름 잘 나오기 떄문에.

+ GD가 Local Minima 문제를 피하는 방법은?
A.
    1. Hmm... local minima 를 피하게 위해 다른 개념들이 사용된 optimzier 가 나오게 되었다. (관성, step size 조절, 등...)

+ 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?
A.
    1. 그건 아~무도 모른다 :). 작년에 GD 로 2 layer MLP 에서 global minima 를 찾는 방법이 증명 되긴 했다.

4. CNN에 대해서 아는대로 얘기하라
A.
    1. Convolutional Neural Network
    2. conv, dilated conv, separable conv, depth,channel-wise conv, coord conv
    3. 이미지 처리를 할 때 주로 사용된다.
    4. 텍스트, 음성 모델에도 사용한다.
    5. 데이터가 많을 때는 다른 네트워크 모듈보다 압도적으로 유리하다
    6. 등등...

+ CNN이 MLP보다 좋은 이유는?
A.
    1. MLP 보다 CNN 이 더 많은 파라메터를 가실 수 있어 표현력이 더 좋다.
    2. 각 filter 에 대해 차원이 높지가 않아 dimension of curse 를 어쩌면 더 피할 수 있다.

+ 어떤 CNN의 파라메터 개수를 계산해 본다면?
A.
    1. in/out filter 32/64 에 kernel size 가 3, stride 1 이면 (3 * 3 * 32) * 64
    2. (kernel height * kernel width * input filter) * output filter

+ 주어진 CNN과 똑같은 MLP를 만들 수 있나?
A.
    1. 음... 가능 할 듯

+ 풀링시에 만약 Max를 사용한다면 그 이유는?
A.
    1. conv pooling 을 제외하고는 제일 compute-wise 하기 때문에

+ 시퀀스 데이터에 CNN을 적용하는 것이 가능할까?
A.
    1. 가능하다.
    2. TextCNN, WaveNet 같은 텍스트, 음성에 Conv1D 를 사용한 approach 들도 많다.

5. Word2Vec의 원리는?
A.
    1. 각 단어를 유의미한 임베딩 vector 로 바꾼다.
    2. CBOW, skip-gram 등의 method 가 있는데 각 method 에 따라 좀 다르다.
    3. CBOW 는 각 단어의 빈도수를 기반으로 트레이닝
    4. skip-gram 은 skip-gram size 만큼 끊어서 set 을 만들어 트레이닝
    
+ 그 그림에서 왼쪽 파라메터들을 임베딩으로 쓰는 이유는?
A.
    1. 그림?
    
+ 그 그림에서 오른쪽 파라메터들의 의미는 무엇일까?
A.
    1. 그림?

+ 남자와 여자가 가까울까? 남자와 자동차가 가까울까?
A.
    1. Word2Vec 방식 트레이닝을 한다면, 트레이닝 데이터에 따라서 달라질 것이다.
    2. case by case :)

+ 번역을 Unsupervised로 할 수 있을까?
A.
    1. 안해봐서 잘 모르겠지만 번역 같은 경우에 label 이 없으면 잘 안될 것 같다.
    2. zero-shot learning 이 가능한 지도 찾아봐야겠다.

6. Auto Encoder에 대해서 아는대로 얘기하라
A.
    1. 간단하게 설명하면 NN 의 unsupervised 버전
    2. input size == output size
    3. Encoder 를 통해 입력 데이터 들의 feature 를 찾고
    4. Decoder 를 통해 compressed 된 feature 를 가지고 새롭게 reconstruct 한다.
    5. training 을 잘 하기 위해서 sparsity 라는 개념이 있다. 

+ MNIST AE를 TF나 Keras등으로 만든다면 몇줄일까?
A.
    1. 짧으면 50줄 길면 100줄 이내가 가능할 것 같다.

+ MNIST에 대해서 임베딩 차원을 1로 해도 학습이 될까?
A.
    1. MNIST 차운이 784 차원인데 이것을 1차원으로 축소를 하고 training 을 한다면. 그 차원 축소를 한 method 에 따라서 다를 것이다.

+ 임베딩 차원을 늘렸을 때의 장단점은?
A.
    1. 장점 : capacity 가 커져서 많고 다양한 word 에 대해 임베딩으로 표현이 가능하다.
    2. 단점 : dimension 이 커지면 계산량이 많아지고 dimension of curse 에 의해 더 성능이 안좋아 질 수도 있다.
    (경험상 한국어 w2v 임베딩을 할 경우 300~400 이 적당한 임베딩 사이즈 같다)

+ AE 학습시 항상 Loss를 0으로 만들수 있을까?
A.
    1. NOP

+ VAE는 무엇인가?
A.
    1. Variational Auto Encoder
    2. 

7. Training 세트와 Test 세트를 분리하는 이유는?

+ Validation 세트가 따로 있는 이유는?
A.
    1. training 시에 over-fitting 여부를 확인해 주는 지표가 된다.

+ Test 세트가 오염되었다는 말의 뜻은?
A.
    1. test 셋도 같이 트레이닝 해 버린 경우

+ Regularization이란 무엇인가?
A.
    1. 정규화.
    2. l1, l2, l1-l2 norm 만큼 갹 weight, bias, gradient 등등에 달아서 해당 값이 업데이트 될 때 over-shooting 하는 것을 방지 해 준다.

8. Batch Normalization의 효과는?
A.
    1. network 를 거친 값들의 분포를 gaussian distribution 에 가깝게 만들어 줘,
    값이 튀지(?) 않는 역할을 해 준다.

+ Dropout의 효과는?
A.
    1. training 시에 모든 neural cell 들을 학습 하지 않고 selective 하게 학습을 해서

+ BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?
A.
    1. BN 에 ```trainable=False``` 로 설정해서 moving variance average 같은 값을 업데이트 하지 않게 해 줘야 한다.

+ GAN에서 Generator 쪽에도 BN을 적용해도 될까?
A.
    1. 실제로 적용 해 본 결과 잘 되는 GAN 이 있고 안되는 GAN 이 있다.

9. SGD, RMSprop, Adam에 대해서 아는대로 설명한다면?
A.
    1. SGD : GD 인데 차이점은 GD 는 전체 데이터를 SGD 는 부분 데이터를 트레이닝 한다.
    2. RMSProp : adagrad 에 exponential moving average 방식의 gradient update 를 한다.
    3. Adam : RMSProp + Momentum.
    4. 물론 이 3개 다 step size wise 한 optimizer 그룹이라고 볼 수 있다. (크게 관성 vs step size 하자면 ㅇㅇ)
    
+ SGD에서 Stochastic의 의미는?
A.
    1. 확률적이란 뜻인데, SGD 측면에서 보면 임의의 batch 를 골라 GD 하겠다는 의미
    
+ 미니배치를 작게 할때의 장단점은?
A.
    1. 장점 : loss sensitive 해 진다.
    2. 단점 : big batch 보다 느리다.
    3. 단점 : batch 가 작게 되면 한 번에 작은 양의 데이터 셋의 feature 밖에 고려를 하지 못하니 한 번에 다양한 feature 트레이닝이 힘들다.

+ 모멘텀의 수식을 적어 본다면?
A.
    1. v = mu * v + - learning_rate * dx; x += v

10. 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?
A.
    1. fc 구현 data loader, optimizer 등등 다 고려 해 보면 ~300줄 정도가 될 듯

+ 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?
A.
    1. reference 있이는 6시간?

+ Back Propagation은 몇줄인가?
A.
    1. fully-connected 만 있으면 이 부분 하나에 대한 back-propagation 은 ~10 줄 내외가 될 거다.

+ CNN으로 바꾼다면 얼마나 추가될까?
A.
    1. conv propagation, back-propagation 정보면 생각 해 보면 ~25줄?

11. 간단한 MNIST 분류기를 TF나 Keras 등으로 작성하는데 몇시간이 필요한가?

+ CNN이 아닌 MLP로 해도 잘 될까?
A.
    1. 잘-은 된다. 그런데 acc 는 비교적 낮을 것이다.

+ 마지막 레이어 부분에 대해서 설명 한다면?
A.
    1. 클래스 사이즈 만큼으로 feature 를 줄여서 실제 classification 에 사용한다.

+ 학습은 BCE loss로 하되 상황을 MSE loss로 보고 싶다면?
A.
    1. loss 는 BCE 를 쓰고 MSE 는 metric 으로 쓰면 된다.

+ 만약 한글 (인쇄물) OCR을 만든다면 데이터 수집은 어떻게 할 수 있을까?
A.
    1. 부분적으로 OCR 되야 할 부분의 이미지와 그에 대한 positional label, (글자 label) 이 필요할 듯

12. 간단한 MNIST DCGAN을 작성한다면 TF 등으로 몇줄 정도 될까?
A.
    1. D/G model 20줄 나머지 3~40줄 해서 ~70줄 내외.

+ GAN의 Loss를 적어보면?
A.
    1. min(d)max(g)V(G,D) = log(D(x)) + log(1 - D(G(z)))

+ D를 학습할때 G의 Weight을 고정해야 한다. 방법은?
A.
    1. TF 라면 각 layer 의 함수에 ```trainable=False``` 를 하면 된다.

+ 학습이 잘 안될때 시도해 볼 수 있는 방법들은?
A.
    1. (모든 case) train,valid,test data verify
    2. over/under-fitting 확인
    3. mode collapse, bn 문제, 등등 확인

13. 딥러닝할 때 GPU를 쓰면 좋은 이유는?
A.
    1. 빠름ㅋ

+ 학습 중인데 GPU를 100% 사용하지 않고 있다. 이유는?
A.
    1. TF 는 내부적으로 최적화 되게 GPU 를 사용하기 때문에 100% 가 아니라고 해서 잘 안되는게 아니다.
    2. 아니면 진짜 GPU 를 안쓰고 CPU 쓰고 있던가 ㅇㅅㅇ

+ GPU를 두개 다 쓰고 싶다. 방법은?
A.
    1. TF 같은 경우면 tf.device('/gpu:N') 으로 지정하던가
    2. keras 면 MultiGPUModel 을 쉽게 해준다. 그냥 n_gpus 에 갯수만 집어 넣으면 된다.

+ 학습시 필요한 GPU 메모리는 어떻게 계산하는가?
A.
    1. 네트워크 크기 + batch_size * 데이터 shape

14. TF 또는 Keras 등을 사용할 때 디버깅 노하우는?
A.
    1. 디버깅을 해 본 경험은 없지만. 실시간 training 에 대한 loss 나 acc 같은 값들은
    tensorboard 를 통해 visualization 을 해 본다.

15. Collaborative Filtering에 대해 설명한다면?
A.
    1. 모르겠다....

16. AutoML이 뭐하는 걸까?
A.
    1. 모델 구조나 하이퍼 파라메터를 자동으로 제일 optimal 한 구조, 값을 찾아주는 것이다.
    2. 방식으로는 R.L, 유전 알고리즘 approach 를 쓴다고 한다.