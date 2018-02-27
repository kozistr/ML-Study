Sedong Nam 1월 14일 오후 9:06
딥러닝이라는 주제에 대해서 개발자 면접을 자주 하다 보니 어떤 질문을 하면 되는지 대략 정리가 되었다.

1년 이하 정도 딥러닝을 열심히 해 본 개발자들이 일정 수준 이상으로 잘 대답할 것으로 기대하는 질문들이다.

Q1. 요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는?

A.
1. ReLU 가 Sigmoid 보다 cost-efficient 하다. ( exp(x) cost 가 비싸다. )
2. Sigmoid 는 zero-centered 되있지 않다. 항상 양수인 값이 들어오게 되면 weight 에 대한 gradient 는 모두
positive 하거나 negative 해진다 (극단적으로).
3. |x| > 10 구간에서 gradient vanishing 이 발생한다.
4. [더 있으면 후에 더 추가]

+ Non-Linearity 라는 말의 의미와 그 필요성은?

A. 말대로 함수가 비선형이라는 뜻과 당연하게 비선형성의 Activation Function 이 사용되어야 한다.
이유는 다양하고 깊게 neural network 를 만드려고.

+ ReLU로 어떻게 곡선 함수를 근사하나?

A. 곡선을 구간 미분해 그 부분을 연결하면 선형이듯(?) 곡선 함수에 부분 적으로 선형 함수인 ReLU 를 근사한다.

+ ReLU의 문제점은?

A. 
1. 여전히 zero-centered 가 아니다.
2. x < 0 에서의 gradient 는 다 0이 되 버린다.

+ Bias 는 왜 있는걸까?

A. 예를 들면, 단순한 선형 함수만으로는 W 와 5 * W 를 구별 할 수가 없기 때문이다.

2. Gradient Descent 에 대해서 쉽게 설명한다면?

A. 곡선 함수를 미분해 가며 최소의 기울기, 즉 loss 를 찾아가는 과정.

+ 왜 꼭 Gradient 를 써야 할까?

A. loss 의 정도를 구하기 위해(?) // 더 찾아볼 것

+ 그 그래프에서 가로축과 세로축 각각은 무엇인가?

A. 가로축은 weight , 세로축은 loss

+ 실제 상황에서는 그 그래프가 어떻게 그려질까?

A. 3차원 굴곡이 있는 표면!

+ GD 중에 때때로 Loss 가 증가하는 이유는?

A. GD 중 여러 굴곡을 만나는 데, 올라가는 굴곡을 만났을 때.

+ 중학생이 이해할 수 있게 더 쉽게 설명 한다면?

A. 그래프 + 위에가 최선..?!

+ Back Propagation 에 대해서 쉽게 설명 한다면?

A. 차이를 줄이기 위해 계산한 값을 출발지에 다시 보내는 과정.

3. Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?

+ GD가 Local Minima 문제를 피하는 방법은?

A. 띠용...!? 읎는거 같다...

+ 찾은 해가 Global Minimum 인지 아닌지 알 수 있는 방법은?

A. 음... 아직까진 Global Minimum 을 구하는 방법은 알 수가 없고 (불가능), 
2 개의 neural net 인 경우에 SGD 를 사용해서 Global Minimum 을 구하는 방법 까지는 증명되었음.

4. CNN 에 대해서 아는대로 얘기하라

A. Convolutional Neural Network 로 주로 image classification 에 자주 사용되는 네트워크다.
사용 되는 하이퍼 파라메터로는 stride 크기, kernel 크기, padding, filter 크기.

+ CNN 이 MLP 보다 좋은 이유는?

A. 직관적으로 딱 하나만 봤을 때 MLP 보다 feature 수용량이 크다. 즉, 네트워크가 크다.

+ 어떤 CNN 의 파라메터 개수를 계산해 본다면?

A. 단순히 하나 CNN 을 본다면, (input_filter * k_s * k_w) * output_filter 정도가 될 거 같다.

+ 주어진 CNN 과 똑같은 MLP 를 만들 수 있나?

A. 흠.... 가능할거 같다.

+ 풀링시에 만약 Max 를 사용한다면 그 이유는?

A. 음... pooling 자체는 sub sampling 을 통해 데이터 사이즈를 줄이고 (cost efficient), 각 영역들이 대푯값을 추출해서 feature 는 그래도 유지한다.
그런데 avg pooling 대신 max pooling 을 사용하는 거의 차이라면 avg 보다 max pooling 이 더 cost efficient 하고 해당 영역의 최댓값이 아닌 평균값을 
계산해서 feature 를 더 잘 보존할 수 있다.

+ 시퀀스 데이터에 CNN 을 적용하는 것이 가능할까?

A. 입력받은 각 sequence data 를 2d vectorize 한 다음에 CNN 을 적용해도 가능하긴 할 거 같다.

5. Word2Vec의 원리는?

A. 단어들을 학습이 가능한 특정한 벡터로 embedding 하는 과정. 

+ 그 그림에서 왼쪽 파라메터들을 임베딩으로 쓰는 이유는?

A. 

+ 그 그림에서 오른쪽 파라메터들의 의미는 무엇일까?

A. 

+ 남자와 여자가 가까울까? 남자와 자동차가 가까울까?

A. 남자와 여자가 가깝다고 생각한다.

+ 번역을 Unsupervised 로 할 수 있을까?

A. 가능하다!

6. Auto Encoder 에 대해서 아는대로 얘기하라

+ MNIST AE를 TF나 Keras 등으로 만든다면 몇줄일까?

A. Keras 기준으로 DataSet load 1줄, CNN 으로 AE 1~20줄, loss,opt,fit ~10줄. 아마 40줄 내외일 거 같다.

+ MNIST 에 대해서 임베딩 차원을 1로 해도 학습이 될까?

A. 가능하다!?

+ 임베딩 차원을 늘렸을 때의 장단점은?

A.

+ AE 학습시 항상 Loss 를 0으로 만들수 있을까?

A.

+ VAE 는 무엇인가?

A. Various Auto Encoder 로,

7. Training 세트와 Test 세트를 분리하는 이유는?

+ Validation 세트가 따로 있는 이유는?

A. training set 의 over-fitting 을 확인하려고.

+ Test 세트가 오염되었다는 말의 뜻은?

A. 알기로는 Test Set 를 training set 으로 사용하였다는 뜻.

+ Regularization 이란 무엇인가?

A. Weight Regularization 이라해서, 예를 들어 선형 함수에서 W 나 5W 나 결과에 차이가 없어 버리게 되니,
 이를 weight 정규화를 해 주는 것이다.

8. Batch Normalization 의 효과는?

A. neural net 들의 결과 값을 잘 zero-centered 하게 모아준다. 값이 튀?는거 방지.

+ Dropout 의 효과는?

A. over-fitting 방지.

+ BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?

A. train 시와 test 시 역할이 다르다.

+ GAN 에서 Generator 쪽에도 BN을 적용해도 될까?

A. 직접 해 봤었던 결과 대체적으로는 적용 안하는 편이 좋은거 같다. 적용 가능은 하나...

9. SGD, RMSprop, Adam 에 대해서 아는대로 설명한다면?

A, SGD (Stochastic Gradient Descent) 는 확률적 경사 하강법, 엄청 느리다, 주로 fine-tuning 할 때 낮은 learning rate 로 사용한다.
RMSprop, Adagrad 의 일종의 upgrade 버전.
Adam, RMSprop + momentum 이라 볼 수 있음.

+ SGD 에서 Stochastic 의 의미는?

A. 확률적 이라는 뜻. 확장 해석해 보면 임의적 샘플을 골라 Gradient Descent 를 하겠다는 뜻.

+ 미니배치를 작게 할때의 장단점은?

A. 장점으로는 loss sensitive 해 진다. 단점으로는 local minimum 에 빠질 가능성이 높아진다.

+ 모멘텀의 수식을 적어 본다면?

A. v = mu * v + - learning_rate * dx; x += v

10. 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy 로 만든다면 몇줄일까?

A. affine_forward/backward 구현 ~15 줄, softmax train/test 구현 ~40줄, relu forward/backward (~10줄)
등등 2~300줄?

+ 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?

A. 1~2시간?

+ Back Propagation 은 몇줄인가?

A. ~100 줄 내로 나올거 같다.

+ CNN 으로 바꾼다면 얼마나 추가될까?

A. conv forward/backward 등등 를 더 추가해야하니 짧게는 ~60줄이면 충분할 거 같다

11. 간단한 MNIST 분류기를 TF나 Keras 등으로 작성하는데 몇시간이 필요한가?

A. 1~30 분?!

+ CNN 이 아닌 MLP 로 해도 잘 될까?

A. 되긴 하겠지만, 성능은 더 떨어질 것이다.

+ 마지막 레이어 부분에 대해서 설명 한다면?

A. dense(classes, activation='softmax') 로 classify 할 클래스 갯수만큼 feature 를 뽑아주고
 softmax 로 각 label 에 대한 prob 를 계산한다.

+ 학습은 BCE loss 로 하되 상황을 MSE loss 로 보고 싶다면?

A. 마지막 softmax 하기 전 dense layer 를 logits 으로 뽑고 따로 MSE 를 구하면 된다.

+ 만약 한글 (인쇄물) OCR 을 만든다면 데이터 수집은 어떻게 할 수 있을까?

A. 한글 한 글자에 대한 hand-write image 를 수만~십만장 crop ?

12. 간단한 MNIST DCGAN 을 작성한다면 TF 등으로 몇줄 정도 될까?

A. DCGAN 논문에 있는 모델 따라서 Celeb-A DataSet 을 사용해 제대로 만들면? 3~400 줄 되겠지만,
mnist dcgan tf 로는 100줄이면 충분할 거다. 

+ GAN 의 Loss 를 적어보면?

A. d_loss = -tf.reduce_mean(d_real) - tf.reduce_mean(1 - d_fake),
g_loss = tf.reduce_mean(d_fake)

+ D를 학습할때 G의 Weight 을 고정해야 한다. 방법은?

A. trainable 를 False 로 해 둔다.

+ 학습이 잘 안될때 시도해 볼 수 있는 방법들은?

A. learning rate, batch_size 조절

13. 딥러닝할 때 GPU를 쓰면 좋은 이유는?

A. 무쟈게 빨라진다.

+ 학습 중인데 GPU 를 100% 사용하지 않고 있다. 이유는?

A. Tensorflow 같은 경우 최대로 효율적인 정도만 알아서 쓴다. 100% 라고 잘 되는 것도 아니다. 만약 할당량을 더 늘리고 싶으면
gpu_allow_growth 를 True 로 설정하면 된다.

+ GPU 를 두개 다 쓰고 싶다. 방법은?

A. tf.device('/gpu:0 또는 1') 로 scope 를 감싸서 진행한다. Keras 로는 더 편하다.

+ 학습시 필요한 GPU 메모리는 어떻게 계산하는가?

A. Input Image Size * batch_size + alpha

14. TF 또는 Keras 등을 사용할 때 디버깅 노하우는?

A. 디버깅 툴이 있는걸 보긴 했지만 써 본적은 아직 없다. StackOverflow 나 다른 커뮤니티로 대부분 해결이 되었다.

15. Collaborative Filtering 에 대해 설명한다면?

A. 처음 들어보는 개념이다 ㅠㅠ

16. AutoML이 뭐하는 걸까?

A. 자동으로 Hyper Parameter 를 optimizing 해주는 것? 잘 모르겠다...


이상 공통 (기본) 질문들만 정리해 봤다.