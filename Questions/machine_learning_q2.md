딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는? <br>
A.
    1.
    
왜 갑자기 딥러닝이 부흥했을까요? <br>
A.
    1. 인간이 해결할 수 없는 또는 기존 알고리즘으로 해결이 어려운 문제들을 
    쉽게 해결
    
마지막으로 읽은 논문은 무엇인가요? 설명해주세요 <br>
A.
    1. WaveGAN, SpecGAN
    
Cost Function과 Activation Function은 무엇인가요? <br>
A.
    1. 
    
Tensorflow, Keras, PyTorch, Caffe, Mxnet 중 선호하는 프레임워크와 그 이유는 무엇인가요? <br>
A.
    1. Tensorflow/Keras
    2. Tensorflow 는 low level 하게 각 layer 마다의 weight 컨트롤도 편하고 tensorboard 라는 visualization 툴이 있어서
    3. Keras 는 high level API 덕분에 빠르게 개발 가능
    
Data Normalization은 무엇이고 왜 필요한가요? <br>
A.
    1. normalize 가 되지 않은 상태의 데이터를 넣게 되면 예를 들어 이미지 같은 경우 [0, 255] 로 스케일 된 데이터를 넣게 되면
    각 값들이 너무 커서 연산이 어려워 질 수도 있다. <- 막기 위해 함
    
알고있는 Activation Function에 대해 알려주세요. (Sigmoid, ReLU, LeakyReLU, Tanh 등) <br>
A.
    1. Activation Function : <br>
    - sigmoid <br>
    - tanh <br>
    - ReLU <br>
    - LeakyReLU <br>
    - ELU <br>
    - PReLU <br>
    - SELU <br>

오버피팅일 경우 어떻게 대처해야 할까요? <br>
A.
    1. network capacity 를 줄여본다.
    2. feature engineering 확인
    3. 

하이퍼 파라미터는 무엇인가요? <br>
A.
    1. 네트워크에 필요한 여러 파라메터들, 성능을 결정한다.
    2. batch_size, learning rate, kerenl size, 등등
    
Weight Initialization 방법에 대해 말해주세요. 그리고 무엇을 많이 사용하나요? <br>
A.
    1. 가중치를 초기화 하는 방식인데, weight 의 초기점을 찾는 것이 성능을 결정하기도 한다.
    최근에는 xavier 나 HE weight initialization 을 주로 쓴다.


볼츠만 머신은 무엇인가요? <br>
A.
    1. 확률적으로 순환하는 네트워크?

뉴럴넷의 가장 큰 단점은 무엇인가? 이를 위해 나온 One-Shot Learning은 무엇인가? <br>
A.
    1.
