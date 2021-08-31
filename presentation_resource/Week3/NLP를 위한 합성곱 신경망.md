**<i>✨Convolutional Neural Network for NLP✨</i>**
========

- Convolution: 주로 이미지, 비전 분야에서 사용되는 알고리즘

# **<i>1. Convolution Neural Network</i>**
- Convolution layer와 Pooling layer로 구성됨

## <i>Advent of Convolution Neural Network</i>
- MLP: 공간적인 구조(spatial structure) 정보가 쉽게 유실됨
  - 공간적인 구조 정보: 거리가 가까운 어떤 픽셀들끼리는 어떤 연관이 있고, 어떤 픽셀들끼리는 값이 비슷하다 등을 포함
- 이미지의 공간적인 구조 정보를 보존하면서 학습할 수 있는 방법 필요

<br><br><br>

## <i>Channel</i>
- 이미지: (H, W, C) 3차원 텐서
    - H: 이미지의 세로 방향 픽셀 수
    - W: 이미지의 가로 방향 픽셀 수
    - C: 색 성분(=깊이)

<br><br><br>

## <i>Convolution operation</i>
- **Convolution**: kernel(=filter)라는 N*M 크기의 행렬로 H*W 크기의 이미지를 처음부터 끝까지 겹치며 훑으면서 N*M 크기의 겹쳐지는 부분의 각 이미지와 커널의 원소의 값을 곱해서 모두 더한 값을 출력으로 하는 것
    - 커널은 일반적으로 3*3 or 5*5

<img width=120% height=120%>![First Step](https://wikidocs.net/images/page/64066/conv4.png)</img>

<img width=120% height=120%>![Second Step](https://wikidocs.net/images/page/64066/conv5.png)</img>

<img width=120% height=120%>![Third Step](https://wikidocs.net/images/page/64066/conv6.png)</img>

<img width=120% height=120%>![Fourth Step](https://wikidocs.net/images/page/64066/conv7.png)</img>

<br>

- **Feature Map**: input으로부터 kernel을 사용하여 convolution을 통해 나온 결과
<img width=120% height=120%>![Feature Map](https://wikidocs.net/images/page/64066/conv8.png)

<br>

- **Stride**: 커널의 이동 범위
- kernel_size=3, stride=2
<img width=120% height=120%>![Stride Example](https://wikidocs.net/images/page/64066/conv9.png)

<br><br><br>

## <i>Padding</i>
- Convolution의 결과로 얻은 Feature map의 크기는 초기 input보다 작아진 상태가 된다.
- **padding**: convolution 연산 이후에도 feature map 크기가 input과 동일하게 유지되도록 하고 싶을 때 사용
- 
<img width=120% height=120%>![Padding Example](https://wikidocs.net/images/page/64066/conv10.png)

<br><br><br>

## <i>Weight and Bias</i>

<br>

### **CNN의 Weight**
- CNN의 Weight = Kernel의 Elements

<img width=120% height=120%>![CNN Weight Example](https://wikidocs.net/images/page/64066/conv13.png)

- MLP보다 훨씬 적은 수의 Weight를 사용하며 Spatial Structure 를 보존
- Hidden Layer에서 ReLU를 주로 사용함

<br>

### **CNN의 Bias**
<img width=120% height=120%>![CNN Bias Example](https://wikidocs.net/images/page/64066/conv14.png)

- bias는 kernel 적용 후에 더해짐

<br><br><br>

## <i>Calculate Feature Map Size</i>

- $O_h = floor((I_h - K_h + 2P) / S + 1)$

- $O_w = floor((I_w - K_w + 2P) / S + 1)$

<br><br><br>

## <i>Convolution in multiple channels (convolution of 3D tensor)</i>
- 커널의 채널 수 = 입력의 채널 수
- Convolution은 채널별로 수행됨

<img width=120% height=120%>![CNN Bias Example](https://wikidocs.net/images/page/64066/conv15.png)

<br><br><br>

## <i>Convolution of 3D Tensor</i>
<img width=120% height=120%>![CNN Bias Example](https://wikidocs.net/images/page/64066/conv16_final.png)

- Total Weight parameter: $K_i * K_o * C_i * C_o$

<br><br><br>

## <i>Pooling</i>
- feature map을 downsampling 하여 크기를 줄임
    - feature map의 weight 개수가 줄어듦
- max pooling, average pooling
<img width=120% height=120%>![CNN Bias Example](https://wikidocs.net/images/page/64066/maxpooling.PNG)

<br>
<br>
<br>
======
<br>

# **<i>1D Convolutional Neural Networks for NLP</i>**

<br>

## <i>1D Convolutions</i>
- 1D CNN의 input: 각 단어가 벡터로 변환된 sentence matrix (LSTM과 동일함)
    - LSTM의 input: embeded sentence
- 'wait for the video and don't rent it' → Tokenization → Padding → Embedding Layer
<img width=120% height=120%>![1D CNN Input Example](https://wikidocs.net/images/page/80437/sentence_matrix.PNG)

<img width=120% height=120%>![1D CNN First Step](https://wikidocs.net/images/page/80437/1d_cnn.PNG)
- width 방향으로는 더 이샹 움직일 곳이 없음
    - 커널이 sentence matrix의 height 방향으로만 움직인다. = 커널이 아래쪽으로만 이동할 것

<img width=120% height=120%>![1D CNN Other Steps](https://wikidocs.net/images/page/80437/%EB%84%A4%EB%B2%88%EC%A7%B8%EC%8A%A4%ED%85%9D.PNG)
- kernel size에 따라서 **참고하는 단어 묶음의 크기가 달라진다.**
    - **참고하는 N-gram이 달라진다.**

<br><br><br>

## <i>Max Pooling</i>
- max pooling: convolution 연산으로부터 얻은 vector에서 가장 큰 값을 가진 scalar 값만 빼내는 연산

<br><br><br>

## <i>Design Neural Network</i>
- binary classification
- softmax
- (kernel_size = 4) * 2, (kernel_size = 3) * 2, (kernel_size=2) * 2
- Max Pooling
- Dense Layer를 이용한 Text Classification

<img width=120% height=120%>![Network Structure](https://wikidocs.net/images/page/80437/conv1d.PNG)

<br>
<br>
<br>
======
<br>

# **<i>Classify IMDB Reviews by 1D CNN</i>**
[Colab NoteBook](https://colab.research.google.com/drive/1s-ogEJxLNCCAqWwkbJbQ0eheUJMhDqSG)

<br>
<br>
<br>
======
<br>

# **<i>Classify Spam Mails by 1D CNN</i>**
[Colab NoteBook](https://colab.research.google.com/drive/1wl3Lk2R7vxE17l_3pKX9DPy0BGI_R36s)

<br>
<br>
<br>
======
<br>

# **<i>Classify Naver Movie Reviews by Multi-Kernel 1D CNN</i>**
[Colab NoteBook](https://colab.research.google.com/drive/12C1e9_9zNLrT9B3FxiUMftD4zMVASEmb)

<br>
<br>
<br>
======
<br>

# **<i>Intent Classification using Pre-trained Word Embedding</i>**
[Colab NoteBook](https://colab.research.google.com/drive/1TquwVdmbYdwor9h67GJ7hwBCYyvYDpWf)

<br>
<br>
<br>
======
<br>

# **<i>Character Embedding</i>**
- OOV 문제를 해결

<br>
<br>
<br>
======
<br>

## <i>Character Embedding by 1D CNN</i>
- 1D CNN: 전체 시퀀스 입력 내부의 더 작은 시퀀스로부터 정보를 얻어냄

![Character Embedding by 1D CNN](https://wikidocs.net/images/page/116193/%EC%BA%A1%EC%B2%981.PNG)

- Embedding: Character에 대해서
- 최종 표현: Character-level representation
- 어떤 단어이든 기본적으로 Character 레벨로 쪼개기 때문에 OOV 문제가 해결됨

<br><br><br>

## <i>Character Embedding by BiLSTM</i>

![Character Embedding by BiLSTM](https://wikidocs.net/images/page/116193/bilstm.PNG)

- Embedding: Character에 대해서
- 최종 표현: Character-level representation
- 정방향 LSTM: 정방향으로, 순차적으로, Character Embedding Vector를 읽음
- 역방향 LSTM: 역방향으로, 순차적으로, Character Embedding Vector를 읽음
- Character Embeding을 Word Embedding과 연결하여 신경망의 입력으로 사용함