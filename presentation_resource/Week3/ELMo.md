# Deep contextualized word representations(ELMo) 논문 리뷰

ELMo는 Embeddings from Language Model의 약자로 해석하면 '언어 모델로 하는 임베딩'이다.
ELMo의 가장 큰 특징은 사전 훈련된 언어 모델(Pre-trained language model)을 사용한다는 점이다.

즉, 엘모는 사전훈련과 문맥을 고려하는 문맥 반영 언어 모델이다.

## Introduction

이 representation은 (문장 내) 각 token이 전체 입력 sequence의 함수인 representation를 할당받는다는 점에서 전통적인 단어 embedding과 다르다.

이를 위해 이어붙여진 language model(LM)로 학습된 bidirectional LSTM(biLM)로부터 얻은 vector를 사용합니다.
또한, lstm의 마지막 레이어만 사용하는 기존의 방법론과는 달리, ELMo는 lstm의 모든 내부 레이어를 사용해서 만들어지기 때문에 더욱 많은 정보를 사용할 수 있다.
- higher-level LSTM : 문맥을 반영한 단어의 의미를 잘 표현
- lower-level LSTM : 단어의 문법적인 측면을 잘 표현


![image](https://user-images.githubusercontent.com/61388801/130611787-b2567202-71d5-49c7-8f2b-0985690d51e2.png)

즉, ElMo는 위의 예시와 같이, '단어의 문맥, 의미, 다양성'을 고려하고, 이전 모델과 달리 이를 구별할 수 있다는 것이다.

## ELMo

ELMo는

- 전체 문장을 input으로 받고, 그에 대한 각 단어들의 representation 생산
- Character convolution로부터 얻은 biLM의 가장 위 2개 layer의 가중합으로 계산
- 큰 사이즈로 biLM pretrain 시켰을 때 semi-supervised learning 가능
- 쉽게 다른 모델에 붙일 수 있음

### 3-1) Bidirectional language models

biLM = forward language model + backward language model

N개의 token (t1,t_2, …,t_N)이 있다고 하면

> forward language model

 (t_1,t_2, …,t(k-1))이 주어졌을 때 token t_k가 나올 확률을 계산하는 것

![image](https://user-images.githubusercontent.com/61388801/130612348-46c03e16-f541-4298-8f2b-9b49ef3f1df9.png)


> backward language model

(𝑡(𝑘+1),𝑡(𝑘+2), …,𝑡𝑁)이 주어졌을 때 token 𝑡𝑘가 나올 확률을 계산

![image](https://user-images.githubusercontent.com/61388801/130613011-c34471a6-fab3-44d3-a8af-6cf6d8cee200.png)

biLM = forward language model + backward language model 이었으니,

두 방향의 log likelihood를 최대화 시키는 방향으로 학습을 진행한다.

![image](https://user-images.githubusercontent.com/61388801/130612823-ca94e1b4-113d-472b-99c8-59af3f40f345.png)

이때 (Θx는 token representation, Θs는 Softmax layer)이며, 이 둘은 parameter와는 다르게 고정된다.

### 3-2) ELMo



