# [Paper Review]GloVe: Global Vectors for Word Representation

> ## 기존 문제점

- Word2Vec
  - 저차원 벡터공간에 임베딩 된 단어벡터 사이의 유사도를 측정하는데는 좋다.
  - 그러나 사용자가 지정한 윈도우 내에서만 학습/분석이 이뤄지기때문에 co-occurrence정보는 반영되기 어렵다. 
- LSA
  - 말뭉치 전체의 통계적인 정보를 모두 활용한다.
  - 그러나 단어/문서 간 유사도를 측정하기는 어렵다.

---

> ## Key Idea

- 동시 등장 확률로써 단어를 표현할 수 있다.
- 전체 코퍼스에서의 동시 등장 확률 로그값과, 임베딩된 중심 단어와 주변 단어 벡터의 내적이 같게 만들자!
- 임베딩 된 단어벡터 간 유사도 측정을 수월하게 하면서도 말뭉치 전체의 통계 정보를 좀 더 잘 반영해보자!

---

> ### 윈도우 기반 동시 등장 행렬

- 행과 열 : 단어 집합의 단어들로 구성
- i단어의 윈도우 크기 내에서 k단어가 등장한 횟수를 적음
- 이 행렬은 전치해도 동일한 행렬이 된다.

<img src="https://user-images.githubusercontent.com/59716219/130346735-27d23267-0b06-42e5-9132-30de804415e8.png" width="400" height="250">

---

> ## 동시 등장 확률

- <img src="https://render.githubusercontent.com/render/math?math=P(k|i)"> : (i가 등장 했을 때 단어 k가 등장한 횟수)/(i의 전체 등장 횟수 카운트)
- 이때 i : 중심단어, k : 주변 단어
- k가 solid일때, ice와 등장할 확률이 steam과 등장할 확률보다 훨씬 크다
- k가 water, fasion인 경우는 비슷한 비율로 동시에 등장한다고 볼 수 있다.
- 동시 등장 확률을 기반으로 손실 함수 설계

<img src="https://user-images.githubusercontent.com/59716219/130346806-9e8e49f1-9dd4-4962-8405-3cda6376968f.png" width="400" height="150">

---

> ## 손실 함수

<img src="https://user-images.githubusercontent.com/59716219/130347049-5709d657-aa2e-4b7c-9259-91d5993585f4.png" width="400" height="200">

- 중심단어와 주변단어 벡터의 내적이 동시 등장확률이 되어야한다.
- 그래서 <img src="https://render.githubusercontent.com/render/math?math=dot product(w_i, \tilde{w_k}) \approx P(k|i) = P_{ik}"> 이렇게 표현
- 이때 앞서 배운 개념을 사용하여 <img src="https://render.githubusercontent.com/render/math?math=w_i, w_j, \tilde{w_k}"> 를 가지고 어떤 함수 F를 수행하면, <img src="https://render.githubusercontent.com/render/math?math=P_{ik}/P_{jk}"> 가 나온다는 초기 식으로부터 전개
  -> <img src="https://render.githubusercontent.com/render/math?math=F(w_i, w_j, \tilde{w_k}) = P_{ik}/P_{jk}"> 
  
- 함수 F는 동시 등장 확률의 크기 관계 비율 정보를 벡터 공간에 인코딩 해야한다. 이를 위해 두 중심벡터의 차이를 함수 F의 입력으로 준다
  -> <img src="https://render.githubusercontent.com/render/math?math=F(w_i - w_j, \tilde{w_k}) = P_{ik}/P_{jk}"> 
  
- 우변은 스칼라값, 좌변은 벡터값이다. 이를 위해 함수 F의 입력을 내적을 해준다.
  -> <img src="https://render.githubusercontent.com/render/math?math=F({(w_i - w_j)}^T \tilde{w_k}) = P_{ik}/P_{jk}"> 
  
- 이때 중심단어와 주변단어는, 학습을 하며 역할이 계속 바뀐다. (현재 주변단어가 중심단어가 되면, 중심단어는 주변단어가 된다). 그렇기에 중심단어와 주변단어는 똑같은 모양이어야 한다.
- 이를 위해 함수 F는 실수의 덧셈과 양수의 곱셈에 대해 준동형을 만족하도록 한다. 
  - (준동형(homomorphishm) :두 개의 Group 사이의 모든 연산과 관계를 보존하는 사상(map))
  - (ex) F(x+y) = F(x)F(y))(대표적 함수 : exp)
- 준동형식에서 x와y가 각 벡터값이라면 F의 결과값이 스칼라값이 나올 수 없다. 그러므로 준동형식을 바꿔보자
  -> <img src="https://render.githubusercontent.com/render/math?math=F(v_1^Tv_2"> + <img src="https://render.githubusercontent.com/render/math?math=v_3^Tv_4) = F(v_1^Tv_2)F(v_3^Tv_4), \forall{v_1, v_2, v_3, v_4 \in V}">
  
- GloVe식에 적용!
  ->  <img src="https://render.githubusercontent.com/render/math?math=F({(w_i - w_j)}^T \tilde{w_k}) = P_{ik}/P_{jk} = F(W_i^T\tilde{w_k})/F(W_j^T\tilde{w_k})">
  ->  <img src="https://render.githubusercontent.com/render/math?math=F(w_i^T\tilde{w_k}) = P_{ik} = X_{ik}/X_i">  
  
- 이때 F를 만족시키는 함수는 exp이다!
  -> <img src="https://render.githubusercontent.com/render/math?math=exp(w_i^T\tilde{w_k}) = P_{ik} = X_{ik}/X_i">
  -> <img src="https://render.githubusercontent.com/render/math?math=w_i^T\tilde{w_k} = logP_{ik} = logX_{ik} - logX_i">
 
- 이때 <img src="https://render.githubusercontent.com/render/math?math=w_i"> 와 <img src="https://render.githubusercontent.com/render/math?math=\tilde{w_i}"> 는 두 값의 위치를 바꾸어도 성립해야한다.
- <img src="https://render.githubusercontent.com/render/math?math=log X_i">를 편향 b에 관한 상수항으로 대체하자
  -> <img src="https://render.githubusercontent.com/render/math?math=w_i^T\tilde{w_k}"> + <img src="https://render.githubusercontent.com/render/math?math=b_i"> + <img src="https://render.githubusercontent.com/render/math?math=\tilde{b_k} = logX_{ik}">
  
> ### <img src="https://render.githubusercontent.com/render/math?math=Loss function = \sum^V_{m,n = 1}{(w_i^T\tilde{w_k}}"> + <img src="https://render.githubusercontent.com/render/math?math=b_i"> + <img src="https://render.githubusercontent.com/render/math?math={\tilde{b_k} - logX_{ik})}^T">

---

> ## 동시 등장 행렬 X는 DTM처럼 희소행렬일 가능성이 크다!

- 동시 등장 행렬 X에는 많은 값이 0이거나, 작은 수치를 가진다. 
- GloVe는 동시 등장 빈도의 값이 낮은 경우는 정보에 거의 도움이 되지 않는다고 판단. 
- <img src="https://render.githubusercontent.com/render/math?math=X_{ik}">의 값에 영향을 받는 가중치 함수 <img src="https://render.githubusercontent.com/render/math?math=f(X_{ik})">를 손실함수에 도입
- <img src="https://render.githubusercontent.com/render/math?math=f(X_{ik})">

<img src="https://user-images.githubusercontent.com/59716219/130348280-844bb322-6f5b-4dd7-9222-3cba51aeb7e3.png" width="300" height="200">

- <img src="https://render.githubusercontent.com/render/math?math=f(x) = min(1,(x/x_{max})^{3/4}">
> ### 최종식 : <img src="https://render.githubusercontent.com/render/math?math=Loss function = \sum^V_{m,n = 1}{f(X_{mn})(w_i^T\tilde{w_k}}"> + <img src="https://render.githubusercontent.com/render/math?math=b_i"> + <img src="https://render.githubusercontent.com/render/math?math={\tilde{b_k} - logX_{ik})}^2">

---

> ## result

- word analogy task(단어 유추)(dog : puppy :: cat : _______)

<img src="https://user-images.githubusercontent.com/59716219/130491608-c20d5782-f053-42a5-81aa-c36758b0e079.png" width="300" height="260">

- word similarity task

<img src="https://user-images.githubusercontent.com/59716219/130493558-af9353c9-9d6d-413b-bcec-8d2fd87688a5.png" width="300" height="200">

- NER task with the CRF-based model

<img src="https://user-images.githubusercontent.com/59716219/130493537-05642bfb-25e2-4a2a-bd40-72ee9802c51f.png" width="300" height="200">
