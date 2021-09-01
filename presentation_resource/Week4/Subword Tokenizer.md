**<i>✨Subword Tokenizer✨</i>**
========
- 단어 집합에 없는 단어 **OOV(Out-of-Vocabulary), UNK(Unknown Token)** 로 표현한다.
- **OOV 문제**: 모르는 단어로 인해 문제를 푸는 것이 까다로워지는 상황
<br><br><br>
- **Subword Segmentation**: 하나의 단어를 여러 서브워드로 분리해서 단어를 인코딩, 임베딩하려는 전처리 작업
    - [Byte Pair Encoding(2015)](https://arxiv.org/pdf/1508.07909v5.pdf), [Word Piece(2016)](https://arxiv.org/pdf/1609.08144v2.pdf), [Unigram Segmentation(2018)](https://arxiv.org/pdf/1804.10959v1.pdf), [Gradient-Based Subword Tokenization(2021)](https://arxiv.org/pdf/2106.12672v2.pdf) 등 다양한 메소드로 수행.
    - Details: [PapersWithCode](https://paperswithcode.com/methods/category/subword-segmentation)
    - <i>예시</i>
    <br>

        |언어|단어|조합|
        |:---|:---|:---|
        |영어|Concentrate|con(=together) + centr(=center) + ate(=make|
        |한국어|집중(集中)|集(모을 집) + 中(가운데 중)|
    - subword detection을 통해 subword 단위로 쪼개면, 어휘수를 줄일 수 있고, sparsity를 감소시킬 수 있다.
    - 신조어나 오타를 subword / character 단위로 쪼개줌으로써 knwon token들의 조합으로 만들어줄 수 있다!

<br>
<br>
<br>
======
<br>

# **<i>1. Byte Pair Encoding, BPE</i>**
## <i>What is Byte Pair Encoding?</i>
- 데이터 압축 알고리즘 → 서브워드 분리 알고리즘

```python
# 예시 문자열
str = "aaabdaaabac"

# 가장 자주 등장하고 있는 byte pair = "aa"
# "aa"를 하나의 byte로 치환
Z=aa
ZabdZabac

# 가장 자주 등장하고 있는 byte pair = "ab"
# "ab"를 하나의 byte로 치환
Z=aa
Y=ab
ZYdZYac

# 가장 자주 등장하고 있는 byte pair = "ZY"
# "ZY"를 하나의 byte로 치환
X=ZY
Y=ab
Z=aa
XdXac

# 더 이상 합칠 수 있는 byte pair 없음.
# 알고리즘 종료
```

<br>
======
<br>

## <i>Byte Pair Encoding in NLP</i>

<img width=120% height=120%>![Byte Pair Encoding의 과정](https://wikidocs.net/images/page/22592/%EA%B7%B8%EB%A6%BC.png)</img>

- character 단위에 점차적으로 vocabulary를 만들어 내는 bottom-up 방식의 접근
    1. training data에 있는 word
    2. character 혹은 UniCode 단위로 vocabulary 생성
    3. 가장 많이 등장하는 Unigram을 하나의 Unigram으로 통합

<br>

### **기존의 접근**
```python
# dictionary
# training data에 있는 word와 frequency
low: 5, lower: 2, newest: 6, widest: 3

# vocabulary: 중복 제거
low, lower, newest, widest

# Out-of-Vocabulary
lowest
```

<br>

### **BPE 알고리즘을 사용한 경우**
![Byte Pair Encoding](https://blog.kakaocdn.net/dn/cam9jp/btqQt4yJsLT/NUK4W9OJ2zY4j4GVSaUxv0/img.png)
1. word를 character로 분리
    - 이제부터 dictionary는 vocabulary 업데이트에도 쓰이고, 자기 자신도 업데이트됨

```python
# character 단위로 split
l o w : 5
l o w e r : 2
n e w e s t : 6
w i d e s t : 3

# 형성되는 vocabulary = 글자 단위로 분리됨. 중복 없음.
l, o, w, e, r, n, w, s, t, i, d
```

2. 알고리즘을 몇 번 반복할지 사용자가 결정.
    - most frequent unigram pair를 하나의 unigram으로 몇 번 합칠까?
    - 10번 합쳐보자!

```python
# step 1. Freq((e, s)) = 9
# (e, s) -> es

# dictionary update!
l o w : 5,
l o w e r : 2,
n e w es t : 6,
w i d es t : 3
```

```python
# step 2. Freq((es, t)) = 9
# (es, t) -> est

# dictionary update!
l o w : 5,
l o w e r : 2,
n e w est : 6,
w i d est : 3
```

```python
# step 3. Freq((l, o)) = 7
# (l, o) -> lo

# dictionary update!
lo w : 5,
lo w e r : 2,
n e w est : 6,
w i d est : 3
```
………

```python
# step 10.

# dictionary update!
low : 5,
low e r : 2,
newest : 6,
widest : 3

# vocabulary update!
l, o, w, e, r, n, w, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest
```
```python
# Not Anymore OOV!
lowest = low + est
```

<br>

### **코드 실습하기**
[Colab NoteBook](https://colab.research.google.com/drive/18hbax12YHXtZN4e-YZbge8rTqabbDpkz#scrollTo=U11TkEMH4ULT)

<br>

### **OOV에 대처하기**
[Colab NoteBook](https://colab.research.google.com/drive/18hbax12YHXtZN4e-YZbge8rTqabbDpkz#scrollTo=jx7Qs90v4ZRm)

<br><br><br>

## <i>Wordpiece Model</i>
- [구글 번역기](https://arxiv.org/pdf/1609.08144.pdf), [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- WPM: BPE의 변형 알고리즘
- completely data-driven and guaranteed to generate a deterministic segmentation for any possible sequence of characters.

<br>

- > **<i>Using wordpieces gives a good balance between the flexibility of single characters and the efficiency of full words for decoding, and also sidesteps the need for special treatment of unknown words.</i>**

<br>
 
|WPM|BPE|
|:---:|:---:|
|merge 되었을 때 corpus의 likelihodd를 가장 높이는 pair를 merge|frequency가 가장 높은 pair를 merge|

- **(before)** Jet makers feud over seat width with big orders at stake
- **(after)** _J et _makers _fe ud _over _seat _width _with _big _orders _at _stake

<br>

- 모든 단어의 맨 앞에 _(문장 복원을 위한 장치)를 붙이고, subword로 구분

<br><br><br>

## <i>Unigram Language Model Tokenizer</i>
1. 각각의 subword들에 대해서 loss를 계산
    - subword의 loss: 해당 subword가 vocabulary에서 제거되었을 경우, corpus의 likelihood가 감소하는 정도
2. 최악의 영향을 주는 10~20%의 토큰을 제거
3. 원하는 vocabulary의 크기에 도달할 때까지 반복

<br><br><br><br>
======
<br>

# **<i>SentencePiece</i>**
- BPE를 포함하여 기타 서브워드 토크나이징 알고리즘을 내장함

## <i>What is SentencePiece?</i>
- 구글이 공개한 센텐스피스 [깃허브](https://github.com/google/sentencepiece)
- BPE, Unigram Language Model Tokenizer
- 사전 토큰화 작업 없이, 전처리를 하지 않은 raw data에 바로 적용
    - 언어에 종속되지 않는다.

<br><br><br>

## <i>Tokenizing IMDB Review</i>
[Colab NoteBook](https://colab.research.google.com/drive/15AdwY_9YIwVRVQvR_BQcV1kYcpwDw-kI#scrollTo=1MG46R5WN1I4)

<br><br><br>

## <i>Tokenizing Naver Movie Review</i>
[Colab NoteBook](https://colab.research.google.com/drive/15AdwY_9YIwVRVQvR_BQcV1kYcpwDw-kI#scrollTo=7rWRjHodQhI2)

<br><br><br><br>
======
<br>

# **<i>SubwordTextEncoder</i>**
- 텐서플로에서 제공하는 subword tokenizer

<br><br><br>

## <i>Tokenizing IMDB Review</i>
[Colab NoteBook](https://colab.research.google.com/drive/1NoTly8RP6f_55_zUYSa2CE2hn_YPn6Q0#scrollTo=OvM_81ngSUyv)

<br><br><br>

## <i>Tokenizing Naver Movie Review</i>
[Colab NoteBook]()