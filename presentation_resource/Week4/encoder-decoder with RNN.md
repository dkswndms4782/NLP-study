# **<i> RNN을 이용한 인코더-디코더</i>**



## 1)시퀀스-투-시퀀스(Sequence-to-Sequence, seq2seq)

1) seq2seq

2) seq2seq 예제 - 글자 레벨(character-level) 기계 번역기 만들기


## 2)Word-Level 번역기 만들기(Neural Machine Translation (seq2seq) Tutorial)


## 3) BLEU Score(Bilingual Evaluation Understudy Score)

>  언어 모델의 성능 측정을 위한 평가 방법
### **1. PPL(펄플렉서티)**

### **2. BLEU(Bilingual Evaluation Understudy)**
BLEU는 기계 번역 결과와 사람이 직접 번역한 결과가 얼마나 유사한지 비교하여 번역에 대한 성능을 측정하는 방법이다.
이때 측정 기준은 n-gram에 기반한다.

BLEU는 완벽한 방법이라고는 할 수는 없지만 다음과 같은 이점을 가진다.

- 언어에 구애받지 않고 사용할 수 있다.
- 계산 속도가 빠르다
-  BLEU는 PPL과는 달리 높을 수록 성능이 더 좋음을 의미한다.
#### 2-1) 단어 개수 카운트로 측정하기(Unigram Precision)

 번역된 문장을 각각 Candidate1, 2라고 할 때, 이 문장의 성능을 평가하기 위해서는 정답으로 비교되는 문장이 있어야 한다. 
 세 명의 사람에게 한국어를 보고 영작해보라고 하여 세 개의 번역 문장을 만들어, 이 세 문장을 각각 Reference1, 2, 3라고 해봅시다.

> Example 1
- Candidate1 : It is a guide to action which ensures that the military always obeys the commands of the party.

- Candidate2 : It is to insure the troops forever hearing the activity guidebook that party direct.

- Reference1 : It is a guide to action that ensures that the military will forever heed Party commands.

- Reference2 : It is the guiding principle which guarantees the military forces always being under the command of the Party.

- Reference3 : It is the practical guide for the army always to heed the directions of the party.
 
 
 #### 유니그램 정밀도(Unigram Precision)
 Ref 1, 2, 3 중 어느 한 문장이라도 등장한 단어의 개수를 Ca에서 count한 후, Ca의 모든 단어의 카운트의 합. 즉, Ca에서의 총 단어의 수으로 나눠준 값을 구한다.
 
 ![image](https://user-images.githubusercontent.com/61388801/131663383-46969ebe-7e8a-4065-811f-d869bce4f042.png)

![image](https://user-images.githubusercontent.com/61388801/131663441-fc88d2cf-3d60-4529-99d4-431a04cb8cbd.png)

 ### 2-2) 중복을 제거하여 보정하기(Modified Unigram Precision)
 
> Example 2
- Candidate : the the the the the the the

- Reference1 : the cat is on the mat

- Reference2 : there is a cat on the mat

위의 Ca는 the만 7개가 등장한 터무니 없는 번역이다. 하지만 이 번역은 앞서 배운 유니그램 정밀도에 따르면 7/7 = 1이라는 최고의 성능 평가를 받게 된다.

 이에 유니그램 정밀도를 다소 보정할 필요가 있다. 이를 보정하기 위해서는 정밀도의 분자를 계산하기 위해 Ref와 매칭하며 카운트하는 과정에서 Ca의 유니그램이 이미 Ref에서 매칭된 적이 있었는지를 고려해야 한다.
 
#### 보정된 유니그램 정밀도(Modified Unigram Precision)

우선, 유니그램이 하나의 Ref에서 최대 몇 번 등장했는지를 카운트합니다. 

이 값을 maximum reference count를 줄인 의미에서 Max_Ref_Count라고 부른다.

Max_Ref_Count가 기존의 단순 카운트한 값보다 작은 경우에는 이 값을 최종 카운트 값으로 대체한다.  정밀도의 분자 계산을 위한 새로운 카운트 방식을 식으로 표현하면 다음과 같다.

![image](https://user-images.githubusercontent.com/61388801/131664010-66545de9-eb06-4e67-a8aa-020fab55f836.png)

위의 카운트를 사용하여 분자를 계산한 정밀도를 보정된 유니그램 정밀도(Modified Unigram Precision)라고 한다.

 
![image](https://user-images.githubusercontent.com/61388801/131663913-b384de3e-5b91-40e8-9e4c-eb3d52e29b96.png)


분모의 경우에는 이전과 동일하게 Ca의 모든 유니그램에 대해서 각각 Count하고 모두 합한 값을 사용한다.



 ### 2-3) 보정된 유니그램 정밀도 (Modified Unigram Precision) 구현하기
 
 
