---
title: Embedding과 Recurrent
category:
  - AI
tags:
 - [Aiffel, DL]
Date: 2023-06-30
Last_modified_at: 2023-07-04
---





# Embedding과 Recurrent

## 분포 가설과 분산 표현

​	**희소 표현(Sparse Representation)**이란, 벡터의 특정 차원에 단어 혹은 의미를 직접 매핑하는 방식을 말한다.



### 단어의 분산 표현(Distributed Representation)

- 분포 가설(distribution hypothesis): 유사한 맥락에서 나타나는 단어는 그 의미도 비슷하다고 가정하는 것
- 분산 표현(Distributed Representation)
  - 유사한 맥락에 나타난 단어끼리의 벡터를 가까이 하고 그렇지 않은 단어는 멀어지게 조정하여 얻어지는 단어 벡터
  - 희소 표현과는 다르게 단어 간의 유사도를 계산으로 구할 수 있다



## Embedding 레이어

- Weight는 (단어의 개수, Embedding 사이즈)로 정의된다.

- Lookup Table = Embedding Table

- Embedding Layer는 미분이 불가능하므로 신경망 설계를 할 때 **입력에 직접 연결되게** 사용해야 한다.

  ![img](https://d3s0tskafalll9.cloudfront.net/media/images/F-24-12.max-800x600.png)

### One Hot Encoding

- N개의 단어를 N차원의 벡터로 표현하는 방식
- 단어가 포함되는 자리에는 1을 넣고 나머지에는 0을 넣는다

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FzGpUq%2FbtrsXgtJf93%2Fk6o6JCgr7cihZSPuCJIVF1%2Fimg.png)

- **치명적 단점:** 단어의 의미 또는 개념 차이를 담을 수 없다. 예를 들면 '과학'과 '공학'의 관계는 '과학'과 '수박'의 관계와 같다.
- 원 - 핫 벡터들은 1인 딱 하나의 요소를 제외하면 모두 0인 희소 벡터 형태를 띤다. 이런 경우 두 단어 벡터의 내적은 0으로 직교를 이룬다. 이는 단어들이 서로 **독립적**으로 존재한다는 것을 의미한다.



## Recurrent 레이어 - RNN

- 순차적(Sequential) 특성: 문장, 영상, 음성 데이터의 특성

- RNN(Recurrent Neural Network): 순차 데이터를 처리하기 위한 모델

- (입력 차원, 출력 차원)에 해당하는 단 하나의 Weight를 순차적으로 업데이트하기 때문에 다른 레이어에 비해 느리다.

- 입력의 앞부분이 뒤로 갈수록 옅어져 손실이 발생한다.(기울기 소실, Vanishing Gradient)

- [Tensorflow 공식 가이드(RNN)](https://www.tensorflow.org/guide/keras/rnn?hl=ko)

  ```python
  sentence = 'What time is it?'
  dic = {
    'is':0,
    'it':1,
    'What':3,
    'time': 3,
    '?':4
  }
  
  print('입력할 문장: ', sentence)
  sentence_tensor = tf.constant([dic[word] for word in sentence.split()])
  
  print('Embedding word mapping: ', sentence_tensor.numpy())
  print('data shape: ', sentence_tensor.shape)
  
  embedding_layer = tf.keras.layers.Embedding(input_dim=len(dic), output_dim=100)
  emb_out = embedding_layer(sentence_tensor)
  
  print('\nEmbedding result: ', emb_out.shape)
  print('Embedding Layer Weight shape: ', embedding_layer.weights[0].shape)
  
  rnn_seq_layer = \
  tf.keras.layers.SimpleRNN(units=64, return_sequence=True, use_bias=False)
  rnn_seq_out = rnn_seq_layer(emb_out)
  
  print('\nRNN result: ', rnn_seq_out.shape)
  print('RNN Layer Weight shape: ', rnn_seq_layer.weights[0].shape)
  
  rnn_fin_layer = tf.keras.layers.SimpleRNN(units=64, use_bias=False)
  rnn_fin_out = rnn_fin_layer(emb_out)
  
  print('\nRNN result(finish): ', rnn_fin_out.shape)
  print('RNN Layer Weight shape: ', rnn_fin_layer.weights[0].shape)
  ```

  