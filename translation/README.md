# Translation

## Architectures

### RNN Encoder-Decoder / GRU (3 Jun 2014; EMNLP)

[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio

- **Summary:** 
  - Introduces: 
    - 1) RNN encoder-decoder architectures for rescoring hypotheses produced by a phrase-based system
    - 2) A recurrent unit similar to a LSTM but with fewer parameters, that are easier and faster to train than their LSTM counterparts
  - GRU + GRU  (context vector is concatenated to the decoder's inputs and outputs )
  - **Blog (2):** [Link](https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb)

### Vanilla Seq2Seq (10 Sep 2014; NIPS)

[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), Ilya Sutskever, Oriol Vinyals, Quoc V. Le

- **Summary:** LSTM + LSTM    (4 layers both)
- **Blog (1):** [Link](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)

### Bahdanau Attention (1 Sep 2014)

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio

- **Summary:** Bidirectional LSTM + attention
- **Blog (3):** [Link](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

```
Concept: Softmax(Q*K)*V
Keys=Values=Encoder; Queries=Decoder
-------------------------------------  Variation from the paper
Attention = Softmax(Wc*tanh( energy ))*V
energy = Wa*K + Wb*Q
```

### Luong Attention (17 Aug 2015)

[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025), Minh-Thang Luong, Hieu Pham, Christopher D. Manning

- **Summary:** Multiple attention mechanisms

``````
Concept: Softmax(Q*K)*V
Keys=Values=Encoder; Queries=Decoder
-------------------------------------  Variation from the paper
Attention = Softmax(Wc*tanh( energy ))*V
content-based function (enery):
  - General: Wa*K + Wb*Q
  - Dot: K*Q
  - Concat: Wa*[K, Q]
``````

### GNMT (26 Sep 2016)

[Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144), Google

- **Summary:** LSTM + LSTM (8 layers, with residual connections and attention)

### Convolution for Seq2Seq (8 May 2017)

 [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122), Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin

- **Summary:** Architecture based entirely on CNNs. (Fully parallelized)
- **Blog (5):** [Link](https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb)

### Transformer (12 Jun 2017)

[Attention is All You Need](https://arxiv.org/abs/1706.03762), Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

- **Summary:** Architecture based solely on attention mechanisms
- **Blog (6):** [Link](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)

````
Concept: Softmax(Q*K)*V
Keys=Values=Encoder; Queries=Decoder
-------------------------------------
Attention = Softmax(energy + Mask)*V
energy = Q*K
````

## Multilingual

### Survey of Multilingual NMT (4 Jan 2020; ACL 2020)

[A Comprehensive Survey of Multilingual Neural Machine Translation](https://arxiv.org/abs/2001.01115), Raj Dabre, Chenhui Chu, Anoop Kunchukuttan

### M2M-100 (21 Oct 2020; Arxiv 2020)

[Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125), Facebook

## Systems

### Iterative training for low-resource machine translation  (15 Oct 2019)

[Facebook AI's WAT19 Myanmar-English Translation Task Submission](https://arxiv.org/abs/1910.06848), Facebook

- **Summary:** 
  1. Train backtranslation system using parallel data
  2. Translate monolingual corpuses using this trained system
  3. Retrain the backtranslation system
  4. Repeat 3-4 times

- **Blog**: [Recent advances in low-resource machine translation](https://ai.facebook.com/blog/recent-advances-in-low-resource-machine-translation/)

### Transformer + BERT (17 Feb 2020)

[Incorporating BERT into Neural Machine Translation](https://arxiv.org/abs/2002.06823), Jinhua Zhu, Yingce Xia, Lijun Wu, Di He, Tao Qin, Wengang Zhou, Houqiang Li, Tie-Yan Liu

## Tools

### VizSeq (12 Sep 2019)

[VizSeq: A Visual Analysis Toolkit for Text Generation Tasks](https://arxiv.org/abs/1909.05424), Changhan Wang, Anirudh Jain, Danlu Chen, Jiatao Gu

- **Summary: ** A Visual Analysis Toolkit for Text Generation (Translation, Captioning, Summarization, etc.)
- **Link:** [https://facebookresearch.github.io/vizseq/](https://facebookresearch.github.io/vizseq/)

