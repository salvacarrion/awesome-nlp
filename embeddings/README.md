# Embeddings

## Word Embedding

### Word2Vec (ICLR 2013)

[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean

**Summary:** Two architectures for computing word embeddings: 

1. Continuous Bag-of-Words (CBOW): Predict centered word given a context.

   1.1 Better syntactic accuracy

   1.2 one-hot inputs (averaged)

   1.3 Arch: Inputs=>Hidden=>Output + Hierarchical Softmax

   1.4 Extracting word embeddings: 1) Use W1, 2) Use W2, or 3) W3=a*W1+b*W2

 2. Skip-gram: Predict context given a word.

    2.1 Better semantic accuracy

**Core idea:** Learning distributed representations of words that try to minimize computational complexity (with neural networks).

**Key achievement:** Learning high-quality word vectors from huge data sets with billions of words, and with millions of words in the vocabulary.

> The model is trickier to implement than it looks due to the huffman tree binary and the hierarchical softmax. For instance, one word can be encoded as "01" and another as "1101001".
>
> To learn more: [word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738), Xin Rong
> 

```
Concept:
W = Vocab
w = word

# Slow version
P(W) = Softmax(input * w1 * w2)

# Hierarchical softmax
P(w) = PROD_j[ Sigmoid((input * w1) * w2[:, j]) = ((1, vocab)x(vocab, dim))x(dim, 1)=(1, dim)x(dim, 1)=(1,1)]. // j=[1..L(w)] => L(w=00101)=5
```

### Word2Vec v2 (NIPS 2013)

[Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)

**Summary:** Extensions that improve the quality of the vectors and the traning speed-up w.r.t to the previous Word2Vec paper.

**Core idea:** 

- Subsampling of frequent words to obtain a significant speedup and more regular word representations. (*Remove % of frequent pairs*)
- The hierarchical softmax can be replaced with Negative Sampling (derived from the Noise Contrastive Estimation). (*Select the positive target and k negative targets randomly, based on their unigram probabilities*)
- Learning phrases: Using a simple data-driven approach to find phrases such as "New York Times" and expand the vocabulary considering them as a single token.

**Key achievement:** Incremental improvements over Word2vec (v1)

### GloVe (EMNLP 2014)

[GloVe: Global Vectors for Word Representation](https://www-nlp.stanford.edu/pubs/glove.pdf), Jeffrey Pennington, Richard Socher, Christopher D. Manning

**Summary:** Analysis of the origin of semantic and syntactic regularities in vector space representation of words. Proposal of a new global log-bilinear regression model.

**Core idea:** Word vector learning should be with ratios of co-occurrence probabilities rather than the probabilities themselves

**Key achievement:** A new global log-bilinear regression model for the unsupervised
learning of word representations that outperforms other models on word analogy, word similarity, and named entity recognition tasks.

> It is a good exercise to derive the equations from scratch.

### FastText (Arxiv 2016)

[Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf), Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov

- **Summary:** Model for text classification and representation learning. Extension of Word2Vec to learn form char n-grams (instead of complete words).
- **Core idea:** Use a bag of n-grams as additional features to capture some partial information about the local word order.
- **Key achievement:** Efficient method for text classification
- **Implementation details:**
  - Hierarchical Softmax, DFS to discart small probabilities, top-k with binary heap, efficient mapping of n-grams with the hashing trick

## Sentence Embedding

### CNN char-embeddings + LSTM (2016)

[Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf), Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, Yonghui Wu

- **Summary:** Extend current models to deal with two key challenges present in this task: corpora and vocabulary sizes, and complex, long term structure of language
- **Core idea:** For large scale LM, use char CNN inputs, LSTM with large hidden states, importance sampling as "large scale softmax" and ensemble models
- **Key achievement:**  Training RNN LMs on large amounts of data

> Importance sampling reduces the computational complexity of the softmax by estimating the normalizing factor. 
>
> When using importance sampling with the CNN softmax, a small correction factor must be added to calculate the logit, otherwise the model cannot differentiate between words with similar spelling but with very different meaning. [See review](http://www.stochasticgrad.com/article/exploring_limits_language_modeling)

### ELMo (NAACL 2018)

[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf), Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer

- **Summary:** Introduces a novel way to use language models for word representation in-context.
- **Core idea:** Different layers learn different abstraction of the text. Hence, the word embedding is the learned weighted sum of all the hidden layers
- **Key achievement:** Learning high-quality deep context-dependent representations

> *Note: The model -per se- was based on prior work: Exploring the Limits of Language Modeling*

### ULMfit (ACL 2018)

[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146), Jeremy Howard, Sebastian Ruder

- **Summary:** d
- **Core idea:** d
- **Key achievement:** d

### BERT (NAACL 2019)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

- **Summary:** State-of-the-art, contextual and bidirectional LM suitable to be fine-tuned on broad range of NLP task
- **Core idea:** 
  - Bidirectionality: Mask some words in the input and predict those masked words
  - Relationships: Given two sentences A and B, predict if B comes after A or if it is just a random one.
- **Key achievement:** State-of-the-art language model

### InferSent

[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364), Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, Antoine Bordes

### Sentence-BERT

[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084), Nils Reimers and Iryna Gurevych

- **Summary:** Sentence embeddings using siamese BERT-like networks
- **Core idea:** Create a simese network using two BERT-base models
- **Key achievement:** Allow ys to compute sentence similarty efficiently using BERT

### Positional embeddings (ICLR 2020)

[Encoding Word Order in Complex Embeddings](https://arxiv.org/pdf/1912.12333.pdf), Benyou Wang, Donghao Zhao, Christina Lioma, Qiuchi Li, Peng Zhang, Jakob Grue Simonsen

### Others

### Embedding models comparisson (TACL 2015)

[Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016), Omer Levy, Yoav Goldberg, Ido Dagan

- **Summary:** Shows that when all word embedding methods are allowed to tune a similar set of hyperparameters, their performance is largely comparable
- **Core idea:** There is no consistent advantage to one algorithmic approach over another
- **Key achievement:** Show insignificant performance differences between the methods, with no global advantage to any single approach over the others.

### Evaluation of word embedding (EMNLP 2015)

[Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036), Tobias Schnabel, Igor Labutov, David Mimno, Thorsten Joachims

- **Summary:** Present new evaluation techniques that directly compare embeddings with respect to specific queries
- **Core idea:** Different tasks favor different embeddings
- **Key achievement:**  Evaluation framework based on direct comparisons between embeddings

### Word Embedding dimensionality (NIPS 2018)

[On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf), Zi Yin, Yuanyuan Shen

- **Summary:** Theoretical understanding of word embedding and its dimensionality
- **Core idea:** There is a fundamental bias-variance trade-off in dimensionality selection for word embeddings (existence of an optimal dimensionality)
- **Key achievement:**  Pairwise Inner Product (PIP) loss, a metric of dissimilarity between word embeddings

## Multilingual sentence embeddings

### mBERT (2018)

Pretrained (BERT) model on the top 104 languages with the largest Wikipedia using a masked language modeling (MLM) objective.

- **BERT paper:** [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- **Multilingual BERT:** [repo](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages)

### LASER (2018)

[Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464), Mikel Artetxe, Holger Schwenk

- **Summary:** Autoencoder architecture using BiLSTMs and a shared BPE vocabulary for 93 languages
- **Core idea:** Language-agnostic representation
- **Key achievement:**  

### XML (NIPS 2019)

[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291), Guillaume Lample, Alexis Conneau

- **Summary:** Learn cross-lingual language models from: 1) an unsupervised method, and 2) supervised method.
- **Core idea:** Translation language modeling (TLM) objective (Mask words in both the source and target sentences, but to predict a word masked in the source sentence, the model can either attend to its surrounding or to the target sentence)
- **Key achievement:**  Demonstrating the effectiveness of cross-lingual language model pretraining on multiple cross-lingual understanding 

### XML-R (NIPS 2019)

[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1901.07291), Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov

- **Summary: **
- **Core idea:**  
- **Key achievement:**  

### m~USE (2019)

[Multilingual Universal Sentence Encoder for Semantic Retrieval](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html)

### LaBSE (2020)

[Language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852), Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, Wei Wang

- **Summary:** Adaptation of multilingual BERT to produce language-agnostic sentence embeddings 
- **Core idea:** Combine masked language model (MLM) and translation language model (TLM) pretraining with a translation ranking task using bi-directional dual encoders
- **Key achievement:**  New state-of-the-art results several bitext retrieval tasks

### Multilingual knowledge distillation

[Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813), Nils Reimers, Iryna Gurevych

- **Summary:** Minimize the MSE loss between the teacher and the student model for the source sentence, and the target sentence: `MSE[MT(si)-MS(sj)]+MSE[(MT(sj)-MS(tj))]`
- **Core idea:** Multilingual knowledge distillation following a teacher-student approach
- **Key achievement:**  