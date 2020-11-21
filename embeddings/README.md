# Embeddings

## Word Embedding

### Word2Vec

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

> The model is trickier to implement than it looks due to the huffman tree binary and the hierarchical softmax. For instance, one word can be encoded as "01" and other as "1101001".
>
> To learn more: [word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738), Xin Rong

<img src="https://render.githubusercontent.com/render/math?math=p(w = w_O) = \prod_{j=1}^{L(w)-1} \sigma( \textbf{[[n(w,j+1) = ch(n(w,j)) ]]} \cdot {v'_{n(w,j)}}^Th)">

```
Concept:
# h = input * w1 //  => (1, dim)

# Slow version
P(:) = Softmax(input * w1 * w2)

# Hierarchical softmax
P(w) = PROD_j[ Sigmoid((input * w1) * w2[:, j]) = ((1, vocab)x(vocab, dim))x(dim, 1)=(1, dim)x(dim, 1)=(1,1)]. // j=[1..L(w)] => L(00101)=5
```

### Word2Vec (follow-up)

[Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)

**Summary:** Extensions that improve the quality of the vectors and the traning speed-up w.r.t to the previous Word2Vec paper.

**Core idea:** 

- Subsampling of frequent words to obtain a significant speedup and more regular word representations. 
- The hierarchical softmax can be replaced with Negative Sampling (derived from the Noise Contrastive Estimation)
- Learning phrases: Using a simple data-driven approach to find phrases such as "New York Times" and expand the vocabulary considering them as a single token.

**Key achievement:** Incremental improvements over Word2vec (v1)

### GloVe

[GloVe: Global Vectors for Word Representation](https://www-nlp.stanford.edu/pubs/glove.pdf), Jeffrey Pennington, Richard Socher, Christopher D. Manning

### FastText

[Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf), Armand Joulin, Edouard Grave, Piotr Bojanowski, Tomas Mikolov

## Sentence Embedding

### ELMo

[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf), Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer

### InferSent

[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364), Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, Antoine Bordes

### Sentence-BERT

[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf), Nils Reimers and Iryna Gurevych

### Positional embeddings (ICLR 2020)

[Encoding Word Order in Complex Embeddings](https://arxiv.org/pdf/1912.12333.pdf), Benyou Wang, Donghao Zhao, Christina Lioma, Qiuchi Li, Peng Zhang, Jakob Grue Simonsen