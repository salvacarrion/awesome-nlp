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

- **Summary: ** Model for text classification and representation learning. Extension of Word2Vec to learn form char n-grams (instead of complete words).
- **Core idea:** Use a bag of n-grams as additional features to capture some partial information about the local word order.
- **Key achievement:** Efficient method for text classification
- **Implementation details:**
  - Hierarchical Softmax, DFS to discart small probabilities, top-k with binary heap, efficient mapping of n-grams with the hashing trick

## Sentence Embedding

### ELMo (NAACL 2018)

[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf), Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer

- **Summary: ** d
- **Core idea:** d
- **Key achievement:** d

### ULMfit (ACL 2018)

[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146), Jeremy Howard, Sebastian Ruder

- **Summary: ** d
- **Core idea:** d
- **Key achievement:** d

### BERT (NAACL 2019)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

- **Summary: ** d
- **Core idea:** d
- **Key achievement:** d

### InferSent

[Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364), Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, Antoine Bordes

### Sentence-BERT

[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf), Nils Reimers and Iryna Gurevych

### Positional embeddings (ICLR 2020)

[Encoding Word Order in Complex Embeddings](https://arxiv.org/pdf/1912.12333.pdf), Benyou Wang, Donghao Zhao, Christina Lioma, Qiuchi Li, Peng Zhang, Jakob Grue Simonsen

### Others

### Embedding models comparisson (TACL 2015)

[Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016), Omer Levy, Yoav Goldberg, Ido Dagan

- **Summary: ** Shows that when all word embedding methods are allowed to tune a similar set of hyperparameters, their performance is largely comparable
- **Core idea:** There is no consistent advantage to one algorithmic approach over another
- **Key achievement:** Show insignificant performance differences between the methods, with no global advantage to any single approach over the others.

### Evaluation of word embedding (EMNLP 2015)

[Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036), Tobias Schnabel, Igor Labutov, David Mimno, Thorsten Joachims

- **Summary: ** Present new evaluation techniques that directly compare embeddings with respect to specific queries
- **Core idea:** Different tasks favor different embeddings
- **Key achievement:**  Evaluation framework based on direct comparisons between embeddings

### Word Embedding dimensionality (NIPS 2018)

[On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf), Zi Yin, Yuanyuan Shen

- **Summary: ** Theoretical understanding of word embedding and its dimensionality
- **Core idea:** There is a fundamental bias-variance trade-off in dimensionality selection for word embeddings (existence of an optimal dimensionality)
- **Key achievement:**  Pairwise Inner Product (PIP) loss, a metric of dissimilarity between word embeddings

