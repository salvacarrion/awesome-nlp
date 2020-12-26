# Architectures

Types:

- Seq2Seq: Encoder+Decoder
- Autoregressive: Decoder
- Autoencoding: Encoder
- Multimodal
- Retrival-based

## Seq2Seq: Enc+Dec

### MarianMT (1 Apr 2018)

[Marian: Fast Neural Machine Translation in C++](https://arxiv.org/abs/1804.00344), Marcin Junczys-Dowmunt et al.

### T5 (23 Oct 2019)

[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683), Colin Raffel et al.

- **Summary:** Transformer-based architecture for text-to-text. This text-to-text approach allow us to perform translation, question answering and classifitcation out of the box.

### BART (29 Oct 2019)

[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461), Mike Lewis et al.

### PEGASUS (18 Dec 2019)

[PEGASUS: Pre-training with Extracted Gap-sentences forAbstractive Summarization](https://arxiv.org/abs/1912.08777), Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu on Dec 18, 2019.

- **Summary:** Transformer for abstractive text summarization

### MBart (22 Jan 2020)

[Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.

- **Summary:** Multilingual BART

### ProphetNet

[ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, Ming Zhou.

## Autoregressive models: Dec

### GPT (11 Jun 2018)

[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), Alec Radford et al.

### Transformer-XL (9 Jan 2019)

[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860), Zihang Dai et al.

- **Summary:** Allow us to model very long-term dependencies by reusesing hidden states from previous segments.

### GPT-2 (14 Feb 2019)

[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Alec Radford et al.

### XLNet (19 Jun 2019)

[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237), Zhilin Yang et al.

- **Summary:** BERT but without the token size limitation

### CTRL (11 Sep 2019)

[CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858), Nitish Shirish Keskar et al.

- **Summary:** Transformer trained to condition on control codes that govern style, content, and task-specific behavior.

  Example:

  ```
  *Horror* A knife handle pulled through the open hole in the front.
  *Reviews* A knife is a tool and this one does the job well.
  
  *Relationships* My neighbor is a jerk and I don’t know what to do
  *Legal* My neighbor is threatening to sue me for not letting him use my pool
  ```

### Reformer (13 Jan 2020)

[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451), Nikita Kitaev et al .

- **Summary:** Efficient transformer architecture which replaces dot-product attention with locality-sensitive hashing and does a trick for the backward pass to reduce the memory consumption (reversible residual layers)

### Linformer (8 Jun 2020)

[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768v3), Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma

- **Summary:** Transformer with a linear self-attention mechanism to tackle the self-attention bottleneck

### Big Bird (28 Jul 2020)

[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062), Google

- **Summary:** Transformer that runs on a sparse attention mechanism that allows it to overcome the quadratic dependency of BERT while preserving the properties of full-attention models.

## Autoencoding: Enc

### BERT

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Jacob Devlin et al.

- **Summary:** Encoder part of the transformer with MLM trained in a self-supervised way.

### ALBERT

[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), Zhenzhong Lan et al.

- **Summary:** BERT-like network to lower the memory footprint and increase the training speed

### RoBERTa

[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692), Yinhan Liu et al.

- **Summary:** BERT optimized

### DistilBERT

[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108), Victor Sanh et al.

- **Summary:** BERT distilled using a teacher-student approach. Same performance, half the size.

### XLM

[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291), Guillaume Lample and Alexis Conneau

- **Summary:** Cross-lingual embeddings using: Shared BPE vocabularies, CLM, MLM and TLM

### XLM-RoBERTa

[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116), Alexis Conneau et al.

- **Summary:** XML with RoBERTa, trained with more data (100 language)

### FlauBERT

[FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372), Hang Le et al.

### ELECTRA

[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555), Kevin Clark et al.

- **Summary:** Efficient alternative to MLM. Instead of masking the input, they replace some tokens plausible alternatives sampled from a small generator network. Then, they train a discriminator.

### Funnel Transformer

[Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://arxiv.org/abs/2006.03236), Zihang Dai et al.

### Longformer

[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150), Iz Beltagy et al.

- **Summary:** Transformer with an attention mechanism that scales linearly with sequence length

### GPT-3

[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165v2), by OpenAI

## Multimodal

### MMBT

[Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/abs/1909.02950), Douwe Kiela et al.

## Retrieval-based models

### DPR

[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), Vladimir Karpukhin et al.

### RAG

[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela