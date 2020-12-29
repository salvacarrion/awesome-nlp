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

## WMT20

[Findings of the 2020 Conference on Machine Translation (WMT20)](https://www.aclweb.org/anthology/2020.wmt-1.1.pdf)

> Classical WTM task
> **This year:** News translation task and the similar language translation task
>
> 38 participans (153 submissions)

- **Summary:** 
  - Many system surpassing human translations
  - Many teams use FairSeq
  - Similar language task(top):
    - INFOSYS: Ensemble of FairSeq models + back-translation.
    - IITDELHI: 1) fine-tuning over pretrained mBART; 2) Transformer architecture with 12 encoder and de- coder layers. (+lots of data)
    - WIPRO-RIT: Single multilingual NMT system based on the transformer architecture
  - News tasks(top):
    - Deepmind: 
      - Document-level translation system built upon noisy channel factorization
      - Lots of data, back-translation, distillation, fine-tuning, domain adaptation, MonteCarlo Tree Search decoding and uncertainty estimation, specialized length models and sentence segmentation
    - Facebook AI: 
      - Low-resource: 1) Exploit all available data. 2) Adapt domain
      - Non-contrained: Iterative backtranslation and self-training and strong language models for noisy channel reranking. (Data from CommonCrawl)
    - WeChat:
      - Transformer with effective variants and the DTMT architecture
      - Data selection, several synthetic data generation approaches (back-translation, knowledge distillation, and iterative in-domain knowledge transfer), advanced fine-tuning approaches and self-bleu based model ensemble.
    - VolcTrans:
      - Very deep Transformer or dynamic convolution models up to 50 encoder layers
      - Strong focus on diversification of the (synthetic) training data
      - Iterative back-translation, knowledge distillation, model ensemble and development set fine-tuning, multiple scalings, dynamic convolution, random upsamplings, back-translated corpus variants, random ensembling which uses not a fixed set of ensembled models but rather a random checkpoint of each of them.
    - Oppo:
      - Corpus filtering, iterative forward and backward translation, fine-tuning on the original parallel data, ensembling of several different models, and complex re-ranking which uses forward scorers, backward scorers) and language models

**Details:**

- Evaluation: SacreBLEU
  - Automatic: BLEU, RIBES, and TER.
  - Manual: Direct assesment by humans, Mechanical Turk and a pool of linguists
- Domain: 
  - Training:
    - Parallel: >500M pairs (Europarl, NewsCommentary, CommonCrawl, ParaCrawl, EU Press, Yandex 1M, CzEng, WikiTitles, CCMT, UN, Extra Tamil-English, Extra Japanese-English, Nunavut Hansard, Opus Corpus, Synthetic, Wikimatrix)
    - Monoloingual: >10,000M senteces (Monolingual Wikipedia, News, Document-Split, Common Crawl)
  - Test domain: Online news websites

[Findings of the First Shared Task on Lifelong Learning Machine Translation](https://www.aclweb.org/anthology/2020.wmt-1.2.pdf)

> Lifelong learning can be defined as the ability to continually acquire new and retain previous knowledge. It allows MT systems to adapt to new vocabularies and topics, and produce accurate translations across time.
>
> => 0 participants (excluding organizers)?

- **Summary:** 
  - Seems that the only participants are the ones who created this task
  - Your system has to be integrated into the BEAT platform and require you to rethink the code so that everything is done in memory
  - **Personal opinion:** Hard to understand, evaluate, too many open challenges, and of out focus (Domain adaptation, Instance-based adaptation, Unsupervised learning, Active learning, Interactive learning)

[Findings of the WMT 2020 Shared Task on Chat Translation](https://www.aclweb.org/anthology/2020.wmt-1.3.pdf)

> This tasks deals with the fact that the conversations are bilingual, less planned, more informal, and often ungrammatical.
>
> => 6 participating teams (14 submissions)

- **Summary:** 
  - Relevance of context-aware systems, document-level models, copy placeholder for rare characters, synthetic noise generation, ensembles, BERT, XML, transfomer-big,...

[Findings of the WMT 2020 Shared Task on Machine Translation Robustness](https://www.aclweb.org/anthology/2020.wmt-1.4.pdf)

> Test current machine translation systems in their ability to handle challenges facing MT models to be deployed in the real world, including domain diversity and non-standard texts common in user generated content, especially in social media. (**This year: Evaluate a general MT system’s in Zero-shot and Few-Shot scenarios**)
>
> =>  11 participating teams (59 submissions)

- **Summary:** 
  - Very few teams introduced specific techniques for robustness, such as augmenting training data with synthetic noise.
  - Few-shot submissions managed to outperform online systems in most test sets
  - Introduced a flag for catastrophic errors (novel way to evaluate translations)
  - Human-translated references contain catastrophic errors as well

[Findings of the WMT 2020 Shared Task on Automatic Post-Editing](https://www.aclweb.org/anthology/2020.wmt-1.75.pdf)

> Automatically correcting the output of a “black-box” machine translation system by learning from existing human corrections of different sentences
>
> => 6 teams (11 runs) for English-German task.  2 teams (2 runs each) for the English-Chinese task 

- **Summary:** 
  - On English-German, the top-ranked system improves over the baseline by -11.35 TER and +16.68 BLEU points
  - On English-Chinese the improvements are respectively up to -12.13 TER and +14.57 BLEU points
  - Winning system (HW-TSC): Multi-source (*src*, *mt*, *auxiliary mt*) + Transformer (+bottleneck adapter layers)

[Findings of the WMT 2020 Biomedical Translation Shared Task: Basque, Italian and Russian as New Additional Languages](https://www.aclweb.org/anthology/2020.wmt-1.76.pdf)

> Access to accurate biomedical information is specifically critical and machine translation (MT) can contribute to making health information available to health professionals and the general public in their own language.
>
> => 20 teams

- **Summary:** 
  - Many teams use Fairseq, trained their models, half fine-tune them, half used backtranslation and a few used language models
  - Overall translation quality is high, with many perfect translation for high-resource languages
  - Some translation require deep background knowledge, abbreviations and acronims are difficult, some medical terms were literally translated, choosing an English synonym of a translated German word altered the original German meaning entirely, problems from unknown vocabularies...

[Results of the WMT20 Metrics Shared Task](https://www.aclweb.org/anthology/2020.wmt-1.77.pdf)

> Automatic metrics that score MT output by comparing them with a reference translation generated by human translators
>
> =>  10 research groups (27 metrics; 4 of which are reference-less “metrics”)

- **Summary:** 
  - Metrics based on word or sentence-level embeddings, achieve the highest performance
  - The performance of the referencefree metrics has improved, and the correlations this year are competitive with the reference-based metrics, and in many cases, outperform BLEU.
  - COMET-QE is good at recognising the high quality of human translations where BLEU falls short.

[Findings of the WMT 2020 Shared Task on Parallel Corpus Filtering and Alignment](https://www.aclweb.org/anthology/2020.wmt-1.78.pdf)

> Improving the quality of parallel corpora by filtering noisy training data.
>
> => 10 participants

- **Summary:** (Incremental improvements) Improvements over the LASER baseline with a variety results measured by translation quality, optimal subset sizes, sentence length,...
  - Techniques based on: XLM-RoBERTa, Yisi-2 scores, GTP-2, LASER-based...

[Findings of the WMT 2020 Shared Task on Quality Estimation](https://www.aclweb.org/anthology/2020.wmt-1.79.pdf)

> Automatic methods for estimating the quality of neural machine translation (MT) output at run-time, without the use of reference translations
>
> => 19 teams (1374 systems)

- **Summary:** (Incremental improvements) Participating systems (WMT) show overall higher correlation with DA labels.

[Findings of the WMT 2020 Shared Tasks in Unsupervised MT and Very Low Resource Supervised MT](https://www.aclweb.org/anthology/2020.wmt-1.80.pdf)

> There is little to no parallel corpora for most of the 7000 languages spoken. This tasks deals with this problem.
>
> => 6 systems for the unsupervised shared task.  10 systems for the very low resource supervised task.

- **Summary:**
  - Transfer learning was a critical in all tasks (Unsupervised, Unsupervised with Multilingual Transfer, and Very Low Resource)
  - For Unsupervised, transfer learning using monolingual corpora was needed
  - For Unsupervised with Multilingual Transfer, the use of English-German bilingual corpora to initialize the Unsupervised system seems to have been highly effective
  - For Very Low Resource, heavy usage of both types of transfer were made as well, with one particular focus being on efforts to leverage the similarity of Czech and Upper Sorbian using Czech/German parallel corpora.
  - Word segmentation and the use of morphological information was useful
  - There was a significant effort focused on trying to make Czech more like Upper Sorbian using a variety of tech- niques, and/or sampling German data like Upper Sorbian data.

