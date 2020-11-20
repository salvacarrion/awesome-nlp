# Metrics

### CheckList (ACL 2020)

[Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://arxiv.org/abs/2005.04118), Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh

- **Summary:** Methodology to comprehensively test NLP models
- **Core idea:** Accuracy on benchmarks is not sufficient for evaluating NLP models
- **Key achievement:** Tool to indentify bugs and critical failures in NLP models

### Adapt LM to you task (ACL 2020)

[Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964), Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith

- **Summary:** Study to know whether state-of-the-art language models (trained on massive datasets) work universally or still need to be fine-tuned on the final task/domain.
- **Core idea:** Pretraining in-domain leads to performance gains
- **Key achievement:** Demonstrating the importance of domain-specific and task-specific pretraining

### Better than BLEU (ACL 2020)

[Tangled up in BLEU: Reevaluating the Evaluation of Automatic Machine Translation Evaluation Metrics](https://arxiv.org/abs/2006.06264), Nitika Mathur, Timothy Baldwin, Trevor Cohn

- **Summary:** Current methods for judging metrics are highly sensitive to the translations used for assessment. Based on Pearson’s correlation coefficient, automatic metrics poorly match human evaluations of translation quality when comparing only a few best systems.
- **Core idea:** Retiring BLEU as the de facto standard metric since small differences have little meaning, and other metrics are prefered such as CHRF, YISI-1, or ESIM.
- **Key achievement:** Thorough analysis of automatic metrics vs. human judgments in machine translation, and providing key recommendations on evaluating MT systems

### BLEU clarity (ACL 2018)

[A Call for Clarity in Reporting BLEU Scores](https://arxiv.org/abs/1804.08771), Matt Post

- **Premise:** Inconsitent BLEU scores due to the variability in parametrization, often not reported

- **Core idea:** Settle on the BLEU scheme used by the WMT. 
- **Key achievement:** Development of a new tool, *SACREBLEU*

### Beam Search width (EMNLP 2018)

[Breaking the Beam Search Curse: A Study of (Re-)Scoring Methods and Stopping Criteria for Neural Machine Translation](https://arxiv.org/abs/1808.09582), Yilin Yang, Liang Huang, Mingbo Ma

- **Summary:** Explanation of the beam search curse, proposal of a new rescoring method to address this problem and a comparisson study between existing alternatives

- **Core idea:** Importance of rescoring methods and stopping criteria when selecting  candidate solutions.
- **Key achievement:** Proposal of new rescoring method and optimal stopping criteria 

### Text Degeneration (ICLR 2020)

[The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf), Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi

- **Summary:** Analysis of text generation strategies and their problems, plus a new decoding strategy improving existing alternatives.

- **Core idea:** Avoiding text degeneration by sampling over a truncated probability distribution
- **Key achievement:** Proposal of a new decoding strategy (Nucleus Sampling)



