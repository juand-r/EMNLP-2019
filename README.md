# EMNLP 2019

- Schedule: https://www.emnlp-ijcnlp2019.org/program/ischedule/
- Papers: https://www.aclweb.org/anthology/events/emnlp-2019

Papers from EMNLP 2019 that look interesting:

- [Tutorials](#tutorials)
- [Workshops](#workshops)
- [Tools, demos, datasets](#tools-demos-datasets)
- [Papers](#papers)
  - [Language learning agents](#language-learning-agents)
  - [Discourse](#discourse)
  - [Text generation and GPT2](#text-generation-and-gpt2)
  - [Explainability](#explainability)
  - [Learning with Less Data](#learning-with-less-data)
  - [Word Embeddings](#word-embeddings)
  - [Knowledge enhanced ML](#knowledge-enhanced-ml)
  - [Relation extraction and Knowledge graphs](#relation-extraction-and-knowledge-graphs)
  - [Information Retrieval](#information-retrieval)
  - [Entity Linking](#entity-linking)
  - [Entities and NER](#entities-and-ner)
  - [Multimodal](#multimodal)
  - [Machine Translation](#machine-translation)
  - [Other](#other)
  
  
# Tutorials

All EMNLP 2019 tutorials: https://www.emnlp-ijcnlp2019.org/program/tutorials/

- Graph-based Deep Learning in Natural Language
Processing (https://github.com/svjan5/GNNs-for-NLP) NOTE: slides available in "Releases".
- Semantic Specialization of Distributional Word Vectors
- Discreteness  in  Neural Natural Language Processing. Slides:
  - https://lili-mou.github.io/resource/emnlp19-1.pdf
  - https://lili-mou.github.io/resource/emnlp19-2.pdf  
  - https://lili-mou.github.io/resource/emnlp19-3.pdf

# Workshops

- [CoNLL 2019](http://www.conll.org/)
- [W-NUT 2019](http://noisy-text.github.io/)
- [LOUHI 2019](http://louhi2019.fbk.eu/)
- [FEVER 2](http://fever.ai/)
- [DiscoMT19](https://www.idiap.ch/workshop/DiscoMT)
- [DeepLo 2019](https://sites.google.com/view/deeplo19/)
- [COIN (Commonsense inference in NLP)](http://www.coli.uni-saarland.de/~mroth/COIN/)
- [WNGT: 3rd workshop on neural generation and translation](https://sites.google.com/view/wngt19/home)
- [NewSum (summarization)](https://summarization2019.github.io/)
- [TextGraphs-19](https://sites.google.com/view/textgraphs2019)
- [NLP4IF: Censorship, Disinformation and Propaganda](http://www.netcopia.net/nlp4if/)

# Tools, demos, datasets

[Multilingual, Multi-scale and Multi-layer Visualization of Sequence-based Intermediate Representations](https://www.aclweb.org/anthology/D19-3026.pdf)
> Code: https://github.com/elorala/interlingua-visualization ; Demo: https://upc-nmt-vis.herokuapp.com/

[VerbAtlas a Novel Large-Scale Verbal Semantic Resource and ItsApplication to Semantic Role Labeling](https://www.aclweb.org/anthology/D19-1058.pdf)

[SEAGLE: A Platform for Comparative Evaluation of Semantic Encoders for Information Retrieval](https://www.aclweb.org/anthology/D19-3034.pdf)

[Joey NMT: A Minimalist **NMT** Toolkit for Novices](https://www.aclweb.org/anthology/D19-3019.pdf)

[OpenNRE: An Open and Extensible Toolkit for **Neural Relation Extraction**](https://www.aclweb.org/anthology/D19-3029.pdf)


## Annotation tools

- [Redcoat: A Collaborative Annotation Tool for **Hierarchical Entity Typing**](https://www.aclweb.org/anthology/D19-3033.pdf)
- [HARE: a Flexible Highlighting Annotator for Ranking and Exploration](https://www.aclweb.org/anthology/D19-3015.pdf)

# Papers: Tuesday Nov 5

## Language learning agents
 
- [Seeded self-play for language learning (LANTERN)](https://www.aclweb.org/anthology/D19-6409.pdf)
- [Learning to request guidance in emergent language (LANTERN)](https://www.aclweb.org/anthology/D19-6407.pdf)
- [EGG: a toolkit for research on Emergence of lanGuage in Games](https://www.aclweb.org/anthology/D19-3010.pdf)
- [Recommendation as a Communication Game: Self-Supervised Bot-Play for Goal-oriented Dialogue](https://www.aclweb.org/anthology/D19-1203.pdf)


## Discourse

[Evaluation Benchmarks and Learning Criteriafor Discourse-Aware Sentence Representations](https://www.aclweb.org/anthology/D19-1060.pdf)


## Text generation and GPT2

[(Male, Bachelor) and (Female, Ph.D) have different connotations:Parallelly Annotated Stylistic Language Dataset with Multiple Personas](https://www.aclweb.org/anthology/D19-1179.pdf)
> PASTEL dataset with 41K parallel sentences annotated across different personas (age, gender, political orientation, education level, ethnicity, country, time-of-day). Data and code: https://github.com/dykang/PASTEL

:boom: [Controlling Text Complexity in Neural Machine Translation](https://www.aclweb.org/anthology/D19-1166.pdf)

:boom: [See et al., Do Massively Pretrained Language Models Make Better Storytellers? (CoNLL)](https://www.aclweb.org/anthology/K19-1079.pdf)

:boom: [Attending to Future Tokens for Bidirectional Sequence Generation](https://www.aclweb.org/anthology/D19-1001.pdf)
> Can generate text from BERT!

[How Contextual are Contextualized Word Representations? Comparing **the Geometry of BERT, ELMo, and GPT-2 Embeddings**](https://www.aclweb.org/anthology/D19-1006.pdf)

[Big Bidirectional Insertion Representations for Documents](https://www.aclweb.org/anthology/D19-5620.pdf)
> Insertion Transformer: well suited for long form text generation; this builds on it to produce BIRD for use in machine translation task.

[Neural Text Summarization: A Critical Evaluation](https://www.aclweb.org/anthology/D19-1051.pdf)

[Neural data-to-text generation: A comparison between pipeline and end-to-end architectures](https://www.aclweb.org/anthology/D19-1052.pdf)

:boom: [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://www.aclweb.org/anthology/D19-1053.pdf)

:boom: [Select and Attend: Towards Controllable Content Selection in Text Generation](https://www.aclweb.org/anthology/D19-1054.pdf)

[Sentence-Level Content Planning and Style Specification for Neural Text Generation](https://www.aclweb.org/anthology/D19-1055.pdf)

:boom: [Neural Naturalist: Generating Fine-Grained Image Comparisons](https://www.aclweb.org/anthology/D19-1065.pdf)

## Explainability

[Knowledge Aware Conversation Generation with Explainable Reasoningover Augmented Graphs](https://www.aclweb.org/anthology/D19-1187.pdf)

:boom: Attention is not not **Explanation**
> Put the description here.

Many Faces of **Feature Importance**: Comparing Built-in and Post-hoc Feature Importance in Text Classification

TellMeWhy: **Learning to Explain** Corrective Feedback for Second Language Learners. Yi-Huei Lai and Jason Chang

Analytical Methods for Interpretable Ultradense Word Embeddings

## Learning with Less Data

Investigating Meta-Learning Algorithms for Low-Resource Natural Language Understanding Tasks

A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-resource Named Entity Recognizers

Learning to Bootstrap for Entity Set Expansion

Hierarchical Attention Prototypical Networks for **Few-Shot Text Classification**

[Look-up and Adapt: A One-shot Semantic Parser](https://www.aclweb.org/anthology/D19-1104.pdf)

Latent-Variable Generative Models for Data-Efficient **Text Classification**

Practical Obstacles to Deploying **Active Learning**

**Transfer Learning** Between Related Tasks Using Expected Label Proportions

To Annotate or Not? **Predicting Performance Drop under Domain Shift**

## Word Embeddings

[Correlations between Word Vector Sets](https://www.aclweb.org/anthology/D19-1008.pdf)

[Game Theory Meets Embeddings (paper on WSD)](https://www.aclweb.org/anthology/D19-1009.pdf)

[Parameter-free Sentence Embedding via Orthogonal Basis](https://www.aclweb.org/anthology/D19-1059.pdf)

## Knowledge enhanced ML

**Knowledge-Enriched** Transformer for Emotion Detection in Textual Conversations

**Knowledge Enhanced** Contextual Word Representations

Improving Relation Extraction with **Knowledge-attention**

## Relation extraction and Knowledge graphs

[Commonsense Knowledge Mining from Pretrained Models](https://www.aclweb.org/anthology/D19-1109.pdf)

[KnowledgeNet: A Benchmark Dataset for Knowledge Base Population](https://www.aclweb.org/anthology/D19-1069/)

CaRe: Open Knowledge Graph Embeddings

Supervising Unsupervised Open Information Extraction Models

Open Relation Extraction: Relational Knowledge Transfer from Supervised Data to Unsupervised Data

**Automatic Taxonomy Induction and Expansion**

Multi-Input Multi-Output Sequence Labeling for **Joint Extraction of Fact and Condition Tuples** from Scientific Text

Tackling Long-Tailed Relations and Uncommon Entities in Knowledge Graph Completion

## Information Retrieval 

Applying BERT to Document Retrieval with Birch

## Entity Linking

Fine-Grained Evaluation for Entity Linking

## Entities and NER

EntEval: A Holistic Evaluation Benchmark for **Entity Representations**

A Boundary-aware Neural Model for **Nested Named Entity Recognition**


## Multimodal


## Machine Translation

[Unsupervised Domain Adaptation for Neural Machine Translation withDomain-Aware Feature Embeddings](https://www.aclweb.org/anthology/D19-1147.pdf)

[Machine Translation for Machines:the Sentiment Classification Use Case](https://www.aclweb.org/anthology/D19-1140.pdf)

[Iterative Dual Domain Adaptation for NMT](https://www.aclweb.org/anthology/D19-1078.pdf)

[Self-Attention with Structural Position Representations](https://www.aclweb.org/anthology/D19-1145.pdf)

[Encoders Help You Disambiguate Word Sensesin Neural Machine Translation](https://www.aclweb.org/anthology/D19-1149.pdf)

[Simple, Scalable Adaptation for Neural Machine Translation](https://www.aclweb.org/anthology/D19-1165.pdf)

[Hierarchical Modeling of Global Context for Document-LevelNeural Machine Translation](https://www.aclweb.org/anthology/D19-1168.pdf)

## Other

[Partners in Crime: Multi-view Sequential Inference for MovieUnderstanding](https://www.aclweb.org/anthology/D19-1212.pdf)

[Latent Suicide Risk Detection on Microblogvia Suicide-Oriented Word Embeddings and Layered Attention](https://www.aclweb.org/anthology/D19-1181.pdf)

[Movie Plot Analysis via Turning Point Identification](https://www.aclweb.org/anthology/D19-1180.pdf)

[Learning Invariant Representations of Social Media Users](https://www.aclweb.org/anthology/D19-1178.pdf)

[The Trumpiest Trump? Identifying a Subject’s Most Characteristic Tweets](https://www.aclweb.org/anthology/D19-1175.pdf)

# Papers: Wednesday, Nov 6
