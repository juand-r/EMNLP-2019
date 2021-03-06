# EMNLP 2019

This contains a selection of papers from EMNLP 2019. The ":boom:" symbol indicates papers that I particularly liked or that I found interesting for one reason or another.

Note: there are papers on neural-rule/logic/template hybrids spread throughout.

Currently missing: many papers on cross-lingual NLP, multi-modal NLP, code-switching, dialogue, machine translation, question answering, question generation, intent classification, stance detection, sentiment/emotion, argument mining, SQL and code generation, rumors, humor, bias, hate speech...

**Full list of papers:**

- Schedule: https://www.emnlp-ijcnlp2019.org/program/ischedule/
- ACL Anthology: https://www.aclweb.org/anthology/events/emnlp-2019
- Computer-selected one-sentence "highlights" of all accepted papers: https://www.paperdigest.org/2019/11/emnlp-2019-highlights/

**Other summaries**:

- Highlights from Naver labs: https://europe.naverlabs.com/blog/emnlp2019-what-we-saw-and-liked
- Highlights - focused on Knowledge Graphs: https://medium.com/@mgalkin/knowledge-graphs-nlp-emnlp-2019-part-i-e4e69fd7957c and https://medium.com/@mgalkin/knowledge-graphs-nlp-emnlp-2019-part-ii-56f5b03ad9ba 
- See also: Notes on ACL 2018: https://shuaitang.github.io/files/acl_2018_notes.html#pf3

# CONTENTS

- [Tutorials](#tutorials)
- [Workshops](#workshops)
- [Tools, demos, datasets](#tools-demos-datasets)
- [Papers](#papers)
  - [Topics](#topics)
    - [Methodological](#methodological)
    - [Analysis](#analysis)
    - [Language learning agents](#language-learning-agents)
    - [Discourse](#discourse)
    - [Explainability](#explainability)
    - [Adversarial ML](#adversarial-ml)
    - [Learning with Less Data](#learning-with-less-data)
      - [Out-of-Domain Detection](#out-of-domain-detection)
      - [Data Augmentation](#data-augmentation)
      - [Domain Adaptation](#domain-adaptation)
      - [Meta-learning](#meta-learning)
      - [Few-shot and zero-shot text classification](#few-shot-and-zero-shot-text-classification)
      - [Active Learning](#active-learning)
    - [Word Embeddings](#word-embeddings)
    - [Sentence Embeddings](#sentence-embeddings)
    - [Knowledge enhanced ML](#knowledge-enhanced-ml)
    - [Multimodal ML](#multimodal-ml)
  - [NLG Tasks](#nlg-tasks)
    - [Summarization and Simplification](#summarization-and-simplification)
    - [Style transfer](#style-transfer)
    - [Text generation and GPT2](#text-generation-and-gpt2)
    - [Machine Translation](#machine-translation)
  - [NLU Tasks](#nlu-tasks)
    - [Word Sense Disambiguation (WSD)](#word-sense-disambiguation-wsd)
    - [Keyphrase extraction](#keyphrase-extraction)
    - [Fact checking](#fact-checking)
    - [Relation extraction and Knowledge graphs](#relation-extraction-and-knowledge-graphs)
    - [Commonsense Reasoning](#commonsense-reasoning)
    - [Information Retrieval](#information-retrieval)
    - [Entity Linking](#entity-linking)
    - [Entities and NER](#entities-and-ner)
    - [Coreference](#coreference)
    - [Text classification](#text-classification)
    - [Propaganda, Persuasion](#propaganda-persuasion)
    - [Social Media and Authorship Attribution](#social-media-and-authorship-attribution)
    - [Other](#other)
  
  
# [Tutorials](#contents)

All EMNLP 2019 tutorials: https://www.emnlp-ijcnlp2019.org/program/tutorials/

A few that look particularly interesting:

- :memo: Graph-based Deep Learning in Natural Language Processing (https://github.com/svjan5/GNNs-for-NLP) NOTE: slides available in "Releases".
- :memo: Semantic Specialization of Distributional Word Vectors (pdf of slides in this repo).
- :memo: Discreteness  in  Neural Natural Language Processing. Slides:
  - https://lili-mou.github.io/resource/emnlp19-1.pdf
  - https://lili-mou.github.io/resource/emnlp19-2.pdf  
  - https://lili-mou.github.io/resource/emnlp19-3.pdf

# [Workshops](#contents)

All workshops: https://www.emnlp-ijcnlp2019.org/program/workshops/

A selection:

- [**CoNLL 2019**](http://www.conll.org/)
- [W-NUT 2019](http://noisy-text.github.io/) (Noisy User-generated Text)
- [LOUHI 2019](http://louhi2019.fbk.eu/) (Health Text Mining and Information Analysis)
- [FEVER 2](http://fever.ai/) (Fact Extraction and VERification)
- [DiscoMT19](https://www.idiap.ch/workshop/DiscoMT) (Discourse in Machine Translation)
- [DeepLo 2019](https://sites.google.com/view/deeplo19/) (Deep Learning for Low-Resource NLP)
- [COIN (Commonsense inference in NLP)](http://www.coli.uni-saarland.de/~mroth/COIN/)
- [WNGT: 3rd workshop on neural generation and translation](https://sites.google.com/view/wngt19/home)
- [NewSum (summarization)](https://summarization2019.github.io/)
- [TextGraphs-19](https://sites.google.com/view/textgraphs2019)
- [NLP4IF: Censorship, Disinformation and Propaganda](http://www.netcopia.net/nlp4if/)

# [Tools, demos, datasets](#contents)

### Annotation tools

:heavy_minus_sign: [MedCATTrainer: A Biomedical Free Text Annotation Interface with **Active Learning** and Research Use Case Specific Customisation](https://www.aclweb.org/anthology/D19-3024.pdf)
> A flexible interactive web interface for NER and linking using active learning. Code: https://github.com/CogStack/MedCATtrainer  Demo: https://www.youtube.com/watch?v=lM914DQjvSo

:heavy_minus_sign: [Redcoat: A **Collaborative** Annotation Tool for **Hierarchical Entity Typing**](https://www.aclweb.org/anthology/D19-3033.pdf)
> Code: https://github.com/Michael-Stewart-Webdev/redcoat ; Demo: https://www.youtube.com/watch?v=igtR8Sfi8oo&feature=youtu.be

:heavy_minus_sign: [HARE: a Flexible Highlighting Annotator for Ranking and Exploration](https://www.aclweb.org/anthology/D19-3015.pdf)
> Code: https://github.com/OSU-slatelab/HARE/

### Development Tools

:boom: [Joey NMT: A Minimalist **NMT** Toolkit for Novices](https://www.aclweb.org/anthology/D19-3019.pdf)
> Code: https://joeynmt.readthedocs.io/en/latest/ Demo: https://www.youtube.com/watch?v=PzWRWSIwSYc&feature=youtu.be

:heavy_minus_sign: [OpenNRE: An Open and Extensible Toolkit for **Neural Relation Extraction**](https://www.aclweb.org/anthology/D19-3029.pdf)
> Code: https://github.com/thunlp/OpenNRE Demo: http://opennre.thunlp.ai

:heavy_minus_sign: [NeuronBlocks: Building Your NLP DNN Models Like Playing Lego](https://www.aclweb.org/anthology/D19-3028.pdf)
> Code: https://github.com/Microsoft/NeuronBlocks Demo: https://youtu.be/x6cOpVSZcdo

:heavy_minus_sign: [UER: An Open-Source Toolkit for **Pre-training Models**](https://www.aclweb.org/anthology/D19-3041.pdf)
> Code: https://github.com/dbiir/UER-py **NOTE:** it appears the available models and datasets are only for Chinese.

:heavy_minus_sign: [CFO: A Framework for Building Production NLP Systems](https://www.aclweb.org/anthology/D19-3006.pdf)
> Code: https://github.com/IBM/flow-compiler Demo: https://www.youtube.com/watch?v=6LOJm5ZcGCs&feature=youtu.be

### Visualization and Evaluation Tools

:heavy_minus_sign: [AllenNLP **Interpret**: A Framework for **Explaining Predictions** of NLP Models](https://www.aclweb.org/anthology/D19-3002.pdf)
> Demos, code and tutorials: https://allennlp.org/interpret

:heavy_minus_sign: [Multilingual, Multi-scale and Multi-layer Visualization of Sequence-based Intermediate Representations](https://www.aclweb.org/anthology/D19-3026.pdf)
> Code: https://github.com/elorala/interlingua-visualization ; Demo: https://upc-nmt-vis.herokuapp.com/

:heavy_minus_sign: [VizSeq: a visual analysis toolkit for **text generation tasks**](https://www.aclweb.org/anthology/D19-3043.pdf)
> Code: https://github.com/facebookresearch/vizseq

:heavy_minus_sign: [SEAGLE: A Platform for **Comparative Evaluation** of Semantic Encoders for **Information Retrieval**](https://www.aclweb.org/anthology/D19-3034.pdf)
> Code: https://github.com/MarkusDietsche/seagle (missing as of January 2020). Demo: https://www.youtube.com/watch?v=8ncTgRqr8w4&feature=youtu.be

### Datasets

:heavy_minus_sign: [(Male, Bachelor) and (Female, Ph.D) have different connotations: Parallelly Annotated Stylistic Language Dataset with Multiple Personas](https://www.aclweb.org/anthology/D19-1179.pdf)
> PASTEL dataset with 41K parallel sentences annotated across different personas (age, gender, political orientation, education level, ethnicity, country, time-of-day). Data and code: https://github.com/dykang/PASTEL
>
> Note: this could be useful for semi-supervised style transfer; e.g., see "Semi-supervised Text Style Transfer: Cross Projection in Latent Space" in the [Style Transfer](#style-transfer) section.

:heavy_minus_sign: [VerbAtlas: a Novel Large-Scale **Verbal Semantic Resource** and Its Application to Semantic Role Labeling](https://www.aclweb.org/anthology/D19-1058.pdf)
> Introduces a hand-crafted lexical resource collecting all verb synsets of WordNet into "semantically-coherent frames". Available at http://verbatlas.org

:heavy_minus_sign: [Automatic **Argument Quality Assessment** - New Datasets and Methods](https://www.aclweb.org/anthology/D19-1564.pdf)
> 6.3k arguments, annotated for "quality". Data available at https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Argument%20Quality Two methods are introduced for the tasks of argument pair classification and argument ranking, both based on BERT.

:heavy_minus_sign: [The Role of Pragmatic and Discourse Context in Determining **Argument Impact**](https://www.aclweb.org/anthology/D19-1568.pdf)
> Introduces a new dataset to study the effect of "pragmatic and discourse context" (*kairos*) on determining argument impact (persuasive power). The dataset contains arguments covering 741 topics and over 47k claims. Models that incorporate the pragmatic and discourse context (i.e., sequences of  of claims which support or oppose other claims in order to defend a certain thesis) outperform models which only rely on claim-specific features.

# [Topics](#contents)

## [Methodological](#contents)

**See also:**
- [Gorman, Kyle, and Steven Bedrick. We need to talk about standard splits. ACL 2019](https://www.aclweb.org/anthology/P19-1267.pdf)
- [Dror, Shlomov and Reichart. Deep Dominance - How to Properly Compare Deep Neural Models. ACL 2019](https://www.aclweb.org/anthology/P19-1266/)
- [Replicability Analysis for Natural Language Processing: Testing Significance with Multiple Datasets, TACL 2017](https://www.aclweb.org/anthology/Q17-1033.pdf)
- [The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing, ACL 2018]()

:boom: [Show Your Work: Improved Reporting of Experimental Results](https://www.aclweb.org/anthology/D19-1224.pdf)
> Proposes an evaluation framework and suggests best practices to fairly report experimental results (this is now requested for submissions to EMNLP 2020; a "reproducibility checklist" will be used by reviewers).  Test set performance is not enough to fairly compare models. It is important (especially in the age of huge language models and heavy computation) to *report expected validation performance as a function of computation budget* (training time or number of hyperparameter trials). This is crucial not only for fair comparison of models but also for reproducibility. Results from some case studies on **text classification** (Stanford Sentiment Treebank):
> - confirm previous results that Logistic Regression does better using less than 10 hyperparameter search trials; for over 10 trials CNNs do better. Also LR has lower variance over all budgets.
> - investigate whether the conclusion of Peters et al. (2019) "To tune or not to tune", that it is better to use fixed ELMo embeddings in a task-specific network than to fine-tune ELMo embeddings during training, depends on the computational budget.  For under 2 hours of training time, GloVE is best. For budgets up to 1 day, fixing the ELMo embeddings is best. For larger budgets, fine-tuning outperforms feature extraction.

:heavy_minus_sign: [Polly Want a Cracker: Analyzing Performance of Parroting on **Paraphrase Generation** Datasets](https://www.aclweb.org/anthology/D19-1611.pdf)
> Notes an issue with standard metrics (BLEU, METEOR, TER) for paraphrase generation: parroting input sentences surpasses SOTA models. Datasets used: MSCOCO (from image captions), Twitter, Quora.  They do not explore whether this is due to SOTA models (mostly) parroting, or due to BLUE and METEOR being inappropriate for evaluating paraphrase generation. Alternative way to evaluate: effectiveness of paraphrasing on data augmentation (Iyyer et al., NAACL 2018)

:heavy_minus_sign: [Are We Modeling the Task or the Annotator? An Investigation of Annotator Bias in Natural Language Understanding Datasets](https://www.aclweb.org/anthology/D19-1107.pdf)
> Do models pick up on annotator artifacts? Yes. On 3 NLP datasets (MNLI, OpenBookQA and CommonsenseQA):
> - using annotator identifiers as features improves performance
> - models often do not generalize well to examples from different annotators (not in training).
**Suggestions:** annotator bias should be monitored during dataset creation, and the "test set annotators should be disjoint from the training set annotators."

## [Analysis](#contents)

:boom: [Revealing the Dark Secrets of BERT](https://www.aclweb.org/anthology/D19-1445.pdf)
> A qualitative and quantitative analysis of the information encoded by the the BERT heads. **Conclusion:** the model is overparametrized (there is a limited set of attention patterns repeated across different heads). In addition, manually disabling attention in some heads leads to a performance increase. This paper also explores how the self-attention weights change when fine-tuning BERT. **Related work**:
> - Paul Michel, Omer Levy, and Graham Neubig. 2019. "Are  sixteen  heads  really  better  than  one?" (some layers can be reduced to a single head without degrading performance much).
> - Frankle and Carbin. 2018. "The lottery ticket hypothesis: Finding sparse, trainable neural networks"
> - Adhikari et al., 2019.  "Rethinking  complex  neural  network  architectures  for  document  classification" (a properly tuned BiLSTM without attention is better or competitive to more complicated architectures on document classification).

:boom: [The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives](https://www.aclweb.org/anthology/D19-1448.pdf)
> How does learning objective affect information flow in a Transformer-based neural network? This paper uses canonical correlation analysis (CCA) and mutual information estimators to study how information flows across Transformer layers, and how it depends on the choice of learning objective. This process is compared across three tasks - three objective functions (machine translation, left-to-right language models and masked language models) while keeping the data and the model architecture fixed.
>
>**Blog post:** https://lena-voita.github.io/posts/emnlp19_evolution.html
>
>**Podcast:** [NLP Highlights 98 - Analyzing Information Flow In Transformers, With Elena Voita](https://soundcloud.com/nlp-highlights/98-analyzing-information-flow-in-transformers-with-elena-voita)

:heavy_minus_sign: [Quantity doesn't buy quality **syntax** with **neural language models**](https://www.aclweb.org/anthology/D19-1592.pdf)
> Can increasing the size of the neural network or the amount of training data improve LMs' syntactic ability? Tests on RNNs, GPT and BERT shows diminishing returns in both network capacity and corpus size, and the training corpus would need to be unrealistically large in order to reach human-level performance. **Conclusion:** "reliable and data-efficient learning of syntax is likely to require external supervision signals or a stronger inductive bias than that provided by RNNs and Transformers".
>
> Details: GPT, BERT Base, and 125 two-layer LSTM language models were trained (using tied input and output embeddings; see Press and Wolf, "Using the Output Embedding to Improve Language Model" and Inan, Khosravi and Socher, ICLR 2017), i.e., on 5 corpus sizes (2M, 10M, 20M, 40M and 80M words), 5 layer sizes (100, 200, 400, 800 or 1600 units in each hidden layer) and 5 corpus subsets of the WikiText-103 corpus. Models were tested on constructions from Marvin and Linzen (2018) (e.g., "the author has/\*have books", "the author that likes the guards has/\*have books", "the authors laugh and have/\*has books", "The manager that the architects like doubted himself/\*themselves").

:heavy_minus_sign: [What Part of the Neural Network Does This? **Understanding LSTMs** by Measuring and Dissecting Neurons](https://www.aclweb.org/anthology/D19-1591.pdf)
> How do individual neurons of an LSTM contribute to the final label prediction? ("neuron" = component of the hidden layer vector of the LSTM). This is studied using the NER task (using an LSTM without the CRF layer, built using NCRF++). **Contributions:** (1) A simple metric is introduced and tested to quantify the "sensitivity" of neurons to each label, and (2) it is found that "each individual neuron is specialized to carry information for a subset of labels, and the information of each label is only carried by a subset of all neurons". 
> **Related work:**
> - Karpathy et al., Visualizing and understanding recurrent networks, arxiv (2015): find individual neurons in LSTM language models that encode specific information such as sequence length.
> - Radford et al., Learning to generate reviews and discovering sentiment (2017): find that some neurons encode sentiment information.

:heavy_minus_sign: [Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel](https://www.aclweb.org/anthology/D19-1443.pdf)
> Introduces a formulation of the attention mechanism in Transformers using kernels, allowing to unify previous variants of attention and create new ones which are competitive to SOTA models on machine translation and sequence prediction (language modeling).

## [Language learning agents](#contents)

:boom: [EGG: a toolkit for research on Emergence of lanGuage in Games](https://www.aclweb.org/anthology/D19-3010.pdf)
> "We introduce EGG, a  toolkit that greatly simplifies the implementation of emergent-language communication games.   EGG’s  modular  design provides a set of building blocks that the user can combine to create new  games, easily navigating  the optimization and architecture space." **Demo**: https://vimeo.com/345470060 **Code**: https://github.com/facebookresearch/EGG
 
:heavy_minus_sign: [Emergent Linguistic Phenomena in Multi-Agent Communication Games](https://www.aclweb.org/anthology/D19-1384.pdf)
> A simple multi-agent communication framework suggests that some of the properties of language evolution "can  emerge  from  simple  social  exchanges between perceptually-enabled agents playing communication games". Specific properties observed:
> - "a symmetric communication proto-col  emerges  without  any  innate,  explicit  mecha-nism built in an agent"
> - linguistic contact leads to either convergence to a majority protocol or to a "creole", which is of lower complexity than the original languages.
> - "a linguistic continuum emerges whereneighboring languages are more mutually in-telligible than farther removed languages".

:heavy_minus_sign: [Seeded self-play for language learning (LANTERN workshop)](https://www.aclweb.org/anthology/D19-6409.pdf)

:heavy_minus_sign: [Learning to request guidance in emergent language (LANTERN workshop)](https://www.aclweb.org/anthology/D19-6407.pdf)

:heavy_minus_sign: [Recommendation as a Communication Game: Self-Supervised Bot-Play for Goal-oriented Dialogue](https://www.aclweb.org/anthology/D19-1203.pdf)

:heavy_minus_sign: [Interactive Language Learning by Question Answering](https://www.aclweb.org/anthology/D19-1280.pdf)

**See also:** this paper on Theory of Mind: "Revisiting the Evaluation of Theory of Mind through Question Answering, https://www.aclweb.org/anthology/D19-1598.pdf

## [Discourse](#contents)

:boom: [Linguistic Versus Latent Relations for Modeling **Coherent Flow** in Paragraphs](https://www.aclweb.org/anthology/D19-1589.pdf)

:boom: [Next Sentence Prediction helps Implicit Discourse Relation Classification within and across Domains](https://www.aclweb.org/anthology/D19-1586.pdf)
> The task of discourse relation classification: given a pair of sentences, to determine their discourse relation such as "comparison", "contingency", "expansion", "temporal". Examples:
> - Comparison relation (explicit) [It’s a great album.][**But** it's probably not their best.]
> - Comparison relation (implicit) [It’s a great album.][It's probably not their best.] 
> 
> The task is much harder (45-48% SOTA accuracy, before this paper) in the implicit case (without explicit connectives such as but, because, however, for example, previously, moreover, etc.)
>
> BERT outperforms the current SOTA in 11-way classification by 8% on the PDTB dataset; this is probably due to the next sentence prediction (NSP) task that BERT was trained with.  Fine-tuning BERT on PDTB without NSP hurts performance.
>
> In order to test how well this transfers to out-of-domain data, the BERT fine-tuned on PDTB was also evaluated on BioDRB (biomedical text) - and outperforms previous SOTA by about 15%. In addition, further pre-training on in-domain data (GENIA, or using BioBERT) performs best.

:boom: [Pretrained Language Models for Sequential Sentence Classification](https://www.aclweb.org/anthology/D19-1383.pdf)
> Task: classification of sentences in a document (a) according to their rhetorical roles (e.g., Introduction, Method, Result, etc.), or (b) according to  whether or not it should be included in an (extractive) summary of the document.
>
> **Code and data:** https://github.com/allenai/sequential_sentence_classification

:heavy_minus_sign: [Evaluation Benchmarks and Learning Criteria for Discourse-Aware Sentence Representations](https://www.aclweb.org/anthology/D19-1060.pdf)
> Introduces DiscoEval, a test suite of 7 tasks to evaluate whether sentence embedding representations contain information about the sentence's discourse context:
> - PDTB-E: explicit discourse relation classification
> - PDTB-I: implicit discourse relation classification
> - RST-DT: predict labels of nodes in RST discourse trees
> - Sentence Position: predict where an out-of-order sentence should fit amongst 4 other sentences
> - Binary sentence ordering: decide on the correct order for a pair of consecutive sentences.
> - Discourse Coherence: determine whether a sequence of 6 sentences forms a coherent paragraph (some paragraphs are left alone; for others one random sentence is replaced with another from another article).
> - Sentence Section Prediction: predict whether a given sentence belongs to the Abstract.
>
> An encoder-decoder model (one encoder, many decoders) is introduced for these tasks, but is outperformed by BERT.

:heavy_minus_sign: [A Unified Neural Coherence Model](https://www.aclweb.org/anthology/D19-1231.pdf)
> Proposes a model that incorporates grammar, inter-sentence coherence relations and global coherence patterns into a common neural framework, which beats SOTA. Code: https://ntunlpsg.github.io/project/coherence/n-coh-emnlp19/

## [Explainability](#contents)

### [Interpretable Word Embeddings](#contents)

:boom: [Learning Conceptual Spaces with Disentangled Facets (CoNLL)](https://www.aclweb.org/anthology/K19-1013.pdf)
> Can vector space embeddings be decomposed into "facets" (domains) of conceptual spaces (Gardenfors, Conceptual Spaces: The Geometry of Thought) *without supervision*? This paper clusters features (word vectors) found via the method of Derrac and Schockaert  2015, into semantically meaningful facets (sets of features, ie., subspaces).
>
> Conceptual spaces are similar to word embeddings in NLP, except that components of the vectors are interpretable properties of entities of a given kind. A conceptual space for movies could have facets for genre, language, etc., and properties such as "scary". This would be very useful for answering specific queries, e.g., rather than finding movies similar to Poltergeist (in general), one could search for movies which are similar in a particular facet (e.g., cinematography, plot, genre), or which are scarier than Poltergeist. Learning conceptual spaces is similar to "disentangled representation learning" (DRL), except that DRL tries to learn factors which are uncorrelated. However, facets might naturally be correlated (e.g., budget and genre).
>
> There hasn't been much work on learning conceptual spaces from data (though see Schockaert's papers, especially Derrac and Schockaert (2015) and his talk at SemDeep-3 "Knowledge Representation with Conceptual Spaces" (2018)).
>
> An example facet that was found for buildings: "architecture,  art,  history,  literature,  architectural,  society, culture, ancient, scholars, vernacular, classical, historical, contemporary, cultural, medieval", which seems to capture "historical vs contemporary". A facet for organizations: "canadian, australian, australia, africa, nations, african, canada, states, countries, asia, united, british, european, competition, world, europe, asian, britain, country, german".
>
> **See also**: disentangled representation learning (mostly work in computer vision; however see Jain et al., EMNLP 2018 "Learning disentangled representations of texts with application to biomedical abstracts" and Banaee et al., Data-driven conceptual spaces: Creating semantic representations for linguistic descriptions of numerical data. JAIR 2018. Both of these use supervised approaches.)
>
> **Code:** https://github.com/rana-alshaikh/Disentangled-Facets

:heavy_minus_sign: [Analytical Methods for Interpretable Ultradense Word Embeddings](https://www.aclweb.org/anthology/D19-1111.pdf)
> This paper investigates three methods for making word embedding spaces interpretable by rotation: Densifier (Rothe et al., 2016), SVMs, and DensRay (introduced here). DensRay is "hyperparameter-free". The methods are evaluated lexical induction (finding words given properties such as concreteness or sentiment) and on a word analogy task.

:heavy_minus_sign: [Interpretable Word Embeddings via Informative Priors](https://www.aclweb.org/anthology/D19-1661.pdf)

:heavy_minus_sign: [Identifying and Explaining Discriminative Attributes](https://www.aclweb.org/anthology/D19-1440.pdf)


### [Explainability](#contents)

:boom: [**Human-grounded Evaluations** of Explanation Methods for Text Classification](https://www.aclweb.org/anthology/D19-1523.pdf)
> Examines "several model-agnostic and model-specific explanation methods for CNNs for text classification" and evaluates them with 3 "human-grounded evaluations" to answer the following questions:
> - 1. Do the explanations reveal the model behavior?
> - 2. Do they justify the model predictions?
> - 3. Does it help people investigate uncertain predictions?
>
> **See also:**
> recent papers by Hanna Wallach on human evaluations of interpretability methods:
> - "Manipulating and Measuring Model Interpretability" (2019)
> - "A Human-Centered Agenda for Intelligible Machine Learning" (chapter by Vaughan and Wallach)
> - Kaur et al., "Interpreting Interpretability: Understanding Data Scientists' Use of Interpretability Tools for Machine Learning" (also co-authored by Rich Caruana and Hanna Wallach).

:boom: [Attention is not not **Explanation**](https://www.aclweb.org/anthology/D19-1002.pdf)
> Challenges the assumptions underlying the paper "Attention is not Explanation" (Jain and Wallace, 2019). Four alternative tests are proposed to determine when attention can be used as explanation. 

:heavy_minus_sign: [Many Faces of **Feature Importance**: Comparing Built-in and Post-hoc Feature Importance in Text Classification](https://www.aclweb.org/anthology/D19-1046.pdf)
> Compares feature importance from built-in mechanisms (e.g., attention values) and post-hoc methods such as LIME for text classification. Results:
> 1. Regardless of the feature importance method, important features from traditional models (SVM, XGBoost, etc) are more similar to each other than deep learning models.
> 2. Post-hoc methods "tend to generate more similar important features" for two given models than  built-in methods
> 3. Important features do not are not always more similar when models have the same label than when they disagree.

:heavy_minus_sign: [Auditing Deep Learning processes through Kernel-based Explanatory Models](https://www.aclweb.org/anthology/D19-1415.pdf)
> Extends the Layerwise Relevance Propagation (LRP) method (used for explaining image classification in Bach et al., 2015) to the linguistically motivated Kernel-Based Deep Architectures (KDA) (Croce et al., 2017). " The result is a mechanism  that ... generates an **argument-by-analogy explanation** based on real training examples". This is tested on question classification and argument classification (subtask of frame semantic role labelling, to assign labels to sentence fragments for individual roles of a frame).
>
> For example, to explain the classification of "What is the capital of Zimbabwe?" as "LOCATION", the system would return "it recalls me of 'What  is  the  capital  of  California?'  which  also refers  to  a Location."

## [Adversarial ML](#contents)

### Adversarial Training

:heavy_minus_sign: [LexicalAT: Lexical-Based **Adversarial Reinforcement Training** for Robust Sentiment Classification](https://www.aclweb.org/anthology/D19-1554.pdf)

The next two papers are about methods to avoid "being right for the wrong reasons", (i.e., relying on dataset artifacts) in order to improve generalizabilty to out-of-domain data.

:boom: [Topics to Avoid: Demoting Latent Confounds in Text Classification](https://www.aclweb.org/anthology/D19-1425.pdf)
> On the **native language identification** task, classifiers often do not generalize well because they learn topical features which are artifacts (superficial patterns); for example if the label "Swedish" is predicted due to the word 'Sweden' appearing in the text. A model is proposed which predicts both the label of the text and the confound; the two predictors are trained adversarially in order to learn a text representation that predicts the correct label but does not use information from the confound.
>
> Check - how similar is this to McHardy, Adel and Klinger, Adversarial Training for Satire Detection: Controlling for Confounding Variables, NAACL 2019?

:boom: [Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases](https://www.aclweb.org/anthology/D19-1418.pdf)
> SOTA models are often right for the wrong reasons (e.g., visual question answering models often ignore the evidence in the image, and predict prototypical answers, and textual entailment models simply learn that certain key words usually imply entailment between sentences). This paper proposes the following method: (1) train a "naive" model that makes predicttions based on dataset biases, and (2) "train a robust model as part of an ensemble with the naive one in order to encourage it to focus on other patterns in the data that are more likely to generalize".
>
> Experiments on 5 datasets shows increased robustness to out-of-domain test sets.
>
> Code and data: https://github.com/chrisc36/debias

### Adversarial Attacks

These two papers (from two different groups, Stanford and Deepmind) discuss "certifying robustness" against some class of adversarial attacks, inspired by the technique in computer vision (Dvijotham et al., Training verified learners with learned verifiers, 2018;  Gowal et al., On the effectiveness of interval bound propagation for training verifiably robust models):

:heavy_minus_sign: [Certified Robustness to Adversarial Word Substitutions](https://www.aclweb.org/anthology/D19-1423.pdf)
> Abstract: "... This paper considers one exponentially large family of label-preserving transformations, in which every word in the input can be replaced with a similar word. We train **the first models that are provably robust to all word substitutions in this family**. Our training procedure uses Interval Bound Propagation (IBP) to minimize an upper bound on the worst-case loss that any combination of word substitutions can induce. To evaluate models’ robustness to these transformations, we measure accuracy on adversarially chosen word substitutions applied to test examples. Our IBP-trained models attain 75% adversarial accuracy on both sentiment analysis on IMDB and natural language inference on SNLI; in comparison, on IMDB, models trained normally and ones trained with data augmentation achieve adversarial accuracy of only 12% and 41%, respectively." **Code and data:** https://worksheets.codalab.org/worksheets/0x79feda5f1998497db75422eca8fcd689

:heavy_minus_sign: [Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation](https://www.aclweb.org/anthology/D19-1419.pdf)
>Abstract: "We ... formally verify a system’s robustness against a predefined class of adversarial attacks. We study text classification under synonym replacements or character flip perturbations. We propose modeling these input perturbations as a simplex and then using **Interval Bound Propagation** – a formal model verification method. We modify the conventional log-likelihood training objective **to train models that can be efficiently verified**, which would otherwise come with exponential search complexity. The resulting models show only little difference in terms of nominal accuracy, but have much improved verified accuracy under perturbations and **come with an efficiently computable formal guarantee on worst case adversaries.**" 

## [Learning with Less Data](#contents)

:heavy_minus_sign: [Latent-Variable Generative Models for Data-Efficient **Text Classification**](https://www.aclweb.org/anthology/D19-1048.pdf)
> Abstract: "Generative classifiers offer potential advantages over their discriminative counterparts, namely in the areas of data efficiency, robustness to data shift and adversarial examples, and zero-shot learning (Ng and Jordan,2002; Yogatama et al., 2017; Lewis and Fan,2019). In this paper, we improve generative text classifiers by introducing discrete latent variables into the generative story, and explore several graphical model configurations. We parameterize the distributions using standard neural architectures used in conditional language modeling and perform learning by directly maximizing the log marginal likelihood via gradient-based optimization, which avoids the need to do expectation-maximization. We empirically characterize the performance of our models on six text classification datasets. The choice of where to include the latent variable has a significant impact on performance, with **the strongest results obtained when using the latent variable as an auxiliary conditioning variable in the generation of the textual input**. This model **consistently outperforms both the generative and discriminative classifiers in small-data settings**. We analyze our model by finding that **the latent variable captures interpretable properties of the data, even with very small training sets.**"

### [Out-of-Domain Detection](#contents)

The next two papers look at the "out-of-domain"/"none-of-the-above" issue, where it is useful to also detect whether an instance does not belong to any of the labels in the training set (this seems useful in real-word settings). Note: the term "out-of-domain" seems a bit misleading...this is not to be confused with "out-of-distribution" or novelty detection (https://arxiv.org/pdf/1610.02136.pdf ; https://arxiv.org/pdf/1911.07421.pdf)

:heavy_minus_sign: [FewRel 2.0: Towards More Challenging Few-Shot Relation Classification](https://www.aclweb.org/anthology/D19-1649.pdf)
> Updates the FewRel dataset to address two issues that FewRel ignored: domain adaptation (i.e., with a biomedical out-of-domain test set) and "none-of-the-above" (NOTA) detection. Results: existing SOTA few-shot models and as popular methods for NOTA detection and domain adaptaion struggle on these new aspects.  **Data and baselines:** https://github.com/thunlp/fewrel

:heavy_minus_sign: [**Out-of-Domain Detection** for **Low-Resource** Text Classification Tasks](https://www.aclweb.org/anthology/D19-1364.pdf)
> Task: text classification with very little in-domain labeled data, and no out-of-domain (OOD) data labels (indicating whether an instance has one of the in-domain labels or not). The authors propose an "*OOD-resistant Prototypical Network*" for the task, which is competitive with other baselines.

**See also:**

### [Data Augmentation](#contents)

Something to keep an eye on: improvements in text generation for data augmentation in NLP (see below). In addition to these, there are also data augmentation papers applied to [domain agnostic question answering](https://www.aclweb.org/anthology/D19-5829.pdf), [detecting adverse drug reactions](https://www.aclweb.org/anthology/D19-1239.pdf), [text normalization](https://www.aclweb.org/anthology/D19-5536.pdf), [intent classification](https://www.aclweb.org/anthology/D19-6101.pdf), [dialogue generation](https://www.aclweb.org/anthology/D19-1132.pdf) and machine translation ([G. Li et al.](https://www.aclweb.org/anthology/D19-1570.pdf), [Z. Li and Specia](https://www.aclweb.org/anthology/D19-5543.pdf), [Tong et al.](https://www.aclweb.org/anthology/D19-5218.pdf)).

:heavy_minus_sign: [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://www.aclweb.org/anthology/D19-1670.pdf)
> Performs experiments on "easy data augmentation" (EDA) strategies on five text classification datasets, and both convolutional and recurrent networks. On average "training with EDA while using only 50% of the available training set achieved the same accuracy as normal training with all available data". **Code:** http://github.com/jasonwei20/eda_nlp
>
> Details of EDA: for each sentence in the training set, randomly perform one of these operations: (1) synonym replacement for n (randomly chosen) non stop-words in the sentence, (2) random insertion: insert a random synonym of a random word into a random position of the sentence, (3) randomly swap two words in the sentence, (4) randomly remove each word in the sentence with some probability. An ablation study shows that all four operations contribute to performance.
 >
 > Note: this paper doesn't compare against other benchmarks, such as paraphrases or agumentation with adversarial examples.
 
:heavy_minus_sign: [Controlled Text Generation for Data Augmentation in Intelligent Artificial Agents](https://www.aclweb.org/anthology/D19-5609.pdf)
> "We investigate the use of text generation techniques to augment the training data of a popular commercial artificial agent" (Alexa?; slot filling task)

:heavy_minus_sign: [Data Augmentation with Atomic Templates for Spoken Language Understanding](https://www.aclweb.org/anthology/D19-1375.pdf)
> Abstract: "Spoken Language Understanding (SLU) converts user utterances into structured semantic representations. Data sparsity is one of the main obstacles of SLU due to the high cost of human annotation, especially when domain changes or a new domain comes. In this work, we propose **a data augmentation method with atomic templates for SLU, which involves minimum human efforts**. The atomic templates produce exemplars for fine-grained constituents of semantic representations. **We propose an encoder-decoder model to generate the whole utterance from atomic exemplars**. Moreover, the generator could be transferred from source domains to help a new domain which has little data. Experimental results show that our method achieves significant improvements on DSTC 2&3 dataset which is a domain adaptation setting of SLU."

### [Domain Adaptation](#contents)

Other keywords: "domain adaptive", "cross-domain", "domain shift"...

:boom: [To Annotate or Not? **Predicting Performance Drop under Domain Shift**](https://www.aclweb.org/anthology/D19-1222.pdf)
> Abstract: "... we study the problem of predicting the performance drop of modern NLP models under domain-shift, in the absence of any target domain labels. We investigate three families of methods (H-divergence, reverse classification accuracy and confidence measures), show how they can be used to predict the performance drop and study their robustness to adversarial domain-shifts. Our results on sentiment classification and sequence labelling show that our method is able to predict performance drops with an error rate as low as 2.15% and 0.89% for sentiment analysis and POS tagging respectively."
> 
> See also:
> 1. [Dai et al., Using Similarity Measures to Select Pretraining Data for NER, NAACL 2019](https://www.aclweb.org/anthology/N19-1149)
> 2. [van Asch and Daelemans, Using Domain Similarity for Performance Estimation, ACL 2010](https://www.aclweb.org/anthology/W10-2605.pdf)
> 3. Jesper Back, thesis (2018), Domain similarity metrics for predicting transfer learning performance

:boom: [**Unsupervised Domain Adaptation of Contextualized Embeddings for Sequence Labeling**](https://www.aclweb.org/anthology/D19-1433.pdf)
> How well does pre-training (e.g., BERT, ELMo) work when the target domain is very different from the pretraining corpus? This paper considers the scenario where we have:
> - target domain unlabeled text (e.g., Twitter, Early Modern English)
> - source domain texts for pre-training (e.g., newswire)
> - source domain labeled text for fine-tuning (e.g., newswire)
>
> Proposes a "domain-adaptive fine-tuning" method (in which the contextual embeddings are "adapted by masked language modeling on text from the target domain"; this is tested on sequence labeling tasks with very strong results, especially for out-of-vocabulary words.
>
> **See also**:
> - Rietzler et al., Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification
> - (the BioBERT results in) [Shi and Demberg, Next Sentence Prediction helps Implicit Discourse Relation Classification within and across Domains, EMNLP 2019](https://www.aclweb.org/anthology/D19-1586.pdf)


:heavy_minus_sign: [Adaptive Ensembling: Unsupervised Domain Adaptation for Political Document Analysis](https://www.aclweb.org/anthology/D19-1478.pdf)
> Presents a new unsupervised domain adaptation framework, *adaptive ensembling*, which outperforms strong benchmarks on  two political text classification tasks, with the Annotated NYT as the source domain and COHA as the target domain. **Code and data**: http://github.com/shreydesai/adaptive-ensembling
>
> This is work co-authored by Jessy Li (with Shrey Desai, Barea Sinno, Alex Rosenfeld).

:heavy_minus_sign: [Weakly Supervised Domain Detection. Yumo Xu and Mirella Lapata (TACL)](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00287)
> Abstract: "... we introduce **domain detection** as a new natural language processing task. We argue that the ability to detect textual segments that are **domain-heavy (i.e., sentences or phrases that are representative of and provide evidence for a given domain)** could enhance the robustness and portability of various text classification applications. We propose an encoder-detector framework for domain detection and **bootstrap classifiers with multiple instance learning**. The model is hierarchically organized and suited to multilabel classification. We demonstrate that despite learning with minimal supervision, our model can be **applied to text spans of different granularities, languages, and genres**. We also showcase the potential of domain detection for text summarization."


:heavy_minus_sign: [Domain Adaptive Text Style Transfer](https://www.aclweb.org/anthology/D19-1325.pdf)
> See under [Style transfer](#style-transfer)

### [Meta-learning](#contents)

:boom: [Investigating Meta-Learning Algorithms for Low-Resource Natural Language Understanding Tasks](https://www.aclweb.org/anthology/D19-1112.pdf)


:heavy_minus_sign: [Learning to Learn and Predict: A Meta-Learning Approach for Multi-Label Classification](https://www.aclweb.org/anthology/D19-1444.pdf)
> "... we propose a meta-learning method to capture these complex label dependencies. More specifically, our method utilizes a meta-learner to jointly learn the training policies and prediction policies for different labels. The training policies are then used to train the classifier with the cross-entropy loss function, and the prediction policies are further implemented for prediction. "

:heavy_minus_sign: [Text Emotion Distribution Learning from Small Sample: A Meta-Learning Approach](https://www.aclweb.org/anthology/D19-1408.pdf)
> Emotion distribution learning (EDL) task: predict the intensity of a sentence across a set of emotion categories. The authors "propose a meta-learning approach to learn text emotion distributions from a small sample", by (1) learning a low-rank sentence embedding via tensor decomposition, (2) generating sample clusters using K-nearest neighbors (KNNs) of sentence embeddings, and (3) "train a meta-learner that can adapt to new data with only a few training samples on the clusters, and further fit the meta-learner on KNNs of a testing sample for EDL."

### [Few-shot and zero-shot text classification](#contents)

:boom: [Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach](https://www.aclweb.org/anthology/D19-1404.pdf)
>

:boom: [Look-up and Adapt: A One-shot Semantic Parser](https://www.aclweb.org/anthology/D19-1104.pdf)
> Paper co-authored by Tom Mitchell. Abstract: "... we propose a semantic parser that generalizes to out-of-domain examples by learning a general strategy for parsing an unseen utterance **through adapting the logical forms of seen utterances, instead of learning to generate a logical form from scratch**. Our parser maintains a memory consisting of a representative subset of the seen utterances paired with their logical forms. Given an unseen utterance, our parser works by looking up a similar utterance from the memory and adapting its logical form until it fits the unseen utterance. Moreover, we present a data generation strategy for constructing utterance-logical form pairs from different domains. **Our results show an improvement of up to 68.8% on one-shot parsing under two different evaluation settings compared to the baselines.**

:heavy_minus_sign: [Hierarchical Attention Prototypical Networks for **Few-Shot Text Classification**](https://www.aclweb.org/anthology/D19-1042.pdf)

:heavy_minus_sign: [Induction Networks for Few-Shot Text Classification](https://www.aclweb.org/anthology/D19-1403.pdf)


### [Active learning](#contents)

:boom: [Practical Obstacles to Deploying **Active Learning**](https://www.aclweb.org/anthology/D19-1003.pdf)
>
> Paper by David Lowell, Zachary Lipton and Byron Wallace. This is a thorough empirical study of various active learning methods on text classification and NER tasks that shows **consistent improved performance for NER**, while for text classification:
> 1. current active learning methods don't "generalize reliably across models and tasks".
> 2. "subsequently training a successor model with an actively-acquired dataset does not consistently outperform training on i.i.d. sampled data"
>
> "While  a  specific  acquisition  function  and model applied to a particular task and domain maybe quite effective, it is not clear that this can be predicted ahead of time. Indeed, there is no way to retrospectively determine the relative success of AL without collecting a relatively large quantity of i.i.d. sampled data, and this would undermine the purpose of AL in the first place. Further, even if such an i.i.d. sample were taken as a diagnostic tool early in the active learning cycle, relative success early in the AL cycle is not necessarily indicative of relative success later in the  cycle ... Problematically, even in successful cases, **an actively sampled training set is linked to the model used  to  acquire  it.**"

:heavy_minus_sign: [Empirical Evaluation of Active Learning Techniques for Neural MT](https://www.aclweb.org/anthology/D19-6110.pdf)
> This is the first thorough investigation of active learning for neural machine translation (NMT). From the abstract: "We demonstrate how recent advancements in unsupervised pre-training and paraphrastic embedding can be used to improve existing AL methods. Finally, we propose a neural extension for an AL sampling method used in the context of phrase-based MT - Round Trip Translation Likelihood (RTTL). RTTL uses a bidirectional translation model to estimate the loss of information during translation and outperforms previous methods."

## [Word Embeddings](#contents)

:boom: [Do NLP Models Know Numbers? Probing Numeracy in Embeddings](https://www.aclweb.org/anthology/D19-1534.pdf)
>

:boom: **Best paper award** [Specializing Word Embeddings (for Parsing) by Information Bottleneck](https://www.aclweb.org/anthology/D19-1276.pdf)
>

:boom: [Still a Pain in the Neck: Evaluating Text Representations on Lexical Composition (TACL)](https://www.aclweb.org/anthology/Q19-1027.pdf)
>

:heavy_minus_sign: [Multiplex Word Embeddings for **Selectional Preference Acquisition**](https://www.aclweb.org/anthology/D19-1528.pdf)
>  A multiplex  network  embedding  model (originally used to model social networks in Zhang et al., Scalable  multiplex  network embedding, IJCAI 2018) is used to encode selectional preference information in a set of embeddings to represent both its general semantics and relation-dependent (e.g., subject, object) semantics. This is used to learn selectional preferences (for example, that "song" is a more plausible object for "sing" than "potato"). Experiments on selectional preference acquisition and word similarity demonstrate the effectiveness of the embeddings.

:heavy_minus_sign: [Feature2Vec: Distributional semantic modelling of human property knowledge](https://www.aclweb.org/anthology/D19-1595.pdf)
> *Feature norm* datasets list properties of words; they "yield highly interpretable models of word meaning and play an important role in neurolinguistic research on semantic cognition".  This paper proposes "a method for mapping human property knowledge onto a distributional semantic space, which adapts the word2vec architecture to the task of modelling concept features".

:heavy_minus_sign: [Text-based inference of moral sentiment change](https://www.aclweb.org/anthology/D19-1472.pdf)
> Uses diachronic word embeddings to explore how moral sentiments change over time (e.g. *slavery*). "We demonstrate how a parameter-free model supports inference of historical shifts in moral sentiment toward concepts such as slavery and democracy over centuries at three incremental levels: moral relevance, moral polarity, and fine-grained moral dimensions..."

:heavy_minus_sign: [Correlations between Word Vector Sets](https://www.aclweb.org/anthology/D19-1008.pdf)
> From the abstract: "Similarity measures based purely on word embeddings are comfortably competing with much more sophisticated deep learning and expert-engineered systems on unsupervised semantic textual similarity (STS) tasks. In contrast to commonly used geometric approaches, **we treat a single word embedding as e.g. 300 observations from a scalar random variable.**"
> 1. "similarities derived from elementary pooling operations and classic correlation coefficients yield excellent results on standard STS benchmarks, outperforming many recently proposed methods **while being much faster and trivial to implement**"
> 2. "we demonstrate how to avoid pooling operations altogether and **compare sets of word embeddings directly via correlation operators between reproducing kernel Hilbert spaces.** Just like cosine similarity is used to compare individual word vectors, we introduce a novel application of the centered kernel alignment (CKA) as a natural generalisation of squared cosine similarity for sets of word vectors. Likewise, CKA is very easy to implement and enjoys very strong empirical results." 

:heavy_minus_sign: [Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation](https://www.aclweb.org/anthology/D19-1009.pdf)
> Builds on Word Sense Disambiguation Games (WSDG), introduced in Tripodi and Pelillo, Computational Linguistics, 2017. This represents words as players in a non-cooperative game, and word senses as strategies. Edges of a graph determine the game interactions. The games are played repeatedly; at each iteration, words update their strategy/sense preferences, which are probability distributions over senses/strategies.
>
> The main difference between this paper and Tripodi and Pelillo (2017) is that they use word co-occurence and tf-idf vectors, resulting in sparse graphs, rather than dense vectors (which results in a dense weighted graph). Each player is involved in many more games, so the interaction among players is also changed.

:heavy_minus_sign: [Rotate King to get Queen: Word Relationships as Orthogonal Transformations in Embedding Space](https://www.aclweb.org/anthology/D19-1354.pdf)

## [Sentence Embeddings](#contents)

:heavy_minus_sign: [Parameter-free Sentence Embedding via Orthogonal Basis](https://www.aclweb.org/anthology/D19-1059.pdf)
> A simple, non-parametrized approach, inspired by the Gram-Schmidt process: "we build an orthogonal basis of the subspace spanned by a word and its surrounding context in a sentence... We evaluate our approach on 11 downstream NLP tasks. Our model shows superior performance compared with non-parameterized alternatives and it is **competitive to other approaches relying on either large amounts of labelled data or prolonged training time.**"

:heavy_minus_sign: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**](https://www.aclweb.org/anthology/D19-1410.pdf)
> Paper by Nils Reimers and Iryna Gurevych. They modify the pretrained BERT network to use siamese network structures to get semantically meaningful sentence embeddings, which can be compared using cosine similarity for semantic textual similarity (STS). This approach ("SBERT") is much faster than using BERT directly (feeding both sentences into the network to calculate a pairwise score): it takes 5 seconds rather than 65 hours has similar accuracy.

:heavy_minus_sign: [Efficient Sentence Embedding using Discrete Cosine Transform](https://www.aclweb.org/anthology/D19-1380.pdf)
> Abstract: "... We propose the use of discrete cosine transform (DCT) to compress word sequences in an order-preserving manner. The lower order DCT coefficients represent the overall feature patterns in sentences, which results in suitable embeddings for tasks that could benefit from syntactic features. Our results in semantic probing tasks demonstrate that DCT embeddings indeed preserve more syntactic information compared with vector averaging. With practically equivalent complexity, the model yields better overall performance in downstream classification tasks that correlate with syntactic features, which illustrates **the capacity of DCT to preserve word order information.**"

## [Knowledge enhanced ML](#contents)

See also many of the papers in [Commonsense Reasoning](#commonsense-reasoning) and [Relation extraction and Knowledge graphs](#relation-extraction-and-knowledge-graphs) sections.

:boom: [**Knowledge Enhanced** Contextual Word Representations](https://www.aclweb.org/anthology/D19-1005.pdf)
> This presents a method to embed multiple knowledge bases (KB) into large-scale models. After integrating WordNet and part of Wikipedia into BERT, **KnowBERT** shows improvements in perplexity, fact recall (via probing tasks) and downstream tasks (relationship extraction, entity typing, WSD).

The next two papers use attention mechanisms to incorporate external knowledge:

:heavy_minus_sign: [**Knowledge-Enriched** Transformer for Emotion Detection in Textual Conversations](https://www.aclweb.org/anthology/D19-1016.pdf)
> Proposes a "Knowledge-Enriched Transformer" (KET), where "utterances are interpreted using hierarchical self-attention and external commonsense knowledge is dynamically leveraged using a context-aware affective graph attention mechanism." This is applied to emotion detection tasks with promising (SOTA) results.

:heavy_minus_sign: [Improving Relation Extraction with **Knowledge-attention**](https://www.aclweb.org/anthology/D19-1022.pdf)
> Abstract: "...We propose a novel knowledge-attention encoder **which incorporates prior knowledge from external lexical resources into deep neural networks for relation extraction** task. Furthermore, we present three effective ways of integrating knowledge-attention with self-attention to maximize the utilization of both knowledge and data..."

## [Multimodal ML](#contents)

:boom: [Integrating Text and Image: Determining Multimodal Document Intent in Instagram Posts](https://www.aclweb.org/anthology/D19-1469.pdf)
> Introduces a multimodal dataset of Instagram posts labeled with author intent and image-text relationships. A baseline deep multimodal classifier is used to validate the proposed taxonomy, and using both the text and the image improves intent detection by almost 10% (over using only the image). Intents include: advocative, promotive, exhibitionist, expressive, informative, entertainment and provocative.
>
> See also at EMNLP 2019: "Dialog Intent Induction with Deep Multi-View Clustering".

:heavy_minus_sign: [Neural Naturalist: Generating Fine-Grained Image Comparisons](https://www.aclweb.org/anthology/D19-1065.pdf)
> 1. Introduces the "Birds-to-Words" datasets of sentences describing differences between pairs of birds.
> 2. Introduces a model, *Neural Naturalist*, to encode the image pairs and generate the comparative text.
>
> **Data:** https://github.com/google-research-datasets/birds-to-words

:heavy_minus_sign: [What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues](https://www.aclweb.org/anthology/D19-1516.pdf)
> "... we formally define the task of visual-aware pronoun coreference resolution (PCR) and introduce VisPro, a large-scale dialogue PCR dataset, to investigate whether and how the visual information can help resolve pronouns in dialogues. We then propose a novel visual-aware PCR model, VisCoref, for this task and conduct comprehensive experiments and case studies on our dataset. Results demonstrate the importance of the visual information in this PCR case and show the effectiveness of the proposed model."

:heavy_minus_sign: [Partners in Crime: Multi-view Sequential Inference for Movie Understanding](https://www.aclweb.org/anthology/D19-1212.pdf)
> "In this paper, we propose a neural architecture coupled with a novel training objective that **integrates multi-view  information for sequence prediction problems.**".  This is used in an incremental inference setup (i.e., making predictions without encoding the full sequence), which better mimics how humans process information. This architecture is tested on three tasks derived from the Frermann et al. (2018) dataset episodes from the CSI TV show:
> - perpetrator mention identification (task from Frermann et al. (2018))
> - crime case segmentation (some episodes alternate between more than one crime case)
> - speaker type tagging (each utterance can come from either a detective, perpretator, suspect, or extra).

# [NLG Tasks](#contents)

## Miscellaneous Tasks:

There are also papers on code generation, particularly SQL generation, that should go here.

:heavy_minus_sign: [Text2Math: End-to-end Parsing Text into Math Expressions](https://www.aclweb.org/anthology/D19-1536.pdf)
> Presents an end-to-end system to translate text (arithmetic word problems) into tree-structured math expressions.
>
> See also: [Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412), and associated blog post [Using neural networks to solve advanced mathematics equations](https://ai.facebook.com/blog/using-neural-networks-to-solve-advanced-mathematics-equations/)

## [Summarization and Simplification](#contents)

### Summarization

There are a huge number of papers for this task. The following paper may be useful to see the big picture:

:boom: [Neural Text Summarization: A Critical Evaluation](https://www.aclweb.org/anthology/D19-1051.pdf)
> Abstract: "...We critically evaluate key ingredients of the current research setup: datasets, evaluation metrics, and models, and highlight three primary shortcomings: 1) automatically collected datasets leave the task underconstrained and may contain noise detrimental to training and evaluation, 2) **current evaluation protocol is weakly correlated with human judgment and does not account for important characteristics such as factual correctness**, 3) models **overfit to layout biases of current datasets** and offer limited diversity in their outputs.

:boom: [BottleSum: Unsupervised and Self-supervised Sentence Summarization using the Information Bottleneck Principle](https://www.aclweb.org/anthology/D19-1389.pdf)
> A new approach to summarization based on the Information Bottleneck principle in information theory: to produce a summary (compression) of information X optimized to predict some other information Y (e.g., the next sentence) conditioned on the summary. The unsupervised *extractive* component (BottleSum-Ex) searches over subsequences of a given sentence; for example:
>
> Source: "Okari, for instance, told CNN that he saw about 200 people sitting in the scorching mid-90-degree heat Thursday in a corner of a Garissa airstrip, surrounded by military officials."
>
> BottleSum-Ex: "Okari, told CNN he saw about 200 people in the scorching corner of Garissa airstrip, surrounded by officials."
>
> The self-supervised abstractive summarization component (BottleSum-Self) uses GPT-2 trained on the output summaries of BottleSum-Ex to rewrite the sentence:
>
> BottleSum-Self: "For instance, told CNN he saw about 200 people sitting in the scorching mid-90-degree heat in a Garissa airstrip,."
>
> Note: in another example, the source contains a name, Griner ("Griner told The Daily Dot that.."), but BottleSum-Self replaces it with "Fletcher told The Daily Dot that...").  Isn't this problematic?

#### Reinforcement Learning

The following two papers focus on finding better reward functions for RL-based summarization:

:heavy_minus_sign: [Deep Reinforcement Learning with Distributional Semantic Rewards for Abstractive Summarization](https://www.aclweb.org/anthology/D19-1623.pdf)
> Rather than use Rouge-L (the conventional reward for RL-based abstractive summarization), this proposes a new "distributional semantics reward", which performs better according to human judgments.

:heavy_minus_sign: [Better Rewards Yield Better Summaries: Learning to Summarise Without References](https://www.aclweb.org/anthology/D19-1307.pdf)
> Abstract: "... summaries with high ROUGE scores often receive low human judgement. To find a better reward function that can guide RL to generate human-appealing summaries, we learn a reward function from human ratings on 2,500 summaries. Our reward function only takes the document and system summary as input. Hence, once trained, it can be used to train RL based summarisation systems without using any reference summaries. We show that our learned rewards have significantly higher correlation with human ratings than previous approaches."

:heavy_minus_sign: [Text Summarization with Pretrained Encoders](https://www.aclweb.org/anthology/D19-1387.pdf) (BERT)
> Abstract: "... we showcase how BERT can be usefully applied in text summarization and propose a general framework for both **extractive** and **abstractive** models. We introduce a novel document-level encoder based on BERT which is able to express the semantics of a document and obtain representations for its sentences..." 

:heavy_minus_sign: [Contrastive Attention Mechanism for Abstractive Sentence Summarization](https://www.aclweb.org/anthology/D19-1301.pdf)
>  Proposes a "contrastive attention mechanism" for seq2seq-based abstractive sententence summarization. Two types of attention are used: the usual one (attending to relevant parts of the sentence) and an "opponent attention that attends to irrelevant or less relevant parts of the source sentence". The contribution of the "opponent attention" is discouraged. **Code:** https://github.com/travel-go/Abstractive-Text-Summarization


:heavy_minus_sign: [Deep copycat Networks for Text-to-Text Generation](https://www.aclweb.org/anthology/D19-1318.pdf)
> Abstract: "We introduce Copycat, **a transformer-based pointer network** for such tasks which obtains competitive results in abstractive text summarisation and generates more abstractive summaries. We propose a further extension of this architecture for **automatic post-editing, where generation is conditioned over two inputs** (source language and machine translation), and the model is capable of deciding where to copy information from. This approach achieves competitive performance when compared to state-of-the-art automated post-editing systems. More importantly, we show that it addresses a well-known limitation of automatic post-editing - overcorrecting translations - and that our novel mechanism for copying source language words improves the results."

:heavy_minus_sign: [How to Write Summaries with Patterns? Learning towards Abstractive Summarization through Prototype Editing](https://www.aclweb.org/anthology/D19-1388.pdf)

:heavy_minus_sign: [A Summarization System for Scientific Documents](https://www.aclweb.org/anthology/D19-3036.pdf)

### Simplification

:heavy_minus_sign: [EASSE: Easier Automatic Sentence Simplification Evaluation](https://www.aclweb.org/anthology/D19-3009.pdf)
> Demo paper introducing EASSE, a Python package for automatic evaluation and comparison of sentence simplification systems. "EASSE provides a single access point to a broad range of evaluation resources: standard automatic metrics for assessing SS outputs (e.g. SARI), word-level accuracy scores for certain simplification transformations, reference-independent quality estimation features (e.g. compression ratio), and standard test data for SS evaluation (e.g. TurkCorpus)."
>
> **Code:** https://github.com/feralvam/easse

:heavy_minus_sign: [Recursive Context-Aware Lexical Simplification](https://www.aclweb.org/anthology/D19-1491.pdf)

### Evaluation

:heavy_minus_sign: [Answers Unite! Unsupervised Metrics for Reinforced Summarization Models](https://www.aclweb.org/anthology/D19-1320.pdf)
> ROUGE is a bad metric for summarization - so the authors "explore and propose alternative evaluation measures: the reported human-evaluation analysis shows that the proposed metrics, based on Question Answering, favorably compare to ROUGE – **with the additional property of not requiring reference summaries**". These evaluation metrics are also useful when used in an RL model: "training a RL-based model on these metrics leads to improvements (both in terms of human or automated metrics) over current approaches that use ROUGE as reward."

:heavy_minus_sign: [SUM-QE: a BERT-based Summary Quality Estimation Mode](https://www.aclweb.org/anthology/D19-1618.pdf)
> A new BERT-based **quality estimation** model for summarization, which also doesn't involve comparison with human references.

## [Style transfer](#contents)

Style transfer in NLP is also a hot area now. I think a lot of interest in this area also comes from adapting NMT methods, and from recent progress on unsupervised neural machine translation (because it is hard/impractical to obtain paired datasets for style transfer). I have a couple questions:

- Why are there so many papers that do "style transfer" for sentiment? The point of sentiment transfer is to preserve "sentence meaning", but changing "I loved the burger" to "I hated the burger" completely changes the meaning... Also, is there any compelling use case for sentiment style transfer?
- Are there any good surveys, reviews, or comparative evaluation papers?

See also: papers on style transfer using disentangled representation learning, e.g., John et al., Disentangled representation learning for text style transfer, 2018.

Most of these papers address the crucial issue of lack of parallel training data for style transfer in different ways: semi-supervised learning, pretrained text generation (GPT2) networks (combined with rules or with Transformers), and creating corpora of "pseudo-parallel" aligned sentences:

:heavy_minus_sign: [Semi-supervised Text Style Transfer: Cross Projection in Latent Space](https://www.aclweb.org/anthology/D19-1499.pdf)
> The authors propose a "semi-supervised text style transfer model that combines the small-scale parallel data with the large-scale nonparallel data". The authors evaluate this on a dataset for style transfer between ancient and modern Chinese.

:heavy_minus_sign: [Harnessing Pre-Trained Neural Networks **with Rules** for **Formality** Style Transfer](https://www.aclweb.org/anthology/D19-1365.pdf)
> Abstract: we study how to harness rules into a state-of-the-art neural network that is typically pretrained on massive corpora. We propose three fine-tuning methods in this paper and achieve a new state-of-the-art on benchmark datasets

:heavy_minus_sign: [Learning to Flip the Sentiment of Reviews from Non-Parallel Corpora](https://www.aclweb.org/anthology/D19-1659.pdf)
> "We introduce a method for acquiring imperfectly aligned sentences from non-parallel corpora and propose a model that learns to minimize the sentiment and content losses in a fully end-to-end manner."

:heavy_minus_sign: [IMaT: Unsupervised Text Attribute Transfer via Iterative Matching and Translation](https://www.aclweb.org/anthology/D19-1306.pdf)
>  Abstract: "... Existing approaches try to explicitly disentangle content and attribute information, but this is difficult and often results in poor content-preservation and ungrammaticality. In contrast, we propose a simpler approach, Iterative Matching and Translation (IMaT), which: (1) constructs a pseudo-parallel corpus by aligning a subset of semantically similar sentences from the source and the target corpora; (2) applies a standard sequence-to-sequence model to learn the attribute transfer; (3) iteratively improves the learned transfer function by refining imperfections in the alignment..."

:heavy_minus_sign: [Transforming Delete, Retrieve, Generate Approach for Controlled Text Style Transfer](https://www.aclweb.org/anthology/D19-1322.pdf)
> Abstract: "... In this work we introduce the Generative Style Transformer (GST) - a new approach to rewriting sentences to a target style **in the absence of parallel style corpora**. GST leverages the power of both, large unsupervised pre-trained language models as well as the Transformer. GST is a part of a larger ‘Delete Retrieve Generate’ framework, in which we also propose a novel method of deleting style attributes from the source sentence by exploiting the inner workings of the Transformer. Our models outperform state-of-art systems across 5 datasets on sentiment, gender and political slant transfer. We also propose the use of the GLEU metric as an automatic metric of evaluation of style transfer, which we found to compare better with human ratings than the predominantly used BLEU score."
>
> See also: Juncen Li, Robin Jia, He He, and Percy Liang. Delete, retrieve, generate: a simple approach to sen-timent and style transfer, NAACL 2018.

:heavy_minus_sign: [Domain Adaptive Text Style Transfer](https://www.aclweb.org/anthology/D19-1325.pdf)
> Abstract: "...The proposed models presumably learn from the source domain to: (i) distinguish stylized information and generic content information; (ii) maximally preserve content information; and (iii) adaptively transfer the styles in a domain-aware manner. We evaluate the proposed models on two style transfer tasks (sentiment and formality) over multiple target domains where only limited non-parallel data is available. Extensive experiments demonstrate the effectiveness of the proposed model compared to the baselines."

## [Text generation and GPT2](#contents)

These are papers on text generation which are not focused on [style transfer](#style-transfer), [summarization](#summarization) or [machine translation](#machine-translation).

:boom: [Generating Natural Anagrams: Towards Language Generation Under Hard Combinatorial Constraints](https://www.aclweb.org/anthology/D19-1674.pdf)
>

:boom: [Denoising based Sequence-to-Sequence Pre-training for Text Generation](https://www.aclweb.org/anthology/D19-1412.pdf)
>

:boom: [Judge the Judges: A Large-Scale Evaluation Study of Neural Language Models for Online Review Generation](https://www.aclweb.org/anthology/D19-1409.pdf)
>

:boom: [Controlling Text Complexity in Neural Machine Translation](https://www.aclweb.org/anthology/D19-1166.pdf)
>

:boom: [See et al., Do Massively Pretrained Language Models Make Better Storytellers? (CoNLL)](https://www.aclweb.org/anthology/K19-1079.pdf)
>

:boom: [Attending to Future Tokens for Bidirectional Sequence Generation](https://www.aclweb.org/anthology/D19-1001.pdf)
> The BISON paper. Can generate text from BERT!

:boom: [How Contextual are Contextualized Word Representations? Comparing **the Geometry of BERT, ELMo, and GPT-2 Embeddings**](https://www.aclweb.org/anthology/D19-1006.pdf)
> See also: Coenen et al., Visualizing and Measuring the Geometry of BERT (2019), (paper: https://arxiv.org/pdf/1906.02715.pdf ; blog post: https://pair-code.github.io/interpretability/bert-tree/).

:boom: [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://www.aclweb.org/anthology/D19-1053.pdf)
>

:boom: [Distributionally Robust Language Modeling](https://www.aclweb.org/anthology/D19-1432.pdf)
> Abstract: "Language models are generally trained on data spanning a wide range of topics (e.g., news, reviews, fiction), but they might be applied to an a priori unknown target distribution (e.g., restaurant reviews). In this paper, we first show that training on text outside the test distribution can degrade test performance when using standard maximum likelihood (MLE) training. To remedy this without the knowledge of the test distribution, we propose an approach which trains a model that performs well over a wide range of potential test distributions. In particular, we derive a new distributionally robust optimization (DRO) procedure which minimizes the loss of the model over the worst-case mixture of topics with sufficient overlap with the training distribution. Our approach, called topic conditional value at risk (topic CVaR), obtains a 5.5 point perplexity reduction over MLE when the language models are trained on a mixture of Yelp reviews and news and tested only on reviews."


:boom: [Select and Attend: Towards Controllable Content Selection in Text Generation](https://www.aclweb.org/anthology/D19-1054.pdf)
>

:heavy_minus_sign: [A Modular Architecture for Unsupervised **Sarcasm Generation**](https://www.aclweb.org/anthology/D19-1636.pdf)
> Presents a new framework for sarcasm generation which take a negative opinion as input (e.g., "I hate it when my bus is late.") and translates it to a sarcastic version (e.g., "Absolutely love waiting for the bus").  *This does not require paired training data*.
>
> The system is trained only with unlabeled data (sarcastic and non-sarcastic opinions) and employs the theory of "context incongruity": sarcasm is the contrast between positive sentiment context and negative situational context. It uses modules for (1) sentiment neutralization (removing sentiment-bearing words), (2) translation to positive-sentiment phrases, (3) negative situation retrieval of a situation related to the input, (4) sarcasm synthesis. Qualitative and quantitative evaluation shows this system is superior to NMT and style transfer baselines.
>
> Examples ("ALL (RL)"): (1) "worrying  because  did  not  finish  my  homework" -> "worrying about finish homework is great". (2) "swimming lessons are very costly in nyc." -> "i am loving the swimming lessons . going to be a very costly in nyc ch."
>
>
> **Code:**  https://github.com/TarunTater/sarcasmgeneration

:heavy_minus_sign: [Generating Personalized Recipes from Historical User Preferences](https://www.aclweb.org/anthology/D19-1613.pdf)
> Task: personalized recipe generation.
>
> Example input: "Name:Pomberrytini;Ingredients:pomegranate-blueberry juice, cranberry juice,vodka; Calorie:Low".
>
> Output: "Combine  all  ingredients  except  for  the  ice  in  a  blender  or  food  processor. Process to make a  smoothpaste and then add the remaining vodka and blend until smooth. Pour into a chilled glass and garnish with a little lemon and fresh mint."
>
> Code: https://github.com/majumderb/recipe-personalization

:heavy_minus_sign: [Learning Rhyming Constraints using Structured Adversaries](https://www.aclweb.org/anthology/D19-1621.pdf)
>

:heavy_minus_sign: [An Empirical Comparison on Imitation Learning and Reinforcement Learning for Paraphrase Generation](https://www.aclweb.org/anthology/D19-1619.pdf)
>

:heavy_minus_sign: [Encode, Tag, Realize: High-Precision Text Editing](https://www.aclweb.org/anthology/D19-1510.pdf)
>

:heavy_minus_sign: [Towards Controllable and Personalized Review Generation](https://www.aclweb.org/anthology/D19-1319.pdf)
>

:heavy_minus_sign: [An End-to-End Generative Architecture for Paraphrase Generation](https://www.aclweb.org/anthology/D19-1309.pdf)
>

:heavy_minus_sign: [Enhancing Neural Data-To-Text Generation Models with External Background Knowledge](https://www.aclweb.org/anthology/D19-1299.pdf)
>

:heavy_minus_sign: [Autoregressive Text Generation Beyond Feedback Loops](https://www.aclweb.org/anthology/D19-1338.pdf)
>

:heavy_minus_sign: [Big Bidirectional Insertion Representations for Documents](https://www.aclweb.org/anthology/D19-5620.pdf)
> Insertion Transformer: well suited for long form text generation; this builds on it to produce BIRD for use in machine translation task.

:heavy_minus_sign: [Neural data-to-text generation: A comparison between pipeline and end-to-end architectures](https://www.aclweb.org/anthology/D19-1052.pdf)
>

:heavy_minus_sign: [Sentence-Level Content Planning and Style Specification for Neural Text Generation](https://www.aclweb.org/anthology/D19-1055.pdf)
>

## [Machine Translation](#contents)

:boom: [Synchronously Generating Two Languages with Interactive Decoding](https://www.aclweb.org/anthology/D19-1330.pdf)
> Simultaneously translating into two languages at once!

:boom: [Machine Translation With Weakly Paired Documents](https://www.aclweb.org/anthology/D19-1446.pdf)
>

:boom: [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://www.aclweb.org/anthology/D19-1331.pdf)
>

:heavy_minus_sign: [FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow](https://www.aclweb.org/anthology/D19-1437.pdf)
> "... In this paper, we propose a simple, efficient, and effective model for non-autoregressive sequence generation using latent variable models. Specifically, we turn to generative flow, an elegant technique to model complex distributions using neural networks, and design several layers of flow tailored for modeling the conditional density of sequential latent variables. We evaluate this model on three neural machine translation (NMT) benchmark datasets, achieving comparable performance with state-of-the-art non-autoregressive NMT models and almost constant decoding time w.r.t the sequence length."

:heavy_minus_sign: [Exploiting Monolingual Data at Scale for Neural Machine Translation](https://www.aclweb.org/anthology/D19-1430.pdf)
> Abstract: "While target-side monolingual data has been proven to be very useful to improve neural machine translation (briefly, NMT) through back translation, source-side monolingual data is not well investigated. In this work, we study how to use both the source-side and target-side monolingual data for NMT, and propose an effective strategy leveraging both of them. ...  Our approach achieves state-of-the-art results on WMT16, WMT17, WMT18 English↔German translations and WMT19 German -> French translations, which demonstrate the effectiveness of our method. We also conduct a comprehensive study on how each part in the pipeline works."

:heavy_minus_sign: [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://www.aclweb.org/anthology/D19-1633.pdf)
>

:heavy_minus_sign: [Unsupervised Domain Adaptation for Neural Machine Translation with Domain-Aware Feature Embeddings](https://www.aclweb.org/anthology/D19-1147.pdf)
>

:heavy_minus_sign: [Machine Translation for Machines: the Sentiment Classification Use Case](https://www.aclweb.org/anthology/D19-1140.pdf)
>

:heavy_minus_sign: [Iterative Dual Domain Adaptation for NMT](https://www.aclweb.org/anthology/D19-1078.pdf)
>

:heavy_minus_sign: [Self-Attention with Structural Position Representations](https://www.aclweb.org/anthology/D19-1145.pdf)
>

:heavy_minus_sign: [Simple, Scalable Adaptation for Neural Machine Translation](https://www.aclweb.org/anthology/D19-1165.pdf)
>

:heavy_minus_sign: [Hierarchical Modeling of Global Context for Document-Level Neural Machine Translation](https://www.aclweb.org/anthology/D19-1168.pdf)
>

:heavy_minus_sign: [Context-Aware Monolingual Repair for Neural Machine Translation](https://www.aclweb.org/anthology/D19-1081.pdf)
>

# [NLU Tasks](#contents)

## [Word Sense Disambiguation (WSD)](#contents)

:heavy_minus_sign: [Game Theory Meets Embeddings: a Unified Framework for Word Sense Disambiguation](https://www.aclweb.org/anthology/D19-1009.pdf)
> See under [Word Embeddings](#word-embeddings)

:heavy_minus_sign: [SyntagNet: Challenging Supervised Word Sense Disambiguationwith Lexical-Semantic Combinations](https://www.aclweb.org/anthology/D19-1359.pdf)
> This presents SyntagNet, a new resource of manually disambiguated "lexical semantic combinations" (sense-annotated lexical combinations, i.e., noun-noun and noun-verb pairs) which can be used in knowledge-based WSD. This supplements resources such as WordNet, with lacks syntagmatic relations ("A syntagmatic relation exists between two words which co-occur in spoken or written language more frequently than would be expected by chance and which have different grammatical roles in the sentences in which they occur"). The resource was extracted from English Wikipedia and the BNC, and disambiguated using WordNet 3.0. Examples: (run.v.19, program.n.07), (run.v.37, race.n.02), (run.v.4, farm.n.01).
>
> SyntagNet enables SOTA WSD (using UKB*) which is competitive with the best supervised systems.
>
> **Data:** http://syntagnet.org/download/SyntagNet1.0.zip
> 
> \*  Agirre, Random Walks for Knowledge-Based WordSense Disambiguation, Computational Linguistics, 2014.

And of course there are a couple BERT-related papers:

:heavy_minus_sign: [GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge](https://www.aclweb.org/anthology/D19-1355.pdf)
>

:heavy_minus_sign: [Improved Word Sense Disambiguation Using Pre-Trained Contextualized Word Representations](https://www.aclweb.org/anthology/D19-1533.pdf)
>

**See also:** the "Word-in-Context" (WiC) challenge, and associated papers: https://pilehvar.github.io/wic/

## [Keyphrase Extraction](#contents)

:heavy_minus_sign: [Open Domain Web Keyphrase Extraction Beyond Language Modeling](https://www.aclweb.org/anthology/D19-1521.pdf)
> Presents a new dataset, OpenKP, of annotated web documents for open domain keyphrase extraction. Also develops BLING-KPE, a neural model that combines the "visual presentation" of documents (e.g., font size, text location, etc.) with language modeling (ELMo) and weak supervision from search logs (sampled from Bing search logs). Ablation shows that visual features and search pretraining improve the precision and recall scores.
>
> **Code and data**: https://github.com/microsoft/OpenKP

## [Fact checking](#contents)

:boom: [Towards Debiasing Fact Verification Models](https://www.aclweb.org/anthology/D19-1341.pdf)
> See in Commonsense Reasoning section.

:heavy_minus_sign: [MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims](https://www.aclweb.org/anthology/D19-1475.pdf)
>

## [Fake News](#contents)

There are a few fake-news related papers from EMNLP 2019 that could go in here.

See also:

- [A Stylometric Inquiry into Hyperpartisan and Fake News, ACL 2018](https://www.aclweb.org/anthology/P18-1022.pdf) - This has a taxonomy of fake news detection techniques and references to related work.
- Robert Dale, NLP in a post-truth world. Natural Language Engineering, 2017
- Papers by Victoria L. Rubin

## [Relation extraction and Knowledge graphs](#contents)

:boom: [Language Models as Knowledge Bases?](https://www.aclweb.org/anthology/D19-1250.pdf)
> Idea: maybe while learning linguistic knowledge, language models are also storing relational knowledge from the training data, and could be used to answer queries such as: "Dante was born in [MASK]". Results:
> 1. "Without fine-tuning, BERT contains relational knowledge competitive with traditional NLP methods that have some access to oracle knowledge".
> 2. "BERT does well on open-domain question answering against a supervised baseline."
> 3. "certain  types  of  factual  knowledge  are  learned much more readily than others by standard language model pretraining approaches."
>
> **Code:** https://github.com/facebookresearch/LAMA
>
>**See also:** a paper showing why using LMs as knowledge bases is a bad idea: [Negated LAMA: Birds cannot fly](https://arxiv.org/pdf/1911.03343.pdf). Also: [A Simple Method for Commonsense Reasoning](https://arxiv.org/abs/1806.02847); [Don’t SayThat! Making Inconsistent Dialogue Unlikely with Unlikelihood Training](https://arxiv.org/pdf/1911.03860.pdf)

:heavy_minus_sign: [Automatic **Taxonomy Induction and Expansion**](https://www.aclweb.org/anthology/D19-3005.pdf)
> (Demo paper). Demo: https://ibm.box.com/v/emnlp-2019-demo

**Papers using graph neural networks:**

:boom: [Learning to Update Knowledge Graphs by Reading News](https://www.aclweb.org/anthology/D19-1265.pdf)
> Abstract: "...  we propose a novel neural network method, GUpdater ...built upon graph neural networks (GNNs) with a text-based attention mechanism to guide the updating message passing through the KG structures. Experiments on a real-world KG updating dataset show that our model can effectively broadcast the news information to the KG structures and perform necessary link-adding or link-deleting operations to ensure the KG up-to-date according to news snippets."

:boom: [Incorporating Graph Attention Mechanism into Knowledge Graph Reasoning Based on Deep Reinforcement Learning](https://www.aclweb.org/anthology/D19-1264.pdf)
>

**Papers doing joint or multi-task learning for relation extraction and other things:**

:heavy_minus_sign: [Entity, Relation, and Event Extraction with Contextualized Span Representations](https://www.aclweb.org/anthology/D19-1585.pdf)
> "We examine the capabilities of a unified, multi-task framework for three information extraction tasks: named entity recognition, relation extraction, and event extraction." **Code:**: https://github.com/dwadden/dygiep

:heavy_minus_sign: [Improving Distantly-Supervised Relation Extraction with Joint Label Embedding](https://www.aclweb.org/anthology/D19-1395.pdf)
> "... we propose a novel multi-layer attention-based model to improve relation extraction with joint label embedding."

**Papers on knowledge graph completion:**

:heavy_minus_sign: [TuckER: Tensor Factorization for Knowledge Graph Completion](https://www.aclweb.org/anthology/D19-1522.pdf)
> Abstract: "... Link prediction is a task of inferring missing facts based on existing ones. We propose TuckER, a relatively **straightforward but powerful linear model based on Tucker decomposition of the binary tensor representation of knowledge graph triples**. TuckER outperforms previous state-of-the-art models across standard link prediction datasets, acting as a strong baseline for more elaborate models. We show that TuckER is a fully expressive model, derive sufficient bounds on its embedding dimensionalities and demonstrate that **several previously introduced linear models can be viewed as special cases of TuckER.**"  Datasets evaluated: FB15k, FB15k-237, WN18 and WN18RR.

:heavy_minus_sign: [Representation Learning with Ordered Relation Paths for Knowledge Graph Completion](https://www.aclweb.org/anthology/D19-1268.pdf)
>

:heavy_minus_sign: [Tackling Long-Tailed Relations and Uncommon Entities in Knowledge Graph Completion](https://www.aclweb.org/anthology/D19-1024.pdf)
>

:heavy_minus_sign: [Using Pairwise Occurrence Information to Improve Knowledge Graph Completion on Large-Scale Datasets](https://www.aclweb.org/anthology/D19-1368.pdf)
>

**Other papers:**

:heavy_minus_sign: [Collaborative Policy Learning for Open Knowledge Graph Reasoning](https://www.aclweb.org/anthology/D19-1269.pdf)
>

:heavy_minus_sign: [DIVINE: A Generative Adversarial Imitation Learning Framework for Knowledge Graph Reasoning](https://www.aclweb.org/anthology/D19-1266.pdf)
>

:heavy_minus_sign: [A Non-commutative Bilinear Model for Answering Path Queries in Knowledge Graphs](https://www.aclweb.org/anthology/D19-1246.pdf)
>

:heavy_minus_sign: [Commonsense Knowledge Mining from Pretrained Models](https://www.aclweb.org/anthology/D19-1109.pdf)
>

:heavy_minus_sign: [KnowledgeNet: A Benchmark Dataset for Knowledge Base Population](https://www.aclweb.org/anthology/D19-1069/)
>

:heavy_minus_sign: [CaRe: Open Knowledge Graph Embeddings](https://www.aclweb.org/anthology/D19-1036.pdf)
>

:heavy_minus_sign: [Supervising Unsupervised Open Information Extraction Models](https://www.aclweb.org/anthology/D19-1067.pdf)
>

:heavy_minus_sign: [Open Relation Extraction: Relational Knowledge Transfer from Supervised Data to Unsupervised Data](https://www.aclweb.org/anthology/D19-1021.pdf)
>


## [Commonsense Reasoning](#contents)

:boom: [Counterfactual Story Reasoning and Generation](https://www.aclweb.org/anthology/D19-1509.pdf)
>

:boom: [Incorporating Domain Knowledge into Medical NLI using Knowledge Graphs](https://www.aclweb.org/anthology/D19-1631.pdf)
>

:boom: [Towards Debiasing Fact Verification Models](https://www.aclweb.org/anthology/D19-1341.pdf)
> Shows that SOTA models for FEVER are in fact relying on artifacts from the data, since claim-only classifiers are competitive with top evidence & claim models. These models are very good at learning from artifacts. The authors suggest a way to create an "unbiased" evaluation dataset based on FEVER, where SOTA performance is much lower (Table 3).
>
> **Data and code:** https://github.com/TalSchuster/FeverSymmetric
>
> **See also:** [Niven and Kao, Probing Neural Network Comprehension of Natural Language Arguments, ACL 2019](https://www.aclweb.org/anthology/P19-1459/) and https://thegradient.pub/nlps-clever-hans-moment-has-arrived/

:heavy_minus_sign: [KagNet: Knowledge-Aware Graph Networks for Commonsense Reasoning](https://www.aclweb.org/anthology/D19-1282.pdf)
>

:heavy_minus_sign: [QUARTZ: An Open-Domain Dataset of Qualitative Relationship Questions](https://www.aclweb.org/anthology/D19-1608.pdf)
>

:heavy_minus_sign: [Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension](https://www.aclweb.org/anthology/D19-1609.pdf)
>

:heavy_minus_sign: [WIQA: A dataset for 'What if...' reasoning over procedural text](https://www.aclweb.org/anthology/D19-1629.pdf)
>

:heavy_minus_sign: [Do Nuclear Submarines Have Nuclear Captains? A Challenge Dataset for Commonsense Reasoning over Adjectives and Objects](https://www.aclweb.org/anthology/D19-1625.pdf)
>

:heavy_minus_sign: [Posing Fair Generalization Tasks for Natural Language Inference](https://www.aclweb.org/anthology/D19-1456.pdf)
>

:heavy_minus_sign: [Self-Assembling Modular Networks for **Interpretable** Multi-Hop Reasoning](https://www.aclweb.org/anthology/D19-1455.pdf)
>

:heavy_minus_sign: ["Going on a vacation" takes longer than "Going for a walk": A Study of Temporal Commonsense Understanding](https://www.aclweb.org/anthology/D19-1332.pdf)
>

:heavy_minus_sign: [Answering questions by learning to rank -Learning to rank by answering questions](https://www.aclweb.org/anthology/D19-1256.pdf)
>

:heavy_minus_sign: [Finding Generalizable Evidence by Learning to Convince Q&A Models](https://www.aclweb.org/anthology/D19-1244.pdf)
>

:heavy_minus_sign: [Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning over Few-Shot Relations](https://www.aclweb.org/anthology/D19-1334.pdf)
>

:heavy_minus_sign: [How Reasonable are Common-Sense Reasoning Tasks: A Case-Study on the Winograd Schema Challenge and SWAG](https://www.aclweb.org/anthology/D19-1335.pdf)
>

:heavy_minus_sign: [COSMOSQA: Machine Reading Comprehensionwith Contextual Commonsense Reasoning](https://www.aclweb.org/anthology/D19-1243.pdf)
>

:heavy_minus_sign: [What’s Missing: A Knowledge Gap Guided Approachfor Multi-hop Question Answering](https://www.aclweb.org/anthology/D19-1281.pdf)
> "given partial knowledge, explicitly  identifying  what’s  missing  substantially outperforms previous approaches".

:heavy_minus_sign: [On the Importance of Delexicalization for Fact Verification](https://www.aclweb.org/anthology/D19-1340.pdf)
>

## [Information Retrieval](#contents)

:heavy_minus_sign: [Bridging the Gap Between Relevance Matching and Semantic Matching for Short Text Similarity Modeling](https://www.aclweb.org/anthology/D19-1540.pdf)
>

:heavy_minus_sign: [Cross-Domain Modeling of Sentence-Level Evidence for Document Retrieval](https://www.aclweb.org/anthology/D19-1352.pdf)
>

:heavy_minus_sign: [Applying BERT to Document Retrieval with Birch](https://www.aclweb.org/anthology/D19-3004.pdf)
>

:heavy_minus_sign: [Modelling Stopping Criteria for Search Results using Poisson Processes](https://www.aclweb.org/anthology/D19-1351.pdf)
>

## [Entity Linking](#contents)

:heavy_minus_sign: [Fine-Grained Evaluation for Entity Linking](https://www.aclweb.org/anthology/D19-1066.pdf)
>

## [Entities and NER](#contents)

:boom: [ner and pos when nothing is capitalized](https://www.aclweb.org/anthology/D19-1650.pdf)
>

:heavy_minus_sign: [A Little Annotation does a Lot of Good: A Study in Bootstrapping **Low-resource Named Entity Recognizers**](https://www.aclweb.org/anthology/D19-1520.pdf)
> Abstract: "... What is the most effective method for efficiently creating high-quality entity recognizers in under-resourced languages? Based on extensive experimentation using both simulated and real human annotation, we settle on a recipe of starting with a cross-lingual transferred model, then performing targeted annotation of only uncertain entity spans in the target language, minimizing annotator effort. Results demonstrate that cross-lingual transfer is a powerful tool when very little data can be annotated, but an entity-targeted annotation strategy can achieve competitive accuracy quickly, with just one-tenth of training data."

:heavy_minus_sign: [Fine-Grained Entity Typing via Hierarchical Multi Graph Convolutional Networks](https://www.aclweb.org/anthology/D19-1502.pdf)
>

:heavy_minus_sign: [An Attentive Fine-Grained Entity Typing Model with Latent Type Representation](https://www.aclweb.org/anthology/D19-1641.pdf)
>

:heavy_minus_sign: [Hierarchically-Refined Label Attention Network for Sequence Labeling](https://www.aclweb.org/anthology/D19-1422.pdf)
>

:heavy_minus_sign: [Feature-Dependent Confusion Matrices for Low-Resource NER Labeling with Noisy Labels](https://www.aclweb.org/anthology/D19-1362.pdf)
>

:heavy_minus_sign: [Small and Practical BERT Models for Sequence Labeling](https://www.aclweb.org/anthology/D19-1374.pdf)
> Paper on **Multilingual sequence labeling**.

:heavy_minus_sign: [Hierarchical Meta-Embeddings for Code-Switching Named Entity Recognition](https://www.aclweb.org/anthology/D19-1360.pdf)
>

:heavy_minus_sign: [EntEval: A Holistic Evaluation Benchmark for **Entity Representations**](https://www.aclweb.org/anthology/D19-1040.pdf)
>

:heavy_minus_sign: [A Boundary-aware Neural Model for **Nested Named Entity Recognition**](https://www.aclweb.org/anthology/D19-1034.pdf)
>

## [Coreference](#contents)

Two new coreference-related datasets:

:heavy_minus_sign: [WikiCREM: A Large Unsupervised Corpus for Coreference Resolution](https://www.aclweb.org/anthology/D19-1439.pdf)
> Abstract:"... In this work, we introduce WikiCREM (Wikipedia CoREferences Masked) a large-scale, yet accurate dataset of pronoun disambiguation instances. We use a language-model-based approach for pronoun resolution in combination with our WikiCREM dataset. We compare a series of models on a collection of diverse and challenging coreference resolution problems, where we match or outperform previous state-of-the-art approaches on 6 out of 7 datasets, such as GAP, DPR, WNLI, PDP, WinoBias, and WinoGender. We release our model to be used off-the-shelf for solving pronoun disambiguation."
>
> **Code:** https://github.com/vid-koci/bert-commonsense ; **Dataset and models:** https://ora.ox.ac.uk/objects/uuid:c83e94bb-7584-41a1-aef9-85b0e764d9e3

:heavy_minus_sign: [Quoref: A Reading Comprehension Dataset with Questions Requiring Coreferential Reasoning](https://www.aclweb.org/anthology/D19-1606.pdf)
>"... most current reading comprehension benchmarks do not contain complex coreferential phenomena and hence fail to evaluate the ability of models to resolve coreference. We present a new crowdsourced dataset containing more than 24K span-selection questions that require resolving coreference among entities in over 4.7K English paragraphs from Wikipedia. Obtaining questions focused on such phenomena is challenging, because it is hard to avoid lexical cues that shortcut complex reasoning. We deal with this issue by using a strong baseline model as an adversary in the crowdsourcing loop, which helps crowdworkers avoid writing questions with exploitable surface cues. We show that state-of-the-art reading comprehension models perform significantly worse than humans on this benchmark—the best model performance is 70.5 F1, while the estimated human performance is 93.4 F1."

And two more papers using BERT for coreference:

:heavy_minus_sign: [BERT for Coreference Resolution: Baselines and Analysis](https://www.aclweb.org/anthology/D19-1588.pdf)

:heavy_minus_sign: [Coreference Resolution in Full Text Articles with BERT and Syntax-based Mention Filtering](https://www.aclweb.org/anthology/D19-5727.pdf)

## [Text classification](#contents)

:boom: [Rethinking Attribute Representation and Injection for Sentiment Classification](https://www.aclweb.org/anthology/D19-1562.pdf)
> **Explores different ways to incorporate metadata** (e.g., user or product information for sentiment classification) into text classifiers with attention. From the abstract: "The de facto standard method is to incorporate them as additional biases in the  attention mechanism ... we  show that the above method is the least effective way to represent and inject attributes." The authors test this using a BiLSTM with attention classifier, and test various places for incorporating attributes into the model (embedding, encoding, attention, classifier).
>
> **Code and data:** https://github.com/rktamplayo/CHIM

:boom: [**Label Embedding using Hierarchical Structure of Labels** for Twitter Classification](https://www.aclweb.org/anthology/D19-1660.pdf)
> Tweets can be classified into pre-definited classes, but these classes can be put into a hierarchical structure. ".. we propose a method that can consider the hierarchical structure of labels and label texts themselves. We conducted evaluation over the Text REtrieval Conference (TREC) 2018 Incident Streams (IS) track dataset, and we found that our method outperformed the methods of the conference participants." **Note:** it seems the hierarchy is very shallow (two layers).

:heavy_minus_sign: [Learning Only from Relevant Keywords and Unlabeled Documents](https://www.aclweb.org/anthology/D19-1411.pdf)
> Task: document classification where no labels are available for training, and only the positive class has a list of keywords associated with the class.  This is a variant of the "dataless classification task" (this has also been called "lightly-supervised one-class classification"). This paper proposes a  "theoretically guaranteed learning framework" for this problem "that is simple to implement and has flexible choices of models".

:heavy_minus_sign: [Heterogeneous **Graph Attention Networks** for **Semi-supervised** Short Text Classification](https://www.aclweb.org/anthology/D19-1488.pdf)
>

:heavy_minus_sign: [Learning Explicit and Implicit Structures for Targeted Sentiment Analysis](https://www.aclweb.org/anthology/D19-1550.pdf)
>

:heavy_minus_sign: [Sequential Learning of Convolutional Features for Effective Text Classification](https://www.aclweb.org/anthology/D19-1567.pdf)
>

:heavy_minus_sign: [Text Level Graph Neural Network for Text Classification](https://www.aclweb.org/anthology/D19-1345.pdf)
>

:heavy_minus_sign: [Delta-training: Simple Semi-Supervised Text Classification using Pretrained Word Embeddings](https://www.aclweb.org/anthology/D19-1347.pdf)
>

## [Propaganda, Persuasion](#contents)

**See also:** Workshop "NLP4IF: Censorship, Disinformation and Propaganda"

:heavy_minus_sign: [Fine-Grained Analysis of Propaganda in News Articles](https://www.aclweb.org/anthology/D19-1565.pdf)
> Introduces a corpus of news articles annotated (at fragment level) with 18 propaganda techniques. A "multi-granularity" network based on BERT achieves the best performance on the task.
> 
> See also: https://propaganda.qcri.org/ , the Fine Grained Propaganda Detection competition at the NLP4IF’19 workshop, and Semeval 2020 Task 11 (https://propaganda.qcri.org/semeval2020-task11/).

:heavy_minus_sign: [Revealing and Predicting Online Persuasion Strategy with Elementary Units](https://www.aclweb.org/anthology/D19-1653.pdf)
>

## [Social Media and Authorship Attribution](#contents)

:heavy_minus_sign: [The Trumpiest Trump? Identifying a Subject’s Most Characteristic Tweets](https://www.aclweb.org/anthology/D19-1175.pdf)
> How to quantify the degree to which tweets are characteristic of their authors (probability a given tweet was written by its author)? This is done via binary classification, and then calculating the probability using the distance from the decision boundary. One possible application of this: to highlight unusual/suspicious activity (e.g., an account was hacked).  This is studied using the tweets of 15 celebrities (including Oprah, JK Rowling, LeBron James, Justin Bieber, Obama and Trump).
>
> Five approaches to authorship identification are tested; in order of increasing performance these are: (1) a compression-based method using Lempel-Ziv-Welch, (2) LDA topic model features used in Logistic Regression or MLP, (3) an n-gram based language model to score the probability of a tweet given positive or negative ngrams, (4) two different document embeddings -- using fastText and BERT -- fed into a linear classifier, and (5) BERT token embeddings fed to an LSTM. The BERT+LSTM approach yields the highest accuracy of (on average across celebrities) 90.37%.
>
> The paper also finds a strong correlation between "characterstic-ness" and tweet popularity for 13 of the 15 celebrities in the study; for some celebrities (Trump, Modi, Jimmy Fallon) the correlation is positive, whereas for others it is negative (Bieber, Kim Kardashian, and Obama).

:heavy_minus_sign: [Learning Invariant Representations of Social Media Users](https://www.aclweb.org/anthology/D19-1178.pdf)
>

:heavy_minus_sign: [You Shall Know a User by the Company It Keeps: Dynamic Representations for Social Media Users in NLP](https://www.aclweb.org/anthology/D19-1477.pdf)
>

:heavy_minus_sign: [A Hierarchical Location Prediction Neural Network for Twitter User Geolocation](https://www.aclweb.org/anthology/D19-1480.pdf)
>

## [Other](#contents)

:heavy_minus_sign: [Humor Detection: A Transformer Gets the Last Laugh](https://www.aclweb.org/anthology/D19-1372.pdf)
> Task: detecting whether a joke is humorous. Trains a model to identify humorous jokes using the r/Jokes thread. A Transformer architecture outperforms previous work on humor identification tasks and is comparable to human performance.

:heavy_minus_sign: [TalkDown: A Corpus for Condescension Detection in Context](https://www.aclweb.org/anthology/D19-1385.pdf)
> Goal: to detect condescending language (e.g., "Are you struggling with this whole English language thing?"). Introduces TalkDown, a new labeled dataset of condescending linguistic acts **in context**. Including discourse representations improves performance of a language-only model. Of course, the models are initialized with BERT.

:heavy_minus_sign: [Movie Plot Analysis via Turning Point Identification](https://www.aclweb.org/anthology/D19-1180.pdf)
> "We  introduce  a  dataset  consisting of screenplays and plot synopses annotated with turning points and present an end-to-end neural network  model  that  identifies turning points in plot synopses and projects them onto scenes in screenplays". **Data:** https://github.com/ppapalampidi/TRIPOD

:heavy_minus_sign: [Context-Aware Conversation Thread Detection in Multi-Party Chat](https://www.aclweb.org/anthology/D19-1682.pdf)
> Task: to group messages in a multi-party chat by thread. This proposes a new model for the task.

# Organize these later:

:boom: [Using Local Knowledge Graph Construction to Scale Seq2Seq Models to Multi-Document Inputs](https://www.aclweb.org/anthology/D19-1428.pdf)
> Abstract: "... We propose **constructing a local graph structured knowledge base for each query**, which compresses the web search information and reduces redundancy. We show that by linearizing the graph into a structured input sequence, models can encode the graph representations within a standard Sequence-to-Sequence setting. For two generative tasks with very long text input, **long-form question answering** and **multi-document summarization**, feeding graph representations as input can achieve better performance than using retrieved text portions."

:boom: [Compositional Generalization for Primitive Substitutions](https://www.aclweb.org/anthology/D19-1438.pdf)
> Abstract: "Compositional generalization is a basic mechanism in human language learning, but current neural networks lack such ability. In this paper, we conduct fundamental research for encoding compositionality in neural networks. Conventional methods use a single representation for the input sentence, making it hard to apply prior knowledge of compositionality. In contrast, our approach leverages such knowledge with two representations, one generating attention maps, and the other mapping attended input words to output symbols. We reduce the entropy in each representation to improve generalization. Our experiments demonstrate significant improvements over the conventional methods in five NLP tasks including instruction learning and machine translation. In the SCAN domain, **it boosts accuracies from 14.0% to 98.8% in Jump task**, and from 92.0% to 99.7% in TurnLeft task. It also beats human performance on a few-shot learning task. We hope the proposed approach can help ease future research towards human-level compositional language learning."

:boom: [Parallel Iterative Edit Models for Local Sequence Transduction](https://www.aclweb.org/anthology/D19-1435.pdf)
> Abstract: "We present a Parallel Iterative Edit (PIE) model for the problem of local sequence transduction arising in tasks like Grammatical error correction (GEC). Recent approaches are based on the popular encoder-decoder (ED) model for sequence to sequence learning. The ED model auto-regressively captures full dependency among output tokens but is slow due to sequential decoding. The PIE model does parallel decoding, giving up the advantage of modeling full dependency in the output, yet it achieves accuracy competitive with the ED model for four reasons: 1. predicting edits instead of tokens, 2. labeling sequences instead of generating sequences, 3. iteratively refining predictions to capture dependencies, and 4. factorizing logits over edits and their token argument to harness pre-trained language models like BERT. Experiments on tasks spanning GEC, OCR correction and spell correction demonstrate that the PIE model is an accurate and significantly faster alternative for local sequence transduction."

:heavy_minus_sign: [A Logic-Driven Framework for Consistency of Neural Models](https://www.aclweb.org/anthology/D19-1405.pdf)
> Abstract: "While neural models show remarkable accuracy on individual predictions, their internal beliefs can be inconsistent across examples. In this paper, we formalize such inconsistency as a generalization of prediction error. We propose a learning framework for constraining models using logic rules to regularize them away from inconsistency. Our framework can leverage both labeled and unlabeled examples and is directly compatible with off-the-shelf learning schemes without model redesign. We instantiate our framework on natural language inference, where experiments show that enforcing invariants stated in logic can help make the predictions of neural models both accurate and consistent."

:heavy_minus_sign: [Transductive Learning of Neural Language Models for Syntactic and Semantic Analysis](https://www.aclweb.org/anthology/D19-1379.pdf)
> Abstract: "In transductive learning, an unlabeled test set is used for model training. Although this setting deviates from the common assumption of a completely unseen test set, it is applicable in many real-world scenarios, wherein the texts to be processed are known in advance. However, despite its practical advantages, transductive learning is underexplored in natural language processing. Here we conduct an empirical study of transductive learning for neural models and demonstrate its utility in syntactic and semantic tasks. Specifically, we fine-tune language models (LMs) on an unlabeled test set to obtain test-set-specific word representations. Through extensive experiments, we demonstrate that despite its simplicity, transductive LM fine-tuning consistently improves state-of-the-art neural models in in-domain and out-of-domain settings."

:heavy_minus_sign: [Neural Gaussian Copula for Variational Autoencoder](https://www.aclweb.org/anthology/D19-1442.pdf)
> Abstract: "Variational language models seek to estimate the posterior of latent variables with an approximated variational posterior. The model often assumes the variational posterior to be factorized even when the true posterior is not. The learned variational posterior under this assumption does not capture the dependency relationships over latent variables. We argue that this would cause a typical training problem called posterior collapse observed in all other variational language models. We propose Gaussian Copula Variational Autoencoder (VAE) to avert this problem. Copula is widely used to model correlation and dependencies of high-dimensional random variables, and therefore it is helpful to maintain the dependency relationships that are lost in VAE. The empirical results show that by modeling the correlation of latent variables explicitly using a neural parametric copula, we can avert this training difficulty while getting competitive results among all other VAE approaches."







