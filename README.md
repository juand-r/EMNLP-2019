# EMNLP 2019

This contains summaries for a selection of papers from EMNLP 2019. What is missing: many cross-lingual papers and multi-modal papers, as well as papers on machine translation, dialogue, question answering, intent classification, sentiment/emotion, stance detection, argument mining, SQL and code generation, code-switching, rumors, question generation, humor, bias and hate speech...

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
      - [Domain Adaptation](#domain-adaptation)
      - [Meta-learning](#meta-learning)
      - [Few-shot and zero-shot text classification](few-shot-and-zero-shot-text-classification)
      - [Active Learning](#active-learning)
    - [Word Embeddings](#word-embeddings)
    - [Knowledge enhanced ML](#knowledge-enhanced-ml)
    - [Multimodal](#multimodal)
  - [NLG Tasks](#nlg-tasks)
    - [Summarization and Simplification](#summarization-and-simplification)
    - [Style transfer](#style-transfer)
    - [Text generation and GPT2](#text-generation-and-gpt2)
    - [Machine Translation](#machine-translation)
  - [NLU Tasks](#nlu-tasks)
    - [WSD](#wsd)
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

:heavy_minus_sign: [A Biomedical Free Text Annotation Interface with Active Learning and Research Use Case Specific Customisation](https://www.aclweb.org/anthology/D19-3024.pdf)
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

:heavy_minus_sign: [VerbAtlas: a Novel Large-Scale **Verbal Semantic Resource** and Its Application to Semantic Role Labeling](https://www.aclweb.org/anthology/D19-1058.pdf)
> Introduces a hand-crafted lexical resource collecting all verb synsets of WordNet into "semantically-coherent frames". Available at http://verbatlas.org

:heavy_minus_sign: [Automatic **Argument Quality Assessment** - New Datasets and Methods](https://www.aclweb.org/anthology/D19-1564.pdf)
> 6.3k arguments, annotated for "quality". Data available at https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Argument%20Quality Two methods are introduced for the tasks of argument pair classification and argument ranking, both based on BERT.

:heavy_minus_sign: [The Role of Pragmatic and Discourse Context in Determining **Argument Impact**](https://www.aclweb.org/anthology/D19-1568.pdf)
> Introduces a new dataset to study the effect of "pragmatic and discourse context" (*kairos*) on determining argument impact (persuasive power). The dataset contains arguments covering 741 topics and over 47k claims. Models that incorporate the pragmatic and discourse context (i.e., sequences of  of claims which support or oppose other claims in order to defend a certain thesis) outperform models which only rely on claim-specific features.

# [Topics](#contents)

## [Methodological](#contents)

**See also:**
- Gorman, Kyle, and Steven Bedrick. We need to talk about standard splits. ACL 2019.
- Dror, Shlomov and Reichart. Deep Dominance - How to Properly Compare Deep Neural Models. ACL 2019.

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
> See also recent papers by Hanna Wallach on human evaluations of interpretability methods:
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


:boom: [Topics to Avoid: Demoting Latent Confounds in Text Classification](https://www.aclweb.org/anthology/D19-1425.pdf)
> On the **native language identification** task, classifiers often do not generalize well because they learn topical features which are artifacts (superficial patterns); for example if the label "Swedish" is predicted due to the word 'Sweden' appearing in the text. A model is proposed which predicts both the label of the text and the confound; the two predictors are trained adversarially in order to learn a text representation that predicts the correct label but does not use information from the confound.
>
> Check - how similar is this to McHardy, Adel and Klinger, Adversarial Training for Satire Detection: Controlling for Confounding Variables, NAACL 2019?

:boom: [Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases](https://www.aclweb.org/anthology/D19-1418.pdf)


:heavy_minus_sign: [Adversarial Reprogramming of Text Classification Neural Networks](https://www.aclweb.org/anthology/D19-1525.pdf)


:heavy_minus_sign: [Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation](https://www.aclweb.org/anthology/D19-1419.pdf)


:heavy_minus_sign: [Certified Robustness to Adversarial Word Substitutions](https://www.aclweb.org/anthology/D19-1423.pdf)


## [Learning with Less Data](#contents)

:heavy_minus_sign: [FewRel 2.0: Towards More Challenging Few-Shot Relation Classification](https://www.aclweb.org/anthology/D19-1649.pdf)


:heavy_minus_sign: [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://www.aclweb.org/anthology/D19-1670.pdf)


:heavy_minus_sign: [A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-resource Named Entity Recognizers](https://www.aclweb.org/anthology/D19-1520.pdf)


:heavy_minus_sign: [Learning to Bootstrap for Entity Set Expansion](https://www.aclweb.org/anthology/D19-1028.pdf)


:heavy_minus_sign: [Latent-Variable Generative Models for Data-Efficient **Text Classification**](https://www.aclweb.org/anthology/D19-1048.pdf)


:heavy_minus_sign: [**Transfer Learning** Between Related Tasks Using Expected Label Proportions](https://www.aclweb.org/anthology/D19-1004.pdf)

### [Domain Adaptation](#contents)

:boom: [To Annotate or Not? **Predicting Performance Drop under Domain Shift**](https://www.aclweb.org/anthology/D19-1222.pdf)


:boom: [**Unsupervised Domain Adaptation of Contextualized Embeddings for Sequence Labeling**](https://www.aclweb.org/anthology/D19-1433.pdf)


:boom: [**Out-of-Domain Detection** for **Low-Resource** Text Classification Tasks](https://www.aclweb.org/anthology/D19-1364.pdf)


:heavy_minus_sign: [Shallow Domain Adaptive Embeddings for Sentiment Analysis](https://www.aclweb.org/anthology/D19-1557.pdf)


:heavy_minus_sign: [Domain-Invariant Feature Distillation for Cross-Domain Sentiment Classification](https://www.aclweb.org/anthology/D19-1558.pdf)


:heavy_minus_sign: [Adaptive Ensembling: Unsupervised Domain Adaptation for Political Document Analysis](https://www.aclweb.org/anthology/D19-1478.pdf)


:heavy_minus_sign: [Weakly Supervised Domain Detection. Yumo Xu and Mirella Lapata (TACL)](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00287)

### [Meta-learning](#contents)

:boom: [Investigating Meta-Learning Algorithms for Low-Resource Natural Language Understanding Tasks](https://www.aclweb.org/anthology/D19-1112.pdf)


:heavy_minus_sign: [Learning to Learn and Predict: A Meta-Learning Approach for Multi-Label Classification](https://www.aclweb.org/anthology/D19-1444.pdf)


:heavy_minus_sign: [Text Emotion Distribution Learning from Small Sample: A Meta-Learning Approach](https://www.aclweb.org/anthology/D19-1408.pdf)


### [Few-shot and zero-shot text classification](#contents)

:boom: [Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach](https://www.aclweb.org/anthology/D19-1404.pdf)


:heavy_minus_sign: [Look-up and Adapt: A One-shot Semantic Parser](https://www.aclweb.org/anthology/D19-1104.pdf)


:heavy_minus_sign: [Induction Networks for Few-Shot Text Classification](https://www.aclweb.org/anthology/D19-1403.pdf)


:heavy_minus_sign: [Hierarchical Attention Prototypical Networks for **Few-Shot Text Classification**](https://www.aclweb.org/anthology/D19-1042.pdf)

### [Active learning](#contents)

:boom: [Practical Obstacles to Deploying **Active Learning**](https://www.aclweb.org/anthology/D19-1003.pdf)


:heavy_minus_sign: [Sampling Bias in Deep Active Classification: An Empirical Study](https://www.aclweb.org/anthology/D19-1417.pdf)


## [Word Embeddings](#contents)

:boom: [Do NLP Models Know Numbers? Probing Numeracy in Embeddings](https://www.aclweb.org/anthology/D19-1534.pdf)


:boom: **Best paper award** [Specializing Word Embeddings (for Parsing) by Information Bottleneck](https://www.aclweb.org/anthology/D19-1276.pdf)


:heavy_minus_sign: [Feature2Vec: Distributional semantic modelling of human property knowledge](https://www.aclweb.org/anthology/D19-1595.pdf)


:heavy_minus_sign: [Multiplex Word Embeddings for **Selectional Preference Acquisition**](https://www.aclweb.org/anthology/D19-1528.pdf)


:heavy_minus_sign: [Text-based inference of moral sentiment change](https://www.aclweb.org/anthology/D19-1472.pdf)
> Uses diachronic word embedding to explore moral biases over time (e.g. *slavery*).


:heavy_minus_sign: [Still a Pain in the Neck: Evaluating Text Representations on Lexical Composition (TACL?)](https://arxiv.org/abs/1902.10618)


:heavy_minus_sign: [Rotate King to get Queen: Word Relationships as Orthogonal Transformations in Embedding Space](https://www.aclweb.org/anthology/D19-1354.pdf)


:heavy_minus_sign: [Correlations between Word Vector Sets](https://www.aclweb.org/anthology/D19-1008.pdf)


:heavy_minus_sign: [Game Theory Meets Embeddings (paper on WSD)](https://www.aclweb.org/anthology/D19-1009.pdf)


:heavy_minus_sign: [Parameter-free Sentence Embedding via Orthogonal Basis](https://www.aclweb.org/anthology/D19-1059.pdf)

## [Knowledge enhanced ML](#contents)

:heavy_minus_sign: [**Knowledge-Enriched** Transformer for Emotion Detection in Textual Conversations](https://www.aclweb.org/anthology/D19-1016.pdf)


:heavy_minus_sign: [**Knowledge Enhanced** Contextual Word Representations](https://www.aclweb.org/anthology/D19-1005.pdf)


:heavy_minus_sign: [Improving Relation Extraction with **Knowledge-attention**](https://www.aclweb.org/anthology/D19-1022.pdf)


## [Multimodal](#contents)

:boom: [Integrating Text and Image: Determining Multimodal Document Intent in Instagram Posts](https://www.aclweb.org/anthology/D19-1469.pdf)
> Introduces a multimodal dataset of Instagram posts labeled with author intent and image-text relationships. A baseline deep multimodal classifier is used to validate the proposed taxonomy, and using both the text and the image improves intent detection by almost 10% (over using only the image). Intents include: advocative, promotive, exhibitionist, expressive, informative, entertainment and provocative.

> See also at EMNLP 2019: "Dialog Intent Induction with Deep Multi-View Clustering".

:heavy_minus_sign: [Partners in Crime: Multi-view Sequential Inference for Movie Understanding](https://www.aclweb.org/anthology/D19-1212.pdf)
> "In this paper, we propose a neural architecture coupled with a novel training objective that **integrates multi-view  information for sequence prediction problems.**".  This is used in an incremental inference setup (i.e., making predictions without encoding the full sequence), which better mimics how humans process information. This architecture is tested on three tasks derived from the Frermann et al. (2018) dataset episodes from the CSI TV show:
> - perpetrator mention identification (task from Frermann et al. (2018))
> - crime case segmentation (some episodes alternate between more than one crime case)
> - speaker type tagging (each utterance can come from either a detective, perpretator, suspect, or extra).

# [NLG Tasks](#contents)

:heavy_minus_sign: [Text2Math: End-to-end Parsing Text into Math Expressions](https://www.aclweb.org/anthology/D19-1536.pdf)

## [Summarization and Simplification](#contents)

### Summarization

:heavy_minus_sign: [Deep Reinforcement Learning with Distributional Semantic Rewardsfor Abstractive Summarization](https://www.aclweb.org/anthology/D19-1623.pdf)


:heavy_minus_sign: [Abstract Text Summarization: A Low Resource Challenge](https://www.aclweb.org/anthology/D19-1616.pdf)


:heavy_minus_sign: [Neural Text Summarization: A Critical Evaluation](https://www.aclweb.org/anthology/D19-1051.pdf)


:heavy_minus_sign: [BottleSum: Unsupervised and Self-supervised Sentence Summarization using the Information Bottleneck Principle](https://www.aclweb.org/anthology/D19-1389.pdf)


:heavy_minus_sign: [How to Write Summaries with Patterns? Learning towards Abstractive Summarization through Prototype Editing](https://www.aclweb.org/anthology/D19-1388.pdf)


:heavy_minus_sign: [Text Summarization with Pretrained Encoders](https://www.aclweb.org/anthology/D19-1387.pdf)


:heavy_minus_sign: [Better Rewards Yield Better Summaries: Learning to Summarise **Without References**](https://www.aclweb.org/anthology/D19-1307.pdf)


:heavy_minus_sign: [Contrastive Attention Mechanism for Abstractive Sentence Summarization](https://www.aclweb.org/anthology/D19-1301.pdf)


:heavy_minus_sign: [A Summarization System for Scientific Documents](https://www.aclweb.org/anthology/D19-3036.pdf)


:heavy_minus_sign: [Deep copycat Networks for Text-to-Text Generation](https://www.aclweb.org/anthology/D19-1318.pdf)


### Simplification

:heavy_minus_sign: [EASSE: Easier Automatic Sentence Simplification Evaluation](https://www.aclweb.org/anthology/D19-3009.pdf)


:heavy_minus_sign: [Recursive Context-Aware Lexical Simplification](https://www.aclweb.org/anthology/D19-1491.pdf)


### Evaluation

:heavy_minus_sign: [Answers Unite! Unsupervised Metrics for Reinforced Summarization Models](https://www.aclweb.org/anthology/D19-1320.pdf)


:heavy_minus_sign: [SUM-QE: a BERT-based Summary Quality Estimation Mode](https://www.aclweb.org/anthology/D19-1618.pdf)


## [Style transfer](#contents)

:heavy_minus_sign: [Semi-supervised Text Style Transfer: Cross Projection in Latent Space](https://www.aclweb.org/anthology/D19-1499.pdf)


:heavy_minus_sign: [Learning to Flip the Sentiment of Reviews from Non-Parallel Corpora](https://www.aclweb.org/anthology/D19-1659.pdf)


:heavy_minus_sign: [Style Transfer for Texts: Retrain, Report Errors, Compare with Rewrites](https://www.aclweb.org/anthology/D19-1406.pdf)


:heavy_minus_sign: [Multiple Text Style Transfer by using Word-level Conditional Generative Adversarial Network with Two-Phase Training](https://www.aclweb.org/anthology/D19-1366.pdf)
> See "erase and replace"


:heavy_minus_sign: [Harnessing Pre-Trained Neural Networks with Rules for Formality Style Transfer](https://www.aclweb.org/anthology/D19-1365.pdf)


:heavy_minus_sign: [IMaT: Unsupervised Text Attribute Transfer via Iterative Matching and Translation](https://www.aclweb.org/anthology/D19-1306.pdf)


:heavy_minus_sign: [Transforming Delete, Retrieve, Generate Approach for Controlled Text Style Transfer](https://www.aclweb.org/anthology/D19-1322.pdf)
> What to do if you don't have parallel corpora?

:heavy_minus_sign: [Domain Adaptive Text Style Transfer](https://www.aclweb.org/anthology/D19-1325.pdf)

## [Text generation and GPT2](#contents)

:boom: [Neural Naturalist: Generating Fine-Grained Image Comparisons](https://www.aclweb.org/anthology/D19-1065.pdf)


:boom: [Generating Natural Anagrams: Towards Language Generation Under Hard Combinatorial Constraints](https://www.aclweb.org/anthology/D19-1674.pdf)


:boom: [Denoising based Sequence-to-Sequence Pre-training for Text Generation](https://www.aclweb.org/anthology/D19-1412.pdf)


:boom: [Judge the Judges: A Large-Scale Evaluation Study of Neural Language Models for Online Review Generation](https://www.aclweb.org/anthology/D19-1409.pdf)


:boom: [Controlling Text Complexity in Neural Machine Translation](https://www.aclweb.org/anthology/D19-1166.pdf)


:boom: [See et al., Do Massively Pretrained Language Models Make Better Storytellers? (CoNLL)](https://www.aclweb.org/anthology/K19-1079.pdf)


:boom: [Attending to Future Tokens for Bidirectional Sequence Generation](https://www.aclweb.org/anthology/D19-1001.pdf)
> Can generate text from BERT!

:boom: [How Contextual are Contextualized Word Representations? Comparing **the Geometry of BERT, ELMo, and GPT-2 Embeddings**](https://www.aclweb.org/anthology/D19-1006.pdf)
> See also: Coenen et al., Visualizing and Measuring the Geometry of BERT (2019), (paper: https://arxiv.org/pdf/1906.02715.pdf ; blog post: https://pair-code.github.io/interpretability/bert-tree/).

:boom: [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://www.aclweb.org/anthology/D19-1053.pdf)


:boom: [Select and Attend: Towards Controllable Content Selection in Text Generation](https://www.aclweb.org/anthology/D19-1054.pdf)


:heavy_minus_sign: [Learning Rhyming Constraints using Structured Adversaries](https://www.aclweb.org/anthology/D19-1621.pdf)


:heavy_minus_sign: [An Empirical Comparison on Imitation Learning and Reinforcement Learning for Paraphrase Generation](https://www.aclweb.org/anthology/D19-1619.pdf)


:heavy_minus_sign: [Encode, Tag, Realize: High-Precision Text Editing](https://www.aclweb.org/anthology/D19-1510.pdf)


:heavy_minus_sign: [Towards Controllable and Personalized Review Generation](https://www.aclweb.org/anthology/D19-1319.pdf)


:heavy_minus_sign: [An End-to-End Generative Architecture for Paraphrase Generation](https://www.aclweb.org/anthology/D19-1309.pdf)


:heavy_minus_sign: [Enhancing Neural Data-To-Text Generation Models with External Background Knowledge](https://www.aclweb.org/anthology/D19-1299.pdf)


:heavy_minus_sign: [Autoregressive Text Generation Beyond Feedback Loops](https://www.aclweb.org/anthology/D19-1338.pdf)


:heavy_minus_sign: [Big Bidirectional Insertion Representations for Documents](https://www.aclweb.org/anthology/D19-5620.pdf)
> Insertion Transformer: well suited for long form text generation; this builds on it to produce BIRD for use in machine translation task.

:heavy_minus_sign: [Neural data-to-text generation: A comparison between pipeline and end-to-end architectures](https://www.aclweb.org/anthology/D19-1052.pdf)


:heavy_minus_sign: [Sentence-Level Content Planning and Style Specification for Neural Text Generation](https://www.aclweb.org/anthology/D19-1055.pdf)

## [Machine Translation](#contents)

:boom: [Synchronously Generating Two Languages with Interactive Decoding](https://www.aclweb.org/anthology/D19-1330.pdf)
> Simultaneously translating into two languages at once!

:boom: [Machine Translation With Weakly Paired Documents](https://www.aclweb.org/anthology/D19-1446.pdf)


:boom: [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://www.aclweb.org/anthology/D19-1331.pdf)


:heavy_minus_sign: [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://www.aclweb.org/anthology/D19-1633.pdf)


:heavy_minus_sign: [Unsupervised Domain Adaptation for Neural Machine Translation with Domain-Aware Feature Embeddings](https://www.aclweb.org/anthology/D19-1147.pdf)


:heavy_minus_sign: [Machine Translation for Machines: the Sentiment Classification Use Case](https://www.aclweb.org/anthology/D19-1140.pdf)


:heavy_minus_sign: [Iterative Dual Domain Adaptation for NMT](https://www.aclweb.org/anthology/D19-1078.pdf)


:heavy_minus_sign: [Self-Attention with Structural Position Representations](https://www.aclweb.org/anthology/D19-1145.pdf)


:heavy_minus_sign: [Encoders Help You Disambiguate Word Sensesin Neural Machine Translation](https://www.aclweb.org/anthology/D19-1149.pdf)


:heavy_minus_sign: [Simple, Scalable Adaptation for Neural Machine Translation](https://www.aclweb.org/anthology/D19-1165.pdf)


:heavy_minus_sign: [Hierarchical Modeling of Global Context for Document-Level Neural Machine Translation](https://www.aclweb.org/anthology/D19-1168.pdf)


:heavy_minus_sign: [Context-Aware Monolingual Repair for Neural Machine Translation](https://www.aclweb.org/anthology/D19-1081.pdf)

# [NLU Tasks](#contents)

## [WSD](#contents)

:heavy_minus_sign: [Improved Word Sense Disambiguation Using Pre-Trained Contextualized Word Representations](https://www.aclweb.org/anthology/D19-1533.pdf)


## [Keyphrase Extraction](#contents)

:heavy_minus_sign: [Open Domain Web Keyphrase Extraction Beyond Language Modeling](https://www.aclweb.org/anthology/D19-1521.pdf)


## [Fact checking](#contents)

:boom: [Towards Debiasing Fact Verification Models](https://www.aclweb.org/anthology/D19-1341.pdf)
> See in Commonsense Reasoning section.

:heavy_minus_sign: [MultiFC: A Real-World Multi-Domain Dataset for Evidence-Based Fact Checking of Claims](https://www.aclweb.org/anthology/D19-1475.pdf)


## [Fake News](#contents)

:heavy_minus_sign: [Different Absorption from the Same Sharing: Sifted Multi-task Learning for Fake News Detection](https://www.aclweb.org/anthology/D19-1471.pdf)


## [Relation extraction and Knowledge graphs](#contents)

:boom: [Automatic **Taxonomy Induction and Expansion**](https://www.aclweb.org/anthology/D19-3005.pdf)


:boom: [Learning to Update Knowledge Graphs by Reading News](https://www.aclweb.org/anthology/D19-1265.pdf)


:boom: [Incorporating Graph Attention Mechanism into Knowledge Graph Reasoning Based on Deep Reinforcement Learning](https://www.aclweb.org/anthology/D19-1264.pdf)


:boom: [Language Models as Knowledge Bases?](https://www.aclweb.org/anthology/D19-1250.pdf)
> See also: A Simple Method for Commonsense Reasoning (https://arxiv.org/abs/1806.02847)

:heavy_minus_sign: [Entity, Relation, and Event Extractionwith Contextualized Span Representations](https://www.aclweb.org/anthology/D19-1585.pdf)


:heavy_minus_sign: [TuckER: Tensor Factorization for Knowledge Graph Completion](https://www.aclweb.org/anthology/D19-1522.pdf)


:heavy_minus_sign: [Improving Distantly-Supervised Relation Extraction withJoint Label Embedding](https://www.aclweb.org/anthology/D19-1395.pdf)


:heavy_minus_sign: [Representation Learning with Ordered Relation Pathsfor Knowledge Graph Completion](https://www.aclweb.org/anthology/D19-1268.pdf)

:heavy_minus_sign: [Collaborative Policy Learning for Open Knowledge Graph Reasoning](https://www.aclweb.org/anthology/D19-1269.pdf)


:heavy_minus_sign: [DIVINE: A Generative Adversarial Imitation Learning Framework for Knowledge Graph Reasoning](https://www.aclweb.org/anthology/D19-1266.pdf)


:heavy_minus_sign: [A Non-commutative Bilinear Model for Answering Path Queries in Knowledge Graphs](https://www.aclweb.org/anthology/D19-1246.pdf)


:heavy_minus_sign: [Commonsense Knowledge Mining from Pretrained Models](https://www.aclweb.org/anthology/D19-1109.pdf)


:heavy_minus_sign: [KnowledgeNet: A Benchmark Dataset for Knowledge Base Population](https://www.aclweb.org/anthology/D19-1069/)


:heavy_minus_sign: [CaRe: Open Knowledge Graph Embeddings](https://www.aclweb.org/anthology/D19-1036.pdf)


:heavy_minus_sign: [Supervising Unsupervised Open Information Extraction Models](https://www.aclweb.org/anthology/D19-1067.pdf)


:heavy_minus_sign: [Open Relation Extraction: Relational Knowledge Transfer from Supervised Data to Unsupervised Data](https://www.aclweb.org/anthology/D19-1021.pdf)

:heavy_minus_sign: [Multi-Input Multi-Output Sequence Labeling for **Joint Extraction of Fact and Condition Tuples** from Scientific Text](https://www.aclweb.org/anthology/D19-1029.pdf)

:heavy_minus_sign: [Tackling Long-Tailed Relations and Uncommon Entities in Knowledge Graph Completion](https://www.aclweb.org/anthology/D19-1024.pdf)

## [Commonsense Reasoning](#contents)

:boom: [Counterfactual Story Reasoning and Generation](https://www.aclweb.org/anthology/D19-1509.pdf)

:boom: [Incorporating Domain Knowledge into Medical NLI using Knowledge Graphs](https://www.aclweb.org/anthology/D19-1631.pdf)

:boom::boom: [Towards Debiasing Fact Verification Models](https://www.aclweb.org/anthology/D19-1341.pdf)

:heavy_minus_sign: [QUARTZ: An Open-Domain Dataset of Qualitative Relationship Questions](https://www.aclweb.org/anthology/D19-1608.pdf)

:heavy_minus_sign: [Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension](https://www.aclweb.org/anthology/D19-1609.pdf)

:heavy_minus_sign: [WIQA: A dataset for 'What if...' reasoning over procedural text](https://www.aclweb.org/anthology/D19-1629.pdf)

:heavy_minus_sign: [Do Nuclear Submarines Have Nuclear Captains? A Challenge Dataset for Commonsense Reasoning over Adjectives and Objects](https://www.aclweb.org/anthology/D19-1625.pdf)

:heavy_minus_sign: [Posing Fair Generalization Tasks for Natural Language Inference](https://www.aclweb.org/anthology/D19-1456.pdf)

:heavy_minus_sign: [Self-Assembling Modular Networks for **Interpretable** Multi-Hop Reasoning](https://www.aclweb.org/anthology/D19-1455.pdf)

:heavy_minus_sign: ["Going on a vacation" takes longer than "Going for a walk": A Study of Temporal Commonsense Understanding](https://www.aclweb.org/anthology/D19-1332.pdf)

:heavy_minus_sign: [Answering questions by learning to rank -Learning to rank by answering questions](https://www.aclweb.org/anthology/D19-1256.pdf)

:heavy_minus_sign: [Finding Generalizable Evidence by Learning to Convince Q&A Models](https://www.aclweb.org/anthology/D19-1244.pdf)


:heavy_minus_sign: [Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning over Few-Shot Relations](https://www.aclweb.org/anthology/D19-1334.pdf)

:heavy_minus_sign: [How Reasonable are Common-Sense Reasoning Tasks: A Case-Study on the Winograd Schema Challenge and SWAG](https://www.aclweb.org/anthology/D19-1335.pdf)

:heavy_minus_sign: [KagNet: Knowledge-Aware Graph Networksfor Commonsense Reasoning](https://www.aclweb.org/anthology/D19-1282.pdf)

:heavy_minus_sign: [COSMOSQA: Machine Reading Comprehensionwith Contextual Commonsense Reasoning](https://www.aclweb.org/anthology/D19-1243.pdf)

:heavy_minus_sign: [What’s Missing: A Knowledge Gap Guided Approachfor Multi-hop Question Answering](https://www.aclweb.org/anthology/D19-1281.pdf)
> "given partial knowledge, explicitly  identifying  what’s  missing  substantially outperforms previous approaches".

:heavy_minus_sign: [On the Importance of Delexicalization for Fact Verification](https://www.aclweb.org/anthology/D19-1340.pdf)

## [Information Retrieval](#contents)

:heavy_minus_sign: [Bridging the Gap Between Relevance Matching and Semantic Matching for Short Text Similarity Modeling](https://www.aclweb.org/anthology/D19-1540.pdf)

:heavy_minus_sign: [Cross-Domain Modeling of Sentence-Level Evidence for Document Retrieval](https://www.aclweb.org/anthology/D19-1352.pdf)

:heavy_minus_sign: [Applying BERT to Document Retrieval with Birch](https://www.aclweb.org/anthology/D19-3004.pdf)

:heavy_minus_sign: [Modelling Stopping Criteria for Search Results using Poisson Processes](https://www.aclweb.org/anthology/D19-1351.pdf)

## [Entity Linking](#contents)

:heavy_minus_sign: [Fine-Grained Evaluation for Entity Linking](https://www.aclweb.org/anthology/D19-1066.pdf)

## [Entities and NER](#contents)

:boom: [ner and pos when nothing is capitalized](https://www.aclweb.org/anthology/D19-1650.pdf)

:heavy_minus_sign: [Fine-Grained Entity Typing via Hierarchical Multi Graph Convolutional Networks](https://www.aclweb.org/anthology/D19-1502.pdf)

:heavy_minus_sign: [An Attentive Fine-Grained Entity Typing Model with Latent Type Representation](https://www.aclweb.org/anthology/D19-1641.pdf)

:heavy_minus_sign: [Hierarchically-Refined Label Attention Network for Sequence Labeling](https://www.aclweb.org/anthology/D19-1422.pdf)

:heavy_minus_sign: [Feature-Dependent Confusion Matrices for Low-Resource NER Labeling with Noisy Labels](https://www.aclweb.org/anthology/D19-1362.pdf)

:heavy_minus_sign: [Small and Practical BERT Models for Sequence Labeling](https://www.aclweb.org/anthology/D19-1374.pdf)
> Multilingual sequence labeling.

:heavy_minus_sign: [Hierarchical Meta-Embeddings for Code-Switching Named Entity Recognition](https://www.aclweb.org/anthology/D19-1360.pdf)

:heavy_minus_sign: [EntEval: A Holistic Evaluation Benchmark for **Entity Representations**](https://www.aclweb.org/anthology/D19-1040.pdf)

:heavy_minus_sign: [A Boundary-aware Neural Model for **Nested Named Entity Recognition**](https://www.aclweb.org/anthology/D19-1034.pdf)

## [Coreference](#contents)

:heavy_minus_sign: [WikiCREM: A Large Unsupervised Corpus for Coreference Resolution](https://www.aclweb.org/anthology/D19-1439.pdf)

:heavy_minus_sign: [BERT for Coreference Resolution: Baselines and Analysis](https://www.aclweb.org/anthology/D19-1588.pdf)

## [Text classification](#contents)

:boom: [**Label Embedding using Hierarchical Structure of Labels** for Twitter Classification](https://www.aclweb.org/anthology/D19-1660.pdf)

:heavy_minus_sign: [Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification](https://www.aclweb.org/anthology/D19-1488.pdf)

:heavy_minus_sign: [Rethinking Attribute Representation and Injectionfor Sentiment Classification](https://www.aclweb.org/anthology/D19-1562.pdf)
> Different ways to incorporate metadata into the text classifier.

:heavy_minus_sign: [Learning Explicit and Implicit Structures for Targeted Sentiment Analysis](https://www.aclweb.org/anthology/D19-1550.pdf)

:heavy_minus_sign: [LexicalAT: Lexical-Based Adversarial Reinforcement Training for Robust Sentiment Classification](https://www.aclweb.org/anthology/D19-1554.pdf)

:heavy_minus_sign: [Sequential Learning of Convolutional Features for Effective Text Classification](https://www.aclweb.org/anthology/D19-1567.pdf)

:heavy_minus_sign: [Text Level Graph Neural Network for Text Classification](https://www.aclweb.org/anthology/D19-1345.pdf)

:heavy_minus_sign: [Delta-training: Simple Semi-Supervised Text Classification using Pretrained Word Embeddings](https://www.aclweb.org/anthology/D19-1347.pdf)

## [Propaganda, Persuasion](#contents)

**See also:** Workshop "NLP4IF: Censorship, Disinformation and Propaganda"

:heavy_minus_sign: [Fine-Grained Analysis of Propaganda in News Articles](https://www.aclweb.org/anthology/D19-1565.pdf)
> Introduces a corpus of news articles annotated (at fragment level) with 18 propaganda techniques. A "multi-granularity" network based on BERT achieves the best performance on the task.
> 
> See also: https://propaganda.qcri.org/ and Semeval 2020 Task 11 (https://propaganda.qcri.org/semeval2020-task11/).

:heavy_minus_sign: [Revealing and Predicting Online Persuasion Strategy with Elementary Units](https://www.aclweb.org/anthology/D19-1653.pdf)

## [Social Media and Authorship Attribution](#contents)

:heavy_minus_sign: [The Trumpiest Trump? Identifying a Subject’s Most Characteristic Tweets](https://www.aclweb.org/anthology/D19-1175.pdf)
> How to quantify the degree to which tweets are characteristic of their authors (probability a given tweet was written by its author)? This is done via binary classification, and then calculating the probability using the distance from the decision boundary. One possible application of this: to highlight unusual/suspicious activity (e.g., an account was hacked).  This is studied using the tweets of 15 celebrities (including Oprah, JK Rowling, LeBron James, Justin Bieber, Obama and Trump).
>
> Five approaches to authorship identification are tested; in order of increasing performance these are: (1) a compression-based method using Lempel-Ziv-Welch, (2) LDA topic model features used in Logistic Regression or MLP, (3) an n-gram based language model to score the probability of a tweet given positive or negative ngrams, (4) two different document embeddings -- using fastText and BERT -- fed into a linear classifier, and (5) BERT token embeddings fed to an LSTM. The BERT+LSTM approach yields the highest accuracy of (on average across celebrities) 90.37%.
>
> The paper also finds a strong correlation between "characterstic-ness" and tweet popularity for 13 of the 15 celebrities in the study; for some celebrities (Trump, Modi, Jimmy Fallon) the correlation is positive, whereas for others it is negative (Bieber, Kim Kardashian, and Obama).

:heavy_minus_sign: [Learning Invariant Representations of Social Media Users](https://www.aclweb.org/anthology/D19-1178.pdf)


:heavy_minus_sign: [You Shall Know a User by the Company It Keeps: Dynamic Representations for Social Media Users in NLP](https://www.aclweb.org/anthology/D19-1477.pdf)


:heavy_minus_sign: [A Hierarchical Location Prediction Neural Network for Twitter User Geolocation](https://www.aclweb.org/anthology/D19-1480.pdf)

## [Other](#contents)

:heavy_minus_sign: [A Modular Architecture for Unsupervised **Sarcasm Generation**](https://www.aclweb.org/anthology/D19-1636.pdf)
> Presents a new framework for sarcasm generation which take a negative opinion as input (e.g., "I hate it when my bus is late.") and translates it to a sarcastic version (e.g., "Absolutely love waiting for the bus").  *This does not require paired training data*.
>
> The system is trained only with unlabeled data (sarcastic and non-sarcastic opinions) and employs the theory of "context incongruity": sarcasm is the contrast between positive sentiment context and negative situational context. It uses modules for (1) sentiment neutralization (removing sentiment-bearing words), (2) translation to positive-sentiment phrases, (3) negative situation retrieval of a situation related to the input, (4) sarcasm synthesis. Qualitative and quantitative evaluation shows this system is superior to NMT and style transfer baselines.
>
> Examples ("ALL (RL)"): (1) "worrying  because  did  not  finish  my  homework" -> "worrying about finish homework is great". (2) "swimming lessons are very costly in nyc." -> "i am loving the swimming lessons . going to be a very costly in nyc ch."
>
>
> **Code:**  https://github.com/TarunTater/sarcasmgeneration

:heavy_minus_sign: [Humor Detection: A Transformer Gets the Last Laugh](https://www.aclweb.org/anthology/D19-1372.pdf)
> Task: detecting whether a joke is humorous. Trains a model to identify humorous jokes using the r/Jokes thread. A Transformer architecture outperforms previous work on humor identification tasks and is comparable to human performance.

:heavy_minus_sign: [Generating Personalized Recipes from Historical User Preferences](https://www.aclweb.org/anthology/D19-1613.pdf)
> Task: personalized recipe generation.
>
> Example input: "Name:Pomberrytini;Ingredients:pomegranate-blueberry juice, cranberry juice,vodka; Calorie:Low".
>
> Output: "Combine  all  ingredients  except  for  the  ice  in  a  blender  or  food  processor. Process to make a  smoothpaste and then add the remaining vodka and blend until smooth. Pour into a chilled glass and garnish with a little lemon and fresh mint."
>
> Code: https://github.com/majumderb/recipe-personalization

:heavy_minus_sign: [TalkDown: A Corpus for Condescension Detection in Context](https://www.aclweb.org/anthology/D19-1385.pdf)
> Goal: to detect condescending language (e.g., "Are you struggling with this whole English language thing?"). Introduces TalkDown, a new labeled dataset of condescending linguistic acts **in context**. Including discourse representations improves performance of a language-only model. Of course, the models are initialized with BERT.

:heavy_minus_sign: [Movie Plot Analysis via Turning Point Identification](https://www.aclweb.org/anthology/D19-1180.pdf)
> "We  introduce  a  dataset  consisting of screenplays and plot synopses annotated with turning points and present an end-to-end neural network  model  that  identifies turning points in plot synopses and projects them onto scenes in screenplays". **Data:** https://github.com/ppapalampidi/TRIPOD

:heavy_minus_sign: [Context-Aware Conversation Thread Detection in Multi-Party Chat](https://www.aclweb.org/anthology/D19-1682.pdf)

:heavy_minus_sign: [A Context-based Framework for Modeling the Role and Function of On-line Resource Citations in Scientific Literature](https://www.aclweb.org/anthology/D19-1524.pdf)


# Organize these later:

:boom: [Meta Relational Learning for Few-Shot Link Prediction in Knowledge Graphs]

:boom: [Using Local Knowledge Graph Construction to Scale Seq2Seq Models to Multi-Document Inputs]

:boom: [Distributionally Robust Language Modeling]

:heavy_minus_sign: [Learning Only from Relevant Keywords and Unlabeled Documents](https://www.aclweb.org/anthology/D19-1411.pdf)

:heavy_minus_sign: [**Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**](https://www.aclweb.org/anthology/D19-1410.pdf)

:heavy_minus_sign: [A Logic-Driven Framework for Consistency of Neural Models](https://www.aclweb.org/anthology/D19-1405.pdf)

:heavy_minus_sign: [Pretrained Language Models for Sequential Sentence Classification]

:heavy_minus_sign: [A Search-based Neural Model for Biomedical Nested and Overlapping Event Detection]

:heavy_minus_sign: [Efficient Sentence Embedding using Discrete Cosine Transform]

:heavy_minus_sign: [Transductive Learning of Neural Language Models for Syntactic and Semantic Analysis]

:heavy_minus_sign: [Semantic Relatedness Based Re-ranker for Text Spotting]

:heavy_minus_sign: [A Multi-Pairwise Extension of Procrustes Analysis for Multilingual Word Translation](https://www.aclweb.org/anthology/D19-1363.pdf)

:heavy_minus_sign: [Neural Gaussian Copula for Variational Autoencoder]

:heavy_minus_sign: [Patient Knowledge Distillation for BERT Model Compression]

:heavy_minus_sign: [Topics to Avoid: Demoting Latent Confounds in Text Classification]

:heavy_minus_sign: [Learning to Ask for Conversational Machine Learning]

:heavy_minus_sign: [Language Modeling for Code-Switching: Evaluation, Integration of Monolingual Data, and Discriminative Training]

:heavy_minus_sign: [Fine-grained Knowledge Fusion for Sequence Labeling Domain Adaptation]

:heavy_minus_sign: [Exploiting Monolingual Data at Scale for Neural Machine Translation]

:heavy_minus_sign: [Parallel Iterative Edit Models for Local Sequence Transduction]

:heavy_minus_sign: [ARAML: A Stable Adversarial Training Framework for Text Generation]

:heavy_minus_sign: [FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow]

:heavy_minus_sign: [Compositional Generalization for Primitive Substitutions]

