# Read Me

**Paper:** Sentence-level Media Bias Analysis with Event Relation Graph<br/>
**Accepted:** The 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang<br/>
**Paper Link:** https://aclanthology.org/2024.naacl-long.292/

<br/>

## Task Description

This paper identifies media bias at sentence level. Specifically, the model takes a whole news article consisting of N sentences as input, and outputs the prediction for each sentence whether it contains bias or not.

<br/>

## Dataset Description

The sentence-level media bias identification is experimented on two datasets:

* **BASIL** contains 300 articles, with both lexical bias and informational bias annotated (https://github.com/launchnlp/BASIL). Because both types of bias can introduce ideological bias to the readers and sway their opinions, we consider them both in our bias sentences identification task. To be specific, we label a sentence as _bias_ if it carries either type of bias, or assign the _non\-bias_ label if neither type of bias exists.

* **BiasedSents** contains 46 articles with crowd-sourcing annotations in four scales: not biased, slightly biased, biased, and very biased (https://github.com/skymoonlight/biased-sents-annotation). We process the first two scales as _non\-bias_ class and the latter two as _bias_ class. The dataset releases the annotations from five different annotators, from which we derive the majority voting label as the ground truth.

<br/>

## Event Relation Graph

We release the code for training the event relation graph algorithm and constructing the event relation graph for a new document:

* **Dataset:** We used MAVEN-ERE dataset for training the event relation graph (https://github.com/THU-KEG/MAVEN-ERE)
* **mavenere_event_relation_label.py:** the code for processing the event relations labels in MAVEN-ERE dataset
* **training_event_relation_graph.py:** the code for training the event relation graph algorithm
* **build_event_relation_graph.py:** the code for constructing the event relation graph for a candidate document

<br/>

## Bias Sentences Identification

We release the code for identifying sentence-level media bias, by incorporating the event relation graph as an extra guidance:

* **bias_event_relation_graph_BASIL.py:** the code for identifying sentence-level media bias on BASIL dataset
* **bias_event_relation_graph_BiasedSents.py:** the code for identifying sentence-level media bias on BiasedSents dataset


<br/>

## Citation

If you are going to cite this paper, please use the form:

Yuanyuan Lei and Ruihong Huang. 2024. Sentence-level Media Bias Analysis with Event Relation Graph. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5225–5238, Mexico City, Mexico. Association for Computational Linguistics.

```bibtex
@inproceedings{lei-huang-2024-sentence,
    title = "Sentence-level Media Bias Analysis with Event Relation Graph",
    author = "Lei, Yuanyuan  and
      Huang, Ruihong",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.292",
    pages = "5225--5238",
    abstract = "Media outlets are becoming more partisan and polarized nowadays. In this paper, we identify media bias at the sentence level, and pinpoint bias sentences that intend to sway readers{'} opinions. As bias sentences are often expressed in a neutral and factual way, considering broader context outside a sentence can help reveal the bias. In particular, we observe that events in a bias sentence need to be understood in associations with other events in the document. Therefore, we propose to construct an event relation graph to explicitly reason about event-event relations for sentence-level bias identification. The designed event relation graph consists of events as nodes and four common types of event relations: coreference, temporal, causal, and subevent relations. Then, we incorporate event relation graph for bias sentences identification in two steps: an event-aware language model is built to inject the events and event relations knowledge into the basic language model via soft labels; further, a relation-aware graph attention network is designed to update sentence embedding with events and event relations information based on hard labels. Experiments on two benchmark datasets demonstrate that our approach with the aid of event relation graph improves both precision and recall of bias sentence identification.",
}
```














