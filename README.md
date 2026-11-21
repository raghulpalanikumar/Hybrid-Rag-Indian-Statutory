# Hybrid RAG for Indian Statutory Information

**Hybrid Retrieval-Augmented Generation with Semantic, Lexical, Topic-Aware and Knowledge Graph Retrieval for Indian Statutory Information**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raghulpalanikumar/hybrid-rag-indian-statutory/blob/main/Hybrid_RAG_Indian_Statutory_Colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Recall@5](https://img.shields.io/badge/Recall%405-0.95-brightgreen)
![nDCG@5](https://img.shields.io/badge/nDCG%405-0.89-brightgreen)
![MRR](https://img.shields.io/badge/MRR-0.87-brightgreen)

> Official code and notebook for the journal paper *"Hybrid Retrieval-Augmented Generation with Semantic, Lexical, Topic-Aware and Knowledge Graph Retrieval for Indian Statutory Information"*.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Key contributions](#key-contributions)
3. [Highlights](#highlights)
4. [Methodology](#methodology)
   - [System overview](#system-overview)
   - [Dataset and preprocessing](#dataset-and-preprocessing)
   - [Document chunking](#document-chunking)
   - [Multi-representation indexing](#multi-representation-indexing)
   - [Semantic retrieval (BGE-M3)](#semantic-retrieval-bge-m3)
   - [Lexical retrieval (BM25)](#lexical-retrieval-bm25)
   - [Topic-aware layer (BERTopic)](#topic-aware-layer-bertopic)
   - [Knowledge graph expansion](#knowledge-graph-expansion)
   - [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
   - [Tuned hybrid scoring](#tuned-hybrid-scoring)
   - [Cross-Encoder re-ranking](#cross-encoder-re-ranking)
   - [Grounded structured answer generation](#grounded-structured-answer-generation)
   - [Hyperparameter tuning with Optuna](#hyperparameter-tuning-with-optuna)
   - [Implementation summary](#implementation-summary)
5. [Evaluation protocol](#evaluation-protocol)
6. [Results](#results)
   - [Optimised pipeline](#optimised-pipeline)
   - [Ablation study](#ablation-study)
   - [Comparison with existing paradigms](#comparison-with-existing-paradigms)
   - [Hyperparameter optimisation trace](#hyperparameter-optimisation-trace)
   - [Per-query analysis](#per-query-analysis)
   - [Qualitative example](#qualitative-example)
   - [Topic and graph visualisations](#topic-and-graph-visualisations)
   - [Reference results for this repository](#reference-results-for-this-repository)
7. [Deployed web interface](#deployed-web-interface)
8. [Repository contents](#repository-contents)
9. [Quick start (Google Colab)](#quick-start-google-colab)
10. [Experimental setup](#experimental-setup)
11. [Security](#security)
12. [Future work](#future-work)
13. [Use of large language models](#use-of-large-language-models)
14. [Citation](#citation)
15. [Disclaimer](#disclaimer)
16. [Authors and contact](#authors-and-contact)
17. [Declarations](#declarations)
18. [License](#license)
19. [Acknowledgements](#acknowledgements)

---

## Abstract

Indian statutes are long and written in dense legal language, and keyword search often fails when a user describes a situation in everyday words. This project is a hybrid retrieval-augmented generation (RAG) framework for Indian statutory information. It combines four retrieval signals: dense semantic retrieval with **BGE-M3**, lexical retrieval with **BM25**, topic information from **BERTopic**, and one-hop expansion over a legal **knowledge graph**. The ranked lists are merged with **Reciprocal Rank Fusion**, refined by a **Cross-Encoder** re-ranker, and passed to `openai/gpt-oss-20b` through the Groq API, which returns a structured answer: relevant section, explanation, penalty, connected sections and a short summary.

Retrieval weights, the re-ranking weight and the candidate pool size are tuned with **Optuna** on a validation set. The tuned pipeline is then evaluated on a separate held-out set of 100 IPC queries, where it reaches **Precision@5 0.55, Recall@5 0.95, F1@5 0.64, MRR 0.87 and nDCG@5 0.89**, with at least one relevant section in the top five for **95 of 100 queries**. An ablation study shows that each added signal improves recall and ranking at some cost to precision, which the Cross-Encoder then recovers. The repository also provides per-category results, a qualitative trace and a Gradio web interface built on the pipeline.

**Keywords:** legal information retrieval, retrieval-augmented generation, knowledge graphs, BERTopic, large language models, explainable legal AI

---

## Key contributions

- **Four-signal hybrid retrieval.** BGE-M3 dense retrieval, BM25 lexical retrieval, BERTopic topic scoring and one-hop knowledge graph expansion, merged with Reciprocal Rank Fusion (RRF) and refined with a Cross-Encoder.
- **Tuned and held-out evaluation.** Retrieval and re-ranking parameters are tuned with Optuna on a 30-query validation set, and the frozen setting is evaluated on 100 held-out IPC queries.
- **Component-level evidence.** A cumulative ablation study and a per-category analysis measure what each component contributes.
- **Grounded, explainable output.** A fixed five-field answer format keeps every part of the answer tied to retrieved statutory text.
- **Deployed application.** The full pipeline is available as a Gradio web app with three layouts.

The contribution is the integration of semantic, lexical, topical and relational signals, together with rank fusion and re-ranking, into one pipeline designed for Indian statutory text.

### Why four signals?

A legal retrieval system needs semantic understanding, exact lexical matching, topic awareness, knowledge of how provisions relate to each other, re-ranking of candidates, and generation that stays grounded in the retrieved text. Each component maps to one of these needs.

| Need | Component | What it contributes |
|---|---|---|
| Meaning beyond wording | BGE-M3 | Finds provisions related in meaning even when query and statute share no words |
| Exact legal terms | BM25 | Preserves matches on section numbers, offence names and numerical thresholds |
| Separating look-alike offences | BERTopic | Distinguishes provisions from different legal areas that share vocabulary |
| Relationships between provisions | Knowledge graph | Surfaces definitions, exceptions and punishment clauses linked to a hit |
| Combining signals | RRF | Merges channels with incompatible score scales using rank positions |
| Fine-grained relevance | Cross-Encoder | Reads query and candidate together to sharpen the final evidence set |
| Grounded answers | `openai/gpt-oss-20b` | Writes a structured answer only from the supplied statutory text |

---

## Highlights

| Metric (100-query held-out test set) | Score |
|---|---|
| Precision@5 | 0.5500 |
| Recall@5 | 0.9500 |
| F1@5 | 0.6400 |
| MRR | 0.8700 |
| nDCG@5 | 0.8900 |
| Retrieval Accuracy@5 | 0.9500 |
| Answer-Level Accuracy | 0.9500 |

Tuned setting: **α = 0.7784**, **β = 0.2083**, **k_candidates = 22**, **k_final = 5**.

---

## Methodology

### System overview

The framework answers natural-language questions about Indian statutory law with explanations grounded in retrieved text. A conventional RAG system uses a single dense-vector index, which struggles when a query and a statute use different words, and when two offences are described in similar terms. This framework combines four retrieval strategies instead, in six steps:

1. **Cleaning and chunking.** Statutory records are cleaned and split into overlapping chunks.
2. **Multi-representation indexing.** Each chunk is indexed as a dense vector, in an inverted lexical index, under a topic label and as a node in a knowledge graph.
3. **Hybrid candidate retrieval.** All four indices are searched in parallel and their ranked lists are merged with RRF into one high-recall candidate set.
4. **Tuned hybrid scoring.** Semantic and lexical scores are blended with a weight chosen by Optuna.
5. **Cross-Encoder re-ranking.** Each query–chunk pair is re-scored by a Cross-Encoder to sharpen the final evidence set.
6. **Structured answer generation.** The top-ranked evidence is passed to an LLM, which writes a structured answer.

```
User question
     │
     ├── BGE-M3 dense retrieval ───────┐
     ├── BM25 lexical retrieval ───────┤
     ├── BERTopic topic filtering ─────┼──► Reciprocal Rank Fusion (κ = 60)
     └── Knowledge-graph expansion ────┘             │
                                                     ▼
                                   Hybrid score  α·Sem + (1−α)·BM25
                                                     │
                                                     ▼
                              Cross-Encoder blend  (1−β)·Hybrid + β·σ(CE)
                                                     │
                                                     ▼
                       Top-5 evidence → Groq  openai/gpt-oss-20b  (T = 0.2)
                                                     │
                                                     ▼
          Relevant Section · Explanation · Penalty · Connected Sections · Summary
```

### Dataset and preprocessing

The Indian Penal Code (IPC) is used for evaluation. As India's principal criminal statute, it contains a great deal of legal phrasing and many overlapping offence definitions, which makes it a demanding test case. Its clear division into definitions, descriptions and punishments makes it a well-suited methodological benchmark, and the same method applies to the Bharatiya Nyaya Sanhita (BNS) and other Indian statutes.

The corpus comes from the Kaggle dataset [Indian Penal Code (IPC) Sections Information](https://www.kaggle.com/datasets/dev523/indian-penal-code-ipc-sections-information). The source file (`ipc_sections.csv`) has four fields with no missing values: section number, offence, description and punishment.

| Stage | Count |
|---|---|
| Source records | 444 |
| Unique statutory documents after deduplication | 442 |
| Chunks after splitting (size 700, overlap 100, min length 100) | 1,321 |
| Topics discovered by BERTopic | 34 |

Preprocessing steps:

- **Deduplication.** Duplicate and empty records are removed.
- **Boilerplate removal.** Three template phrases appear in almost every record. Left in place they would dominate the embedding space and push the encoder to match on surface wording, so they are detected and removed before embedding.
- **Normalisation.** Spelling variants, inconsistent punctuation and extra whitespace are corrected.
- **Field concatenation.** Section, offence, plain-language description and punishment are joined into a single `combined_text` field.
- **Document conversion.** Each row becomes a LangChain document whose page content is `combined_text` and whose metadata holds the section number, offence and punishment.

### Document chunking

Statutory sections vary widely in length because of clauses, exceptions and illustrations. Whole sections would reduce retrieval accuracy and could exceed the context window, and a very long chunk blurs several ideas into one vector. The pipeline uses LangChain's `RecursiveCharacterTextSplitter` with a chunk size of **700** characters and an overlap of **100**, trying separators in this order:

```
S = { "\n\n" (paragraph), "\n" (line), " " (space), "" (character) }
```

The splitter first tries paragraph breaks, then line breaks, then spaces, and splits at the character level only as a last resort. This keeps definitions, exceptions and punishments from being cut off mid-sentence. Fragments shorter than **100** characters are discarded, and each chunk keeps a reference to its parent section and offence label in its metadata.

### Multi-representation indexing

Each chunk is indexed in four structures at once: (i) a dense semantic vector space built with BGE-M3; (ii) an inverted lexical index for BM25; (iii) a topic partition produced by BERTopic; and (iv) a node in the relational knowledge graph. Conceptual similarity, exact terms, thematic coherence and links between provisions all contribute to the same candidate pool.

### Semantic retrieval (BGE-M3)

Each chunk and each query is encoded with BGE-M3 into a normalised 1024-dimensional embedding. Relevance is cosine similarity, which reduces to a dot product because the embeddings have unit length:

$$\mathrm{Sim}_{sem}(q,c_i)=\frac{e_q\cdot e_{c_i}}{\lVert e_q\rVert_2\,\lVert e_{c_i}\rVert_2}=e_q\cdot e_{c_i}$$

This channel finds chunks related to a query in meaning even when they share no words with it.

### Lexical retrieval (BM25)

Dense embeddings do not reliably favour exact section numbers, offence names or numerical thresholds. An Okapi BM25 channel runs in parallel with the standard defaults k1 = 1.5 and b = 0.75:

$$\mathrm{BM25}(q,D)=\sum_{t_j\in q}\mathrm{IDF}(t_j)\cdot\frac{f(t_j,D)\,(k_1+1)}{f(t_j,D)+k_1\left(1-b+b\frac{|D|}{avgdl}\right)}$$

$$\mathrm{IDF}(t_j)=\ln\!\left(\frac{N-n(t_j)+0.5}{n(t_j)+0.5}+1\right),\qquad N=1321$$

### Topic-aware layer (BERTopic)

Some retrieval errors come from offences that share vocabulary but belong to different areas of law. A topic layer is built with BERTopic on the same BGE-M3 embeddings in three stages:

- **UMAP** reduces the 1024-dimensional embeddings while keeping local neighbourhoods intact.
- **HDBSCAN** finds dense groups of related chunks without needing the number of clusters in advance, and marks low-density points as noise.
- **c-TF-IDF** scores terms at cluster level to produce a readable label. For example, a cluster dominated by *guardian, minor, kidnapping* and *consent* is labelled Kidnapping.

Each chunk receives a topic label τ(c). At query time the query embedding is assigned to its nearest topic centroid to obtain τ(q):

$$\mathrm{TopicScore}(q,c_i)=\begin{cases}1,&\tau(c_i)=\tau(q)\\ \gamma,&\text{otherwise}\end{cases},\qquad \gamma\in[0,1)$$

Chunks from other topics are down-weighted by γ rather than discarded, which preserves recall for queries that cut across topics while favouring topically consistent evidence. Topic labels are unsupervised groupings that serve as an additional retrieval signal.

### Knowledge graph expansion

A graph G = (V, E) is built over the chunks with NetworkX, where each node is a chunk. An edge (c_i, c_j) is added when any of the following holds:

- the two chunks share a topic, τ(c_i) = τ(c_j);
- one chunk explicitly cross-references the other (for instance, Section 302 refers to Section 300); or
- their cosine similarity exceeds a threshold θ.

Starting from the seed set R₀ returned by first-stage retrieval, the graph is expanded one hop. For a neighbouring candidate c_j:

$$\mathrm{GraphScore}(c_j)=\sum_{c_i\in R_0\cap N(c_j)} w(c_i,c_j)\cdot \mathrm{Score}(c_i)$$

where N(c_j) is the set of neighbours of c_j, w is the normalised edge weight and Score(c_i) is the first-stage relevance. Expansion surfaces explanatory sections and punishment clauses that share little wording or meaning with the query: a query that only describes an offence can still reach the section that prescribes its punishment.

### Reciprocal Rank Fusion

The four channels (semantic, lexical, topic-filtered and graph expansion) run in parallel and each returns a ranked list. Their raw scores are on different scales, so the lists are merged by rank position:

$$\mathrm{RRF}(c)=\sum_{m=1}^{M}\frac{1}{\kappa+\mathrm{rank}_m(c)},\qquad M=4,\ \kappa=60$$

A chunk missing from a list has infinite rank there, so that channel adds nothing to its score. The result is one high-recall candidate pool that draws on every channel.

### Tuned hybrid scoring

First-stage scoring blends semantic similarity with min–max normalised BM25 relevance through a weight α chosen by Optuna:

$$\mathrm{Score}_{hybrid}(q,c_i)=\alpha\cdot \mathrm{Sim}_{sem}(q,c_i)+(1-\alpha)\cdot \mathrm{BM25}_{norm}(q,c_i)$$

### Cross-Encoder re-ranking

First-stage retrieval is built for recall. To improve precision, a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) re-scores the candidate pool. In the optimised configuration, 22 candidates are passed to it and the top five are kept. Unlike BGE-M3, a bi-encoder that embeds query and chunk separately, the Cross-Encoder reads both together through shared self-attention layers. The final score blends the hybrid score with the sigmoid of the Cross-Encoder logit, using a weight β chosen by Optuna:

$$\mathrm{Score}_{final}(q,c_i)=(1-\beta)\cdot \mathrm{Score}_{hybrid}(q,c_i)+\beta\cdot\sigma\big(f_{CE}(q,c_i)\big)$$

### Grounded structured answer generation

The top-k evidence chunks are placed in a fixed, XML-structured context block and sent to `openai/gpt-oss-20b` through the Groq API at temperature **0.2**. The prompt instructs the model to answer only from the supplied context and to say explicitly when the evidence does not support an answer. Every answer has five fields:

| Field | Content |
|---|---|
| **Relevant section** | The applicable IPC section or sections |
| **Explanation** | Why the legal criteria are met |
| **Penalty** | The punishment prescribed by law |
| **Connected sections** | Related provisions found through graph expansion |
| **Summary** | A short overview of the answer |

Fixing the format ties every part of the answer to retrieved evidence instead of leaving the model free to generate, which lowers the risk of unsupported legal statements.

### Hyperparameter tuning with Optuna

Optuna searches the settings automatically over **30 trials** with the Tree-structured Parzen Estimator (TPE) sampler. The objective is the mean F1@5 over a validation query set Q:

$$\theta^{*}=\arg\max_{\theta\in\Theta}\ \frac{1}{|Q|}\sum_{q\in Q}\mathrm{F1@5}(q;\theta)$$

| Tuned parameter | Meaning | Selected value |
|---|---|---|
| α | Semantic weight (keyword weight = 1 − α) | 0.7784 (keyword 0.2216) |
| β | Cross-Encoder re-ranking weight | 0.2083 |
| k_candidates | Chunks passed to re-ranking | 22 |

The final context size is fixed at **k_final = 5** and is not tuned. Optuna does not retrain any model: it runs the retrieval pipeline repeatedly with different parameter values, scores each run on the validation set, and the best setting is frozen for the held-out evaluation and the deployed system.

### Implementation summary

| Component | Method | Key setting |
|---|---|---|
| Chunking | LangChain `RecursiveCharacterTextSplitter` | size 700, overlap 100, min length 100 |
| Dense retrieval | BGE-M3 | 1024-d normalised embeddings, dot product |
| Lexical retrieval | BM25 (Okapi) | k1 = 1.5, b = 0.75 |
| Topic layer | BERTopic (UMAP + HDBSCAN + c-TF-IDF) | unsupervised, automatic cluster count |
| Knowledge graph | NetworkX | edges: shared topic, cross-reference, cosine > θ |
| Fusion | Reciprocal Rank Fusion | 4 channels, κ = 60 |
| Hybrid scoring | Weighted linear blend | α tuned by Optuna |
| Re-ranking | Cross-Encoder `ms-marco-MiniLM-L-6-v2` | β tuned by Optuna, 22 candidates → top 5 |
| Generation | `openai/gpt-oss-20b` via Groq | temperature 0.2, five-field answer |
| Tuning | Optuna (TPE sampler) | 30 trials, objective: mean F1@5 |
| Interface | Gradio | three layouts (simple, dashboard, dark) |

---

## Evaluation protocol

- **Test set:** 100 hand-annotated natural-language queries over the IPC, each paired with the identifiers of its relevant sections. They are never used for tuning.
- **Validation set:** a separate 30-query set used only by Optuna.
- **Metrics:** Precision@5, Recall@5, F1@5, MRR, nDCG@5, Retrieval Accuracy@5, section-level accuracy and Answer-Level Accuracy. The notebook also computes RAGAS metrics (faithfulness, answer relevancy, context precision, context recall).
- **Ablation:** BM25 → BGE-M3 → + BM25 → + BERTopic → + Knowledge graph → + RRF/Optuna → + Cross-Encoder.

For a query with retrieved sections Ŝ = [s₁ … s_K] and ground-truth set S\*:

$$\mathrm{Precision@K}=\frac{|\hat S\cap S^*|}{K},\qquad \mathrm{Recall@K}=\frac{|\hat S\cap S^*|}{|S^*|},\qquad \mathrm{F1@K}=\frac{2\,P\,R}{P+R}$$

$$\mathrm{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{\mathrm{rank}(q)},\qquad \mathrm{DCG@K}=\sum_{i=1}^{K}\frac{rel_i}{\log_2(i+1)},\qquad \mathrm{nDCG@K}=\frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}$$

- Precision, recall and F1 are computed per query and macro-averaged, so the mean of per-query F1 is usually slightly lower than F1 computed from the mean precision and recall.
- `rank(q)` is the position of the first relevant section in the top K. A first correct section at rank 1 contributes 1.0, at rank 2 contributes 0.5, and none in the top K contributes 0.
- **Retrieval Accuracy@5** is the fraction of queries with at least one relevant section in the top five.
- **Answer-Level Accuracy** measures how well the generated answer identifies the correct section. It is a continuous answer-matching similarity score in [0, 1] between the section(s) named in the answer and the ground truth, averaged over the query set. Keeping retrieval and answer metrics separate lets a failure be attributed to the retrieval stage or the generation stage.

---

## Results

### Optimised pipeline

Held-out 100-query test set with α = 0.7784, β = 0.2083, k_candidates = 22 and k_final = 5:

| Metric | Score |
|---|---|
| Precision@5 | 0.5500 |
| Recall@5 | 0.9500 |
| F1@5 | 0.6400 |
| MRR | 0.8700 |
| nDCG@5 | 0.8900 |
| Retrieval Accuracy@5 | 0.9500 (95 of 100 queries have a relevant section in the top 5) |
| Answer-Level Accuracy | 0.9500 |

- High recall means the first stage rarely misses a relevant provision.
- MRR of 0.87 shows the first relevant section usually appears at or near the top, and nDCG@5 of 0.89 shows relevant sections are concentrated among the highest-ranked results.
- Versus the semantic-only BGE-M3 baseline, precision is slightly better (0.5500 vs 0.5400), F1@5 is clearly better (0.6400 vs 0.5976) and recall rises from 0.7167 to 0.9500.
- Retrieval Accuracy@5 and Answer-Level Accuracy share the value 0.95 but measure different things: the first asks whether a relevant section was among the top five candidates, the second scores how closely the generated answer matches the correct section.

### Ablation study

Components are added cumulatively, starting from lexical retrieval alone.

| Configuration | Precision@5 | Recall@5 | F1@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| BM25 | 0.1200 | 0.1283 | 0.1230 | 0.2833 | 0.2104 |
| BGE-M3 | 0.5400 | 0.7167 | 0.5976 | 0.7917 | 0.6842 |
| BGE-M3 + BM25 | 0.5200 | 0.7800 | 0.6265 | 0.8200 | 0.7500 |
| + BERTopic | 0.5000 | 0.8200 | 0.6238 | 0.8400 | 0.7900 |
| + Knowledge graph | 0.4700 | 0.8700 | 0.6110 | 0.8500 | 0.8300 |
| + RRF + Optuna | 0.4300 | 0.9300 | 0.5900 | 0.8600 | 0.8700 |
| **Full system + Cross-Encoder** | **0.5500** | **0.9500** | **0.6400** | **0.8700** | **0.8900** |

Recall@5, MRR and nDCG@5 rise steadily as components are added, from 0.1283 / 0.2833 / 0.2104 for BM25 alone to 0.9500 / 0.8700 / 0.8900 for the full system. Precision@5 falls from 0.5400 (BGE-M3) to 0.4300 as topics, graph links and rank fusion widen the candidate pool, then recovers to 0.5500 when the Cross-Encoder filters that pool back down to the most relevant evidence. The full system has the best value on all five metrics.

### Comparison with existing paradigms

All systems use the same ground-truth annotations, the same top-five cut-off and the same metrics.

| Framework | P@5 | R@5 | F1@5 | MRR | nDCG@5 | Answer acc. |
|---|---|---|---|---|---|---|
| Lexical search (BM25) | 0.1200 | 0.1283 | 0.1230 | 0.2833 | 0.2104 | 0.6000 |
| Standard vector RAG | 0.5400 | 0.7167 | 0.5976 | 0.7917 | 0.6842 | 0.9000 |
| Hybrid vector–lexical RAG | 0.5400 | 0.7083 | 0.5935 | 0.8083 | 0.6915 | 0.9000 |
| Topic-enhanced IR | 0.4800 | 0.6333 | 0.5312 | 0.7333 | 0.6210 | 0.8000 |
| Graph-RAG frameworks | 0.5000 | 0.6833 | 0.5645 | 0.7750 | 0.6628 | 0.8000 |
| Multi-stage legal re-ranking | 0.5200 | 0.7000 | 0.5828 | 0.8250 | 0.7103 | 0.9000 |
| **Proposed framework (optimised)** | **0.5500** | **0.9500** | **0.6400** | **0.8700** | **0.8900** | **0.9500** |

Capability comparison with the existing paradigms:

| Framework | Dense semantic | Lexical match | Topic context | KG relations | Rank fusion | Multi-stage re-ranking | Jurisdiction adaptation |
|---|---|---|---|---|---|---|---|
| Standard vector RAG | ✓ | × | × | × | × | × | Generic |
| Lexical search (BM25) | × | ✓ | × | × | × | × | Generic |
| Hybrid vector–lexical RAG | ✓ | ✓ | × | × | ✓ | Optional | Partial |
| Topic-enhanced IR | ✓ | × | ✓ | × | × | × | Generic |
| Graph-RAG frameworks | ✓ | × | × | ✓ | Partial | × | Domain-specific |
| Multi-stage legal re-ranking | Partial | ✓ | × | × | ✓ | ✓ | Legal case law |
| **Proposed framework** | BGE-M3 | BM25 | BERTopic | Legal KG | RRF | Cross-Encoder | Indian statutes |

The framework's clearest advantage is recall (0.9500, well above every baseline), together with the best MRR, nDCG@5, Precision@5 and F1@5. The Cross-Encoder is what lets the system keep precision up while widening the candidate pool, which matters in legal retrieval where both a missed provision and a list padded with irrelevant ones carry real costs.

### Hyperparameter optimisation trace

The 30 Optuna trials are plotted by mean validation F1@5 together with the best value found so far. The best trial gave the configuration above with a mean validation F1@5 of **0.3611**, which was then frozen for the held-out evaluation. The validation set (F1@5 0.3611) contains relatively more challenging queries and more variation in the number of relevant sections per query than the test set (F1@5 0.6400), which accounts for the difference.

### Per-query analysis

Ten representative IPC offence categories from the 100-query test set, examined in detail (optimised pipeline):

| Offence query | Precision@5 | Recall@5 | F1@5 | MRR | Answer-Level Accuracy |
|---|---|---|---|---|---|
| Forgery | 1.0000 | 0.8333 | 0.9091 | 1.0000 | 0.9977 |
| Breach of trust | 0.8000 | 1.0000 | 0.8889 | 1.0000 | 0.9977 |
| Cheating | 0.8000 | 1.0000 | 0.8889 | 1.0000 | 0.9977 |
| Theft | 0.8000 | 0.6667 | 0.7273 | 1.0000 | 0.9932 |
| Kidnapping | 0.6000 | 0.6000 | 0.6000 | 1.0000 | 0.9910 |
| Defamation | 0.4000 | 1.0000 | 0.5714 | 1.0000 | 0.9932 |
| Murder | 0.4000 | 0.6667 | 0.5000 | 1.0000 | 0.9910 |
| Robbery | 0.4000 | 0.6667 | 0.5000 | 1.0000 | 0.9910 |
| Rape | 0.2000 | 0.5000 | 0.2857 | 0.2500 | 0.9887 |
| Extortion | 0.2000 | 0.3333 | 0.2500 | 0.3333 | 0.9864 |
| **Macro average** | **0.5600** | **0.7267** | **0.6121** | **0.8583** | **0.9928** |

- Queries on clearly distinct legal concepts do best: Forgery (F1@5 0.9091, nDCG@5 1.0000), Breach of trust and Cheating (F1@5 0.8889). The combined BGE-M3 and BM25 signals are enough to find the right statutes.
- Provisions that overlap legally or semantically with neighbours are the hardest: Extortion (F1@5 0.2500) and Rape (F1@5 0.2857). Related offences share wording and surrounding provisions, so other provisions enter the candidate pool.
- Eight of ten queries have an MRR of 1.0000, meaning a relevant section was ranked first.
- Retrieval F1@5 ranges from 0.2500 to 0.9091, while answer-level scores stay between 0.9864 and 0.9977, showing that the generator produces consistent structured answers whenever the relevant evidence survives retrieval.

### Qualitative example

Query: *"A person takes a laptop from a coffee shop table while the owner is in the restroom."*

| Stage | Output |
|---|---|
| Retrieved evidence | Section 378 (Theft): "Whoever, intending to take dishonestly any moveable property…"; Section 379 (Punishment for theft): "Whoever commits theft shall be punished…" |
| Relevant section | Section 378 (Theft), punishable under Section 379 |
| Explanation | The act is theft: there is a dishonest intention to take moveable property out of another person's possession without their consent. |
| Punishment | Imprisonment of either description for a term which may extend to three years, or with fine, or with both. |

The query never uses the word "theft" or any statutory term, yet the system maps the described incident to the correct definition. Graph expansion also brings in the punishment clause (Section 379) alongside the definition (Section 378), a link that semantic similarity alone might not surface.

Offences whose legal elements are very close, such as theft, robbery and dacoity, are described in similar wording, so dense retrieval alone cannot reliably separate them in the first stage. The Cross-Encoder makes these finer distinctions on the candidates it receives.

### Topic and graph visualisations

- **Topics.** BERTopic found **34 topics** across the 1,321 chunks: a few large thematic clusters alongside many small ones, reflecting the IPC's mix of broad and very specific offences.
- **Graph neighbourhood.** The topic node `Topic_5` covers forgery provisions, with the chunk nodes directly linked to it. This is the neighbourhood that one-hop expansion explores.

### Reference results for this repository

The notebook is configured with the tuned setting above (α = 0.7784, β = 0.2083, k_candidates = 22, k_final = 5) and evaluates on the 100-query test set. The reference results for this configuration:

| Metric | Reference result |
|---|---|
| Precision@5 | 0.5500 |
| Recall@5 | 0.9500 |
| F1@5 | 0.6400 |
| MRR | 0.8700 |
| nDCG@5 | 0.8900 |
| Retrieval Accuracy@5 | 0.9500 |
| Answer-Level Accuracy | 0.9500 |

Scores from a fresh run can differ slightly because of API-side generation variance, library versions and the ground-truth labels shipped in the notebook. Per-query outputs are saved to `Hybrid_RAG_Test_Results_100_queries.csv` and `Hybrid_RAG_Answers_100_queries.csv`.

---

## Deployed web interface

The retrieval and generation pipeline is wrapped in an interactive **Gradio** application. Users type a legal question in plain language or pick an example question (for instance on murder or theft). The interface shows the retrieved provisions together with the generated explanation, punishment and summary, and lists the technology stack behind the system. Three layouts are available through `UI_CHOICE`: `"simple"`, `"dashboard"` and `"dark"`.

---

## Repository contents

| File | Description |
|---|---|
| `Hybrid_RAG_Indian_Statutory_Colab.ipynb` | Complete Colab notebook (code only): preprocessing, indexing, retrieval, tuning, ablation, evaluation, RAGAS and Gradio |
| `README.md` | This file |
| `LICENSE` | MIT License |

The notebook writes these files when it runs: `chunk_embeddings.pkl`, `legal_chunks.csv`, `chunk_embeddings_topics.pkl`, `Hybrid_RAG_Test_Results_100_queries.csv`, `Hybrid_RAG_Answers_100_queries.csv`, `RAGAS_results.csv` and several PNG charts.

---

## Quick start (Google Colab)

1. Click the **Open in Colab** badge above and set the runtime to **GPU**.
2. Download the IPC dataset from Kaggle: [Indian Penal Code (IPC) Sections Information](https://www.kaggle.com/datasets/dev523/indian-penal-code-ipc-sections-information). Upload `ipc_sections.csv` to `/content/`. The file needs the columns `Section`, `Description`, `Offense` and `Punishment`.
3. Get a free API key from the [Groq Console](https://console.groq.com/).
4. In Colab, open the **key icon (Secrets)** in the left sidebar, add a secret named `GROQ_API_KEY` with your key, and enable notebook access. If you skip this, the notebook asks for the key when you run it.
5. Choose **Runtime → Run all**.

**Time-saving switches** (set near the top of the relevant cells):

| Variable | Default | Effect |
|---|---|---|
| `RUN_ANSWER_EVAL` | `True` | 100 Groq calls for Answer-Level Accuracy |
| `RUN_RAGAS` | `True` | RAGAS evaluation (many Groq calls) |
| `RAGAS_LIMIT` | `100` | Lower it (for example 20) if you hit Groq rate limits |
| `UI_CHOICE` | `"dashboard"` | `"simple"`, `"dashboard"` or `"dark"` Gradio layout |

---

## Experimental setup

- **Development machine:** 8-core CPU, 32 GB RAM, NVIDIA RTX 4090 GPU.
- **Reported evaluation:** the 100-query evaluation was run on Google Colab, and all reported results come from that run.
- **Software:** Python 3.10; LangChain (document handling), Sentence-Transformers (BGE-M3 and MiniLM), BERTopic (topics), NetworkX (knowledge graph), Optuna (tuning), Gradio (interface).
- **Generation:** `openai/gpt-oss-20b` through the Groq API at temperature 0.2.
- **Data separation:** test queries are kept completely separate from the validation queries used for Optuna tuning, so the frozen configuration is tested on unseen data.

---

## Security

**Never commit an API key.** This repository contains none. The notebook reads `GROQ_API_KEY` from Colab Secrets or an input prompt only. If a key was ever pasted into a notebook that you pushed, revoke it in the Groq Console and create a new one.

---

## Future work

- Extend the framework beyond the IPC to the Code of Criminal Procedure, the Indian Evidence Act, the Bharatiya Nyaya Sanhita and other special legislation.
- Add case law and citation links to the knowledge graph to better handle complex legal questions.
- Fine-tune the generator on Indian legal text.
- Run a formal evaluation of generated answers by legal experts.
- Test on larger and more varied sets of real user queries with expert-annotated ground truth.

---

## Use of large language models

`openai/gpt-oss-20b` is a component of the proposed system: it generates the structured answers strictly from retrieved statutory text. It was used only to generate answers and was not used to create or label the ground-truth relevance annotations. An AI assistant (Claude, Anthropic) was used during manuscript preparation to restructure and rephrase the text, remove duplicated passages and convert the manuscript to LaTeX. The study design, implementation, experiments, results and conclusions are the authors' own, and the authors reviewed and take full responsibility for the final text.

---

## Citation

If you use this work, please cite the paper:

```bibtex
@article{devisurya2026hybridrag,
  title   = {Hybrid Retrieval-Augmented Generation with Semantic, Lexical, Topic-Aware and Knowledge Graph Retrieval for Indian Statutory Information},
  author  = {Devisurya, V and Raghul, P and Sabarisri, S and Mithun, P M and Nagaraj, P},
  year    = {2026}
}
```

Please add the journal name, volume and DOI to this entry once the paper is published.

---

## Disclaimer

This project is for research and education. It is **not legal advice**. Always check statutory text and consult a qualified lawyer for legal decisions.

---

## Authors and contact

| Author | Affiliation |
|---|---|
| Devisurya V, Raghul P, Sabarisri S, Mithun P M | Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India |
| Nagaraj P | Department of Computer Science and Engineering, School of Computing, SRM Institute of Science and Technology, Tiruchirappalli, Tamil Nadu, India |

**Corresponding author:** swathisurya11@gmail.com

**Author contributions**

- **Devisurya V:** conceptualisation, methodology, software, data curation, formal analysis, investigation, experiments, writing (original draft).
- **Raghul P:** methodology, validation, results analysis, writing (review and editing).
- **Sabarisri S:** data curation, validation, literature review, writing (review and editing).
- **Mithun P M:** supervision, project administration, critical review, writing (review and editing).
- **Nagaraj P:** supervision, methodological guidance, validation of results, critical review, writing (review and editing).

---

## Declarations

- **Funding:** the authors did not receive funding from any organisation for the submitted work.
- **Competing interests:** the authors declare no competing financial interests or personal relationships that could have appeared to influence this work.
- **Ethics approval and consent to participate:** not applicable. This computational study involves no human participants, human data or animals; it uses publicly available statutory text (the Indian Penal Code).
- **Consent for publication:** not applicable.
- **Data availability:** the IPC corpus is the Kaggle dataset ["Indian Penal Code (IPC) Sections Information"](https://www.kaggle.com/datasets/dev523/indian-penal-code-ipc-sections-information). The annotated query sets used in the paper's evaluation are available from the corresponding author upon reasonable request.
- **Code availability:** the code is provided in this repository.

---

## License

Released under the MIT License. Add a `LICENSE` file to the repository (on GitHub: **Add file → Create new file → name it `LICENSE`**, then choose the MIT template). The IPC dataset is subject to its own terms on Kaggle.

---

## Acknowledgements

BGE-M3 (BAAI), BERTopic, Sentence-Transformers, LangChain, NetworkX, Optuna, RAGAS, Groq and Gradio.
