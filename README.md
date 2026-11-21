# Hybrid RAG for Indian Statutory Information

**Hybrid Retrieval-Augmented Generation with Semantic, Lexical, Topic-Aware and Knowledge Graph Retrieval for Indian Statutory Information**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raghulpalanikumar/hybrid-rag-indian-statutory/blob/main/Hybrid_RAG_Indian_Statutory_Colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Domain](https://img.shields.io/badge/Domain-Legal%20IR%20%2F%20RAG-informational)

---

## Table of Contents

- [Abstract](#abstract)
- [Pipeline](#pipeline)
- [Evaluation protocol](#evaluation-protocol)
- [Results](#results)
- [Repository contents](#repository-contents)
- [Dataset](#dataset)
- [Quick start (Google Colab)](#quick-start-google-colab)
- [Security](#security)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Abstract

Indian statutes are long and written in dense legal language, and keyword search often fails when a user describes a situation in everyday words. This project is a hybrid retrieval-augmented generation (RAG) framework for Indian statutory information. It combines four retrieval signals:

1. dense semantic retrieval with **BGE-M3**,
2. lexical retrieval with **BM25**,
3. topic information from **BERTopic**, and
4. one-hop expansion over a legal **knowledge graph**.

The ranked lists are merged with **Reciprocal Rank Fusion**, refined by a **Cross-Encoder** re-ranker, and passed to `openai/gpt-oss-20b` through the Groq API. The model returns a structured answer: relevant section, explanation, penalty, connected sections and a short summary. Retrieval weights, the re-ranking weight and the candidate pool size are tuned with **Optuna** on a validation set, and the tuned pipeline is evaluated on a held-out set of 100 IPC queries.

**Keywords:** legal information retrieval, retrieval-augmented generation, knowledge graphs, BERTopic, large language models, explainable legal AI

---

## Pipeline

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

| Component | Method | Key setting |
|---|---|---|
| Chunking | LangChain `RecursiveCharacterTextSplitter` | size 700, overlap 100, min length 100 |
| Dense retrieval | BGE-M3 | 1024-d normalised embeddings, dot product |
| Lexical retrieval | BM25 (Okapi) | k1 = 1.5, b = 0.75 |
| Topic layer | BERTopic (UMAP + HDBSCAN + c-TF-IDF) | unsupervised, automatic cluster count |
| Knowledge graph | NetworkX | edges: shared topic, cross-reference, cosine > θ |
| Fusion | Reciprocal Rank Fusion | 4 channels, κ = 60 |
| Re-ranking | Cross-Encoder `ms-marco-MiniLM-L-6-v2` | β tuned by Optuna |
| Generation | `openai/gpt-oss-20b` via Groq | temperature 0.2, five-field answer |
| Tuning | Optuna (TPE sampler) | 30 trials, objective: mean F1@5 |
| Interface | Gradio | three layouts (simple, dashboard, dark) |

Optuna tunes three values: the semantic weight **α**, the re-ranking weight **β**, and the candidate pool size **k_candidates**. The final context size is fixed at 5. Optuna does not retrain any model: it re-runs the retrieval pipeline with different parameter values, scores each run on the validation set and freezes the best setting for the held-out test.

---

## Evaluation protocol

- **Test set:** 100 hand-labelled natural-language queries over the IPC, each with its ground-truth sections. They are never used for tuning.
- **Validation set:** a separate 30-query set, used only by Optuna.
- **Metrics:** Precision@5, Recall@5, F1@5, MRR, nDCG@5, Retrieval Accuracy@5, section-level accuracy and Answer-Level Accuracy. The notebook also computes RAGAS metrics (faithfulness, answer relevancy, context precision, context recall).
- **Ablation:** BM25 → BGE-M3 → + BM25 → + BERTopic → + Knowledge graph → + RRF/Optuna → + Cross-Encoder.

> ⚠️ The ground-truth section lists in the notebook were prepared for this repository. Review them before reporting results in a paper or report.

---

## Results

### Results reported in the paper

On the 100-query held-out test set, with the tuned setting **α = 0.7784** (keyword weight 0.2216), **β = 0.2083**, **k_candidates = 22** and **k_final = 5**:

| Metric | Paper |
|---|---|
| Precision@5 | 0.5500 |
| Recall@5 | 0.9500 |
| F1@5 | 0.6400 |
| MRR | 0.8700 |
| nDCG@5 | 0.8900 |
| Retrieval Accuracy@5 | 0.9500 (95 of 100 queries have a relevant section in the top 5) |
| Answer-Level Accuracy | 0.9500 |

Retrieval Accuracy@5 and Answer-Level Accuracy happen to share the value 0.95, but they measure different things. The first asks whether a relevant section was among the top five candidates. The second scores how closely the generated answer matches the correct section.

### Ablation (components added cumulatively, from the paper)

| Configuration | Precision@5 | Recall@5 | F1@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| BM25 | 0.1200 | 0.1283 | 0.1230 | 0.2833 | 0.2104 |
| BGE-M3 | 0.5400 | 0.7167 | 0.5976 | 0.7917 | 0.6842 |
| BGE-M3 + BM25 | 0.5200 | 0.7800 | 0.6265 | 0.8200 | 0.7500 |
| + BERTopic | 0.5000 | 0.8200 | 0.6238 | 0.8400 | 0.7900 |
| + Knowledge graph | 0.4700 | 0.8700 | 0.6110 | 0.8500 | 0.8300 |
| + RRF + Optuna | 0.4300 | 0.9300 | 0.5900 | 0.8600 | 0.8700 |
| **Full system + Cross-Encoder** | **0.5500** | **0.9500** | **0.6400** | **0.8700** | **0.8900** |

Each added signal improves recall and ranking at some cost to precision, and the Cross-Encoder then recovers precision.

### Results from running this repository

The paper's numbers come from its own annotated queries, labels and run. This repository ships its own query set, so your scores will differ somewhat. Run the notebook and record them here.

| Metric | This repo's run |
|---|---|
| Precision@5 | _fill after run_ |
| Recall@5 | _fill after run_ |
| F1@5 | _fill after run_ |
| MRR | _fill after run_ |
| nDCG@5 | _fill after run_ |
| Retrieval Accuracy@5 | _fill after run_ |
| Answer-Level Accuracy | _fill after run_ |

---

## Repository contents

| File | Description |
|---|---|
| `Hybrid_RAG_Indian_Statutory_Colab.ipynb` | Complete Colab notebook (code only): preprocessing, indexing, retrieval, tuning, ablation, evaluation, RAGAS and Gradio |
| `README.md` | This file |
| `LICENSE` | MIT License (add this file, see [License](#license)) |

The notebook writes these files when it runs: `chunk_embeddings.pkl`, `legal_chunks.csv`, `chunk_embeddings_topics.pkl`, `Hybrid_RAG_Test_Results_100_queries.csv`, `Hybrid_RAG_Answers_100_queries.csv`, `RAGAS_results.csv` and several PNG charts.

---

## Dataset

The IPC corpus comes from the Kaggle dataset [Indian Penal Code (IPC) Sections Information](https://www.kaggle.com/datasets/dev523/indian-penal-code-ipc-sections-information).

| Stage | Count |
|---|---|
| Source records in `ipc_sections.csv` (section number, offence, description, punishment) | 444 |
| Unique documents after deduplication | 442 |
| Chunks after splitting (700 / 100 / min 100) | 1,321 |

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

## Security

**Never commit an API key.** This repository contains none. The notebook reads `GROQ_API_KEY` from Colab Secrets or an input prompt only. If a key was ever pasted into a notebook that you pushed, revoke it in the Groq Console and create a new one.

---

## Limitations

- The IPC is used as a methodological benchmark. It is not India's current operative criminal code (the Bharatiya Nyaya Sanhita replaced it). The same method can be applied to the BNS and other statutes.
- Precision@5 is moderate because each query usually has only a few relevant sections, while the system returns five.
- Topic labels are unsupervised groupings, not official legal categories.
- Answer-Level Accuracy is a section-matching score, not a certificate of legal correctness.

## Disclaimer

This project is for research and education. It is **not legal advice**. Always check statutory text and consult a qualified lawyer for legal decisions.

---

## Authors

- Devisurya V, Raghul P, Sabarisri S, Mithun P M: Department of Information Technology, Kongu Engineering College, Perundurai, Tamil Nadu, India
- Nagaraj P: Department of Computer Science and Engineering, School of Computing, SRM Institute of Science and Technology, Tiruchirappalli, Tamil Nadu, India

## License

Released under the MIT License. Add a `LICENSE` file to the repository (on GitHub: **Add file → Create new file → name it `LICENSE`**, then choose the MIT template). The IPC dataset is subject to its own terms on Kaggle.

## Acknowledgements

BGE-M3 (BAAI), BERTopic, Sentence-Transformers, LangChain, NetworkX, Optuna, RAGAS, Groq and Gradio.
