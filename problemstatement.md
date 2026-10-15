# SemEval-2027 Task 2: Directional-Consistent Fine-Grained Natural Language Inference (DiCo-NLI)
## Problem Statement, Formal Specification & Engineering Benchmark Guide

---

## 1. Executive Summary & Overview

Natural Language Inference (NLI) benchmarks (e.g., SNLI, MNLI, XNLI) traditionally formulate inference as an independent, single-direction classification task over full sentence pairs: given an ordered premise-hypothesis pair $(P, H)$, predict whether $P$ entails $H$, contradicts $H$, or is neutral to $H$. 

However, modern foundation models and bi-encoders exhibit systematic reasoning failures when evaluated on **directional consistency**:
1. **Symmetric Bias & Shallow Matching**: Models frequently rely on bidirectional token overlap, co-occurrence statistics, and semantic similarity cues, ignoring the strict logical directionality of entailment.
2. **The Reversal Curse**: As demonstrated by [Berglund et al. (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5178b2f2d7c44aa390c0777dc77b3f0c-Abstract-Conference.html), models trained on statements in one direction (e.g., "$A$ entails $B$") fail to infer or generalize the symmetric or converse relation when presented with the inverted input ("$B$ is entailed by $A$").
3. **Coarse Sentence Granularity**: Conventional NLI conflates phrase-level semantic relations, quantifiers, and syntactic modifiers into a single aggregate judgment, masking local inference failures.

**DiCo-NLI (Directional-Consistent Fine-Grained Natural Language Inference)**, accepted as **SemEval-2027 Task 2**, establishes a unified, rigorous multilingual benchmark designed to evaluate whether NLI systems make **consistent, direction-sensitive decisions** over paired phrase instances across 4 distinct tracks (English, Spanish, Basque, and Mixed Multilingual).

* **Official Repository**: [ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI)
* **Team Repository**: [NIKHIL-07-CYBER/SemEval-2027-Task-2-Directional-Consistent-Fine-Grained-Natural-Language-Inference-DiCo-NLI-](https://github.com/NIKHIL-07-CYBER/SemEval-2027-Task-2-Directional-Consistent-Fine-Grained-Natural-Language-Inference-DiCo-NLI-)
* **CodaBench Competition**: [CodaBench Development Phase #18038](https://www.codabench.org/competitions/18038/)

---

## 2. Core Problem Formulation & Architecture

```
                          ┌────────────────────────────────────────────────────────┐
                          │               Source Phrase Pair: (A, B)               │
                          └──────────────────────────┬─────────────────────────────┘
                                                     │
                          ┌──────────────────────────┴────────────────────────────┐
                          │                                                       │
                          ▼                                                       ▼
            ┌───────────────────────────┐                           ┌───────────────────────────┐
            │   Forward Instance: x     │                           │   Reversed Instance: x_rev│
            │    Premise: A             │                           │    Premise: B             │
            │    Hypothesis: B          │                           │    Hypothesis: A          │
            └─────────────┬─────────────┘                           └─────────────┬─────────────┘
                          │                                                       │
                          ▼                                                       ▼
            ┌───────────────────────────┐                           ┌───────────────────────────┐
            │   Model Prediction: f(x)  │                           │ Model Prediction: f(x_rev)│
            └─────────────┬─────────────┘                           └─────────────┬─────────────┘
                          │                                                       │
                          └──────────────────────────┬────────────────────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │   Consistency Verification   │
                                      │   f(x) == Rev(f(x_rev))?     │
                                      └──────────────┬───────────────┘
                                                     │
                       ┌─────────────────────────────┴─────────────────────────────┐
                       ▼                                                           ▼
         ┌──────────────────────────────┐                            ┌──────────────────────────────┐
         │ SoftCons Check (Self-Align)  │                            │ HardCons Check (Dual Correct)│
         │ Checks internal consistency  │                            │ Both f(x) == y AND           │
         │ independent of gold labels   │                            │ f(x_rev) == y_rev            │
         └──────────────────────────────┘                            └──────────────────────────────┘
```

### 2.1 Fine-Grained Phrase Units (PhrasIS Lineage)
DiCo-NLI operates on short phrase pairs derived from **PhrasIS**, built from naturally occurring image captions and news headlines following the Interpretable STS annotation framework. PhrasIS decomposes sentences into aligned chunks:
- Noun phrases (`NP`)
- Verb chains (`VP`)
- Prepositional phrases (`PP`)
- Adverbial expressions (`ADVP`)

This guarantees that inference is assessed over **compositional semantic units** rather than bag-of-words or unconstrained multi-sentence paragraphs.

### 2.2 Reversal Operator & Semantic Relations

For any ordered phrase pair $(P, H)$ and its inverted counterpart $(H, P)$, the semantic labels obey a deterministic reversal operator $\text{Rev}(y)$:

| Relation Label | Semantics | Inverted Counterpart $\text{Rev}(y)$ | Consistency Class |
| :--- | :--- | :--- | :--- |
| `EQUIVALENCE` | $P \iff H$ (Mutual entailment) | `EQUIVALENCE` | Reversible Set |
| `FORWARD_ENTAILMENT` | $P \implies H$ ($P$ is strictly more specific than $H$) | `BACKWARD_ENTAILMENT` | Reversible Set |
| `BACKWARD_ENTAILMENT` | $H \implies P$ ($H$ is strictly more specific than $P$) | `FORWARD_ENTAILMENT` | Reversible Set |
| `NEGATIVE_OTHER` | Non-entailment, independence, or noise outside reversible set | N/A (Item-level only) | Irreversible Noise |

#### Illustrative Example:
* Forward Pair:
  * Premise: `"Everyone is hungry"`
  * Hypothesis: `"Someone is hungry"`
  * Gold Label: `FORWARD_ENTAILMENT`
* Reversed Pair:
  * Premise: `"Someone is hungry"`
  * Hypothesis: `"Everyone is hungry"`
  * Gold Label: `BACKWARD_ENTAILMENT`

---

## 3. Official Evaluation Metrics

The official benchmark reports three primary complementary scores for each track:

### 3.1 Weighted F1
Standard item-level multi-class classification performance across all 4 labels:
$$\text{Weighted F1} = \sum_{c \in \mathcal{C}} w_c \cdot \text{F1}_c$$
where $w_c = \frac{N_c}{N_{\text{total}}}$, including `NEGATIVE_OTHER`.

### 3.2 Soft-Consistency ($\text{SoftCons}$)
Measures whether a system is internally logically consistent with its own predictions across reversal, independent of gold ground truth.
$$\text{SoftCons}(x) = \begin{cases} 1 & \text{if } f(x) = \text{Rev}(f(x_{\text{rev}})) \\ 0 & \text{otherwise} \end{cases}$$
$$\text{SoftCons}_{\text{Macro}} = \frac{1}{|\mathcal{P}_{\text{rev}}|} \sum_{(x, x_{\text{rev}}) \in \mathcal{P}_{\text{rev}}} \mathbb{I}[f(x) = \text{Rev}(f(x_{\text{rev}}))]$$
*Evaluated exclusively on reversible source pairs.*

### 3.3 Hard-Consistency ($\text{HardCons}$)
Measures paired directional correctness: both the forward and reversed instances must match their respective gold labels.
$$\text{HardCons}(x) = \begin{cases} 1 & \text{if } f(x) = y \land f(x_{\text{rev}}) = y_{\text{rev}} \\ 0 & \text{otherwise} \end{cases}$$
$$\text{HardCons}_{\text{Macro}} = \frac{1}{|\mathcal{P}_{\text{rev}}|} \sum_{(x, x_{\text{rev}}) \in \mathcal{P}_{\text{rev}}} \mathbb{I}[f(x) = y_{(x)} \land f(x_{\text{rev}}) = y_{(x_{\text{rev}})}]$$

---

## 4. Benchmark Tracks & Cross-Lingual Structure

DiCo-NLI provides 4 evaluation tracks designed to test monolingual and cross-lingual robustness:

| Track ID | Track Name | Target Setting | Language Pairings |
| :--- | :--- | :--- | :--- |
| **Track 1** | **English** | Monolingual English | EN $\to$ EN |
| **Track 2** | **Spanish** | Monolingual Spanish | ES $\to$ ES |
| **Track 3** | **Basque** | Monolingual Basque (Agglutinative / Low-Resource) | EU $\to$ EU |
| **Track 4** | **Mixed Multilingual** | Cross-Lingual Directional Consistency | EN $\leftrightarrow$ ES, EN $\leftrightarrow$ EU, ES $\leftrightarrow$ EU |

### Zero-Leakage Split Design
All datasets strictly enforce **source-pair grouping**:
For every original phrase pair $(A, B)$, all derived instances are placed strictly into the same split (Train or Dev):
1. Original ordered item $(A, B)$
2. Reversed item $(B, A)$
3. Spanish and Basque translated counterparts
4. Mixed-language cross-lingual permutations

*This prevents directional or cross-lingual leakage across train and test splits.*

---

## 5. Dataset Statistics & Splits

### Instance Counts per Track
| Split | Track 1 (EN) | Track 2 (ES) | Track 3 (EU) | Track 4 (Mixed) | Total Instances |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 3,042 | 3,042 | 3,042 | 18,252 | **27,378** |
| **Dev** | 660 | 660 | 660 | 3,960 | **5,940** |
| **Total** | 3,702 | 3,702 | 3,702 | 22,212 | **33,318** |

### Label Distribution
| Label | Tracks 1–3 Train | Track 4 Train | Tracks 1–3 Dev | Track 4 Dev |
| :--- | :---: | :---: | :---: | :---: |
| `FORWARD_ENTAILMENT` | 881 (29.0%) | 5,286 (29.0%) | 190 (28.8%) | 1,140 (28.8%) |
| `BACKWARD_ENTAILMENT`| 881 (29.0%) | 5,286 (29.0%) | 190 (28.8%) | 1,140 (28.8%) |
| `EQUIVALENCE` | 802 (26.4%) | 4,812 (26.4%) | 174 (26.4%) | 1,044 (26.4%) |
| `NEGATIVE_OTHER` | 478 (15.7%) | 2,868 (15.7%) | 106 (16.1%) | 636 (16.1%) |
| **Total** | **3,042** | **18,252** | **660** | **3,960** |

---

## 6. Pilot Benchmark Results (LREC 2026 Reference)

Prior work on the English pilot benchmark established baseline performance across major architecture families:

| Architecture Family | Representative Model | Weighted F1 | SoftCons | HardCons |
| :--- | :--- | :---: | :---: | :---: |
| **Encoder-Only** | **DeBERTa v3-base** | **0.75** | **0.84** | **0.79** |
| **Encoder-Decoder** | BART-large | 0.68 | 0.80 | 0.72 |
| **Decoder-Only** | OPT-125M | 0.61 | 0.65 | 0.58 |

**Key Finding**: Pre-trained encoder-only models with bidirectional cross-attention (e.g., DeBERTa-v3) significantly outperform causal autoregressive decoders on fine-grained directional consistency.

---

## 7. Submission Specifications

Submissions are scored via CodaBench or local CLI.
A submission file must be a 2-column CSV formatted as:

```csv
instance_id,label
dico_phrasis_0000002__en-en__flipped,FORWARD_ENTAILMENT
dico_phrasis_0000002__en-en__original,BACKWARD_ENTAILMENT
...
```

Valid label vocabulary (case-sensitive):
- `EQUIVALENCE`
- `FORWARD_ENTAILMENT`
- `BACKWARD_ENTAILMENT`
- `NEGATIVE_OTHER`

---

## 8. Development & Scorer CLI Quickstart

### Local Evaluation
```bash
python -m evaluation_functions \
  --gold final_data/dev/dico_nli_dev_track1_reference.csv \
  --predictions results/my_track1_dev_predictions.csv \
  --output-dir results/track1_eval_scores
```

### Starter Kit Training
```bash
cd starter_kit
python main.py \
  --train-file ../final_data/train/dico_nli_train_track1_participant_labeled.csv \
  --dev-file ../final_data/dev/dico_nli_dev_track1_participant_labeled.csv \
  --model-name microsoft/deberta-v3-base \
  --output-dir results/deberta_track1
```
