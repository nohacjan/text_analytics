# Earnings Call Analysis

**Team:** Jan Noháč, Šimon Kačer, Tobias Moulis, Viktor Přibyl  
**Course:** 4IT347 Text Analytics 2 (VŠE Prague)  
**Prefinal snapshot:** Tagged as `prefinal` (preserved for peer review)

An end-to-end NLP pipeline and interactive Streamlit dashboard for inspecting S&P 500 quarterly earnings call transcripts (typically 15,000–20,000 words). The tool strips legal Safe Harbor boilerplate, parses speakers and analyst Q&A pairs, scores management tone and evasiveness, generates structured summaries, and answers transcript questions with verifiable citations. Alongside the dashboard, a separate quantitative backtester evaluates whether call sentiment and evasiveness predict post-earnings stock excess returns across 4,000 calls.

---

## Repository map

Everything below can be inspected directly on GitHub without running any code:

| Path | Description |
| :--- | :--- |
| `Earnings_Call_Demo_Dashboard_v7-1.ipynb` | Two-phase GPU pipeline (Colab T4): Safe Harbor cleaning, Regex + DeBERTa-v3 speaker parsing, FinBERT tone, Q&A evasion, BGE-M3 + BM25 + Cross-Encoder RAG, Qwen2.5-7B JSON summaries & 5-driver ABSA, and Streamlit launch. |
| `Earnings_Call_Backtest.ipynb` | Standalone quantitative backtester (NVIDIA L4): evaluates 4,000 S&P 500 calls vs. SPY using a purged chronological split, Loughran-McDonald lexicon, FinBERT, and regularized ML models. |
| `app.py` | 4-tab Streamlit dashboard (`Executive summary`, `Q&A explorer`, `Ask the transcript`, `Evaluation & diagnostics`) with side-by-side call comparison and multi-turn Q&A. |
| `rag_utils.py` | 3-stage retrieval module (`BAAI/bge-m3` dense + stopword-filtered `BM25Okapi` + `ms-marco-MiniLM-L-6-v2` cross-encoder reranking). |
| `docs/nsc-example.md` | Complete input query, model answer, and ground-truth source passage illustrating the Norfolk Southern CapEx error case. |
| `docs/screenshots/` | Screenshots of the running Streamlit dashboard (summary, Q&A explorer, and transcript Q&A views). |
| `report.pdf`, `failure-journal.pdf`, `contributions.pdf` | Prefinal submission documents. |

---

## Dashboard preview

The Streamlit app (`app.py`) serves the five saved v7 demonstration calls (**NSC**, **VLO**, **XYL**, **HD**, **ROL**) across four views:

1. **Executive summary:** Separates completed-quarter highlights from future guidance and risk factors, alongside Prepared Remarks vs. Q&A FinBERT tone.
2. **Q&A explorer:** Plots Q&A tone trajectory and per-question evasion scores with interactive tier/score filters and full exchange inspection.
3. **Ask the transcript:** Conversational RAG interface supporting `lookup`, `explain` (follow-ups), and `define` modes, Czech-to-English query rewriting, and expandable source chunks.
4. **Evaluation & diagnostics:** Displays saved retrieval ablation tables, summary factuality checks, and out-of-sample backtest scorecards.

<!-- Add your dashboard PNG screenshots to docs/screenshots/ and verify the filenames below -->
![Dashboard Executive Summary](docs/screenshots/01_executive_summary.png)
![Dashboard Q&A Explorer](docs/screenshots/02_qa_explorer.png)
![Dashboard Ask the Transcript](docs/screenshots/03_ask_transcript.png)

---

## Data schema & sample outputs

We do **not** republish raw scraped transcripts. Transcripts are loaded directly via Hugging Face from [`glopardo/sp500-earnings-transcripts`](https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts) (`train` split, 20,681 S&P 500 calls, 2013–2025), paired with daily adjusted OHLC stock and `SPY` benchmark prices via `yfinance`.

### Processed data schemas

- **Speaker turns (`all_demo_turns.parquet`):** `call_id`, `turn_id`, `speaker`, `role` (`Executive` | `Analyst` | `Operator`), `section` (`Prepared Remarks` | `Q&A`), `word_count`, `text`
- **Q&A exchanges (`all_demo_qa_analysis.csv`):** `call_id`, `pair_id`, `thread_id`, `is_followup`, `analyst`, `executives`, `handoff_count`, `question`, `answer`, `evasion_score` (0–100), `evasion_tier`, `semantic_alignment`, `answer_finbert_net`, `answer_finbert_label`
- **Backtest features (`backtest_features_4000.csv`):** `call_id`, `ticker`, `quarter`, `earnings_date`, `n_qa_pairs`, `lm_net_score`, `lm_neg_pct`, `lm_uncertainty_rate`, `finbert_score`, `finbert_prep_score`, `finbert_qa_score`, `finbert_tone_delta`, `qa_neg_pct`, `evasion_rate_pct`, `pre_5d_excess_ret`, `actual_T1_ret`, `spy_T1_ret`, `actual_T5_ret`, `spy_T5_ret`

### Sample saved RAG output (Norfolk Southern, `NSC 2015Q3`)

- **Query:** *"What did management say about revenue growth, margins, or capital allocation?"*
- **Model answer (`Qwen2.5-7B-Instruct`, excerpt):**
  > **Revenue Growth:** Management aims to grow the top line through market-based pricing at a rate better than rail inflation; revenue growth excluding fuel surcharge impacts was positive in Q2 and expected to continue for the remainder of the year [3].  
  > **Capital Allocation:** Management discussed a 5% CapEx reduction from the previous year and a target to manage capital spending to about 15% to 17% of sales going forward, shifting toward disciplined spending focused on revenue growth and profit margin [1, 3].

*(See Failure Case 1 below for the two subtle errors exposed when auditing this answer against the raw transcript.)*

---

## Evaluation metrics

### 1. Quantitative market backtest (blind out-of-sample, $n = 1,200$)

To eliminate look-ahead bias, 4,000 parsed calls are split chronologically: **Train** ($n = 2,796$, May 2013–Dec 2017), a **7-day purge/embargo gap** (4 boundary calls dropped to prevent $T+5$ label overlap), and **Blind OOS Test** ($n = 1,200$, Dec 2017–Jan 2025). All trades enter strictly at next-day `T+1 Open`. Baselines use frozen Train Z-score thresholds ($\mu \pm 0.5\sigma$); Tier 3 ML models tune regularization via 5-fold `TimeSeriesSplit` CV strictly inside Train and trade top/bottom 20% tails.

| Strategy (Blind OOS, $n = 1,200$) | Active Trades (Buy / Short) | T+1 Win | T+5 Win | Avg T+5 Alpha (vs. SPY) | T+5 IR | T+5 t-test ($H_0: \mu = 0$) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Tier 1: LM Lexicon ($\mu \pm 0.5\sigma$) | 742 (456 / 286) | 50.3% | 51.9% | +0.09% | +0.02 | $t = +0.54, p = 0.591$ (ns) |
| Tier 2a: FinBERT Whole Call | 775 (537 / 238) | 49.4% | 51.5% | -0.05% | -0.01 | $t = -0.36, p = 0.720$ (ns) |
| Tier 2b: FinBERT Q&A Only | 762 (356 / 406) | 47.6% | 48.7% | -0.24% | -0.06 | $t = -1.52, p = 0.128$ (ns) |
| **Tier 3a: Full LogReg (8 feats, Tail 20%)** | **565 (276 / 289)** | **52.0%** | **49.4%** | **+0.16%** | **+0.04** | **$t = +0.85, p = 0.397$ (ns)** |
| Tier 3b: Compact LogReg (4 feats, Tail) | 589 (228 / 361) | 52.5% | 47.4% | -0.10% | -0.02 | $t = -0.54, p = 0.591$ (ns) |
| Tier 3c: Full Ridge (8 feats, Tail 20%) | 545 (188 / 357) | 52.8% | 47.9% | +0.11% | +0.03 | $t = +0.59, p = 0.557$ (ns) |

**Key takeaway:** No strategy achieves statistically significant excess alpha ($p < 0.10$), and feature correlations with five-day excess returns stay between $-0.0487$ and $+0.0453$. Our **Downside Risk Audit** also refuted our starting hypothesis: the top-25% highest evasion + LM uncertainty cohort ($n = 1,000$) had a *lower* big-drop rate ($< -3\%$ vs. SPY: **15.7%**, avg excess $-0.09\%$) than the bottom-25% low-risk cohort (**17.5%**, avg excess $-0.16\%$). Public earnings news is largely priced into the overnight gap before `T+1 Open`, so we position this project as an analytical verification tool rather than a trading bot.

### 2. Retrieval ablation ($n = 20$ queries)

Chunk-agnostic gold answer snippet match (first 18 words of the recorded executive answer in v7).

| Configuration | Embedder | Queries ($n$) | Hit@1 | Hit@3 | MRR@3 |
| :--- | :--- | :---: | ---: | ---: | ---: |
| v3 Baseline: Fixed 500-word chunks (Dense) | `all-MiniLM-L6-v2` (256 tok) | 20 | 60.0% | 65.0% | 0.625 |
| v3 Window-fit: 220-word chunks (Dense) | `all-MiniLM-L6-v2` (256 tok) | 20 | 75.0% | 85.0% | 0.800 |
| v3 Hybrid: 220-word chunks (Dense + BM25) | `all-MiniLM-L6-v2` (256 tok) | 20 | 80.0% | 95.0% | 0.867 |
| v7 Baseline: Fixed 500-word chunks (Dense) | `BAAI/bge-m3` (8,192 tok) | 20 | 90.0% | 100.0% | 0.942 |
| v7 Turn-aware: Intact Q&A pairs (Dense) | `BAAI/bge-m3` (8,192 tok) | 20 | 95.0% | 100.0% | 0.975 |
| v7 Optimal: Intact Q&A pairs (BM25 + Cross-Encoder) | `BAAI/bge-m3` + MS-MARCO | 20 | 95.0% | 100.0% | 0.975 |

*Methodological note:* Under `MiniLM`, shrinking chunks to fit the 256-token window (+20 pp Hit@3) and adding BM25 (+10 pp) drove clear gains. Under v7, reusing verbatim analyst turns stored inside the target Q&A chunks saturates Hit@3 at 100%, so paraphrased and unanswerable queries are planned before Day 3.

### 3. Five saved v7 demo calls & summary factuality

| Call (Ticker, Quarter) | Q&As | FinBERT | Evasion % | Summary Outlook | ABSA Conf. / Hedging | Num. Grounding | Sem. Sim. | ROUGE-L |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: |
| **NSC** (2015Q3) | 48 | +25.2 | 1.0% | Cautious/Mixed | 7.0/10 · Moderate | 100.0% | 0.888 | 0.215 |
| **VLO** (2017Q2) | 29 | +16.2 | 3.4% | Cautious/Mixed | 7.0/10 · Moderate | 100.0% | 0.959 | 0.146 |
| **XYL** (2015Q1) | 23 | +33.1 | 2.2% | Cautious/Mixed | 7.0/10 · Moderate | 100.0% | 0.788 | 0.156 |
| **HD** (2017Q3) | 33 | +32.8 | 1.5% | Positive | 8.0/10 · Low | 100.0% | 0.906 | 0.263 |
| **ROL** (2018Q2) | 8 | +35.6 | 0.0% | Positive | 8.0/10 · Low | 100.0% | 0.776 | 0.227 |

---

## Analyzed failure cases

1. **Dropped scope qualifier & invented comparison period (`NSC 2015Q3`, RAG QA):**  
   For the CapEx query above, `Qwen2.5-7B` reported *"a 5% reduction from the previous year"* and *"a target to manage capital spending to about 15% to 17% of sales."* The source sentence actually reads: *"Going forward, we will try to manage capital spending **ex-PTC** down to about 15% to 17% of sales."* Dropping `ex-PTC` (Positive Train Control compliance spending) overstates the target scope, and *"from the previous year"* was fabricated from a mid-year reduction. Both errors pass automated numeric grounding (`5%`, `15%`, `17%` appear verbatim). Full trace in [`docs/nsc-example.md`](docs/nsc-example.md).
2. **Speaker parsing & post-2018 dropout (`Backtest`):**  
   Early spaCy NER parsing mislabelled banks named after people (*Goldman Sachs*, *Morgan Stanley*) as speakers. Switching to body-first `Name:` extraction plus a `DeBERTa-v3` NLI fallback recovered rosterless demo calls (`TROW`, `FIS`, `PANW`). At 4,000-call scale, however, the NLI step over-classified technical analyst questions about financials as `Executive`, causing `extract_qa_pairs` to reject calls with zero analysts and dropping annual coverage tenfold after 2018 (501–794 calls/year in 2014–2018 vs. 58–89 calls/year in 2019–2024). As a result, 64.7% of the OOS test set comes from 2018 alone.
3. **Prompt formatting over-caution (`24.11`, Dashboard RAG):**  
   When we adjusted the RAG prompt to answer in connected prose rather than bullet points, the model began asking users for clarification even when the question already named the exact figures to compare.
4. **Evasion compression under `BGE-M3`:**  
   Because `BGE-M3` assigns high baseline cosine similarity across financial text, the unadjusted `(1 - sim) * 25` term compressed call evasion rates to 0.0%–3.4% (in NSC, 47 of 48 exchanges scored below 38; only Q8 reached 40.0).

---

## Known limits & roadmap before Day 3 (December 10)

- **Human evaluation:** Complete 100 human-labelled Q&A sentiment chunks (25 per team member) + 20 shared chunks for Cohen's $\kappa$ (FinBERT vs. Loughran-McDonald), plus a 20-query paraphrased/unanswerable RAG test set and speaker-role accuracy audit.
- **Parser & evasion calibration:** Fix the DeBERTa NLI role threshold causing post-2018 attrition and Z-normalize `BGE-M3` evasion scores.
- **App integration:** Display saved `absa_sentiment.json` in `app.py`, wire `app.py` to `rag_utils.py` with `BGE-M3` offloaded to CPU (avoiding T4 OOM risk), unify backtest CSV paths, and replace 2,000-word summary truncation with guidance-preserving selection.

---

## Running the project

### Option 1: Google Colab (recommended, requires GPU)
1. Open [`Earnings_Call_Demo_Dashboard_v7-1.ipynb`](Earnings_Call_Demo_Dashboard_v7-1.ipynb) on a **T4 GPU** runtime and run all cells to generate processed data, ChromaDB vectors, summaries, and a live Cloudflare tunnel URL for Streamlit.
2. Open [`Earnings_Call_Backtest.ipynb`](Earnings_Call_Backtest.ipynb) on an **L4 GPU** runtime to reproduce the 4,000-call backtest.

### Option 2: Local Streamlit dashboard (using exported data)
```bash
pip install streamlit chromadb sentence-transformers plotly pandas pyarrow
export EARNINGS_PROJECT_DIR="./TA2_Earnings_Project_v7"
streamlit run app.py
```
*(In the sidebar, select **Google Gemini** or **OpenAI** API mode if running on a CPU-only machine without pre-cached local Qwen weights.)*
