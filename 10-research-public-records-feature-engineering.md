---
source: Multiple (15+ sources)
author: Various
date: 2026-04-07
type: research
tags: [public-records, feature-engineering, credit-risk, insurance, inquiry, property-assessment, regulatory]
---

# Research: Public Records Feature Engineering for Credit Risk & Insurance ML

## Search Keywords
public records, machine learning, credit risk, insurance underwriting, property assessment, inquiry embedding, feature engineering, event sequence, regulatory, fair lending

## Sources Consulted

### 1. Reverse Engineering FICO 8 (Michael Fowlie, Medium)
- URL: https://medium.com/@mfow/reverse-engineering-fico-8-d2d68315d20
- Type: Applied analysis
- Key contribution: Empirical feature weights for FICO 8 subscore categories; inquiry decay curves; public record coefficients

**Findings:**
- FICO 8 decomposes into 5 subscore categories: Payment History (most important, ~0.198 per 0.001), New Credit (~0.059), History Length (~0.040), Credit Mix (~0.041), Amounts Owed (~0.029)
- Public records encoding: `pub_rec_bankruptcies` coefficient = -12.4333; `tax_liens` = -0.9490
- Inquiry decay: last 6 months = -1.1 points; 6-12 months = -0.8 points; 12-24 months = -0.13 points
- Inquiry features: `inq_last_6mths` coefficient = -0.3191; `mths_since_recent_inq` = +0.2098
- Collections under $100 have minimal effect — threshold-based encoding, not linear
- Multi-task neural network (5 parallel subscore heads) achieved 14-point MAE vs OLS 19-point std dev

### 2. Sequential Deep Learning for Credit Risk Monitoring (Babaev et al., arXiv 2012.15330)
- URL: https://arxiv.org/abs/2012.15330
- Type: Academic paper
- Key contribution: TCN vs GRU vs GBDT on sequential credit card data

**Findings:**
- Traditional GBDT models plateau without new data sources or highly engineered features
- Temporal convolutional network (TCN) outperformed benchmark GBDT on credit card transaction sequences
- Proposed new credit card transaction sampling technique for exploiting long historical sequences
- Sequence-based learning avoids the need for hand-crafted time-windowed aggregations
- Architectures tested: RNN, TCN, GBDT (benchmark)

### 3. Embedding-Aware Feature Discovery (EAFD, arXiv 2603.15713, 2026)
- URL: https://arxiv.org/html/2603.15713v1
- Type: Academic paper (state of the art)
- Key contribution: Bridges pretrained event-sequence embeddings with LLM-driven interpretable feature generation

**Findings:**
- Production systems still rely heavily on handcrafted statistical features (interpretability, robustness, latency)
- CoLES embeddings struggle with temporal dynamics (R² = 0.327); NTP poorly captures amounts and categories
- EAFD uses alignment score (embedding-feature consistency) and downstream utility score (complementary signal)
- Embeddings capture aggregate statistics well; handcrafted features encode recency, burstiness, seasonality better
- EAFD achieves +5.8% over state-of-the-art embeddings, +19% over weaker representations
- Encoder refinements (log, exp, PLE, Time2Vec) improved temporal reconstruction by 85.68%
- Datasets: banking (Age/Gender Prediction, Rosbank Churn, DataFusion) + proprietary multi-target banking data

### 4. Sequence Embeddings for Insurance Fraud Detection (Fursov et al., arXiv 1910.03072)
- URL: https://arxiv.org/pdf/1910.03072
- Type: Academic paper
- Key contribution: Treatment embeddings for healthcare insurance fraud; SWEM vs TF-IDF+GBDT

**Findings:**
- 381,013 outpatient care claims from major international health insurer; ~2% fraud rate
- Treatment IDs: 2,205 anonymized categories grouped into 17 upper-level categories
- Architecture: Treatment Embedding (300d) → Aggregation (max-pool) → MLP → Extra Tower (meta-features) → Output
- Learned embeddings (SWEM-max) outperform TF-IDF+GBDT: AUC 0.9062 vs 0.8948
- PR AUC improvement more dramatic: 0.2445 vs 0.2036 (crucial for imbalanced fraud)
- Treatment distributions follow modified Zipf's law with heavier tails than NLP — need higher dimensionality (300d)
- Order is irrelevant for bill treatments (unlike temporal event sequences)
- Meta-features (age, gender, insurance type, doctor specialty) handled via separate "extra tower"

### 5. FICO Blog: Combining ML with Credit Risk Scorecards
- URL: https://www.fico.com/blogs/combining-machine-learning-credit-risk-scorecards
- Type: Industry blog (FICO)
- Key contribution: Teacher-Student methodology for hybrid ML+scorecard

**Findings:**
- Teacher-Student learning: ML model (Teacher) identifies complex patterns, segmented scorecards (Student) recode those insights
- Tree Ensemble Modeling (TEM): each tree on subset of data with handful of characteristics; shallow trees reduce overfitting
- ML excels with sparse default data (e.g., 0.2% default rate for home equity — too sparse for reliable WoE binning)
- Traditional scorecard WoE distributions become "noisy, choppy" with sparse data; ML extracts signal
- Neural networks mentioned as alternative to TEM with similar effectiveness

### 6. Inquiry Anatomy and Time Decay (SmartTimeless, 2025)
- URL: https://www.smarttimeless.com/2025/12/new-credit-inquiry-anatomy.html
- Type: Industry analysis
- Key contribution: Detailed inquiry decay windows and clustering patterns

**Findings:**
- 90-day window: Recent inquiries carry "meaningful predictive weight"
- 12-month window: Inquiries typically influence scores for up to 12 months (visible for 24 months)
- Single inquiry = benign; 5 inquiries across 30 days = "elevated risk behavior"
- Clustering is strongest risk marker in new credit category
- Rate-shopping windows: mortgage and auto inquiries grouped as single event within standardized window
- VantageScore inquiries lose weight faster than FICO 8/10T
- FICO 10T uses inquiry timing + behavioral context for default probability bands

### 7. US Fair Lending Perspective on Machine Learning (PMC 8216763)
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8216763/
- Type: Academic review
- Key contribution: Comprehensive regulatory framework for ML in lending

**Findings:**
- Prohibited features: race, color, religion, national origin, sex, marital status, age (direct use)
- Proxy discrimination: features correlating with protected classes create disparate impact even without explicit inclusion
- Two standards: disparate treatment (always illegal) and disparate impact (may be justified by business necessity)
- Adverse action notices must cite principal reasons; "not required to state how or why" the feature contributed
- Mitigation methods requiring explicit protected class data in training/inference are "unlikely to be considered acceptable"
- Measurement: Marginal Effect (ME), Adverse Impact Ratio (AIR), Standardized Mean Difference (SMD/Cohen's d)
- Multi-objective evolutionary search to balance accuracy vs disparate impact without protected class info
- Model governance: document input features, behavior, decisions for audit

### 8. CFPB Innovation Spotlight: Adverse Action Notices with AI/ML
- URL: https://www.consumerfinance.gov/about-us/blog/innovation-spotlight-providing-adverse-action-notices-when-using-ai-ml-models/
- Type: Regulatory guidance (CFPB)
- Key contribution: Official CFPB position on ML model explainability requirements

**Findings:**
- ECOA + FCRA require adverse action notice regardless of technology used
- 2022 Circular correction: "ECOA and Regulation B do not permit creditors to use technology for which they cannot provide accurate reasons for adverse actions"
- Notices must "accurately describe the features actually considered"
- Creditors need not explain the mechanism; relationship to creditworthiness may be unclear to applicant
- More than 4 reasons may not be meaningful to consumer
- Built-in flexibility compatible with AI algorithms, but accuracy of reasons is non-negotiable

### 9. Kaggle Home Credit Default Risk — Feature Engineering
- URL: https://randlow.github.io/posts/machine-learning/kaggle-home-loan-credit-risk-feat-eng-p2/
- Related: https://www.kaggle.com/c/home-credit-default-risk
- Type: Competition analysis
- Key contribution: Practical feature engineering from bureau data

**Findings:**
- Bureau data flattened via groupby + agg([count, mean, median, min, max, sum]) across numeric columns
- Key temporal fields: DAYS_CREDIT, DAYS_CREDIT_UPDATE, DAYS_CREDIT_ENDDATE
- DAYS_CREDIT_UPDATE median: -348.9 for defaulters vs -492.8 for non-defaulters (more recent bureau activity = higher risk)
- One-hot encoding for categorical: CREDIT_TYPE (consumer, credit card, car loan), CREDIT_ACTIVE (closed, active)
- GRU networks trained on time-series tables (installments, POS-cash, credit card, bureau balance) to extract sequence features
- Winning solutions combined handcrafted aggregated features + GRU sequence features as inputs to GBDT

### 10. Time Series Credit Risk Assessment (Advancing Credit Risk, ScienceDirect 2025)
- URL: https://www.sciencedirect.com/science/article/pii/S0169023X25000850
- Type: Academic paper
- Key contribution: Daily balance time series vs monthly aggregations for credit risk

**Findings:**
- CIF (Canonical Interval Forests) + XGBoost hybrid achieved best results (AUC 0.81 vs 0.79 CIF alone vs 0.77 XGBoost without temporal)
- Performance dropped significantly using aggregated monthly data — preserving high-frequency signals is critical
- Shapelets, LSTM, CIF, Logistic Regression, XGBoost all tested
- Model stacking via complementary ROC regions provides practical benefit
- Daily end-of-day balance data captures temporal patterns lost in snapshots

### 11. CAPE Analytics — AI Property Insurance Underwriting
- URL: https://capeanalytics.com/blog/ai-property-insurance-underwriting/
- Type: Industry blog
- Key contribution: Property features from aerial imagery + public records

**Findings:**
- Features: roof condition/age, wildfire risk, pool, trampoline, yard debris, 100s of other risk factors
- Data sources: aerial imagery, public records, county tax assessor, crime statistics, repair permit applications, claims history
- Replacement cost estimates from ML models
- COPE framework: Construction, Occupancy, Protection, Exposure for commercial properties
- Computer vision for structural assessment, material identification, wear detection

### 12. Verisk ISO Risk Analyzer — Insurance Scoring
- URL: https://www.verisk.com/products/iso-risk-analyzer-suite/
- Type: Industry product
- Key contribution: Insurance scoring model architecture

**Findings:**
- ISO Risk Analyzer Homeowners examines 100s of indicators per policy by peril
- Building characteristics module: age of roof, number of rooms, square footage, lot size, pool, finished basement
- 9 major homeowners peril categories with per-peril loss cost relativities
- Variable interaction effects modeled (which variables matter, how much, how they interact)
- ZIP code/Census block group granularity for territory risk
- TransUnion + ISO A-PLUS combined credit + loss history scores for auto and property underwriting

### 13. Experian Public Records & Inquiry Data
- URL: https://www.experian.com/blogs/ask-experian/credit-education/report-basics/understanding-your-experian-credit-report/
- Related: https://www.experian.com/business/solutions/data-solutions/public-data-records
- Type: Bureau documentation
- Key contribution: Data schema for public records and inquiries

**Findings:**
- Public records on consumer credit: ONLY bankruptcy (since 2018 — tax liens and civil judgments removed)
- Business records: 1,400+ government municipalities at county/state/federal levels
- Hard inquiries: auto loans, mortgages, credit applications — shared with third parties, impact scores
- Soft inquiries: account reviews, prescreening, self-checks — not shared, no score impact
- Inquiry retention: 2 years on report; bankruptcy: up to 10 years
- Trade line distribution: bank cards (40%), retail (18%), collections (13%), education (7%), auto (4%), mortgage (7%)

### 14. ATTOM Data — Property Assessor Data
- URL: https://www.attomdata.com/data/property-data/assessor-data/
- Type: Data provider documentation
- Key contribution: Property assessment data schema

**Findings:**
- Property assessor data includes: bedrooms, bathrooms, construction material, property identification, addresses, ownership history, legal descriptions, features, values, taxes
- Critical baseline for generating property and casualty insurance policies
- Used for automated valuations, BPOs, and appraisals
- Census tract and ZIP code level construction/sales data available

### 15. Credit Risk Feature Engineering — Deep Learning Feature Extraction
- URL: Various (FinRegLab, arXiv, PMC)
- Type: Multiple academic sources
- Key contribution: DNN/autoencoder approaches for credit feature extraction

**Findings:**
- SAFE-DNN identifies: number of inquiries past 6 months, personal finance inquiries, credit inquiries past 12 months as key features
- Autoencoders for unsupervised latent feature extraction from complex credit datasets
- Deep belief networks achieve higher accuracy than shallower networks on credit scoring
- Multi-source data merging: applicant details, credit card details, payment history, bureau reports
