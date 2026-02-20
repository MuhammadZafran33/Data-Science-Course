<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0d0221,40:0a1628,70:1a1035,100:2d1b69&height=240&section=header&text=Statistics%20%26%20Probability&fontSize=52&fontColor=a78bfa&fontAlignY=38&desc=%F0%9F%94%AE%20The%20Mathematical%20Engine%20Behind%20Every%20Data%20Science%20Model&descAlignY=60&descSize=17&animation=fadeIn"/>

<br/>

<p>
  <img src="https://img.shields.io/badge/Module-Statistics%20%26%20Probability-7c3aed?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Course-WsCube%20Tech%20DS-a78bfa?style=for-the-badge&logo=youtube&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Notebooks-12+-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Topics-30%2B%20Concepts-blueviolet?style=flat-square"/>
  <img src="https://img.shields.io/badge/Level-Beginner%20%E2%86%92%20Intermediate-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/github/last-commit/MuhammadZafran33/Data-Science-Course?style=flat-square&color=violet"/>
</p>

<br/>

> ### 🎯 *"Statistics is the grammar of science"* — Karl Pearson
>
> This module covers the complete **Statistics & Probability** foundation required for Data Science —
> from measures of central tendency all the way to hypothesis testing & probability distributions.

<br/>

</div>

---

## 📚 Table of Contents

| # | Section | Quick Link |
|---|---------|-----------|
| 01 | 🗺️ Module Overview | [Jump](#️-module-overview) |
| 02 | 🧭 Learning Roadmap | [Jump](#-learning-roadmap) |
| 03 | 📖 Topic Deep Dive | [Jump](#-topic-deep-dive) |
| 04 | 📊 Coverage Charts | [Jump](#-coverage-charts) |
| 05 | 🔬 Probability Distributions | [Jump](#-probability-distributions-at-a-glance) |
| 06 | 🧪 Hypothesis Testing Guide | [Jump](#-hypothesis-testing-decision-guide) |
| 07 | 📐 Key Formulas Cheatsheet | [Jump](#-key-formulas-cheatsheet) |
| 08 | 📁 Folder Structure | [Jump](#-folder-structure) |
| 09 | 🛠️ Tools & Libraries | [Jump](#️-tools--libraries) |
| 10 | 🚀 Getting Started | [Jump](#-getting-started) |

---

## 🗺️ Module Overview

<div align="center">

| 📌 Attribute | 📋 Details |
|-------------|-----------|
| 🎓 **Parent Course** | Data Science Full Course — WsCube Tech |
| 📂 **Module Name** | Statistics & Probability |
| 📍 **Position in Course** | Module 14–15 (Foundations before ML) |
| ⏱️ **Study Duration** | ~2–3 Weeks · 15+ Hours |
| 📓 **Notebooks** | 12+ Jupyter Notebooks |
| 🎯 **Why It Matters** | Every ML algorithm is built on statistical principles |
| 🔗 **Leads To** | EDA → Machine Learning → Model Evaluation |

</div>

---

## 🧭 Learning Roadmap

```mermaid
flowchart TD
    START(["📊 START\nStatistics & Probability"])

    START --> A["📌 Introduction to Statistics\nData · Sample · Population\nTypes of Data"]

    A --> B["📈 Descriptive Statistics\nSummarize & describe data"]

    B --> B1["📍 Measures of Central Tendency\nMean · Median · Mode"]
    B --> B2["📏 Measures of Dispersion\nRange · Variance · Std Dev"]
    B --> B3["🔗 Bivariate Analysis\nCovariance · Correlation"]

    B1 & B2 & B3 --> C["🎲 Probability Theory\nCore concepts & rules"]

    C --> C1["🎰 Random Variables\nDiscrete & Continuous"]
    C --> C2["📐 Probability Distributions\nPDF & CDF"]

    C1 & C2 --> D["🔔 Key Distributions"]
    D --> D1["Normal\nDistribution"]
    D --> D2["Binomial\nDistribution"]
    D --> D3["Poisson\nDistribution"]

    D1 & D2 & D3 --> E["📉 Inferential Statistics"]
    E --> E1["🔄 Central Limit Theorem"]
    E --> E2["📊 Skewness & Kurtosis"]

    E1 & E2 --> F["🧪 Hypothesis Testing"]
    F --> F1["⚖️ Z-Test\nOne Sample"]
    F --> F2["📐 T-Test\nStudent's t"]
    F --> F3["🔲 Chi-Square\nTest"]

    F1 & F2 & F3 --> END(["✅ COMPLETE\nReady for EDA & ML"])

    style START fill:#7c3aed,stroke:none,color:#fff
    style END fill:#059669,stroke:none,color:#fff
    style A fill:#1e1b4b,stroke:#7c3aed,color:#c4b5fd
    style B fill:#1e1b4b,stroke:#7c3aed,color:#c4b5fd
    style C fill:#1e1b4b,stroke:#a78bfa,color:#c4b5fd
    style D fill:#1e1b4b,stroke:#a78bfa,color:#c4b5fd
    style E fill:#1e1b4b,stroke:#7c3aed,color:#c4b5fd
    style F fill:#1e1b4b,stroke:#7c3aed,color:#c4b5fd
    style B1 fill:#2d1b69,stroke:none,color:#e9d5ff
    style B2 fill:#2d1b69,stroke:none,color:#e9d5ff
    style B3 fill:#2d1b69,stroke:none,color:#e9d5ff
    style C1 fill:#2d1b69,stroke:none,color:#e9d5ff
    style C2 fill:#2d1b69,stroke:none,color:#e9d5ff
    style D1 fill:#4c1d95,stroke:none,color:#e9d5ff
    style D2 fill:#4c1d95,stroke:none,color:#e9d5ff
    style D3 fill:#4c1d95,stroke:none,color:#e9d5ff
    style E1 fill:#2d1b69,stroke:none,color:#e9d5ff
    style E2 fill:#2d1b69,stroke:none,color:#e9d5ff
    style F1 fill:#4c1d95,stroke:none,color:#e9d5ff
    style F2 fill:#4c1d95,stroke:none,color:#e9d5ff
    style F3 fill:#4c1d95,stroke:none,color:#e9d5ff
```

---

## 📖 Topic Deep Dive

### 🔷 PART 1 — Introduction to Statistics

| Topic | Description | Notebook |
|-------|-------------|:--------:|
| 📌 What is Statistics? | Definition, uses in data science & real-world examples | `01_intro_statistics.ipynb` |
| 👥 Data, Sample & Population | Difference between population parameters and sample statistics | `01_intro_statistics.ipynb` |
| 🏷️ Types of Data | Qualitative (nominal, ordinal) vs Quantitative (discrete, continuous) | `01_intro_statistics.ipynb` |

---

### 🔷 PART 2 — Descriptive Statistics

#### 📍 Measures of Central Tendency

| Measure | Formula | Best Used When | Notebook |
|---------|---------|---------------|:--------:|
| **Mean** | `Σx / n` | Symmetric data, no outliers | `02_central_tendency.ipynb` |
| **Median** | Middle value when sorted | Skewed data or outliers present | `02_central_tendency.ipynb` |
| **Mode** | Most frequent value | Categorical data | `02_central_tendency.ipynb` |

#### 📏 Measures of Dispersion

| Measure | Formula | What It Tells You | Notebook |
|---------|---------|------------------|:--------:|
| **Range** | `Max − Min` | Total spread of data | `03_dispersion.ipynb` |
| **Variance** | `Σ(x−μ)² / n` | Average squared deviation from mean | `03_dispersion.ipynb` |
| **Std Deviation** | `√Variance` | Spread in original units | `03_dispersion.ipynb` |
| **IQR** | `Q3 − Q1` | Middle 50% spread, robust to outliers | `03_dispersion.ipynb` |

#### 🔗 Bivariate Analysis

| Concept | Range | Interpretation | Notebook |
|---------|-------|---------------|:--------:|
| **Covariance** | `-∞ to +∞` | Direction of linear relationship | `04_bivariate.ipynb` |
| **Pearson Correlation** | `-1 to +1` | Strength + direction of relationship | `04_bivariate.ipynb` |

---

### 🔷 PART 3 — Probability & Distributions

#### 🎲 Probability Foundations

| Concept | Key Idea | Notebook |
|---------|---------|:--------:|
| **Random Variable** | Variable whose value is determined by a random experiment | `05_probability.ipynb` |
| **PDF** (Prob. Density Function) | Probability for continuous variables | `05_probability.ipynb` |
| **CDF** (Cumulative Distribution) | Probability that X ≤ x | `05_probability.ipynb` |
| **Normal Distribution** | Bell curve — the most important distribution in stats | `06_normal_dist.ipynb` |
| **Binomial Distribution** | Success/failure over n trials | `07_binomial_dist.ipynb` |
| **Poisson Distribution** | Count of events in a fixed interval | `08_poisson_dist.ipynb` |
| **Skewness** | Asymmetry of distribution around mean | `09_skewness.ipynb` |

---

### 🔷 PART 4 — Inferential Statistics & Hypothesis Testing

| Concept | Key Idea | Notebook |
|---------|---------|:--------:|
| **Central Limit Theorem** | Sample means are normally distributed regardless of population shape | `10_CLT.ipynb` |
| **Null Hypothesis (H₀)** | Default claim — no effect or difference exists | `11_hypothesis.ipynb` |
| **Alternate Hypothesis (H₁)** | What we aim to prove — effect or difference exists | `11_hypothesis.ipynb` |
| **p-value** | Probability of observing results if H₀ is true | `11_hypothesis.ipynb` |
| **Level of Significance (α)** | Threshold (typically 0.05) to reject H₀ | `11_hypothesis.ipynb` |
| **Confidence Interval** | Range that contains true parameter with (1−α)% certainty | `11_hypothesis.ipynb` |
| **One-Sample Z-Test** | Test population mean when σ is known, n ≥ 30 | `12_zttest.ipynb` |
| **Student's T-Test** | Test means when σ unknown or small sample | `12_zttest.ipynb` |
| **Chi-Square Test** | Test independence between categorical variables | `13_chi_square.ipynb` |

---

## 📊 Coverage Charts

### Content Distribution by Topic

```mermaid
pie title Statistics & Probability — Topic Coverage (Concepts)
    "Descriptive Statistics" : 25
    "Probability Theory" : 20
    "Probability Distributions" : 22
    "Inferential Statistics" : 15
    "Hypothesis Testing" : 18
```

### Time Investment Per Section

```mermaid
xychart-beta
    title "Estimated Study Hours per Section"
    x-axis ["Intro to Stats", "Central Tendency", "Dispersion", "Bivariate", "Probability", "Distributions", "CLT & Skewness", "Hypothesis Tests"]
    y-axis "Hours" 0 --> 4
    bar [1, 1.5, 2, 1.5, 2, 3, 2, 3]
    line [1, 1.5, 2, 1.5, 2, 3, 2, 3]
```

### Module Position in Full Course

```mermaid
gantt
    title Module 14-15 Position in WsCube Tech Data Science Course
    dateFormat  X
    axisFormat  Module %s

    section Core Python
    Python Fundamentals         :done,    m1,  1,  4
    Web Scraping                :done,    m2,  4,  5

    section Data Science Intro
    Intro to Data Science       :done,    m3,  5,  6
    Statistics and Probability  :active,  m4,  6,  8

    section Data Layer
    NumPy                       :         m5,  8,  9
    Pandas                      :         m6,  9, 11

    section Visualization
    Matplotlib and Seaborn      :         m7, 11, 13

    section Analysis
    EDA Projects                :         m8, 13, 15

    section ML
    Machine Learning            :         m9, 15, 22
```

---

## 🔬 Probability Distributions at a Glance

```mermaid
flowchart LR
    subgraph DISCRETE ["🎲 Discrete Distributions"]
        direction TB
        B["📊 Binomial\nBin(n, p)\nFixed trials,\nSuccess/Fail"]
        P["📈 Poisson\nPois(λ)\nEvents per\ninterval"]
    end

    subgraph CONTINUOUS ["〰️ Continuous Distributions"]
        direction TB
        N["🔔 Normal\nN(μ, σ²)\nBell curve\n68-95-99.7 rule"]
        U["⬜ Uniform\nU(a, b)\nEqual\nprobability"]
    end

    subgraph USE ["📌 When to Use?"]
        direction TB
        U1["n trials,\np(success)?\n→ Binomial"]
        U2["Count events\nin time/space?\n→ Poisson"]
        U3["Heights, test\nscores, errors?\n→ Normal"]
    end

    DISCRETE --> USE
    CONTINUOUS --> USE

    style B fill:#4c1d95,stroke:none,color:#e9d5ff
    style P fill:#4c1d95,stroke:none,color:#e9d5ff
    style N fill:#1e1b4b,stroke:#7c3aed,color:#c4b5fd
    style U fill:#1e1b4b,stroke:#7c3aed,color:#c4b5fd
    style U1 fill:#2d1b69,stroke:none,color:#e9d5ff
    style U2 fill:#2d1b69,stroke:none,color:#e9d5ff
    style U3 fill:#2d1b69,stroke:none,color:#e9d5ff
```

### Distribution Properties Comparison Table

| Distribution | Type | Parameters | Mean | Variance | Real-World Example |
|-------------|------|-----------|------|----------|-------------------|
| 🔔 **Normal** | Continuous | μ, σ | μ | σ² | Heights, IQ scores, measurement errors |
| 📊 **Binomial** | Discrete | n, p | np | np(1−p) | Coin flips, pass/fail tests, click-through rates |
| 📈 **Poisson** | Discrete | λ | λ | λ | Calls per hour, bugs per code file, accidents per day |
| ⬜ **Uniform** | Continuous | a, b | (a+b)/2 | (b−a)²/12 | Random number generation, dice rolls |

---

## 🧪 Hypothesis Testing Decision Guide

```mermaid
flowchart TD
    A(["🧪 Start:\nWhat do you want to test?"]) --> B{"Data type?"}

    B -- "Numerical\n(means)" --> C{"How many\ngroups?"}
    B -- "Categorical\n(frequencies)" --> G["🔲 Chi-Square Test\nIndependence or\ngoodness of fit"]

    C -- "1 group vs\nknown μ" --> D{"Sample size\n& σ known?"}
    C -- "2 independent\ngroups" --> E["📐 Independent\nSamples T-Test"]
    C -- "1 group\nbefore & after" --> F["📐 Paired\nT-Test"]

    D -- "n≥30 &\nσ known" --> H["⚡ One-Sample\nZ-Test"]
    D -- "n<30 or\nσ unknown" --> I["📐 One-Sample\nT-Test"]

    H & I & E & F & G --> J(["📋 Compute p-value"])

    J --> K{"p-value < α\n(typically 0.05)?"}
    K -- "YES ✅" --> L(["✅ Reject H₀\nResult is Statistically\nSignificant"])
    K -- "NO ❌" --> M(["❌ Fail to Reject H₀\nInsufficient Evidence"])

    style A fill:#7c3aed,stroke:none,color:#fff
    style L fill:#059669,stroke:none,color:#fff
    style M fill:#dc2626,stroke:none,color:#fff
    style J fill:#1e1b4b,stroke:#7c3aed,color:#c4b5fd
    style K fill:#4c1d95,stroke:none,color:#e9d5ff
    style B fill:#2d1b69,stroke:none,color:#e9d5ff
    style C fill:#2d1b69,stroke:none,color:#e9d5ff
    style D fill:#2d1b69,stroke:none,color:#e9d5ff
    style E fill:#1e1b4b,stroke:#a78bfa,color:#c4b5fd
    style F fill:#1e1b4b,stroke:#a78bfa,color:#c4b5fd
    style G fill:#1e1b4b,stroke:#a78bfa,color:#c4b5fd
    style H fill:#1e1b4b,stroke:#a78bfa,color:#c4b5fd
    style I fill:#1e1b4b,stroke:#a78bfa,color:#c4b5fd
```

### Hypothesis Tests Quick-Reference

| Test | Use Case | Condition | Python Function |
|------|---------|-----------|----------------|
| ⚡ **Z-Test** | Test one mean vs known μ | σ known, n ≥ 30 | `statsmodels.stats.weightstats.ztest` |
| 📐 **1-Sample T-Test** | Test one mean vs value | σ unknown / small n | `scipy.stats.ttest_1samp` |
| 📐 **2-Sample T-Test** | Compare two group means | Independent groups | `scipy.stats.ttest_ind` |
| 📐 **Paired T-Test** | Before vs after comparison | Same group, two measures | `scipy.stats.ttest_rel` |
| 🔲 **Chi-Square Test** | Association between categories | Expected freq ≥ 5 | `scipy.stats.chi2_contingency` |

---

## 📐 Key Formulas Cheatsheet

<div align="center">

| 📌 Concept | 🔢 Formula |
|-----------|-----------|
| **Population Mean** | `μ = Σxᵢ / N` |
| **Sample Mean** | `x̄ = Σxᵢ / n` |
| **Population Variance** | `σ² = Σ(xᵢ − μ)² / N` |
| **Sample Variance** | `s² = Σ(xᵢ − x̄)² / (n−1)` |
| **Standard Deviation** | `σ = √σ²` |
| **Z-Score (standardize)** | `z = (x − μ) / σ` |
| **Covariance** | `Cov(X,Y) = Σ(xᵢ−x̄)(yᵢ−ȳ) / (n−1)` |
| **Pearson Correlation** | `r = Cov(X,Y) / (σₓ · σᵧ)` |
| **Binomial PMF** | `P(X=k) = C(n,k) · pᵏ · (1−p)ⁿ⁻ᵏ` |
| **Poisson PMF** | `P(X=k) = (λᵏ · e⁻λ) / k!` |
| **Normal PDF** | `f(x) = (1/σ√2π) · e^(−(x−μ)²/2σ²)` |
| **Z-Test Statistic** | `z = (x̄ − μ₀) / (σ/√n)` |
| **T-Test Statistic** | `t = (x̄ − μ₀) / (s/√n)` |
| **Confidence Interval** | `x̄ ± z*(σ/√n)` |

</div>

---

## 📁 Folder Structure

```
📂 Statistics and Probability/
│
├── 📓 01_intro_to_statistics.ipynb
│   └── → What is stats · Data types · Sample vs Population
│
├── 📓 02_measures_central_tendency.ipynb
│   └── → Mean · Median · Mode · Weighted Mean
│
├── 📓 03_measures_of_dispersion.ipynb
│   └── → Range · Variance · Std Dev · IQR · Outliers
│
├── 📓 04_bivariate_analysis.ipynb
│   └── → Covariance · Pearson Correlation · Heatmaps
│
├── 📓 05_probability_fundamentals.ipynb
│   └── → Rules · Random Variables · PDF · CDF
│
├── 📓 06_normal_distribution.ipynb
│   └── → Bell curve · Z-score · Empirical Rule (68-95-99.7)
│
├── 📓 07_binomial_distribution.ipynb
│   └── → PMF · CDF · Simulation · Visualisation
│
├── 📓 08_poisson_distribution.ipynb
│   └── → Rate events · λ parameter · Real-world examples
│
├── 📓 09_skewness_and_kurtosis.ipynb
│   └── → Positive/Negative skew · Transformations
│
├── 📓 10_central_limit_theorem.ipynb
│   └── → CLT simulation · Sampling distributions
│
├── 📓 11_hypothesis_testing_basics.ipynb
│   └── → H₀ H₁ · p-value · α · Type I & II errors
│
├── 📓 12_z_test_and_t_test.ipynb
│   └── → One-sample Z · One & Two-sample T · Scipy
│
├── 📓 13_chi_square_test.ipynb
│   └── → Independence test · Contingency tables
│
└── 📄 README.md
```

---

## 🛠️ Tools & Libraries

<div align="center">

| 📦 Library | 🎯 Purpose | 💡 Key Functions Used |
|-----------|-----------|----------------------|
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Numerical computing & array operations | `np.mean()`, `np.std()`, `np.random.*` |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | Data manipulation & summary stats | `df.describe()`, `df.corr()`, `df.cov()` |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) | Static visualizations | `plt.hist()`, `plt.boxplot()`, `plt.scatter()` |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square) | Statistical visualizations | `sns.distplot()`, `sns.heatmap()`, `sns.boxplot()` |
| ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white) | Statistical tests | `scipy.stats.norm`, `ttest_ind`, `chi2_contingency` |
| ![Statsmodels](https://img.shields.io/badge/Statsmodels-4051B5?style=flat-square) | Advanced statistics | `ztest()`, `OLS()`, `anova_lm()` |

</div>

---

## 🚀 Getting Started

### Clone & Navigate

```bash
git clone https://github.com/MuhammadZafran33/Data-Science-Course.git
cd "Data-Science-Course/Data Science Full Course By WsCube Tech/Statistics and Probability"
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels jupyter
```

### Launch Notebooks

```bash
jupyter notebook
```

> ☁️ **No setup?** Run everything directly in the browser:

<div align="center">

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuhammadZafran33/Data-Science-Course/)

</div>

### 📋 Recommended Study Order

```mermaid
journey
    title Recommended Learning Path Through This Module
    section Week 1 — Descriptive Stats
      Intro to Statistics: 9: Learner
      Central Tendency: 9: Learner
      Measures of Dispersion: 8: Learner
      Bivariate Analysis: 7: Learner
    section Week 2 — Probability
      Probability Fundamentals: 8: Learner
      Normal Distribution: 9: Learner
      Binomial Distribution: 8: Learner
      Poisson Distribution: 7: Learner
    section Week 3 — Inference
      Skewness and Kurtosis: 7: Learner
      Central Limit Theorem: 8: Learner
      Hypothesis Testing: 7: Learner
      Z-Test and T-Test: 6: Learner
      Chi-Square Test: 6: Learner
```

---

## 🧠 Why Statistics Matters for Data Science

```mermaid
mindmap
  root((Statistics\nin Data Science))
    Machine Learning
      Model assumptions
      Feature selection
      Regularization logic
    EDA
      Outlier detection
      Distribution analysis
      Correlation mapping
    A/B Testing
      Hypothesis tests
      Confidence intervals
      Significance levels
    Deep Learning
      Weight initialization
      Batch normalization
      Loss function math
    Data Quality
      Anomaly detection
      Missing value treatment
      Data validation
```

---

## 🔗 Navigation

<div align="center">

| ⬅️ Previous Module | 📍 You Are Here | ➡️ Next Module |
|-------------------|----------------|---------------|
| [🌐 Web Scraping](../Web%20Scraping/) | **📊 Statistics & Probability** | [🔢 NumPy →](../NumPy/) |

</div>

---

<div align="center">

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-MuhammadZafran33-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MuhammadZafran33)

<br/>

> *"Statistical thinking will one day be as necessary for efficient citizenship*
> *as the ability to read and write."*
>
> **— H.G. Wells**

<br/>

**⭐ Found this helpful? Drop a star on the repo — it keeps the learning journey going! ⭐**

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:2d1b69,50:1e1b4b,100:0d0221&height=140&section=footer"/>

</div>
