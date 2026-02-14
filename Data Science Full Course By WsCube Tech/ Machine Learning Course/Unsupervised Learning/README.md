# 🤖 Unsupervised Learning - Complete Mastery Guide

<div align="center">

![Unsupervised Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised%20Learning-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate%20to%20Advanced-orange?style=for-the-badge)

**Master the Art of Learning Without Labels** 🎯

A comprehensive, hands-on course on Unsupervised Learning algorithms, techniques, and real-world applications.

[📚 What's Inside](#whats-inside) • [🎓 Learn](#learning-path) • [💻 Algorithms](#algorithms-overview) • [📊 Resources](#resources)

</div>

---

## 📖 Overview

Unsupervised Learning is a powerful machine learning paradigm where models learn patterns, structures, and relationships directly from unlabeled data. Unlike supervised learning, there's **no predefined target variable** — instead, algorithms discover hidden patterns and groupings on their own.

This course provides a comprehensive exploration of unsupervised learning techniques, from foundational clustering algorithms to advanced dimensionality reduction methods.

```
┌─────────────────────────────────────────────────────────┐
│        UNSUPERVISED LEARNING LANDSCAPE                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🎯 GOAL: Find Hidden Patterns in Unlabeled Data        │
│                                                         │
│  ├─ Clustering      → Group similar data points         │
│  ├─ Dimensionality  → Reduce feature complexity         │
│  │  Reduction                                           │
│  └─ Association     → Discover relationships            │
│     Rules                                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 What's Inside

This repository contains:

| 📂 Component | 📝 Description | 🔧 Difficulty |
|:---|:---|:---:|
| **Clustering Algorithms** | K-Means, Hierarchical, DBSCAN, GMM | ⭐⭐⭐ |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP, Autoencoders | ⭐⭐⭐⭐ |
| **Association Rules** | Apriori, Eclat, Market Basket Analysis | ⭐⭐ |
| **Anomaly Detection** | Isolation Forest, LOF, Statistical Methods | ⭐⭐⭐ |
| **Practical Projects** | Real-world datasets & implementations | ⭐⭐⭐⭐ |
| **Visualizations** | Interactive plots & 3D representations | ⭐⭐ |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MuhammadZafran33/Data-Science-Course.git

# Navigate to Unsupervised Learning folder
cd "Data-Science-Course/Data Science Full Course By WsCube Tech/Machine Learning Course/Unsupervised Learning"

# Install required packages
pip install -r requirements.txt
```

### Required Libraries

```python
# Core Data Science
numpy                # Numerical computing
pandas               # Data manipulation
scikit-learn         # ML algorithms
matplotlib           # Visualization
seaborn              # Statistical visualization

# Advanced Techniques
umap-learn           # Dimensionality reduction
plotly               # Interactive plots
scipy                # Scientific computing
tensorflow/keras     # Deep learning (autoencoders)
```

---

## 🧠 Algorithms Overview

### 1️⃣ CLUSTERING ALGORITHMS

Clustering is the task of grouping similar data points together without predefined labels.

#### Algorithm Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                    CLUSTERING ALGORITHMS                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  K-MEANS CLUSTERING        HIERARCHICAL CLUSTERING             │
│  ├─ Time: O(n·k·i)         ├─ Time: O(n² to n³)                │
│  ├─ Space: O(n·k)          ├─ Space: O(n²)                     │
│  ├─ Scalability: High       ├─ Scalability: Low                │
│  ├─ Clusters: Spherical     ├─ Clusters: Dendrogram            │
│  └─ Best for: Large data    └─ Best for: Small data            │
│                                                                │
│  DBSCAN                    GAUSSIAN MIXTURE MODELS             │
│  ├─ Time: O(n log n)       ├─ Time: O(n·k·i)                   │
│  ├─ Space: O(n)            ├─ Space: O(n·k)                    │
│  ├─ Scalability: Medium     ├─ Scalability: Medium             │
│  ├─ Clusters: Any shape     ├─ Clusters: Probabilistic         │
│  └─ Best for: Arbitrary     └─ Best for: Soft clustering       │
│     density shapes                                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### Detailed Comparison Table

| Algorithm | Time Complexity | Scalability | Shape | Interpretability | Use Case |
|:---|:---:|:---:|:---:|:---:|:---|
| **K-Means** | O(nki) | ⭐⭐⭐⭐⭐ | Spherical | ⭐⭐⭐⭐ | Large datasets, e-commerce |
| **Hierarchical** | O(n²) to O(n³) | ⭐⭐ | Any | ⭐⭐⭐⭐⭐ | Gene analysis, taxonomy |
| **DBSCAN** | O(n log n) | ⭐⭐⭐⭐ | Any | ⭐⭐⭐ | Anomaly detection, spatial data |
| **GMM** | O(nki) | ⭐⭐⭐ | Elliptical | ⭐⭐⭐⭐ | Soft clustering, generative models |
| **Spectral** | O(n³) | ⭐ | Any | ⭐⭐⭐ | Image segmentation, non-convex clusters |

---

### 2️⃣ DIMENSIONALITY REDUCTION

Reduce data complexity while preserving important information.

#### Method Comparison

```
DIMENSIONALITY REDUCTION TECHNIQUES
│
├─── LINEAR METHODS
│    ├─ PCA (Principal Component Analysis)
│    │  └─ Preserves variance, orthogonal components
│    │     ⏱️  Speed: Fast | 📊 Interpretability: High
│    │
│    ├─ ICA (Independent Component Analysis)
│    │  └─ Finds independent components
│    │     ⏱️  Speed: Moderate | 📊 Interpretability: Medium
│    │
│    └─ NMF (Non-Negative Matrix Factorization)
│       └─ For non-negative data
│          ⏱️  Speed: Fast | 📊 Interpretability: High
│
├─── NON-LINEAR METHODS
│    ├─ t-SNE (t-Distributed Stochastic Neighbor Embedding)
│    │  └─ Excellent for 2D/3D visualization
│    │     ⏱️  Speed: Slow | 🎨 Visualization: Excellent
│    │
│    ├─ UMAP (Uniform Manifold Approximation & Projection)
│    │  └─ Faster t-SNE alternative, preserves structure
│    │     ⏱️  Speed: Fast | 🎨 Visualization: Excellent
│    │
│    └─ Autoencoders (Deep Learning)
│       └─ Neural network-based compression
│          ⏱️  Speed: Depends | 📊 Power: Excellent
│
└─── MANIFOLD LEARNING
     ├─ Isomap
     ├─ Locally Linear Embedding (LLE)
     └─ Laplacian Eigenmaps
```

#### Performance Metrics

| Technique | Speed | Interpretability | Non-linear | Preserve Global | Best For |
|:---|:---:|:---:|:---:|:---:|:---|
| **PCA** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ❌ | ✅ | Quick analysis, preprocessing |
| **t-SNE** | ⚡⚡ | ⭐⭐ | ✅ | ❌ | Visualization, exploration |
| **UMAP** | ⚡⚡⚡⚡ | ⭐⭐⭐ | ✅ | ✅ | Visualization, large datasets |
| **ICA** | ⚡⚡⚡ | ⭐⭐⭐ | ❌ | ✅ | Blind source separation |
| **Autoencoders** | ⚡ | ⭐⭐ | ✅ | ✅ | Complex patterns, deep learning |

---

### 3️⃣ ANOMALY DETECTION

Identify unusual patterns and outliers in data.

```
ANOMALY DETECTION APPROACHES
│
├─ STATISTICAL METHODS
│  ├─ Z-Score: Distance from mean
│  ├─ IQR Method: Interquartile range
│  └─ Mahalanobis Distance: Multi-dimensional distance
│
├─ PROXIMITY-BASED
│  ├─ Isolation Forest
│  │  └─ Randomly isolates anomalies
│  │
│  ├─ Local Outlier Factor (LOF)
│  │  └─ Compares local density
│  │
│  └─ K-Nearest Neighbors (KNN)
│     └─ Based on neighbor distances
│
├─ MODEL-BASED
│  ├─ One-Class SVM
│  │  └─ Learns boundary of normal data
│  │
│  └─ Autoencoders
│     └─ High reconstruction error = anomaly
│
└─ ENSEMBLE METHODS
   └─ Combination of multiple techniques
```

---

### 4️⃣ ASSOCIATION RULES & MARKET BASKET ANALYSIS

Discover relationships between variables in transaction data.

| Algorithm | Approach | Time | Memory | Best For |
|:---|:---|:---:|:---:|:---|
| **Apriori** | Bottom-up, candidate generation | O(2^n) | High | Small-medium datasets |
| **Eclat** | Depth-first search, vertical format | O(2^n) | Medium | Finding frequent itemsets |
| **FP-Growth** | Prefix tree, pattern growth | O(n log n) | Low | Large datasets |

---

## 📊 Visual Learning Paths

### Beginner Path

```
START
  │
  ├─→ Understand Clustering Concepts
  │    └─→ K-Means Clustering (Iris, Wine datasets)
  │         └─→ Visualize clusters in 2D/3D
  │
  ├─→ Dimensionality Reduction
  │    └─→ PCA (Principal Component Analysis)
  │         └─→ Reduce high-dimensional data
  │
  └─→ Project: Customer Segmentation
       └─→ Apply K-Means + PCA on real data
```

### Intermediate Path

```
START
  │
  ├─→ Advanced Clustering
  │    ├─→ Hierarchical Clustering (Dendrograms)
  │    ├─→ DBSCAN (Density-based)
  │    └─→ Gaussian Mixture Models
  │
  ├─→ Non-linear Dimensionality Reduction
  │    ├─→ t-SNE for visualization
  │    └─→ UMAP for large datasets
  │
  ├─→ Anomaly Detection
  │    ├─→ Isolation Forest
  │    └─→ Local Outlier Factor
  │
  └─→ Project: Fraud Detection System
       └─→ Detect anomalies in transaction data
```

### Advanced Path

```
START
  │
  ├─→ Deep Unsupervised Learning
  │    ├─→ Autoencoders
  │    ├─→ Variational Autoencoders (VAE)
  │    └─→ Generative Adversarial Networks (GAN)
  │
  ├─→ Advanced Association Rules
  │    ├─→ FP-Growth Algorithm
  │    └─→ Sequential Pattern Mining
  │
  ├─→ Manifold Learning
  │    ├─→ Isomap
  │    ├─→ LLE (Locally Linear Embedding)
  │    └─→ Spectral Clustering
  │
  └─→ Project: Advanced Data Discovery
       └─→ Multi-algorithm ensemble approach
```

---

## 🎓 Learning Path

### Module 1: Clustering Fundamentals ⭐⭐
- [ ] Introduction to clustering
- [ ] K-Means algorithm from scratch
- [ ] Elbow method & silhouette score
- [ ] Practical: Iris & Wine datasets
- [ ] **Duration:** 3-4 hours | **Difficulty:** Beginner

### Module 2: Advanced Clustering ⭐⭐⭐
- [ ] Hierarchical clustering
- [ ] DBSCAN algorithm
- [ ] Gaussian Mixture Models
- [ ] Choosing the right algorithm
- [ ] **Duration:** 5-6 hours | **Difficulty:** Intermediate

### Module 3: Dimensionality Reduction ⭐⭐⭐
- [ ] PCA concepts & implementation
- [ ] t-SNE & UMAP
- [ ] Feature extraction vs selection
- [ ] Practical: High-dimensional datasets
- [ ] **Duration:** 4-5 hours | **Difficulty:** Intermediate

### Module 4: Anomaly Detection ⭐⭐⭐
- [ ] Statistical methods
- [ ] Isolation Forest
- [ ] Local Outlier Factor
- [ ] Real-world applications
- [ ] **Duration:** 4-5 hours | **Difficulty:** Intermediate

### Module 5: Deep Unsupervised Learning ⭐⭐⭐⭐
- [ ] Autoencoders
- [ ] Variational Autoencoders
- [ ] Generative models
- [ ] Applications in industry
- [ ] **Duration:** 6-8 hours | **Difficulty:** Advanced

### Module 6: Association Rules & Mining ⭐⭐⭐
- [ ] Market basket analysis
- [ ] Apriori algorithm
- [ ] FP-Growth
- [ ] Business applications
- [ ] **Duration:** 3-4 hours | **Difficulty:** Intermediate

---

## 💻 Implementation Examples

### 1. K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
```

### 2. PCA for Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Explained variance
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

### 3. Anomaly Detection with Isolation Forest

```python
from sklearn.ensemble import IsolationForest
import numpy as np

X = np.random.randn(100, 2)
X = np.vstack([X, [10, 10]])  # Add outlier

clf = IsolationForest(contamination=0.01)
outliers = clf.fit_predict(X)
```

---

## 📈 Algorithm Effectiveness Chart

```
Performance Across Different Scenarios
│
│ 100% ├─────────────────────────────────────
│      │
│  80% ├────┬─────┬─────────────────────────
│      │ K-M│ Hier│
│      │    │    │
│  60% ├────┼──┬──┤────┬────────────────────
│      │    │GM│D │tSNE│
│      │    │M │B │    │
│  40% ├────┼──┼──┼────┼───┬─────────────────
│      │    │  │  │ PCA│UMAP│
│      │    │  │  │    │    │
│  20% ├────┼──┼──┼────┼────┼───┬───────────
│      │    │  │  │    │    │AE │
│      │    │  │  │    │    │   │
│   0% └─────┴──┴──┴────┴────┴───┴───────────
│      Spherical Non-linear Complex Non-euclidean
│      Clusters  Patterns   Patterns  Spaces
│
Legend:
K-M    = K-Means
Hier   = Hierarchical
GMM    = Gaussian Mixture Models
DBSCAN = DBSCAN
PCA    = Principal Component Analysis
t-SNE  = t-Distributed SNE
UMAP   = Uniform Manifold Approximation
AE     = Autoencoder
```

---

## 📁 Folder Structure

```
Unsupervised Learning/
│
├── 1_Clustering/
│   ├── 1_K_Means.ipynb
│   ├── 2_Hierarchical_Clustering.ipynb
│   ├── 3_DBSCAN.ipynb
│   ├── 4_Gaussian_Mixture_Models.ipynb
│   └── datasets/
│
├── 2_Dimensionality_Reduction/
│   ├── 1_PCA.ipynb
│   ├── 2_t_SNE.ipynb
│   ├── 3_UMAP.ipynb
│   ├── 4_Autoencoders.ipynb
│   └── visualizations/
│
├── 3_Anomaly_Detection/
│   ├── 1_Statistical_Methods.ipynb
│   ├── 2_Isolation_Forest.ipynb
│   ├── 3_Local_Outlier_Factor.ipynb
│   └── datasets/
│
├── 4_Association_Rules/
│   ├── 1_Apriori.ipynb
│   ├── 2_Eclat.ipynb
│   └── market_basket_analysis.ipynb
│
├── 5_Real_World_Projects/
│   ├── Customer_Segmentation.ipynb
│   ├── Fraud_Detection.ipynb
│   └── Image_Compression.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🎯 Key Concepts Summary

| Concept | What It Does | When To Use | Pros | Cons |
|:---|:---|:---|:---|:---|
| **K-Means** | Partitions data into k clusters | Fixed, balanced clusters | Fast, scalable | Requires k specification |
| **Hierarchical** | Creates cluster dendrogram | Tree-like structures | Visual, interpretable | Computationally expensive |
| **DBSCAN** | Density-based clustering | Arbitrary shapes, outliers | Finds anomalies, flexible | Parameter tuning needed |
| **PCA** | Linear dimensionality reduction | Quick analysis, preprocessing | Fast, orthogonal | Loses non-linear info |
| **t-SNE** | Non-linear visualization | 2D/3D exploration | Beautiful visualizations | Slow, loses global structure |
| **UMAP** | Fast non-linear reduction | Large datasets, visualization | Fast, preserves structure | Newer, less studied |
| **Autoencoders** | Deep neural compression | Complex patterns | Very flexible, powerful | Requires large data |
| **Isolation Forest** | Anomaly detection | Outlier identification | Works with any data | Limited interpretability |

---

## 🔗 Resources & References

### 📚 Recommended Books
- "Pattern Recognition and Machine Learning" - Christopher M. Bishop
- "Unsupervised Learning" - Ethem Alpaydin
- "Machine Learning: A Probabilistic Perspective" - Kevin P. Murphy

### 🌐 Online Resources
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai Deep Learning](https://www.fast.ai/)

### 📰 Research Papers
- "A Tutorial on Clustering" - Andrew Moore
- "The Art and Science of Tuning Machine Learning Algorithms" - Limsoon Wong
- "UMAP: Uniform Manifold Approximation and Projection" - Leland McInnes

### 🎥 YouTube Channels
- [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
- [3Blue1Brown](https://www.youtube.com/@3blue1brown)
- [Code Basics](https://www.youtube.com/@codebasics)

---

## 📊 Comparison Matrix

### Algorithm Selection Guide

```
CHOOSE K-MEANS IF:
✓ You need fast clustering
✓ Clusters are roughly spherical
✓ You know the number of clusters
✓ Working with large datasets
✗ Clusters have irregular shapes

CHOOSE HIERARCHICAL IF:
✓ Need interpretable dendrogram
✓ Clusters are nested
✓ Small to medium dataset
✓ Want to explore different k
✗ Scalability is a concern

CHOOSE DBSCAN IF:
✓ Clusters are arbitrary shape
✓ Need to detect outliers
✓ Density varies across space
✓ Don't know number of clusters
✗ High-dimensional data

CHOOSE PCA IF:
✓ Need quick dimensionality reduction
✓ Interpretability is important
✓ Linear relationships exist
✓ Large dataset
✗ Need to preserve non-linear structure

CHOOSE t-SNE IF:
✓ Creating exploratory visualizations
✓ Want beautiful 2D/3D plots
✓ Cluster separation is important
✗ Need to preserve global structure
✗ Have very large dataset (slow)

CHOOSE UMAP IF:
✓ Need to preserve both local and global structure
✓ Visualization with larger datasets
✓ Want faster than t-SNE
✓ Scalability is important
```

---

## 🚀 Quick Tips & Best Practices

### Data Preprocessing
- ✅ Always scale/normalize features (StandardScaler, MinMaxScaler)
- ✅ Handle missing values before clustering
- ✅ Remove or handle outliers based on context
- ✅ Feature selection can improve results

### Choosing K (Number of Clusters)
- 📊 **Elbow Method**: Plot inertia vs k, look for "elbow"
- 📊 **Silhouette Score**: Closer to 1 is better
- 📊 **Davies-Bouldin Index**: Lower is better
- 📊 **Domain Knowledge**: Use business requirements

### Evaluation Metrics
- **Silhouette Score**: -1 to 1 (higher is better)
- **Davies-Bouldin Index**: Lower is better
- **Calinski-Harabasz Index**: Higher is better
- **Homogeneity, Completeness, V-measure** (when labels known)

### Common Pitfalls
- ❌ Not scaling features before clustering
- ❌ Using k-means for non-spherical clusters
- ❌ Forgetting to validate results
- ❌ Ignoring computational complexity
- ❌ Not visualizing results

---

## 🎁 Bonus: Interactive Decision Tree

```
START: Choose Your Algorithm
│
└─ What's your goal?
   │
   ├─ GROUPING DATA
   │  └─ Do you know clusters shape?
   │     ├─ YES, spherical
   │     │  └─→ K-MEANS ⭐⭐⭐
   │     │
   │     ├─ NO, arbitrary shape
   │     │  └─→ DBSCAN ⭐⭐⭐⭐
   │     │
   │     └─ Want tree structure?
   │        └─→ HIERARCHICAL ⭐⭐⭐
   │
   ├─ REDUCING DIMENSIONS
   │  └─ Need interpretability?
   │     ├─ YES
   │     │  └─→ PCA ⭐⭐⭐⭐
   │     │
   │     ├─ NO, want visualization
   │     │  ├─ Speed important?
   │     │  │  ├─ YES
   │     │  │  │  └─→ UMAP ⭐⭐⭐⭐
   │     │  │  │
   │     │  │  └─ NO, beautiful plots
   │     │  │     └─→ t-SNE ⭐⭐⭐⭐
   │     │  │
   │     │  └─ Complex patterns?
   │     │     └─→ AUTOENCODER ⭐⭐⭐⭐
   │
   └─ FINDING OUTLIERS
      └─→ ISOLATION FOREST ⭐⭐⭐

```

---

## 📞 Support & Contribution

### Questions? Issues?
- 📧 Open an issue on GitHub
- 💬 Check existing discussions
- 📖 Review notebook comments for explanations

### Want to Contribute?
- Fork the repository
- Add improvements or new algorithms
- Submit pull requests
- Share your projects!

### Code Standards
- Clear, commented code
- Docstrings for functions
- Example notebooks for implementations
- README updates for new content

---

## 📜 License & Attribution

This course material is based on the WsCube Tech Machine Learning curriculum, with enhancements and practical implementations.

**Created with ❤️ for Data Science Enthusiasts**

---

## 🎯 Your Learning Journey

```
Week 1-2: Understand Clustering
├─ Concepts
├─ K-Means
└─ Evaluation

Week 3-4: Advanced Clustering & PCA
├─ Hierarchical & DBSCAN
├─ Dimensionality Reduction
└─ Visualization

Week 5-6: Anomaly Detection & Deep Learning
├─ Outlier Detection
├─ Autoencoders
└─ Advanced Applications

Week 7-8: Real-World Projects
├─ Customer Segmentation
├─ Fraud Detection
└─ Capstone Project

🏆 MASTER UNSUPERVISED LEARNING! 🏆
```

---

<div align="center">

### **Happy Learning! 🚀**

> "Data is talking. Unsupervised learning helps you listen." - Unknown

**Star ⭐ this repository if you found it helpful!**

[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red?style=flat-square)](https://github.com/MuhammadZafran33)

</div>

---

*Last Updated: February 2026 | WsCube Tech ML Course*
