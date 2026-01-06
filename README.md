# 🧪 Project 2: Hierarchical Clustering (Step-by-Step)

## 📌 Project Overview

This project demonstrates **Hierarchical Clustering**, an unsupervised machine learning technique, using the classic **Iris Flower Dataset**. The goal is to group flowers based solely on their physical measurements and observe whether natural clusters emerge — *without using labels*.

This repository is designed to be:

* ✅ Beginner-friendly
* ✅ Interview-ready
* ✅ Concept-focused with visual intuition

---

## 🎯 What You Will Learn

* How hierarchical clustering works internally
* How to **read and interpret a dendrogram**
* How to **choose the number of clusters visually**
* Differences between **Hierarchical Clustering vs K-Means**
* How to **explain this project confidently in interviews**

---

## 📂 Dataset Information

**Dataset Name:** Iris Flower Dataset
**Source:** Kaggle

### 🔹 Features Used

| Column Name | Description            |
| ----------- | ---------------------- |
| SepalLength | Sepal length of flower |
| SepalWidth  | Sepal width of flower  |
| PetalLength | Petal length of flower |
| PetalWidth  | Petal width of flower  |

🚫 **Species column is NOT used** (unsupervised learning)

---

## 🧠 Problem Statement

> Group flowers based only on their physical measurements and analyze whether meaningful natural clusters appear.

---

## 🛠️ Step-by-Step Implementation

### 🔹 Step 0: Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### 🔹 Step 1: Load the Dataset

```python
df = pd.read_csv("Iris.csv")
df.head()
```

---

### 🔹 Step 2: Drop the Label Column

```python
X = df.drop(columns=["Species"])
```

🧠 **Why?**
Hierarchical clustering is an **unsupervised learning** algorithm — no target variable is used.

---

### 🔹 Step 3: Feature Scaling (Mandatory)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

📌 Distance-based algorithms require scaling to prevent bias toward larger features.

---

### 🔹 Step 4: Create the Dendrogram (Key Step)

```python
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
```

### 🧠 How to Read a Dendrogram

* Vertical lines → clusters merging
* Large vertical gap → optimal place to cut
* Horizontal cut → decides number of clusters

📌 For Iris Dataset → **3 clusters** is optimal

---

### 🔹 Step 5: Apply Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

labels = model.fit_predict(X_scaled)
```

---

### 🔹 Step 6: Add Cluster Labels to Dataset

```python
df["Cluster"] = labels
df.head()
```

---

### 🔹 Step 7: Visualize the Clusters

```python
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df["PetalLength"],
    y=df["PetalWidth"],
    hue=labels,
    palette="viridis"
)
plt.title("Hierarchical Clustering of Iris Flowers")
plt.show()
```

---

### 🔹 Step 8: Cluster Interpretation

| Cluster | Interpretation       |
| ------- | -------------------- |
| 0       | Small petal flowers  |
| 1       | Medium petal flowers |
| 2       | Large petal flowers  |

📌 Even without labels, **natural grouping is clearly visible**.

---

## 🚀 Key Takeaways

* Hierarchical clustering does **not require predefined k**
* Dendrogram provides **strong visual intuition**
* Works well for **small to medium datasets**
* Excellent for **exploratory data analysis**

---
\

⭐ If you found this helpful, consider starring the repo and connecting with me!

Happy Learning & Building 🚀
