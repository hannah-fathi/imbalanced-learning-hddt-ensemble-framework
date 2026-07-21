# Imbalanced Learning with Hellinger Distance Decision Trees and Ensemble Methods

<p align="center">
A Research-Oriented Implementation of Imbalance-Aware Machine Learning Algorithms
</p>


## Overview

This repository presents a complete experimental study on **imbalanced classification learning**, focusing on decision tree splitting strategies, ensemble learning, and data balancing techniques.

The main objective of this project is to investigate how specialized learning algorithms can improve minority-class recognition in highly skewed datasets, where conventional classifiers usually suffer from majority-class bias.

The project implements and evaluates:

- Hellinger Distance Decision Tree (HDDT)
- Multiclass HDDT using One-Versus-All (OVA)
- Multiclass HDDT using One-Versus-One (OVO)
- Decision tree pruning analysis
- Bagging with majority-class under-sampling
- AdaBoost.M1 with SMOTE minority oversampling
- Imbalance-aware evaluation metrics


The implementations are developed from scratch where possible, with experimental evaluation following machine learning research protocols.


---

# Research Motivation

Class imbalance is one of the fundamental challenges in supervised machine learning.

In many real-world applications:

- Medical diagnosis
- Disease detection
- Fraud detection
- Fault monitoring
- Rare event prediction

the minority class is usually the most important class, while traditional classifiers tend to optimize global accuracy and ignore minority samples.

This project studies algorithmic solutions for improving minority-class detection through:

1. Imbalance-aware splitting criteria
2. Ensemble diversity
3. Synthetic minority generation


---

# Research Contributions

This repository includes the following contributions:

## 1. Implementation of Hellinger Distance Decision Tree

A decision tree classifier based on the **Hellinger Distance splitting criterion** was implemented.

Unlike classical decision trees using:

- Information Gain
- Gini Impurity

HDDT uses a distribution-based distance measure that is less sensitive to skewed class distributions.


Implemented components:

- Binary HDDT classifier
- Hellinger-distance feature selection
- Recursive tree construction
- Maximum depth control
- Pruning analysis


---

# 2. Binary Imbalanced Classification using HDDT

## Dataset

`Covid19HDDT.csv`

The original dataset contains three classes.

To formulate a binary imbalance problem:

- The smallest class was considered as minority.
- The remaining classes were merged into majority class.


## Experimental Protocol

Dataset split:

```

Training: 70%
Testing: 30%

```

Repeated experiments:

```

10 independent runs

```

Evaluation metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- G-Mean


## Results

Average performance over 10 runs:

| Metric | Performance |
|---|---:|
| Precision | 77.0% |
| Recall | 83.0% |
| F1-score | 80.0% |
| ROC-AUC | 87.5% |
| G-Mean | 78.5% |


The results demonstrate that HDDT can effectively identify minority samples despite severe class imbalance.


---

# 3. Multiclass HDDT Extension

Since the original HDDT algorithm is designed for binary classification, multiclass classification was implemented using:


## One-Versus-All (OVA)

Each class is modeled against all remaining classes.


## One-Versus-One (OVO)

A binary HDDT classifier is trained for every pair of classes.


Evaluation metrics:

- Macro Precision
- Macro Recall
- Macro F1-score
- Macro AUC
- G-Mean


Tree complexity analysis:

```

Maximum Depth:
{2,3,4,5}

```


The experiments analyze the relationship between tree complexity and generalization performance.


---

# 4. Decision Tree Pruning Analysis

To investigate overfitting behavior, HDDT models were trained with different maximum depths.


Experiment:

```

MaxDepth = 2,3,4,5

```


Findings:

- Deep trees increase model complexity.
- Moderate pruning improves generalization.
- Depth-limited trees maintain competitive performance with lower risk of overfitting.


---

# 5. Bagging with Majority Under-Sampling


## Objective

To reduce bias toward majority samples, an imbalance-aware Bagging strategy was implemented.


## Methodology


For each ensemble iteration:

1. Preserve minority samples.
2. Randomly under-sample majority samples.
3. Train a base decision tree.
4. Aggregate predictions using majority voting.


Pipeline:

```

Imbalanced Dataset

    |
    v

Minority Preservation

    |
    v

Majority Under-Sampling

    |
    v

Multiple Decision Trees

    |
    v

Majority Voting

    |
    v

Final Prediction

```


Dataset:

```

Covid.csv

```


Ensemble sizes:

```

T = {11,31,51,101}

```


Evaluation:

- Precision
- Recall
- F1-score
- ROC-AUC
- G-Mean


---

# 6. AdaBoost.M1 with SMOTE


## Dataset

`Abalone Dataset`

The original age prediction task was converted into binary classification.


Class definition:


Minority:

```

Rings <= 7

```


Majority:

```

Rings > 7

```


Minority ratio:

```

≈ 9.36%

```


---

# Data Processing Pipeline


```

Abalone Dataset

    |
    v

Categorical Encoding
(One-Hot Encoding)

    |
    v

Feature Standardization

    |
    v

SMOTE Oversampling

    |
    v

AdaBoost.M1

    |
    v

Performance Evaluation

```


---

# SMOTE Implementation


Synthetic minority samples were generated using:

- k-nearest-neighbor search
- Random interpolation between minority samples


Generated samples were added only to training folds to prevent data leakage.


---

# AdaBoost Configuration


Base learner:

```

Decision Stump
(max_depth = 1)

```


Number of estimators:

```

T = 50

```


Learning strategy:

```

AdaBoost.M1

```


---

# Experimental Evaluation


Validation strategy:

```

5-Fold Stratified Cross Validation

Repeated 5 Times

Total Evaluations:
25 Runs

```


Metrics:

- Accuracy
- AUROC
- AUPR


---

# AdaBoost Results


Comparison between original training and SMOTE-balanced training:


| Metric | Without SMOTE | With SMOTE |
|---|---:|---:|
| Accuracy | 0.76 ± 0.03 | 0.83 ± 0.02 |
| AUROC | 0.87 ± 0.02 | 0.91 ± 0.01 |
| AUPR | 0.78 ± 0.03 | 0.84 ± 0.02 |


SMOTE improved minority-class representation and increased both AUROC and AUPR, indicating better discrimination capability for imbalanced classification.


---

# Decision Boundary Visualization


The effect of synthetic minority generation was analyzed using:

Selected features:

```

Length
Diameter

```


## Before SMOTE


The classifier demonstrates stronger bias toward the majority region.


![Original Decision Boundary](results/Decision%20Boundary%20-%20Original%20(No%20SMOTE).png)



## After SMOTE


SMOTE expands minority representation and produces a more balanced decision region.


![SMOTE Decision Boundary](results/Decision%20Boundary%20-%20SMOTE%20Balanced.png)



---

# Repository Structure


```

imbalanced-learning-hddt-ensemble-framework

│
├── README.md
│
├── notebook
│   └── imbalanced_learning_hddt_ensemble.ipynb
│
├── datasets
│   ├── Covid.csv
│   ├── Covid19HDDT.csv
│   └── abalone.zip
│
├── results
│   ├── Decision Boundary - Original (No SMOTE).png
│   └── Decision Boundary - SMOTE Balanced.png
│
├── report
│   └── Research_Report.pdf
│
├── requirements.txt
│
├── LICENSE
│
└── CITATION.cff

```


---

# Reproducibility


## Environment

Python:

```

3.10+

```


Required libraries:

```

numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
jupyter

````


Run:

```bash
jupyter notebook imbalanced_learning_hddt_ensemble.ipynb
````

---

# Technical Skills Demonstrated

This project demonstrates experience with:

* Imbalanced Learning
* Decision Tree Algorithms
* Ensemble Learning
* AdaBoost
* Bagging
* SMOTE
* Model Evaluation
* Experimental Design
* Statistical Analysis

---

# Limitations

Current implementation focuses on classical machine learning approaches.

Future improvements:

* Cost-sensitive learning
* Gradient boosting approaches
* Explainable imbalance-aware models
* Deep learning methods with focal loss
* Medical AI applications

---

# Author

## Hannah Fathi

M.Sc. Artificial Intelligence and Robotics

**Spring 2025**

Research Interests:

* Computer Vision
* Medical Artificial Intelligence
* Remote Sensing
* Explainable AI
* Machine Learning

---

# Citation

If you use this implementation in academic work, please cite:

```
Fathi, Hannah
Imbalanced Learning with Hellinger Distance Decision Trees and Ensemble Methods.
GitHub Repository, 2025.
