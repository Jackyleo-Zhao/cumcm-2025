# 2025 CUMCM Modeling Project

Solution to **Problem C** of the **2025 China Undergraduate Mathematical Contest in Modeling (CUMCM)**: **Timing Selection for NIPT and Fetal Abnormality Determination**.

This project combines **regression analysis, clustering, risk-aware optimization, and machine learning classification** to address a multi-stage decision problem in NIPT. Although the application background is biomedical, the core of this work is a complete **data-driven modeling workflow**, including quantitative analysis, personalized decision-making, and reproducible implementation.

## Highlights

- Built a **multivariate regression model** to analyze the relationship between fetal Y-chromosome concentration and maternal factors.
- Developed a **BMI-based timing strategy** using **K-means clustering** and risk optimization.
- Extended the framework to **multi-factor soft clustering** with a **Gaussian Mixture Model (GMM)**.
- Constructed an **XGBoost-based classifier** for abnormality detection in female fetuses.

## Key Results

- Optimal testing times under BMI-based grouping: **13.8, 15.2, 16.7 weeks**
- Optimal testing times under multi-factor GMM grouping: **13.2, 15.6, 18.3 weeks**
- Female fetal abnormality detection: **AUC = 0.962**

## Repository Structure

```text
2025-cumcm-modeling-project/
├── README.md
├── problem/
│   └── C题.pdf
├── paper/
│   └── paper.pdf
├── src/
│   ├── problem1.py
│   ├── problem2.py
│   ├── problem3.py
│   └── problem4.py
├── docs/
│   ├── format2025.doc
│   └── ai-usage-statement.pdf
└── .gitignore
