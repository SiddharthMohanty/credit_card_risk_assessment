# 📌 Credit Default Prediction with Imbalanced Data  

## 📖 Project Overview  
This project focuses on building machine learning models to **predict credit default**.  
The dataset is highly **imbalanced**, meaning the number of non-defaulters greatly exceeds defaulters.  
To handle this challenge, I explored **EDA, feature engineering, multiple ML algorithms, and SMOTE-based balancing techniques**.  

The objective is to:  
- Accurately identify **default cases** (minimize false negatives).  
- Compare model performance across different techniques.  
- Suggest improvements and next steps.  

---

## ⚙️ Steps Followed  

### 1. **Exploratory Data Analysis (EDA)**  
- Performed univariate and bivariate analysis to identify feature distributions and correlations.  
- Detected **unknown categories** in some categorical columns which were not mapped in the data dictionary.  
- Tried transformations and binning, but some categorical mappings did not provide significant improvements.  

**Challenges Faced in EDA:**  
- Imbalanced distribution caused misleading trends.  
- Presence of **unknown values** in categorical columns (no mapping available).  
- Tried different binning strategies for continuous variables, but they did not improve separability.  

---

### 2. **Data Preprocessing**  
- Applied **StandardScaler** on continuous variables.  
- Encoded categorical features.  
- Handled missing values and unknown categories.  

---

### 3. **Imbalance Handling (SMOTE)**  
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples for minority class.  
- Ensured balancing did not cause overfitting by validating results on test data.  

---

### 4. **Modeling Approaches**  
Tested multiple ML models with and without SMOTE:  
- Logistic Regression  
- XGBoost  
- Random Forest  
- XGBoost (SMOTE)  
- Random Forest (SMOTE)  

---

### 5. **Model Evaluation**  

| Model                | ROC-AUC | Accuracy | Precision | Recall | F1 |
|-----------------------|---------|----------|-----------|--------|----|
| Logistic Regression   | 0.6558  | 0.5467   | 0.2898    | 0.7229 | 0.4137 |
| XGBoost              | 0.7866  | 0.7510   | 0.4559    | 0.6453 | 0.5342 |
| Random Forest        | 0.7842  | 0.7798   | 0.5021    | 0.5841 | 0.5399 |
| XGBoost (SMOTE)      | 0.9179  | 0.8449   | 0.8800    | 0.7988 | 0.8374 |
| Random Forest (SMOTE)| **0.9343** | **0.8619** | **0.8827** | **0.8349** | **0.8581** |

---

## 🔎 Key Insights  
- Logistic Regression struggled due to **non-linearity and imbalance**.  
- XGBoost and Random Forest performed well, but still biased towards majority class.  
- **SMOTE significantly improved performance** across all metrics.  
- **Random Forest (SMOTE)** performed slightly better than XGBoost (SMOTE), likely because:  
  - RF is more robust with synthetic samples.  
  - RF handles noise from SMOTE better due to averaging across multiple trees.  

---

## 🚧 Challenges Faced  
- Severe **class imbalance** → models biased towards majority class.  
- Unknown categorical values without mapping in data dictionary.  
- Some EDA techniques (binning, feature grouping) did not yield improvements.  
- ROC-AUC alone was misleading → Recall and F1 were critical to assess real performance.  

---

## 📈 Next Steps for Improvement  
- Explore more advanced models:  
  - **KNN**  
  - **Artificial Neural Networks (ANNs)**  
- Use **PCA for dimensionality reduction and noise removal**.  
- Try **ensemble techniques** (stacking/blending).  
- Explore **cost-sensitive learning** instead of just SMOTE.  
- Evaluate models using **PR-AUC**, in addition to ROC-AUC, Recall, and F1.  

---
