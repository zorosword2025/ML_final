# ML_assignment_2

# Machine Learning Assignment 2 - Income Prediction Classification

 
Student: VELURI REDDY PRATHYUSHA
BITS ID : 2025AB05051 

---

Problem Statement

The objective of this assignment is to build, evaluate, and deploy multiple classification models to predict whether an individual's income exceeds $50K/year based on census data. 
This is a binary classification problem that demonstrates the complete machine learning pipeline from data preprocessing to model deployment on Streamlit Community Cloud.

The task involves:
- Implementing 6 different classification algorithms
- Evaluating models using 6 performance metrics
- Building an interactive web application using Streamlit
- Deploying the application to the cloud for public access

---
Dataset Overview
- Name: Adult Income Prediction Dataset
- Source: Kaggle ([mosapabdelghany/adult-income-prediction-dataset](https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset))
- Type: Binary Classification
- Target Variable: Income (<=50K or >50K)



### Data Preprocessing Steps
1. Handling Missing Values:Removed rows with missing values (marked as '?')
2. Label Encoding: Encoded all categorical features using LabelEncoder
3. Target Encoding: Binary encoding (<=50K → 0, >50K → 1)
4. Feature Scaling: StandardScaler applied for model training
5. Train-Test Split: 70-30 split with stratification

---

Models Used

Comparison Table - Evaluation Metrics

| Model               | Accuracy   | AUC        | Precision  | Recall     | F1 Score   | MCC        |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
|  XGBoost            | 0.8719     | 0.9288     | 0.7851     | 0.6683     | 0.7220     | 0.6430     |
| Random Forest       | 0.8550     | 0.9192     | 0.8052     | 0.5506     | 0.6540     | 0.5827     |
| Decision Tree       | 0.8447     | 0.8840     | 0.7907     | 0.5115     | 0.6212     | 0.5490     |
| Logistic Regression | 0.8300     | 0.8612     | 0.7532     | 0.4716     | 0.5800     | 0.5011     |
| K-Nearest Neighbor  | 0.8280     | 0.8554     | 0.6738     | 0.5990     | 0.6342     | 0.5239     |
| Naive Bayes         | 0.7947     | 0.8556     | 0.6759     | 0.3361     | 0.4490     | 0.3712     |

---

### Model Performance Observations

| ML Model Name           | Observation                                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| XGBoost             | Best overall performer with highest Accuracy (0.8719), AUC (0.9288), and MCC (0.6430), indicating excellent predictive power and strong class discrimination. |
| Random Forest       | Strong ensemble model with high AUC (0.9192) and MCC (0.5827), showing good generalization and reduced overfitting compared to single trees.                  |
| Decision Tree       | Moderate performance with decent Accuracy (0.8447) but lower Recall (0.5115), suggesting overfitting and weaker generalization.                               |
| Logistic Regression | Reliable baseline with balanced metrics and good interpretability, but lower Recall (0.4716) indicates difficulty detecting positive class.                   |
| K-Nearest Neighbor | Balanced Precision-Recall trade-off (0.6738 / 0.5990) with stable performance, though slightly lower accuracy than ensemble methods.                          |
| Naive Bayes        | Lowest performance overall, especially Recall (0.3361) and MCC (0.3712), showing independence assumption does not hold well for this dataset.                 |



---




