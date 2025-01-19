# **Credit Card Fraud Detection Workflow**
This document details a comprehensive process for analyzing and detecting fraudulent credit card transactions using Python. The workflow includes data preprocessing, balancing class distribution, and evaluating multiple machine learning models.

---

## **Process Overview**

### **1. Import Libraries and Load Data**

The first step is to import essential libraries for data manipulation, visualization, and model development. The dataset is then loaded for further analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('Creditcard_data.csv')
```
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization.
- **SMOTE**: To handle imbalanced datasets.
- **scikit-learn**: For machine learning model training and evaluation.

---

### **2. Explore Data**

In this step, we explore the dataset to understand its structure and characteristics. We examine the first few rows, check for missing values, and analyze the distribution of the target variable (`Class`).

```python
print(data.head())
print(data.info())
print(data.describe())
print(data['Class'].value_counts())
```

- **`data.head()`**: Displays the first 5 rows of the dataset to understand its structure.
- **`data.info()`**: Provides metadata, including column names, data types, and null value counts.
- **`data.describe()`**: Summarizes numerical columns with statistics like mean, median, and standard deviation.
- **`data['Class'].value_counts()`**: Shows the number of transactions classified as fraudulent (`Class = 1`) and non-fraudulent (`Class = 0`).

---

### **3. Handle Class Imbalance**

Fraudulent transactions often form a small portion of the dataset, leading to a class imbalance. We address this using the Synthetic Minority Oversampling Technique (SMOTE), which generates synthetic samples for the minority class.

```python
X = data.drop('Class', axis=1)
y = data['Class']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verify new class distribution
balanced_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['Class'])], axis=1)
print(balanced_data['Class'].value_counts())
```
- **SMOTE**: Balances the dataset by creating synthetic samples of the minority class based on its nearest neighbors.
- **Balanced Data Check**: Ensures that both classes have an equal number of samples after resampling.

---

### **4. Train and Evaluate Models**

Several machine learning models are trained on the balanced dataset to predict fraudulent transactions. The performance of these models is evaluated using accuracy as the metric.

```python
# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Split balanced data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results[name] = accuracy_score(y_test, predictions)

print(results)
```
- **Model Selection**: Three models are evaluated:
  - Logistic Regression: A linear model for binary classification.
  - Decision Tree: A tree-based model that splits data based on feature thresholds.
  - Gradient Boosting: An ensemble model combining weak learners to improve accuracy.
- **Train-Test Split**: The balanced dataset is split into training and testing subsets.
- **Accuracy Score**: Measures the proportion of correctly predicted instances.

---

### **5. Save Results**

The results of the model evaluations are saved to a CSV file for documentation and future analysis.

```python
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df.to_csv('model_performance.csv', index=False)
```
- **Output File**: Stores model names and their corresponding accuracy scores in `model_performance.csv`.

---

### **Observations**

- **Gradient Boosting** consistently delivers the highest accuracy across the dataset.
- **Logistic Regression** serves as a robust baseline, performing well despite its simplicity.
- **Decision Tree** achieves competitive results but may overfit small datasets due to its hierarchical splitting.

---

This detailed workflow provides a robust framework for detecting fraudulent credit card transactions. It emphasizes data preparation, class balance, and model evaluation to ensure reliable results.
