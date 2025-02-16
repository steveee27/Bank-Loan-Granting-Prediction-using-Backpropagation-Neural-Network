# **Bank Loan Granting Prediction Using Backpropagation Neural Network (BPNN)**

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Architecture](#model-architecture)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results and Discussion](#results-and-discussion)
  - [Model Performance](#model-performance)
  - [Key Insights](#key-insights)
  - [Limitations](#limitations)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview
This project aims to predict whether a loan application will be approved or rejected by a bank based on customer information such as income, age, and credit history. By using **Backpropagation Neural Network (BPNN)**, the project focuses on binary classification—whether a loan is granted (1) or not granted (0).

The dataset consists of **5000 records** with **14 features** and is used to train and evaluate the model. The classification is done by preprocessing the data, applying a neural network model, and evaluating it on a test dataset.

## Dataset Overview
The dataset for this project contains information about customers applying for a loan, with the following features:

1. **Age**: Age of the customer
2. **Experience**: Years of professional experience
3. **Income**: Annual income of the customer
4. **Family**: Number of family members
5. **CCAvg**: Average credit card spending per month
6. **Education**: Level of education (1 = undergraduate, 2 = graduate, 3 = professional)
7. **Mortgage**: Amount of mortgage the customer has
8. **Personal Loan**: Target variable (1 = loan granted, 0 = loan denied)
9. **Securities Account**: Whether the customer has a securities account (binary)
10. **CD Account**: Whether the customer has a CD account (binary)
11. **Online**: Whether the customer uses online banking (binary)
12. **CreditCard**: Whether the customer uses a credit card with the bank (binary)
13. **ZIP Code**: Customer's postal code (removed during preprocessing)
14. **ID**: Customer's unique ID (removed during preprocessing)

The dataset is available for download from [this link](https://tinyurl.com/UTSDeepLearning2024No1).

## Data Preprocessing
Data preprocessing was performed to ensure the dataset is suitable for the neural network model:

- **Handling Missing Data**: No missing values were found in the dataset.
- **Incorrect Data Types**: The `CCAvg` column was identified as an object type instead of numeric. It was corrected by replacing '/' with '.' and converting it into a float type.
- **Outliers**: The features `Income`, `CCAvg`, and `Mortgage` showed some outliers, but these were kept as they represent real-world customer data.
- **Dropping Unnecessary Columns**: The `ID` and `ZIP Code` columns were removed since they did not contribute to the prediction task.

## Exploratory Data Analysis (EDA)
Exploratory data analysis (EDA) was conducted to understand the characteristics of the data:

- **Feature Distribution**: The distribution of numerical features such as `Income`, `CCAvg`, and `Mortgage` was plotted. Outliers were identified but not removed, as they represent valid cases.
  ![image](https://github.com/user-attachments/assets/64d6ced2-e8f9-44ef-af13-a32adb5858d9)
  ![image](https://github.com/user-attachments/assets/8b5547fc-6bdf-454c-976a-41e93b5cd5c7)
  ![image](https://github.com/user-attachments/assets/134ba46a-f576-42a7-bde3-c68326413989)

- **Correlation Analysis**: Strong correlations were observed between `Age` and `Experience`, indicating that older customers often have more years of professional experience.
  ![image](https://github.com/user-attachments/assets/8694f07a-9113-4f37-b30c-b945a2e59d4e)

- **Class Imbalance**: The target variable (`Personal Loan`) showed a slight imbalance between the classes (loan granted vs. loan denied), which was noted for evaluation.

  ![image](https://github.com/user-attachments/assets/51bf0215-98d6-4bbe-a53e-1de388d21d24)

## Model Architecture
For this binary classification task, a **Backpropagation Neural Network (BPNN)** was designed with the following architecture:

- **Input Layer**: 11 nodes (for each of the 11 features).
- **Hidden Layers**: Two hidden layers, each with 22 nodes and **ReLU** activation functions.
- **Output Layer**: One output node with a **Sigmoid** activation function to predict a binary outcome (loan granted or not).
  
The model was built using **Keras** and compiled using the **Stochastic Gradient Descent (SGD)** optimizer and the **binary cross-entropy loss** function:

```python
model = Sequential([
    Dense(11, input_shape=(11,)),
    Dense(22, activation='relu'),
    Dense(22, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
```

## Model Training and Evaluation
The dataset was split into:
- **80%** for training
- **10%** for validation
- **10%** for testing

The model was trained for **50 epochs**, with early stopping to prevent overfitting. The training process showed a steady increase in accuracy and a decrease in loss.

### Evaluation Metrics:
- **Accuracy**: 98% accuracy on the test set.
- **Precision**: Measures the proportion of true positives out of all predicted positives.
- **Recall**: Measures the proportion of actual positives that were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.

## Results and Discussion

### Model Performance
The model performed excellently with an accuracy of **98%** on the test set, meaning it successfully predicted whether the loan was granted or not for most of the cases. However, precision and recall were also evaluated to understand the performance across different classes.

### Key Insights
1. **Accuracy**: The high accuracy indicates that the model generalizes well to unseen data.
2. **Class Imbalance**: Even though the accuracy is high, the class imbalance in the target variable (`Personal Loan`) still affects performance. Further adjustments such as oversampling the minority class could be beneficial.
3. **Feature Importance**: Features such as `Income`, `CCAvg`, and `Mortgage` were found to have significant influence on predicting loan approval.

### Limitations
1. **Class Imbalance**: Although accuracy is high, the imbalance between loan approvals and rejections could lead to biased predictions. Techniques like oversampling or class-weight adjustments might improve recall for the minority class.
2. **Model Generalization**: While the model shows good performance, it may benefit from further tuning and additional features to capture more customer information.

## Conclusion
This project successfully demonstrates the use of **Backpropagation Neural Networks (BPNN)** for predicting loan approval decisions based on customer features. The model achieves high accuracy, making it a promising tool for banking institutions to automate loan decision-making processes. Future improvements could focus on addressing class imbalance and fine-tuning the model.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
