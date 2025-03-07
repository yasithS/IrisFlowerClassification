# Iris Flower Classification

A machine learning project that classifies Iris flowers using multiple classification algorithms.

## Overview

This project implements and compares three different machine learning algorithms for classifying the classic Iris flower dataset:
- Support Vector Machine (SVM)
- Logistic Regression
- Decision Tree Classifier

The Iris dataset contains measurements for 150 iris flowers from three different species: Setosa, Versicolor, and Virginica. Each flower has four features measured: sepal length, sepal width, petal length, and petal width.

## Dataset

The dataset consists of:
- 150 samples
- 4 features (sepal length, sepal width, petal length, petal width)
- 3 classes (Iris Setosa, Iris Versicolor, Iris Virginica)

## Implementation Details

### Data Preprocessing
- The dataset is loaded using pandas with appropriate column names
- Features (X) and target labels (Y) are separated
- Data is split into training (80%) and testing (20%) sets using sklearn's train_test_split

### Data Visualization
- Dataset statistics are examined using pandas' describe() function
- Visual analysis is performed using seaborn's pairplot to show relationships between features with color-coding by class

### Models Implemented

#### 1. Support Vector Machine (SVM)
```python
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)
```

#### 2. Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
```

#### 3. Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
```

### Model Evaluation
- Each model is evaluated using accuracy score
- Classification reports provide precision, recall, and F1-score metrics
- Models are tested on unseen data samples

## How to Run the Project

1. Set up the environment:
```bash
# Clone the repository
git clone https://github.com/yourusername/iris-classification.git
cd iris-classification

# Install required packages
pip install numpy pandas seaborn scikit-learn matplotlib
```

2. Ensure you have the Iris dataset in the correct path:
```
data/iris.data
```

3. Run the Jupyter notebook:
```bash
jupyter notebook "iris Short.ipynb"
```

## File Structure

```
iris-classification/
│
├── Data/
│   └── iris.data           # The Iris dataset
│
├── iris_Flower_Classification.ipynb        # Main Jupyter notebook with implementation
│
└── README.md               # This file
```

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

## Results

The project evaluates and compares the performance of SVM, Logistic Regression, and Decision Tree algorithms on the Iris dataset. The models are compared based on:
- Accuracy scores
- Classification reports (precision, recall, F1-score)
- Prediction capabilities on new data samples

## Making Predictions with New Data

The notebook demonstrates how to make predictions on new, unseen data:

```python
X_new = np.array([[3, 2, 1, 0.2], [5.3, 2.5, 4.6, 1.9], [4.9, 2.2, 3.8, 1.1], [3.2, 3.1, 2.3, 0.3]])
prediction = model_svc.predict(X_new)
```

## Future Improvements

- Implement hyperparameter tuning for optimizing model performance
- Add more visualization methods to better understand the dataset
- Incorporate more advanced models for comparison
- Implement cross-validation for more robust evaluation

## License

[MIT License](LICENSE)

## Contact

Your Name - gunawardhanayasith@gmail.com

Project Link: https://github.com/yasithS/IrisFlowerClassification

## Author

# Yasith Gunawardhana
