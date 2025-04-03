# Titanic Survivor Prediction

## Overview
This project aims to predict whether a passenger survived the Titanic disaster using machine learning techniques. The dataset contains information about passengers such as age, gender, ticket class, and other features that can help in survival prediction.

## Dataset
The dataset used for this project is the famous **Titanic dataset** from Kaggle. It consists of the following features:
- `PassengerId`: Unique identifier for each passenger
- `Survived`: Target variable (0 = Did not survive, 1 = Survived)
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Name`: Passenger's name
- `Sex`: Gender
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Ticket fare
- `Cabin`: Cabin number
- `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Methodology
The project follows these main steps:
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
2. **Exploratory Data Analysis (EDA)**
   - Visualizing feature distributions
   - Identifying correlations
3. **Model Selection & Training**
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
   - Gradient Boosting (XGBoost, LightGBM)
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-AUC curve
5. **Hyperparameter Tuning**
   - GridSearchCV and RandomizedSearchCV for optimal model performance
6. **Deployment (Optional)**
   - Deploying as a web app using Flask or Streamlit

## Installation
To run this project locally, install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage
Run the following script to train and test the model:
```bash
python titanic_prediction.py
```

## Results
After training multiple models, the best-performing model achieved an accuracy of **X%** on the test set. Feature importance analysis showed that `Pclass`, `Sex`, and `Age` were key predictors of survival.

## Future Improvements
- Implement deep learning with TensorFlow/Keras
- Improve feature engineering
- Deploy as an interactive web application

## References
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- [Scikit-Learn Documentation](https://scikit-learn.org/)

## License
This project is open-source and available under the MIT License.

