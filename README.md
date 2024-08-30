# Titanic_Classification

Project descibes about to Make a system which tells whether the person will be save from sinking. What factors were most likely lead to success-socio-economic status, age, gender
and more.

## Data Set
The dataset used in this project is the Titanic dataset from Kaggle. It contains information about the passengers such as their age, sex, class, fare, and whether they survived or not.

## Installation
Install Jupyter Notebook/Goggle Collab/VS Code  
in that install python. To run the notebook, you need to have Python installed along with the following libraries:
pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
    
## Appendix

# 1. Importing Necessary Libraries
    import numpy

    import pandas as pd manipulation and analysis

    import matplotlib.pyplot as plt plotting as visualizations

    import seaborn as sns              
# 2. Loading and Exploring the Dataset

    titanic = pd.read_csv('train.csv')
    
# 3. Handling Missing Data

    Adding fillna() operations
    
# 4. Exploratory Data Analysis (EDA)

    Using boxplot,countplot plotting graphs.
    
# 5.  Data Preprocessing

     Initializing and training the Logistic Regression model

    Predicting and evaluating the model on the training set

    Feature importance based on Logistic Regression coefficients

# 6.  Prediction Function

     Initialize the StandardScaler

     Convert sex to numeric

     Determine if the passenger is alone

     Create a DataFrame for the input data

     Scale the input data
    
     Make prediction

     XGBoost Model (Optional for better accuracy)

    
     Get survival probability
     
# 7.  Saving Accuracy Data

     Saving the training accuracy to a CSV file

    accuracy_df = pd.DataFrame({'Model': ['Logistic Regression', 'XGBoost'],
                            'Training Accuracy': [training_accuracy,       
    training_accuracy],
                            'Test Accuracy': [test_accuracy, 
    test_accuracy_xgb]})

    accuracy_df.to_csv('training_accuracy.csv', index=False)
##  License
    This project is licensed under the MIT License - see the LICENSE file 
    for details.

##  Contributing
    Feel free to fork the repository and submit pull requests. For any 
    questions or suggestions, open an issue or contact the maintainer.



