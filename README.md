## Churn Detection

### Introduction
- Given a dataset <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data">`Telco-Customer-Churn.csv`</a> with 7043 rows and 21 columns, we are going to predict whether a customer will churn or not.

### Data Preprocessing
- `TotalCharges` column has 11 missing values. We will replace them with the median value of the column.
- Additionally, we will convert the `TotalCharges` column to float type.
- Drop the `customerID` column as it is not useful for prediction.

### Exploratory Data Analysis
- The dataset is `imbalanced` with `73% of the customers not churning`.
- `Gender` does not seem to have any major effect on churn.
- `Younger customers` tend `not to churn` withing the age group.
- `Customers with partners` tend `not to churn` compared to those without partners
- Similarly, `customers with dependents` tend `not to churn` compared to those without dependents
- `Churn rate decreases` drastically with `increase in tenure`
- `Customers with Phone Services` availed tend to `remain with the company`
- A large number of customers with Fiber Optic Internet Service tend to churn; suggesting that the company `needs to look into the quality of Fiber Optic services`
- Customers with Online Security, Online Backup, Device Protection, Tech Support and Streaming TV services tend not to churn; suggesting that `additional services are a good way to retain` customers
- Customers with `Month-to-Month contract tend to churn`; as expected to try out the services before committing to a long term contract
- `Different Payment Methods` do not have a significant impact on churn
- Customers with lower monthly charges tend to churn more which can be justified for a trial period
  - although, `churn rate is high` for customers for `almost all charges slab`; suggesting that the company `needs to look into the overall quality of services`

### Encoding Categorical Variables
- Using a `hybrid` approach of `One-Hot Encoding` and `Label Encoding`
- One-Hot Encoding for following columns `["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]`
- Label Encoding for `["gender", "Partner", "Dependents", "PhoneService", "InternetService", "PaperlessBilling", "Churn"]`

### Correlation and Mutual Information
- a few features are highly correlated with each other
    - `TotalCharges` and `tenure`; obviously
    - `MonthlyCharges` and `InternetService`; obviously
    - `churn` and `InternetService` have highest correlation; whereas `churn` and `tenure` have the lowest correlation

    ##### Conclusion
    - features do not show a strong correlation with other independent features and target
    - hence, feature selection/engineering is not expected to bring about a significant change in the model performance
    - with more data, we can expect to see a better correlation between the features

    ##### Note:
    - data augmentation could provide more data points
    - but randomized data augmentation would not be good estimate against the real world data

### Model Building
- Split the data into train, val and test sets
- Using three different training approaches:
    1. Logistic Regression Classifier with Grid Search Cross Validation
    2. Random Forest Classifier with Grid Search Cross Validation
    3. Random Forest Classifier with Randomized Search Cross Validation
- preparing a pipeline for `Grid Search Cross Validation`
    - Logistic Regression Classifier has a set of parameter grid
    ```
        lrc_param_grid = {
            'lrc' : [LogisticRegression()],
            'lrc__penalty' : ['l1', 'l2'],
            'lrc__C' : np.logspace(-4, 4, 20),
            'lrc__solver' : ['liblinear'],
            'lrc__max_iter' : [200, 400, 600, 800, 1000],
        }
    ```
    - Random Forest Classifier has a set of parameter grid
    ```
        rfc_param_grid = {
            'rfc' : [RandomForestClassifier()],
            'rfc__criterion' : ['gini', 'entropy', 'log_loss'],
            'rfc__max_depth' : list(range(7,22,2)),
            'rfc__min_samples_split' : list(range(2,10,2)),
            'rfc__min_samples_leaf' : list(range(2,25,5)),
            'rfc__n_estimators' : list(range(10,200,50)),
            'rfc__bootstrap' : [True],
            'rfc__max_features' : ['sqrt', 'log2'],
            'rfc__max_samples' : [0.5, 0.8, 1.0]
        }
    ```
- For `Randomized Search Cross Validation`, RFC picks up a random configuration and repeats the process for a given number of iterations
    - Random Forest Classifier has a set of parameters
    ```
        params_set = {
            "criterion": ["entropy", "gini"],
            "max_depth": [9,11,13,15,17,19],
            "min_samples_split": [10,20,40,60],
            "min_samples_leaf": [2,5,20,10],
            "n_estimators": [50,100,150],
            "bootstrap": [True],
            "max_features": ["sqrt","log2"],
            "max_samples": [0.5,0.8,1.0],
        }
    ```

### Model Evaluation
|   Model   | Logistic Regression + GridSearchCV | Random Forest + RandomizedSearchCV | Random Forest + GridSearchCV |
| :-------: | :--------------------------------: | :--------------------------------: | :--------------------------: |
| Accuracy  |              0.79886               |              0.80810               |           0.79957            |
|  Recall   |              0.55072               |              0.51304               |           0.49275            |
| Precision |              0.59748               |              0.63441               |           0.61372            |
| F1 Score  |              0.57315               |              0.56731               |           0.54662            |
|    AUC    |              0.71510               |              0.70850               |           0.69600            |

- Random Forest Classifier has the best accuracy
- Although, Logistic Regression Classifier with GridSearchCV has the best ROC AUC score

### Conclusion
- Given the dataset, `Logistic Regression Classifier with GridSearchCV` is the best model to predict churn given a moderate feature set
- With more data, we can expect to see a better model performance; especially with Random Forest Classifier
- Additionally, for a skewed class problem, data augmentation requires feature importance to be considered to build a weighted loss function, which can improve the `accuracy` of the model
- But, for the given problem, `accuracy` is not a good metric to evaluate the model performance
- Instead, `ROC AUC score` and `F1 score` are better metrics to evaluate the model performance here