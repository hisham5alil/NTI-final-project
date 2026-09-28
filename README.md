# Income & Car Price Prediction System

An end-to-end Machine Learning system that combines **income tier classification**, **car recommendation**, and **car price prediction** into one integrated workflow.

The project takes customer information to predict an income tier, recommends cars that match that tier, and then predicts the price of a selected car based on its specifications.

---

## Project Overview

The system combines two Machine Learning tasks:

### 1. Income Tier Classification

A Census Income dataset is used to classify customers into five income tiers:

* Bin 1
* Bin 2
* Bin 3
* Bin 4
* Bin 5

The classification pipeline includes:

* Missing value handling
* Numerical feature imputation
* Standard Scaling
* Categorical feature imputation
* One-Hot Encoding
* PCA where applicable
* Machine Learning classification

Several classification algorithms were evaluated before selecting the best-performing pipeline.

### 2. Car Price Prediction

A car listings dataset is used to predict the price of a car in EGP.

The regression pipeline includes:

* Data cleaning
* Numerical feature extraction
* Missing value handling
* Rare category grouping
* One-Hot Encoding
* Standard Scaling
* Machine Learning regression

Several regression models were evaluated before selecting the best-performing pipeline.

---

## End-to-End Workflow

```text
Customer Information
        │
        ▼
Income Tier Classification
        │
        ▼
Income Tier
        │
        ▼
Car Recommendation
        │
        ▼
Select / Modify Car Specifications
        │
        ▼
Car Price Prediction
        │
        ▼
Predicted Price (EGP)
```

---

## Datasets

### Census Income Dataset

The Census dataset contains demographic, educational, occupational, and financial-related features.

Important features include:

* Age
* Workclass
* Education
* Education Number
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Capital Gain
* Capital Loss
* Hours per Week
* Native Country

The original income label was used to engineer a five-tier income target.

### Car Listings Dataset

The car dataset contains vehicle information such as:

* Brand
* Model
* Year
* Condition
* Kilometers
* Color
* Power Transmission
* Fuel Type
* Location
* Price

---

## Data Cleaning & Preprocessing

### Census Dataset

The preprocessing included:

* Removing extra spaces from column names and string values
* Handling missing values
* Replacing `?` with missing values
* Removing duplicate rows
* Preparing the income target
* Creating a five-tier income target

### Car Dataset

The preprocessing included:

* Extracting numerical values from price and mileage strings
* Converting year into a numerical feature
* Removing duplicate listings
* Handling missing values
* Removing invalid years
* Removing extreme price outliers
* Creating `car_age`
* Grouping rare categorical values

---

## Feature Engineering

For the income classification task, a five-tier target was engineered using an income score based on:

* Original income category
* Capital gain
* Capital loss
* Hours per week
* Education number
* Age

The resulting target contains:

```text
Bin 1
Bin 2
Bin 3
Bin 4
Bin 5
```

For the car prediction task:

```text
car_age = 2026 - year
```

was created and used instead of the original year feature.

---

## Machine Learning Pipelines

A major part of the project is the use of **Scikit-learn Pipelines and ColumnTransformer**.

### Classification Pipeline

Numerical features:

```text
Median Imputation
      ↓
StandardScaler
```

Categorical features:

```text
Most Frequent Imputation
      ↓
OneHotEncoder
```

These transformations are combined using `ColumnTransformer`.

Different classifiers were then evaluated, including:

* Logistic Regression
* Naive Bayes
* KNN
* SVC
* Random Forest
* Gradient Boosting
* LightGBM

---

## Classification Results

| Model               | Accuracy | F1 Macro | ROC-AUC |
| ------------------- | -------: | -------: | ------: |
| Logistic Regression |   0.9900 |   0.9900 |  0.9998 |
| LightGBM            |   0.9826 |   0.9825 |  0.9995 |
| Gradient Boosting   |   0.9436 |   0.9433 |  0.9956 |
| SVC                 |   0.9390 |   0.9389 |  0.9961 |
| Random Forest       |   0.9276 |   0.9274 |  0.9948 |
| KNN                 |   0.8111 |   0.8122 |  0.9651 |
| Naive Bayes         |   0.4793 |   0.4456 |  0.8116 |

The best-performing classification pipeline was:

**Logistic Regression**

Accuracy: **99.00%**

F1 Macro: **99.00%**

ROC-AUC: **0.9998**

---

## Car Price Regression

The regression pipelines included:

* Linear Regression
* Polynomial Regression
* KNN Regressor
* SVR
* Random Forest Regressor
* Gradient Boosting Regressor
* LightGBM Regressor

For categorical features, a custom `RareCategoryGrouper` transformer was implemented to group low-frequency categories into `Other` before One-Hot Encoding.

### Regression Results

| Model                       |      R² | RMSE (EGP) | MAE (EGP) |
| --------------------------- | ------: | ---------: | --------: |
| LightGBM Regressor          |  0.8516 |    233,773 |   148,137 |
| Random Forest Regressor     |  0.8164 |    259,958 |   165,404 |
| Gradient Boosting Regressor |  0.7900 |    278,028 |   192,050 |
| Polynomial Regression       |  0.7634 |    295,109 |   209,225 |
| Linear Regression           |  0.7476 |    304,853 |   223,939 |
| KNN Regressor               |  0.7036 |    330,353 |   223,064 |
| SVR                         | -0.0499 |    621,694 |   466,486 |

The best-performing regression pipeline was:

**LightGBM Regressor**

R²: **0.8516**

MAE: **148,137 EGP**

RMSE: **233,773 EGP**

---

## Recommendation System

The project goes beyond individual predictions.

After predicting the customer's income tier, the system maps that tier to a corresponding car-price range.

The system then recommends several cars from that price segment.

```text
Income Tier
     ↓
Price Range
     ↓
Recommended Cars
     ↓
Selected Car
     ↓
Price Prediction
```

This creates an integrated recommendation and prediction workflow rather than treating the models as completely separate components.

---

## Model Export

The final trained pipelines are exported using `joblib`:

```python
joblib.dump(best_clf_pipeline, "best_income_classifier.joblib")

joblib.dump(best_reg_pipeline, "best_car_price_regressor.joblib")
```

Because preprocessing is included inside the pipelines, the saved models contain the required preprocessing steps along with the trained estimators.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

---

## Project Structure

```text
Income-Car-Prediction/
│
├── data/
│   ├── census_income.csv
│   └── data.csv
│
├── models/
│   ├── best_income_classifier.joblib
│   └── best_car_price_regressor.joblib
│
├── notebooks/
│   └── main.ipynb
│
├── app/
│   └── app.py
│
└── README.md
```

---

## Key Learning Outcomes

Through this project, I practiced:

* End-to-end Machine Learning workflow
* Data cleaning and EDA
* Feature engineering
* Classification
* Regression
* Pipeline design
* ColumnTransformer
* Hyperparameter tuning with GridSearchCV
* Cross-validation
* Model evaluation
* Handling categorical variables
* Rare category grouping
* Model serialization with Joblib
* Building an integrated ML prediction workflow

---

## Future Improvements

* Deploy the application as a web application
* Add more real-world car listings
* Improve the recommendation logic
* Add model monitoring
* Experiment with additional boosting algorithms
* Add explainability features
* Improve the user interface and visualization

---

## Author

**Hisham Mohamed Khalil**

Machine Learning / AI Developer

Interested in building practical AI-powered solutions for businesses.
