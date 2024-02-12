# **NABII - DIMBA FOOTBALL PREDICTION PROJECT.**
![Alt Text](/images/Nabii-Dimba.jpg)

## **AUTHORS**
1.Wayne Korir

2.Rose Kyalo

3.Dennis Kobia

4.Jane Mwangi

5.Brytone Omare

6.Ivy Ndunge


## **Table of Contents**
1. [Introduction](#introduction)
2. [Data Source](#data-source)
3. [Features](#features)
4. [Methodology](#methodology)
   - [EDA](#eda)
   - [Data Preprocessing](#data-preprocessing)
     - [Handling Null Values](#handling-null-values)
     - [Feature Engineering](#feature-engineering)
     - [Dealing with Multicollinearity](#dealing-with-multicollinearity)
     - [Normalizing Feature Distributions](#normalizing-feature-distributions)
     - [Handling Outliers](#handling-outliers)
   - [Modelling](#modelling)
     - [Total Goal Count Predictions](#total-goal-count-predictions)
     - [Fouls](#fouls)
     - [Match Outcomes](#match-outcomes)
5. [Getting Started](#getting-started)
   - [Installation](#installation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)
10. [Contact](#contact)


## **Introduction**

In Kenya, the betting industry has grown significantly, especially in soccer betting. Many Kenyan sports fans actively engage in predicting match outcomes, seeking to turn their passion into profitable decisions. This project responds to the need for better-informed betting by developing a user-friendly system for predicting English Premier League match outcomes.

Our goal is to create a straightforward tool where users can input for example a match ID, and the system, using machine learning, will provide a prediction. By leveraging advanced algorithms and historical data, we aim to offer reliable predictions, filling a gap in the Kenyan soccer betting scene. This project introduces cutting-edge data science techniques to make match predictions more accessible and practical for soccer enthusiasts in Kenya.

## **Data Source**
Our dataset is a comprehensive collection sourced from the football API, FootyStats, and web scraping of relevant data from a dedicated website. The information encompasses football data for the English Premier League spanning from the 2016 to 2024 seasons. Through a meticulous cleaning process, the data was merged into a unified dataframe, ensuring accuracy and coherence for subsequent modeling and analysis.

## **Features**
Utilizing the Recursive Feature Elimination (RFE) technique, we systematically identified and selected the most crucial features for our analysis. This method enhances the model's performance by focusing on the key variables that contribute significantly to the predictive outcomes.Our target variables included total goal count, fouls, and match outcomes.


## **Methodology**
The methodology employed in this project encompasses Exploratory Data Analysis (EDA), Data Preprocessing, and Modelling phases to develop a soccer match outcome prediction system for the English Premier League. Here's a brief description of each phase:

### EDA
In 46% of the matches, the home team emerged victorious, while the away team secured victory in 30%, and approximately 24% resulted in draws. Notably, during the 2020/2021 and 2021/2022 seasons, there was a noticeable decline in the home team's winning rate, coinciding with the global pandemic.

This decrease in home team success coincided with the absence of fans from matches, emphasizing the potential influence of fan presence on match outcomes. The observed correlation suggests that the dynamic interaction between fans and players may play a significant role in determining the success of the home team. The absence of this factor during the pandemic period might have contributed to the shift in match outcomes.

![Alt Text](/images/eda_into_one.png)

### Data Preprocessing

#### Handling Null Values
During the data extraction process, null values were initially encoded as -1. To ensure consistency, the encoding was standardized by replacing null values with zeros. Subsequently, instances with missing data were dropped. Incomplete seasons and suspended matches were also excluded from the dataset.

#### Feature Engineering
To address the need for data on upcoming matches, feature engineering was implemented by calculating the average of the last fifteen matches, ensuring a more comprehensive and predictive dataset.

#### Dealing with Multicollinearity
Identifying multicollinearity among features prompted the application of the Recursive Feature Elimination (RFE) technique and penalization methods to reduce dependencies and enhance model accuracy.

#### Normalizing Feature Distributions
To tackle the non-normal distribution of most features, scaling techniques, including Min-Max scaling and Standard Scaler, were employed to normalize distributions and improve model performance.

#### Handling Outliers
The presence of outliers was addressed by applying log transformation, considering the absence of a strong linear relationship between features and the target variable.

### **Modelling**

In the modeling phase, various machine learning algorithms were employed to predict key outcomes in soccer matches. The project focused on three main predictions: total goal count, fouls, and match outcomes. Regression models, such as Support Vector Regressor (SVR) and XGBoost Regressor, were utilized for total goal count predictions. For fouls, the Support Vector Machine (SVM) and Artificial Neural Network (ANN) demonstrated promising results. Additionally, classification models, including Decision Tree Classifier and Random Forest Classifier, were explored for match outcome predictions.

Model selection was based on rigorous evaluation metrics, such as Mean Absolute Error (MAE),Root Mean Squared Error (RMSE), Accuracy and F1 Score to ensure optimal performance. The chosen models underwent further optimization through hyperparameter tuning and fine-tuning of model architectures. The iterative process aimed to enhance predictive accuracy and produce reliable insights for soccer enthusiasts engaging in match outcome predictions.

Visualizations of model performances, such as learning curves and hyperparameter tuning results, were instrumental in assessing and improving the models. The overall goal was to deploy models that could provide valuable insights, assist in making informed betting decisions, and contribute to the user-friendly soccer match outcome prediction system.

#### Total Goal Count Predictions

For predicting total goal count, we utilized regression models. Various models were considered, and after rigorous evaluation, the Support Vector Regressor (SVR) emerged as the most effective, achieving a Mean Absolute Error (MAE) of 0.3864. To enhance the performance of the SVR model, hyperparameter tuning was conducted, ensuring optimal parameter settings for improved predictive accuracy.

![Alt Text](/images/totalgoalcount_best_model.jpg)

#### Fouls
The Support Vector Machine (SVM) achieved a Root Mean Squared Error (RMSE) of 1.97, while the Artificial Neural Network (ANN) exhibited an RMSE of 1.95. These promising results suggest the potential effectiveness of both models, but additional tuning could enhance their predictive accuracy even further.

![Alt Text](/images/fouls_best_model.jpg)

#### Match Outcomes
The base models using Stochastic Gradient Descent (SGD) and Long Short-Term Memory (LSTM) demonstrated significant promise, achieving a commendable accuracy of 60% for goal predictions. We implemented a two-fold strategy:

1. Hyperparameter Tuning:Decision Tree Classifier (DTC), Random Forest Classifier (RFC), SGD Classifier, and xGBoost Classifier underwent rigorous tuning using Grid Search. This systematic exploration of hyperparameter combinations aimed to unlock improved performance and fine-tune each model for optimal results.
LSTM Models Optimization:

2. Various LSTM model architectures were explored, incorporating adjustments to dropout rates and learning rates. The goal was to identify the most effective configuration for the LSTM models, optimizing their performance for accurate goal predictions.


![Alt Text](/images/matchoutcome_best_model.jpg)

## **Getting Started**

### Installation

Follow these steps to set up and run the project locally:

#### 1. Clone the Repository

To get a copy of this project on your local machine, clone the repository using the following command:

```bash
git clone https://github.com/username/repository.git
```
#### 2. Navigate to the Project Directory
```bash
cd repository
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### **Prerequisites**

Before using this project, make sure you have the following dependencies installed:

```
Python (Version 3.0 or higher)
Scikit-learn
Numpy
Pandas
Matplotlib
Seaborn
TensorFlow
XGBoost

```
### **Contributing**
We welcome contributions to improve the project. If you'd like to contribute, please follow these guidelines:

**1. Bug Reports:** If you encounter a bug, open an issue describing the problem and steps to reproduce it.

**2. Feature Requests:** Suggest new features or enhancements by opening an issue with a detailed description.

**3.Code Contributions:** Fork the repository, create a new branch, make your changes, and submit a pull request.

**4.Code Style:** Follow the project's coding style and conventions.

### **License**
This project is licensed under the MIT license - see the LICENSE.md file for details.

### **Acknowledgements**
We acknowledge and thank the following libraries, tools, or resources that contributed to the development of this project:

**1. Python (Version 3.0 or higher):** The core programming language used for the project, providing a robust foundation for development.

**2. Scikit-learn:** A powerful machine learning library that facilitated the implementation of various models and algorithms.

**3. Numpy:** Essential for numerical computing and array operations, Numpy played a crucial role in data manipulation and analysis.

**4. Pandas:** A versatile data manipulation library that simplified handling and processing of structured data.

**5. Matplotlib:** A comprehensive plotting library that enabled the creation of visualizations for data exploration and analysis.

**6. Seaborn:** Built on top of Matplotlib, Seaborn enhanced the aesthetics of visualizations, making them more informative and appealing.

**7. TensorFlow:** An open-source machine learning framework that supported the implementation of deep learning models and neural networks.

**8. XGBoost:** An efficient and scalable gradient boosting library that contributed to the development of robust predictive models.

### Contact
For questions, feedback, or additional information, feel free to reach out:

Rose Kyalo

Email: rosekyalo94@gmail.com

Twitter: @rose_kawila


