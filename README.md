# NABII DIMBA FOOTBALL PREDICTIONS PROJECT.
## AUTHORS
1.Wayne Korir

2.Rose Kyalo

3.Dennis Kobia

4.Jane Mwangi

5.Brytone Omare

6.Ivy Ndunge


## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Acknowledgements](#acknowledgements)

## Introduction

In Kenya, the betting industry has grown significantly, especially in soccer betting. Many Kenyan sports fans actively engage in predicting match outcomes, seeking to turn their passion into profitable decisions. This project responds to the need for better-informed betting by developing a user-friendly system for predicting English Premier League match outcomes.

Our goal is to create a straightforward tool where users can input for example a match ID, and the system, using machine learning, will provide a prediction. By leveraging advanced algorithms and historical data, we aim to offer reliable predictions, filling a gap in the Kenyan soccer betting scene. This project introduces cutting-edge data science techniques to make match predictions more accessible and practical for soccer enthusiasts in Kenya.

## Data Source
Our dataset is a comprehensive collection sourced from the football API, FootyStats, and web scraping of relevant data from a dedicated website. The information encompasses football data for the English Premier League spanning from the 2016 to 2024 seasons. Through a meticulous cleaning process, the data was merged into a unified dataframe, ensuring accuracy and coherence for subsequent modeling and analysis.

## Features
Utilizing the Recursive Feature Elimination (RFE) technique, we systematically identified and selected the most crucial features for our analysis. This method enhances the model's performance by focusing on the key variables that contribute significantly to the predictive outcomes.Our target variables included total goal count, fouls, and match outcomes.
![Alt Text](/capstone_project/images/totalgoalcount/RFE)

## Methodology

### EDA


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

### Modelling
#### Total Goal Count Predictions

For predicting total goal count, we utilized regression models. Various models were considered, and after rigorous evaluation, the Support Vector Regressor (SVR) emerged as the most effective, achieving a Mean Absolute Error (MAE) of 0.3864. To enhance the performance of the SVR model, hyperparameter tuning was conducted, ensuring optimal parameter settings for improved predictive accuracy.

![Alt Text](/capstone_project/images/totalgoalcount/models)

#### Fouls

#### Match Outcomes

## Getting Started

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

To get a copy of this project on your local machine, clone the repository using the following command:

```bash
git clone https://github.com/username/repository.git

### 2. Navigate to the Project Directory

cd repository
3. Install Dependencies
pip install -r requirements.txt

### Prerequisites

List any dependencies or prerequisites that users need to install before using your project.

### Installation

Step-by-step guide on how to install and configure your project.

## Usage

Demonstrate how users can use your project. Include code snippets, examples, or screenshots.

## Contributing

Guidelines for contributing to your project, including information on how others can submit bug reports, feature requests, or contribute code.

## License

State the license under which your project is distributed.

## Acknowledgements

Acknowledge and give credit to any third-party libraries, tools, or resources you used in your project.

## Contact

Provide contact information for users to reach out with questions or feedback.

---

**Note:** Include badges or additional sections as needed for your specific project, such as a build status badge, demo link, or a roadmap for future development.
