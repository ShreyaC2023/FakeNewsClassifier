# Fake News Classifier


## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Decision Tree](#decision-tree)
- [Manual Testing](#manual-testing)


## About

The Fake News Classifier is a Python-based project that uses machine learning techniques to identify fake and real news articles. It preprocesses text data, trains multiple machine learning models, and provides a way to manually test news articles for authenticity.

## Getting Started

### Prerequisites

- Python 3
- Jupyter Notebook
- Required Python libraries (numpy, pandas, seaborn, matplotlib, scikit-learn)


## Installation

To get started with the Fake News Classifier, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ShreyaC2023/FakeNewsClassifier.git
   cd FakeNewsClassifier
   
## Usage

### Data Preparation

- Two datasets, `Fake.csv` and `True.csv`, are used.
- Data preprocessing includes lowercasing, special character removal, and URL removal.
- The data is split into training and testing sets.

### Machine Learning Models

Three machine learning models are implemented for classification:

#### Logistic Regression

- Logistic regression model for news classification.
- Provides accuracy and classification report.

#### Random Forest Classifier

- Random forest classifier for news classification.
- Provides accuracy and classification report.

#### Decision Tree

- Decision tree classifier for news classification.
- Provides accuracy and classification report.

### Manual Testing

To manually test a news article:

1. **Run the program**.
2. Enter a news article when prompted.
3. The program will display predictions from the logistic regression, decision tree, and random forest classifiers.
