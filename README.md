# **Text Sentiment Analysis**

## **Overview**

This project focuses on performing sentiment analysis on tweets to classify them as positive or negative. Utilizing machine learning techniques, the model processes textual data through various preprocessing steps and applies classification algorithms to determine the sentiment polarity of the input text.

## **Dataset**

* **Source**: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  
* **Description**: The dataset contains 1.6 million tweets, each labeled as positive or negative. It includes fields such as the tweet text, user information, and sentiment label.

## **Objective**

Develop a binary classification model that accurately predicts the sentiment (positive or negative) of a given tweet.

## **Methodology**

### **1\. Data Preprocessing**

* **Text Cleaning**:  
  * Removal of URLs, mentions, hashtags, and special characters  
  * Conversion to lowercase  
  * Tokenization  
  * Removal of stop words  
  * Stemming using the Porter Stemmer

  ### **2\. Feature Extraction**

* **Vectorization Techniques**:  
  * Bag of Words (BoW)  
  * Term Frequency-Inverse Document Frequency (TF-IDF)

  ### **3\. Model Building**

* **Algorithms Implemented**:  
  * Logistic Regression  
  * Naive Bayes  
  * Support Vector Machine (SVM)  
  * Random Forest  
  * LSTM

  ### **4\. Model Evaluation**

* **Metrics Used**:  
  * Accuracy  
  * Precision  
  * Recall  
  * F1-Score  
  * Confusion Matrix

## **Results**

**Traditional ML Models**

* **Best Performing Model:** Logistic Regression  
* **R² Score (Test):** 0.80  
* **R² Score (CV):** 0.77

**Deep Learning Model (LSTM)**

* **Validation Accuracy Achieved:** \~76% (as per LSTM\_val in the accuracy plot)  
* **Validation Loss:** \~0.48

*Note*: Replace the placeholders with actual results obtained from your models.

## **Dependencies**

* Python 3  
* pandas  
* numpy  
* matplotlib  
* seaborn  
* scikit-learn  
* nltk

  ## **Future Work**

* Implement deep learning models like LSTM and BERT for improved accuracy  
* Deploy the model using Flask or Streamlit for real-time sentiment analysis  
* Integrate the model into a web application for user-friendly interaction

  ## **Acknowledgements**

* [Kaggle \- Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  
* [NLTK Documentation](https://www.nltk.org/)  
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)