# FactFinder
A Fact validity check!


# About the Project
In an era where misinformation is rampant, detecting fake news has become crucial to maintaining the integrity of information online. This project presents a comprehensive solution for identifying fake news using an ensemble machine learning model, providing users with a reliable tool to verify the authenticity of news articles.

## Project Overview
This web application, built using Python and Flask, allows users to check whether a given news article is real or fake. The project utilizes a powerful ensemble model that combines the strengths of **Naive Bayes, Passive Aggressive Classifier, and Logistic Regression** to deliver accurate predictions. Additionally, a **BERT (Bidirectional Encoder Representations from Transformers) model** was developed and included in the repository for comparison purposes, though it was not integrated with the frontend.

## Key Features
- The primary model used in the application is an ensemble of Naive Bayes, Passive Aggressive Classifier, and Logistic Regression.
- A BERT model was also developed as part of the project. While not integrated with the frontend, the BERT model's code and implementation are available in the repository for users interested in experimenting with it.
- To enhance the user experience, the application integrates Hugging Face transformers, which generate concise summaries of the news articles.
- Users can either upload the full text of a news article or provide a URL link to the article. The application processes both input types, offering flexibility and ease of use.
- The frontend, designed using HTML and CSS, provides an intuitive interface where users can effortlessly upload articles or URLs and receive instant feedback on the authenticity of the news.

## Dataset
The project uses the [WelFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification), a well-known collection of news articles labeled as either fake or real. This dataset provided a robust foundation for training and testing the models, ensuring reliable performance.


# Installation
