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
To use this project perform the following tasks

Make sure you are using python 3.9.6 or better (https://www.python.org/downloads/release/python-396/)

Clone the repository using
```
git clone https://github.com/sahilbrid/fake-news-detection.git
```

Navigate to the project directory
```
cd fake-news-detection
```

Install the required packages
```
pip install -r requirements.txt
```

Run the Flask application
```
python main.py
```


# Usage
Once the application (main.py) file is running

- Open your browser and go to http://127.0.0.1:5000/

  It will open the landing page
  
  ![image](https://github.com/user-attachments/assets/4303b142-0cd0-411f-a890-bb7290380acf)
  
  It should look something like this

- Now go to the FactChecker section. Here you can paste the news article or the url of the article that you want to check

  ![image](https://github.com/user-attachments/assets/99396c3c-04d2-4659-b692-9e360a4d192a)

- After uploading the article, click on analyse button

  ![image](https://github.com/user-attachments/assets/83ba745f-d7ce-4e0a-abd3-e2205cd25fd9)

  The summary of the article along with its authenticity score will be displayed.


# Technology Used
- **Frontend**: HTML, CSS
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Tensorflow, Transformers, Pandas, Numpy
- **Web Scraping**: BeautifulSoup4


# Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.
