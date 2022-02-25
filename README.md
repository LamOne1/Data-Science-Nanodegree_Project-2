# Data-Science-Nanodegree_Project#2

# Table of contents:
1. [Introduction](https://github.com/LamyaMK/Data-Science-Nanodegree_Project-2/edit/main/README.md#introduction)
2. [Installations](https://github.com/LamyaMK/Data-Science-Nanodegree_Project-2/edit/main/README.md#installations)
3. [Project Components](https://github.com/LamyaMK/Data-Science-Nanodegree_Project-2/edit/main/README.md#project-components)
4. [File Descriptions](https://github.com/LamyaMK/Data-Science-Nanodegree_Project-2/edit/main/README.md#file-descriptions)
5. [Licensing, Authors, Acknowledgements, etc.](https://github.com/LamyaMK/Data-Science-Nanodegree_Project-2/edit/main/README.md#licensing-authors-acknowledgements-etc)


# Introduction

In this project, we will analyze disaster data to build a model for an API that classifies disaster messages. The data contains real messages that were sent during disaster events. We will create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.
Below are a few screenshots of the web app.

![image](https://user-images.githubusercontent.com/97054802/155796723-4d5c6603-657e-45bd-a194-953b6d0e20a1.png)
![image](https://user-images.githubusercontent.com/97054802/155796782-2d9f9c77-02af-4af5-898d-4af66499d499.png)


# Installations
You need to install Python3 and the following packages:
- pandas
- tqdm
- numpy
- sklearn
- nltk
- sqlalchemy
- pickle
- flask
- plotly

# Project Components

There are three componentsof this project:

**1. ETL Pipeline**
In a Python script, process_data.py does data cleaning pipeline that:

1. Loads the `messages` and categories `datasets`
2. Merges the two datasets
3. Cleans the data
3. Stores it in a SQLite database

**2. ML Pipeline**
In a Python script, train_classifier.py writes a machine learning pipeline that:

1. Loads data from the SQLite database
2. Splits the dataset into training and test sets
3. Builds a text processing and machine learning pipeline
4. Trains and tunes a model using GridSearchCV
5. Outputs results on the test set
6. Exports the final model as a pickle file

**3. Flask Web App**
This components will display the results in a Flask web app.

# File Descriptions

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

```

to run the app, go to the app folder then `python run.py`

# Licensing, Authors, Acknowledgements, etc.

Thanks for Udacity for providing this fun project :)!
