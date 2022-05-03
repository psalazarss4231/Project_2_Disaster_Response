# Project_2_Disaster_Response
In the following project, disaster response messages are taken, cleaned with an ETL, a predictive model is trained and then summary graphs of the results are made in an app.


Project 2 Responses to disasters
Index
1. [Motivation. ] (#p1)
2. [Description of attachments. ] (#p2)
3. [Steps. ] (#p3)
4. [Results. ] (#p4)

<a name="p1"></a>

Motivation

This project is the second for the Data Scientist Nanodegre, which seeks to take a series of messages given in the face of natural disasters, cleans them, classifies them, trains a preductive algorithm to categorize new messages and displays a summary of the results.

<a name="p2"></a>
Description of attachments

3 folders are required:
Data: In the following folder you will find both the messages and their categories, the data_process.py file, which cleans the data, as well as the DisasterResponse database with the cleaned and transformed data.

Model: The folder contains a file called train_classifier.py, which is a procedure with functions to build and evaluate a preductive model with Random forest.

App: The folder contains the run.py file with the html to run the app.

<a name="p3"></a>
Steps

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

<a name="p1"></a>
Results:

Several images of the model application are attached to the app.


![I1](/images/I1.png "I 1")
![I2](/images/I2.png "I 2")
![I3](/images/I3.png "I 3")



