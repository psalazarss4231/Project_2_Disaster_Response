# import libraries
import sys
import pandas as pd
import numpy as np
import re
import os
import requests
import sqlite3
from nltk.corpus import stopwords
import pickle
import nltk
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_multilabel_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, accuracy_score
nltk.download(['wordnet', 'punkt', 'stopwords','averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
         Function:
         load the data
         Arguments:
         database_filepath: path
         Return:
         X (DataFrame) : the data of the independent variables
         Y (data frame): dependent variable
         category (list of str): the categories
         """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages_tbl',engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis = 1)
    return X, Y

def tokenize(text):
    """
    Function:generates the model classifications
    Input: text
    Output: the classification
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text_words = word_tokenize(text)   

    stop_words = stopwords.words("english")
    words = [i for i in text_words if i not in stop_words]
    
    clean_tokens = [WordNetLemmatizer().lemmatize(x) for x in text_words]
    
    return clean_tokens


def build_model():
    """ Function: makes a model to classify messages 
    Returns: cv(list of str): model """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [10, 20], 
              'clf__estimator__min_samples_split': [2, 4]} 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test):
    """
      Function: Evaluate the model.
      Arguments:
      model: the model
      X_test: test messages
      Y_test: test dependent variable
      """    
    y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    """
      Function: Save the model file
      Arguments:
      model: the model
      model_filepath (str): path
      """    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()