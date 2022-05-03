import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import requests

def load_data(messages_filepath, categories_filepath):
    """
      Function:
      load data from csv and join them
      Arguments:
      message: path of the csv with the messages
      Categories: path of the csv with the categories
      Return:
      df: the joined data
      """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df


def clean_data(df):
    """
      Function:
      clean to df
      Arguments:
      df (DataFrame): raw data
      Return:
      df (DataFrame): clean data
      """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    # select the first row of the categories dataframe
    row =  categories.head(1)
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =row.applymap(lambda x: x[:-2]).iloc[0, :].tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:   
        # set each value to be the last character of the string
        categories[column] =  categories[column].astype(str).str[-1]   
        # convert column from string to numeric   
        categories[column] =  categories[column].astype(int)
        
        
    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset='id', inplace=True)
    return df

def save_data(df, database_filename):
    """
       Function:
       Save df to a database
       Arguments:
       df (DataFrame): the data
       database_filename (str): the final filename
       """
    database_filepath = database_filename
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()