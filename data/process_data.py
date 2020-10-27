# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 22:54:58 2020

@author: ZY
"""
import sys
import pandas as pd
from sqlalchemy import create_engine
# print ('Number of arguments:', len(sys.argv), 'arguments.')
# print ('Argument List:', str(sys.argv))


def print_input(messages_filepath, categories_filepath, database_filepath):
    """  
    A function to print out the input file paths. 
    
    Parameters
    ----------
    messages_filepath : String
        The file path for the message data.
    categories_filepath : String
        The file path for the category data
    database_filepath : String
        The file path for the database 

    Returns
    -------
    messages : pandas.dataframe
        a dataframe that contains the message data
    categories : pandas.dataframe
        a dataframe that contains the category data

    """
    print ('The message file is:', messages_filepath)
    messages = pd.read_csv(messages_filepath)
    print ('The category file is:', categories_filepath)
    categories = pd.read_csv(categories_filepath)
    print ('The saving database is:', database_filepath)    
    return messages, categories

def prepare_data(messages, categories):
    """
    Merge the two dataframe

    Parameters
    ----------
    messages : pandas.dataframe
        a dataframe that contains the message data
    categories : pandas.dataframe
        a dataframe that contains the category data

    Returns
    -------
    df : pandas.dataframe
        merged dataframe

    """    
    
    df = pd.concat([messages.set_index('id'),categories.set_index('id')], axis=1).reset_index()
    
    # make a new dataframe that contains splitting of categories column
    categories = df['categories'].str.split(';', expand = True)
    row = categories.loc[1,:]
    category_colnames = [xx[:-2] for xx in row]
    categories.columns = category_colnames
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = [xx[-1:] for xx in categories[column]]       
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original 'categories' and merge with new one. 
    df.drop(labels=['categories'], axis=1,inplace=True)
    df = pd.concat([df,categories], axis =1)
    
    # drop duplicates
    df.drop_duplicates(subset=['message'], inplace= True)
    
    # replace '2' by '1' in the column 'related'
    df['related'].replace(2,1, inplace=True)
    
    # the column 'military' has only zero. Cannot be used in training. Delete the column
    # df.drop(labels=['military'],axis=1,inplace=True)
    
    return df


def save_database(df, database_filepath):
    '''
    Save the data to local database

    Parameters
    ----------
    df : pandas.dataframe
        data to be saved
    database_filepath : String
        The file path for the database 

    Returns
    -------
    None.

    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.replace('/', '.').split('.')[-2]
    df.to_sql(table_name, engine, index=False)

def main():
    '''
    The main function for the file

    Returns
    -------
    None.

    '''
    if len(sys.argv) == 4:
    
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        # Read the input files
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = print_input(messages_filepath, 
                                           categories_filepath, 
                                           database_filepath)
    
        # Prepare the output data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        print('Cleaning data...')
        df = prepare_data(messages, categories)
    
        # Save to a local sql file
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_database(df, database_filepath)
        
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