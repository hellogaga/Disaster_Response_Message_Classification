# Import necessary modules
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd 
from sqlalchemy import create_engine
import re
import pickle
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''
    Load the data from the database

    Parameters
    ----------
    database_filepath : String
        The file path for the database

    Returns
    -------
    X : numpy.array
        Data array that contians the messages
    Y : numpy.array
        Data array that contians the categories
    category_name : List
        A list that contains the names of categories

    '''
    print ('The database file is:', database_filepath)
    engine = create_engine('sqlite:///' + database_filepath)
    # Get the table name
    table_name = database_filepath.replace('/', '.').split('.')[-2]
    df = pd.read_sql("SELECT * FROM " + table_name, engine)  
    X = df['message'].values
    Y = df[df.columns[4:]].values
    category_name = df.columns[4:]
    return X, Y, category_name


def tokenize(text):
    '''
    A function that tokennize text

    Parameters
    ----------
    text : String
        The text string to tokenize

    Returns
    -------
    clean_tokens : List
        The tokenized text.

    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    Use pipeline to build the model and parameters to be optimized.
    Use grid search to build the traing model

    Returns
    -------
    cv : grid search cv object
        The model to be trained and the model to predict. 

    '''
    pipeline2 =  Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                           ('tfidf', TfidfTransformer()),
                           ('clf', MultiOutputClassifier(MultinomialNB()))
                          ])
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.5, 0.75, 1.0),
                  'tfidf__use_idf': (True, False),
                  'clf__estimator':(MultinomialNB(), KNeighborsClassifier())}
    cv =  GridSearchCV(pipeline2, param_grid=parameters, verbose=10, n_jobs=-1)
    return cv

def print_results(y_test, y_pred, category_name):
    '''
    Print out the trained model test results. 

    Parameters
    ----------
    y_test : numpy.array
        The real value
    y_pred : numpy.array
        The predicted value
    category_name : List
        A list that contains the names of categories

    Returns
    -------
    None.

    '''
    
    report = dict()
    for col in range(y_test.shape[1]):
        print ('the result summary for the category', category_name[col])
        report[col] = classification_report(y_test[:,col], y_pred[:,col])
        print (report[col])

def save_model(trained_model, model_filepath):
    '''
    To save the trained model to the specified file

    Parameters
    ----------
    trained_model : TYPE
        DESCRIPTION.
    model_filepath : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    pickle.dump(trained_model, open(model_filepath, 'wb'))

def main():
    '''
    The main function for the file

    Returns
    -------
    None.

    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        # Read the input files
        X,Y,category_name = load_data(database_filepath) 
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        cv = build_model()
        
        print('Training model...')
        cv.fit(X_train, y_train)
        
        print('Evaluating model...')
        y_pred = cv.predict(X_test)
        print_results(y_test, y_pred, category_name)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()