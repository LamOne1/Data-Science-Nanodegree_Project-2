import sys, os
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import pickle
import pandas as pd
from sqlalchemy import create_engine
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC

lemmatizer = WordNetLemmatizer()
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """
    load dataset from database into pandas dataframe
    Inputs:
        database_filepath: path to database file
    Outputs:
        X: dataset features
        Y: dataset labels
        category_names: labels
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath.split(os.sep)[-1].replace('.db',''),engine)
    #conn = engine.connect()
    #print(pd.read_sql('show tables', con = conn))
    #df = pd.read_sql('SELECT * FROM DisasterResponse', con = conn)

    #X = df[['message','genre']].copy()
    X = df['message'].copy()
    Y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']].copy()
    
    return X, Y, Y.columns

'''
class select_column(BaseEstimator, TransformerMixin):
    """
    Transformer class to select a specific column from the dataframe    
    """

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, data):
        if self.column == 'genre':
            labels, _ = pd.factorize(data[self.column])
            return pd.DataFrame(labels)#[np.newaxis]
        return data[self.column]
'''

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try:
                first_word, first_tag = pos_tags[0]
            except:
                return False
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def tokenize(text):
    """
    Process the text and clean it to prepare it for training
    Inputs:
        text: text to process
    Outputs:
        clean_tokens: text split into tokens after cleaning
    """
    # this list is copied from source https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python
    contractions = {
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    pat = re.compile(r"\b(%s)\b" % "|".join(contractions))
    text = pat.sub(lambda m: contractions.get(m.group()), text)

    text = re.sub('[^ \a-zA-Z0-9]', ' ', text)
    text = re.sub(' +', ' ',text)
    tokens = word_tokenize(text)

    clean_tokens = [lemmatizer.lemmatize(tok.lower()) for tok in tokens if tok not in stop_words]

    return clean_tokens


def build_model():
    """
    build the model pipeline
    Inputs:
        ----
    Outputs:
        pipeline: model pipeline 
    """
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline', Pipeline([
            #('select_message', select_column('message')), 
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        
            ('starting_verb', StartingVerbExtractor()),
            #('select_genre', select_column('genre')),

    ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evalute the model on the test set and show the results
    Inputs:
        model: trained model
        X_test: test features
        Y_test: test labels
        category_names: labels names
    Outputs:
        ---- 
    """
    y_pred = model.predict(X_test)
    results = []
    columns_name = list(category_names)
    Y_test = np.asarray(Y_test)
    for i in range(len(columns_name)):
        acc = accuracy_score(Y_test[:,i], y_pred[:,i])
        f1 = f1_score(Y_test[:,i], y_pred[:,i], average='macro')
        precision = precision_score(Y_test[:,i], y_pred[:,i], average='macro')
        recall = recall_score(Y_test[:,i], y_pred[:,i], average='macro')
        
        results.append([acc, f1, precision, recall])
        
    
    df = pd.DataFrame(data = np.array(results), index=columns_name, columns=["Accuracy", "F1_score", "Precision" ,"Recall"])
    print(df)


def save_model(model, model_filepath):
    """
    Save the trained model to a pickle file 
    Inputs:
        model: trained model
        model_filepath: path of the pickle file
    """       
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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