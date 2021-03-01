#****************************************************************
# import libraries
from flask import Flask
from flask import request
from flask import make_response
from flask import jsonify
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# NLP libraries
import nltk
import re
from textblob import TextBlob, Word
from sklearn.preprocessing import MultiLabelBinarizer

from nltk.corpus import stopwords
nltk.download('stopwords')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Deep learning libraries
import keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, GlobalMaxPool1D, Dropout

#****************************************************************
# Functions definition

## 1. DATA PREPARATION 

### 1.1 function for text cleaning 
def preprocess_text(text):
    print('preprocessing text...')
    stop_words = set(stopwords.words('english'))
    text = text.lower() # lowercase
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text) #line breaks
    #text = re.sub(r"\'\xa0", " ", text) # xa0 Unicode representing spaces
    #text = re.sub('\s+', ' ', text) # one or more whitespace characters
    text = text.strip(' ') # spaces
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    #lemmatize and remove stopwords
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    text = ' '.join(no_stopword_text) 
        
    return text

### 1.2 function for lemmatization
def lemma(text): # Lemmatization of cleaned body
        print('lemmmatizing...')
        sent = TextBlob(text)
        tag_dict = {"J": 'a', 
                    "N": 'n', 
                    "V": 'v', 
                    "R": 'r'}
        words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        seperator=' '
        lemma = seperator.join(lemmatized_list) 
        return lemma
      
## 2. MODEL

def build_model(max_features, maxlen, inp, embed_size):
    x = Embedding(max_features, embed_size)(inp)
    
    x = LSTM(64, return_sequences=True,name='lstm_layer')(x)
    
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(19, activation="softmax")(x)
    ### 2.3 build the model
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    print("LSTM neural network compiled")
    return model

## 3. Extraction of 5 m,ost probable genres tags from the predictions matrix
def top_5_predictions(df):
        N = 5
        cols = df.columns[:-1].tolist()
        a = df[cols].to_numpy().argsort()[:, :-N-1:-1]
        c = np.array(cols)[a]
        #d = df[cols].to_numpy()[np.arange(a.shape[0])[:, None], a]
        df1 = pd.DataFrame(c).rename(columns=lambda x : f'max_{x+1}_col')
    
        predicted_genres = df1["max_1_col"] + ' ' + df1["max_2_col"]+ ' ' +df1["max_3_col"]+ ' ' + df1["max_4_col"]+ ' '+df1["max_5_col"]
        return predicted_genres
#****************************************************************
# run the API

## 1. Define train and predict folders
app = Flask(__name__)
UPLOAD_FOLDER = '/home/marco/Documents/CV/home assignments/Radix/challenge/challenge/genres/train'
UPLOAD_FOLDER1 = '/home/marco/Documents/CV/home assignments/Radix/challenge/challenge/genres/predict'

# 2. Upload endpoints
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/genres/train', methods=['POST','PUT']) 
def upload_train():
    file = request.files['csv']
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], "train.csv"))
    resp = jsonify({'message' : 'File successfully uploaded'})
    resp.status_code = 201
    return resp

app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1
@app.route('/genres/predict', methods=['POST','PUT']) 
def upload_test():
    file = request.files['csv']
    file.save(os.path.join(app.config['UPLOAD_FOLDER1'], "test.csv"))
    resp = jsonify({'message' : 'File successfully uploaded'})
    resp.status_code = 201
    return resp

# Process and output endpoint

@app.route('/', methods=['POST','PUT'])
def endpoint_process():
    
    # Create train dataframe from the uploaded csv file
    print('train.csv uploaded to /genres/train')
    train= pd.read_csv('genres/train/train.csv')
    print("train.csv transformed into Pandas dataframe")
           
    ## 1. Preprocess text for training matrix
    train['clean_plot'] = train['synopsis'].apply(lambda x: preprocess_text(x))
    train['lemma'] = train['clean_plot'].apply(lambda x: lemma(x))
    X = train['lemma']
    ## 2. define train matrix parameters and tokenize the text
    max_features = 5000
    maxlen = 150
            
    tokenizer = Tokenizer(num_words=max_features)
        
    tokenizer.fit_on_texts(list(X))
    list_tokenized_train = tokenizer.texts_to_sequences(X)
        
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen) # this is the final training matrix
    
        
    ## 3. Apply the onehot transformation for the target vector
    y = train['genres']
    one_hot = MultiLabelBinarizer() # encoder for the  tags 
    y_onehot = one_hot.fit_transform(y.str.split(' ')) # this is the target vector for training
    y_bin = pd.DataFrame(y_onehot, columns=one_hot.classes_ ) # transform it to Pandas object
    
    ## 1. Define the model parameters
    inp = Input(shape=(maxlen, )) #maxlen defined earlier
    embed_size = 128
    batch_size = 16
    epochs = 3
    
    ## 2. Compile the model
    model = build_model(max_features, maxlen, inp, embed_size)
    
    ## 3. Train the model
    print('training the LSTM model...')
    model.fit(X_t,y_onehot, batch_size=batch_size, epochs=epochs, validation_split=0.1)                 
    print('LSTM neural network weights updated, model trained!')

    # Create test dataframe from the uploaded test.csv file
    
    print('test.csv uploaded to /genres/predict')
    test= pd.read_csv('genres/predict/test.csv')
    print("test.csv transformed into Pandas dataframe")
    ## 1. preprocess text
    test['clean_plot'] = test['synopsis'].apply(lambda x: preprocess_text(x))


    test['lemma'] = test['clean_plot'].apply(lambda x: lemma(x))
    print('preprocessing and lemmatization done!')
           
    X_test = test['lemma']    
       
    list_tokenized_test = tokenizer.texts_to_sequences(X_test)
        
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    
    # Predict genres tags appplying the model to test set
    print('prediction...')
    y_pred = model.predict(X_te, batch_size= 16, verbose=0)
    print(y_pred.shape)
    print('obtained probability matrix')
    
    # Obtain a dataframe for the predictions
    df_probs_all = pd.DataFrame(y_pred,columns=y_bin.columns)
            
    pred_gen = top_5_predictions(df_probs_all) # function defined earlier that returns the 5 most probable genres for each movie
           
    submission = pd.DataFrame(data= {'movie_id':test.movie_id,'predicted_genres':pred_gen})
    
    # Return a csv file with predictions as response to the user  
    csv =  submission.to_string()
    
    response = make_response(csv)
    response.headers["Content-Disposition"] = "attachment; filename=submission.csv"
    response.headers["Content-type"] = "text/csv"
      
    return response

#run command

if __name__ == "__main__":
    app.debug=True
    app.run()


