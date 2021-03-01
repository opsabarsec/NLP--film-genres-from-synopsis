import keras
from flask import Flask
from flask import request
# ## 1. Import libraries and load data
#packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# NLP libraries
import nltk
import re
from textblob import TextBlob, Word
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Deep learning libraries

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, GlobalMaxPool1D, Dropout
# Loading tokenizer

app = Flask(__name__)


#if __name__ == '__main__':
#    app.run(host='localhost', debug=True, port=5000)



# load data
@app.route('/genres/train', methods=['GET','POST']) 
def endpoint_train():
    filename = request.files['csv'].filename
    train= pd.read_csv(filename)
    return train

@app.route('/genres/predict', methods=['GET','POST'])
def endpoint_test():
    filename = request.files['csv'].filename
    test = pd.read_csv(filename)
    return test

train= pd.read_csv('genres/train/train.csv')
test= pd.read_csv('genres/predict/test.csv')

# ## 2. DATA PREPARATION 
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function for text cleaning 
def preprocess_text(text):
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
print('preprocessing text...')
train['clean_plot'] = train['synopsis'].apply(lambda x: preprocess_text(x))
test['clean_plot'] = test['synopsis'].apply(lambda x: preprocess_text(x))

def lemma(text): # Lemmatization of cleaned body
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
        
train['lemma'] = train['clean_plot'].apply(lambda x: lemma(x))
test['lemma'] = test['clean_plot'].apply(lambda x: lemma(x))
print('preprocessing and lemmatization done!')
## 3. Variables preparation 

X = train['lemma']
X_test = test['lemma']    

# ### 3.1 Target variable one hot encoding

#apply the onehot transformation for the genres vector
y = train['genres']
one_hot = MultiLabelBinarizer() # encoder for the  tags 
y_onehot = one_hot.fit_transform(y.str.split(' ')) 
y_bin = pd.DataFrame(y_onehot, columns=one_hot.classes_ ) # transform it to Pandas object


# tokenize
max_features = 5000
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(X))
list_tokenized_train = tokenizer.texts_to_sequences(X)
list_tokenized_test = tokenizer.texts_to_sequences(X_test)


maxlen = 150
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# ## 4.The Model

# Load from file
#model = load_model('my_model.h5')
#initialize parameters
inp = Input(shape=(maxlen, )) #maxlen defined earlier
embed_size = 128
# Neural network backbone
x = Embedding(max_features, embed_size)(inp)

x = LSTM(64, return_sequences=True,name='lstm_layer')(x)

x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(len(y_bin.columns), activation="softmax")(x)
# build the model
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
# train the model
batch_size = 16
epochs = 3
print('training the LSTM model...')
hist = model.fit(X_t,y_onehot, batch_size=batch_size, epochs=epochs, validation_split=0.1)                 
print('LSTM neural network weights updated, model trained!')
# ## 5.The prediction
print('prediction...')
y_pred = model.predict(X_te, batch_size= 16, verbose=0)
print(y_pred.shape)
print('obtained probability matrix')
df_probs_all = pd.DataFrame(y_pred,columns=y_bin.columns)

def top_5_predictions(df):
    N = 5
    cols = df.columns[:-1].tolist()
    a = df[cols].to_numpy().argsort()[:, :-N-1:-1]
    c = np.array(cols)[a]
    d = df[cols].to_numpy()[np.arange(a.shape[0])[:, None], a]
    df1 = pd.DataFrame(c).rename(columns=lambda x : f'max_{x+1}_col')

    predicted_genres = df1["max_1_col"] + ' ' + df1["max_2_col"]+ ' ' +df1["max_3_col"]+ ' ' + df1["max_4_col"]+ ' '+df1["max_5_col"]
    return predicted_genres


pred_gen = top_5_predictions(df_probs_all)

submission = pd.DataFrame(data= {'movie_id':test.movie_id,'predicted_genres':pred_gen})

submission.to_csv('genres/predict/submission.csv',index=False)
print('submission.csv created and saved. Ready to be submitted to Kaggle')

