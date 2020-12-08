from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import numpy as np
import re
import codecs
import pickle
import pandas as pd
from nltk.corpus import stopwords
from configs import configs
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import string


def load_word2vec(file): 
    #load pre-trained  Word2Vec vectors on GoogleNews-vectors-negative300.bin
    model = KeyedVectors.load_word2vec_format(file, binary=True,limit=10**5)
    return model


def expand_contraction(text):
    #perform contraction expansion
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"c\+\+", "cplusplus", text)
    text = re.sub(r"c \+\+", "cplusplus", text)
    text = re.sub(r"c \+ \+", "cplusplus", text)
    text = re.sub(r"c#", "csharp", text)
    text = re.sub(r"f#", "fsharp", text)
    text = re.sub(r"g#", "gsharp", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)
    text = re.sub(r",000", '000', text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"didn’t", "do not", text)

    return text

def preprocess(sentence,stemmer):
    #perform preprocessing on sentence 
    #lowercase 
    #stop words deletion
    # lemmatization or stemming using stemmer ; ex : studies ==> study / studi
    #contraction expansion using above function
    #punctuation deletion 

    lower= configs['preprocess']['lower']
    removestopwords =  configs['preprocess']['removestopwords']
    stemming = configs['preprocess']['stemming']
    contraction_expand = configs['preprocess']['contract_expand']
    remove_punctuation = configs['preprocess']['remove_punctuation']

    new_sent = sentence

    if lower:
        new_sent = new_sent.lower() 


    if remove_punctuation:
        new_sent = ' '.join([ch for ch in new_sent.split() if ch not in PUNCTUATIONS])
        

    if contraction_expand:
        new_sent = expand_contraction(new_sent)

    if stemming:
        new_sent = stemmer.lemmatize(new_sent)

    if removestopwords:
        new_sent = remove_stopwords(new_sent)


    return new_sent
            
    

def load_msrp_data(file,stemmer,preprocessing= False,training=True): 
    #load MSRP dataset from MSRPC/msr_paraphrase_train.txt or MSRPC/msr_paraphrase_test.txt'
    #file : file name
    #stemmer : lemmatizer / stemmer
    #preprocessing : if True, it performs preprocessing before 
    #Training : if training, load labels if available
    #returns left sentences list, right sentences lists and labels

    sentences_left = []
    labels = []
    sentences_right = []

    with codecs.open(file, 'r', encoding='utf-8') as raw:
            for idx, line in enumerate(raw):
                line = line.strip()
                if idx != 0 and len(line) != 0:
                    label, id1, id2, sentence1, sentence2 = line.split('\t')

                    sentence1 = re.sub(r'\W', ' ', str(sentence1))
                    sentence1 = re.sub(r'\s+', ' ', sentence1, flags=re.I)

                    sentence2 = re.sub(r'\W', ' ', str(sentence2))
                    sentence2 = re.sub(r'\s+', ' ', sentence2, flags=re.I)

                    labels.append(int(label))
                    sentences_left.append(sentence1)
                    sentences_right.append(sentence2)

    ##additionnal preprocessing
    if preprocessing:
        
        sentences_left = [preprocess(str(sentence),stemmer).split() for sentence in sentences_left]

        sentences_right = [preprocess(str(sentence),stemmer).split() for sentence in sentences_right]
            
    return sentences_left,sentences_right,labels


def load_quora_data(file, stemmer,preprocessing= False,training=True): 
    #load MSRP dataset from 'Quora/train.csv or / 'Quora/test.csv'
    #file : file name
    #stemmer : lemmatizer / stemmer
    #preprocessing : if True, it performs preprocessing before 
    #Training : if training, load labels if available
    #returns left sentences list, right sentences lists and labels

    data = pd.read_csv(file)

    labels = []

    data = data.iloc[np.random.permutation(len(data))]

    data = data.iloc[:configs["global"]["MAX_SAMPLE_SIZE"]]

    if training:
        
        labels = list(data['is_duplicate'])


    sentences_left = list(data['question1'])

    sentences_right = list(data['question2'])


    sentences_left = [re.sub(r'\W', ' ', str(sentence)) for sentence in sentences_left]
    sentences_left = [re.sub(r'\s+', ' ', sentence, flags=re.I) for sentence in sentences_left]

    sentences_right = [re.sub(r'\W', ' ', str(sentence)) for sentence in sentences_right]
    sentences_right = [re.sub(r'\s+', ' ', sentence, flags=re.I) for sentence in sentences_right]

    ##additionnal preprocessing
    if preprocessing:
        
        sentences_left = [preprocess(sentence,stemmer) for sentence in sentences_left]

        sentences_right = [preprocess(sentence,stemmer) for sentence in sentences_right]

    return sentences_left,sentences_right,labels


def from_seq_to_vec(X, embedding_matrix):
    #convert padded word_index sequences to matrix of embeddings
    #embedding_matrix : build from training dataset using pre-trained word vectors
    X = np.array(X)
    X_emb = np.zeros(X.shape+(embedding_matrix.shape[1],),dtype='float32')
    for i, val in np.ndenumerate(X):
        if val < embedding_matrix.shape[0]:
            X_emb[i] = embedding_matrix[val]
    return X_emb


###################################################################################
#Word2vec pre-trained word vectors
model_w2v = load_word2vec('GoogleNews-vectors-negative300.bin')
#dataset name
DATASET = configs["global"]["DATASET"] 
#max sequence lenght == max words in sentence
MAX_SEQ_LENGTH = configs["global"]["MAX_SEQ_LENGTH"] 

TRAIN_FILE = configs[DATASET]["TRAIN_FILE"] 
TEST_FILE =  configs[DATASET]["TEST_FILE"] 
#lemmatizer instanciation
lemm = WordNetLemmatizer()

#punctuations from string
PUNCTUATIONS = set(string.punctuation) 
###################################################################################
#loading training and test datasets
dataset_load_func = load_quora_data if DATASET == "quora" else load_msrp_data

train_sentences_left, train_sentences_right, train_labels = dataset_load_func(TRAIN_FILE,lemm,True,True) 
train_sentences = train_sentences_left + train_sentences_right

test_sentences_left, test_sentences_right, test_labels = dataset_load_func(TEST_FILE,lemm,True,False)

#Tokenization
tokenizer_inputs = Tokenizer(filters='')
#generating words index from training set
tokenizer_inputs.fit_on_texts(train_sentences)
#wrds index dict
word2idx_inputs = tokenizer_inputs.word_index 
#sequences padding
train_left = pad_sequences(tokenizer_inputs.texts_to_sequences(train_sentences_left),maxlen=MAX_SEQ_LENGTH, padding='post') 
train_right = pad_sequences(tokenizer_inputs.texts_to_sequences(train_sentences_right),maxlen=MAX_SEQ_LENGTH, padding='post') 

test_left = pad_sequences(tokenizer_inputs.texts_to_sequences(test_sentences_left),maxlen=MAX_SEQ_LENGTH, padding='post')
test_right = pad_sequences(tokenizer_inputs.texts_to_sequences(test_sentences_right),maxlen=MAX_SEQ_LENGTH, padding='post')


#########################################################################################

VEC_DIM = configs["global"]["VEC_DIM"] #size of one word embedding
VOCABULARY_SIZE = configs["global"]["VOCABULARY_SIZE"] #max vocabulary size
TRAIN_VOCABULARY_SIZE = min(VOCABULARY_SIZE, len(word2idx_inputs) +1) #actual vocabulary size

#initializing embedding matrix
embeddings_matrix = np.zeros((TRAIN_VOCABULARY_SIZE,VEC_DIM),dtype='float32')

#building embedding matrix
dictionnary_ = model_w2v.wv.vocab

for word, i in word2idx_inputs.items():
 if word in list(dictionnary_.keys()) and i < TRAIN_VOCABULARY_SIZE:
    embedding_vector = model_w2v[word]
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector


#saving preprocessed data, embedding matrix
dest_embeddings = "{}_w2v_29_11.pickle".format(str(DATASET)) 
preprocessed_train_test ="{}_prep_train_test.pickle".format(str(DATASET))
sequences_train = "{}_seq_train.pickle".format(str(DATASET))
sequences_test = "{}_seq_test.pickle".format(str(DATASET))

with open(dest_embeddings, 'wb') as f:
    pickle.dump([embeddings_matrix,word2idx_inputs], f)
    f.close()


with open(sequences_train, 'wb') as f:
    pickle.dump([train_left,train_right,train_labels], f)
    f.close()


with open(sequences_test, 'wb') as f:
    pickle.dump([test_left,test_right,test_labels], f)
    f.close()


with open(preprocessed_train_test, 'wb') as f:
    pickle.dump([train_sentences_left,train_sentences_right,test_sentences_left,test_sentences_right],f)
    f.close()
     


