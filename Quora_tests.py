import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras import optimizers
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
import pickle
import lexdecomp_model
import data_generation
import datetime
from configs import configs
import tensorflow as tf


###################
#  PARAMETERS #
###################
# Selecting the model

##DATA PARAMETERS
DATASET_NAME = configs["global"]["DATASET"]
DATASET = configs[DATASET_NAME]
TRAIN_FILE = DATASET["TRAIN_FILE"]
TEST_FILE = DATASET["TEST_FILE"]

Xtrain1_inputs = None
Xtrain2_inputs = None
Y_train = None

print("################### Loading embedding matrix #######################")
with open("{}_w2v_29_11.pickle".format(DATASET_NAME), 'rb') as f:
    embedding_matrix = pickle.load(f)[0]
    
print("################### Loading train sequences #######################")
with open("{}_seq_train.pickle".format(DATASET_NAME), 'rb') as f:
    ff = pickle.load(f)
    Xtrain1_inputs,Xtrain2_inputs, Y_train = ff[0],ff[1], ff[2]


## MODEL PARAMETERS 

epochs = configs["global"]["EPOCHS"]
batch_size = configs["global"]["BATCH_SIZE"]
#window for local-w 
window = configs["global"]["WINDOW"]
# method = linear or orthogonal
method = configs["global"]["METHOD"]
#500 filters of each size (unigram to trigram)
filters =configs["global"]["FILTERS"]
use_class_weight = configs["global"]["use_class_weight"]

#######################
# END PARAMETERS #
#######################


print("################# list of sequences to matrix #############################")

print(Xtrain1_inputs.shape)

print(Xtrain2_inputs.shape)

Y_train = np.array(Y_train)
print(embedding_matrix.shape)

X_train1 = data_generation.from_seq_to_vec(Xtrain1_inputs, embedding_matrix)
X_train2 = data_generation.from_seq_to_vec(Xtrain2_inputs, embedding_matrix)


# Decomposing train and test data
print("Decomposing training data")
X_train1, X_train2 = lexdecomp_model.decompose_data(X_train1, X_train2, window, method)

print("Decomposed data")
print("X_train:", X_train1.shape)
print("Y_train:", Y_train.shape)

embeddings_dim = embedding_matrix.shape[1]
max_seq_length = Xtrain1_inputs.shape[1]

model = lexdecomp_model.main_model((max_seq_length, embeddings_dim,2),
                                        embeddings_dim, max_seq_length, filters)

optimizer = optimizers.Adam(learning_rate= 0.001)
model.compile(optimizer=optimizer,
              loss="binary_crossentropy",
              metrics=["accuracy"])

#loading the best model weights
checkpoint_filepath = './runs/wang_lexdecomp/tmp/{}_model_weights.hdf5'.format("msrpc")

model.load_weights(checkpoint_filepath)

print("Evaluation (loss, acc)")
loss, acc = model.evaluate(x=[X_train1, X_train2], y=Y_train)

print("loss: {:.4f}   acc: {:.4f}".format(loss, acc))

