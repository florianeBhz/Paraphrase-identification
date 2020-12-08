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

##DATA PARAMETERS
DATASET_NAME = configs["global"]["DATASET"]
DATASET = configs[DATASET_NAME]
TRAIN_FILE = DATASET["TRAIN_FILE"]
TEST_FILE = DATASET["TEST_FILE"]

embedding_matrix = None
Xtrain1_inputs = None
Xtrain2_inputs = None
Y_train = None

Xtest1_inputs = None 
Xtest2_inputs = None 
Y_test = None 
print("################### Loading embedding matrix #######################")
with open("{}_w2v_29_11.pickle".format(DATASET_NAME), 'rb') as f:
    embedding_matrix = pickle.load(f)[0]
    
print("################### Loading train sequences #######################")
with open("{}_seq_train.pickle".format(DATASET_NAME), 'rb') as f:
    ff = pickle.load(f)
    Xtrain1_inputs,Xtrain2_inputs, Y_train = ff[0],ff[1], ff[2]

print("################### Loading test sequences #######################")
with open("{}_seq_test.pickle".format(DATASET_NAME), 'rb') as f:
    ff = pickle.load(f)
    Xtest1_inputs,Xtest2_inputs, Y_test = ff[0],ff[1], ff[2]



## MODEL PARAMETERS 

epochs = configs["global"]["EPOCHS"]
batch_size = configs["global"]["BATCH_SIZE"]
#window for local-l 
window = configs["global"]["WINDOW"]
# method = linear or orthogonal
method = configs["global"]["METHOD"]
#500 filters of each size (unigram to trigram)
filters =configs["global"]["FILTERS"]

#######################
# END PARAMETERS #
#######################


print("################# list of sequences to matrix #############################")

Y_test = np.array(Y_test)
print(Xtrain1_inputs.shape)

print(Xtrain2_inputs.shape)
print(Xtest1_inputs.shape)
print(Xtest2_inputs.shape)

Y_train = np.array(Y_train)
print(embedding_matrix.shape)

X_train1 = data_generation.from_seq_to_vec(Xtrain1_inputs, embedding_matrix)
X_train2 = data_generation.from_seq_to_vec(Xtrain2_inputs, embedding_matrix)
X_test1 = data_generation.from_seq_to_vec(Xtest1_inputs, embedding_matrix)
X_test2 = data_generation.from_seq_to_vec(Xtest2_inputs, embedding_matrix)


# Decomposition into similar and dissimilar parts
print("Decomposing training data")
X_train1, X_train2 = lexdecomp_model.decompose_data(X_train1, X_train2, window, method)
print("Decomposing test data")
X_test1, X_test2 = lexdecomp_model.decompose_data(X_test1, X_test2, window, method)
print("Decomposed data")
print("X_train:", X_train1.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test1.shape)
print("Y_test:", Y_test.shape)


embeddings_dim = embedding_matrix.shape[1]
max_seq_length = Xtrain1_inputs.shape[1]

# Creation of the model
model = lexdecomp_model.main_model((max_seq_length, embeddings_dim,2),
                                        embeddings_dim, max_seq_length, filters)
# Printing summaries
model.summary(line_length=100)

# Compiling model
optimizer = optimizers.Adam(learning_rate= 0.001)
model.compile(optimizer=optimizer,
              loss="binary_crossentropy",
              metrics=["accuracy"])

class prediction_history(Callback):
    def __init__(self):
        self.acchis = []
        self.f1his = []
    def on_epoch_end(self, epoch, logs={}):
        pred=self.model.predict([self.validation_data[0], self.validation_data[1]])
        predclass = np.where(pred>0.5, 1, 0).reshape(-1)
        acc = accuracy_score(self.validation_data[2],predclass)
        print(acc)
        self.acchis.append(acc)
        f1 = f1_score(self.validation_data[2],predclass)
        print(f1)
        self.f1his.append(f1)

per_epoch_preds = prediction_history()


# Training model
print("Training model ...")

checkpoint_filepath = './runs/wang_lexdecomp/tmp/{}_model_weights.hdf5'.format(DATASET_NAME)
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor= 'val_accuracy',
    mode= 'max',
    save_best_only=True)

#es = EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=4,verbose=1,restore_best_weights=True)

my_calls = [model_checkpoint_callback,es,#per_epoch_preds
           tf.keras.callbacks.TensorBoard(log_dir='./logs/mrsp')]

history = model.fit(x=[X_train1, X_train2],
                    y=Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=my_calls)


#visualizing traing and val losses
# Plot history: BCEntropy
plt.plot(history.history['loss'], label='BCE (training data)')
plt.plot(history.history['val_loss'], label='BCE (validation data)')
plt.title('BCE on {}'.format(DATASET_NAME))
plt.ylabel('BCE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig('Training.png')
plt.close()

#loading the best model weights
model.load_weights(checkpoint_filepath)

print("Evaluation (loss, acc)")
loss, acc = model.evaluate(x=[X_test1, X_test2], y=Y_test)
print("loss: {:.4f}   acc: {:.4f}".format(loss, acc))
with open("tmp.p", "wb") as fid:
    pickle.dump(model.history.history, fid)
pred = np.where(model.predict(x=[X_test1, X_test2])>0.5, 1, 0).reshape(-1)
f1 = f1_score(Y_test, pred)
print("f1: {:.4f}".format(f1))
print("confusion matrix")
cf_mat = confusion_matrix(Y_test, pred)
print(cf_mat)
history.history["test_loss"] = loss
history.history["test_acc"] = acc
history.history["f1"] = f1
history.history["cf_mat"] = cf_mat
history.history["pred"] = pred

hdir = "./runs/wang_lexdecomp/"
date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
hfname = hdir + "hist_" + date + "_wang_lexdecopm.p"
with open(hfname, "wb") as fid:
    pickle.dump(history.history, fid)
