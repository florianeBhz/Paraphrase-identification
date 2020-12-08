import numpy as np
import keras.backend as K
import tensorflow as tf 
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Lambda, Conv2D
from keras.layers import MaxPooling2D, Flatten, Concatenate, Dense
from keras.layers import Activation, BatchNormalization, Dropout, LeakyReLU, ELU
from configs import configs

MAX_SEQ_LENGTH = configs["global"]["MAX_SEQ_LENGTH"] 
VEC_DIM = configs["global"]["VEC_DIM"]

#### Code from https://github.com/CubasMike/paraphrase_identification/blob/master/wang_lexdecomp_approach.py ####

def semantic_match_word_to_sentence(S, T, A, w): #local-w 
    """
    shape X: (s,n,d), Y: (s,m,d), A: (s, n, m)
    A is the semantic matching at word-word level
    window : w (=3)
    """
    # shape Pivot, lower_lim, upper_lim: (s,n,1)
    #Pivot = k : the max matching between each word of X and 
    #a whole sentence of Y for each sentence of X
    #for each sentence of Y
    
    k = np.expand_dims(np.argmax(A, axis=-1), axis=-1) 
    left_bound = np.maximum(0, k-w) #
    right_bound = np.minimum(A.shape[-1], k+w)

    # shape indices: (s,n,m)
    indices = np.tile(np.arange(A.shape[-1]), A.shape[:-1]+(1,))
    # NOTE: To replicate "mcrisc" implementation in github use: indices < upper_lim
    mask = ((indices >= left_bound) & (indices <= right_bound)).astype(np.float32)

    # shape X_hat: (n,d)
    S_hat = np.matmul(A*mask, T)

    return S_hat #ligne 7 de l'algo 

def decompose(X, X_hat, method="linear"):
    """Decompose a dataset into pos and neg components 
    with regards to its semantic match version
    
    shape X, X_hat: (s,n,d)
    """
    assert method in ("linear", "orthogonal")
    if method == "linear":
        # shape alpha: (s,n,1)
        denom = (np.linalg.norm(X, axis=-1, keepdims=True) *
                 np.linalg.norm(X_hat, axis=-1, keepdims=True))
        alpha = np.divide(np.sum(X * X_hat, axis=-1, keepdims=True),
                          denom, where=denom!=0)

        # shape X_pos, X_neg: (s,n,d)
        X_pos = alpha * X
        X_neg = (1 - alpha) * X
    elif method == "orthogonal": #the chosen one (line 8)
        # shape X_pos, X_neg: (s,n,d)
        denom = np.sum(X_hat * X_hat, axis=-1, keepdims=True)
        X_pos = np.divide(np.sum(X * X_hat, axis=-1, keepdims=True),
                          denom, where=denom!=0) * X_hat
        X_neg = X - X_pos
    X_pos = np.expand_dims(X_pos, axis=-1)
    X_neg = np.expand_dims(X_neg, axis=-1)
    # shape X_decomp: (s,n,d,2)
    X_decomp = np.concatenate([X_pos, X_neg], axis=-1)
    return X_decomp


def decompose_data(X, Y, window=3, method="linear"): 
    """Decompose datasets X, Y into positive and negative
    channels with regards to each other
    shape X: (s,n,d), Y: (s,m,d)
    """
    # Cosine similarity
    # shape A: (s,n,m)
    norm_X = np.linalg.norm(X, axis=-1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=-1, keepdims=True)
    A = np.matmul(np.divide(X, norm_X, where=norm_X!=0), np.swapaxes(np.divide(Y, norm_Y, where=norm_Y!=0), -1, -2))
    A = np.matmul(np.divide(X, norm_X, where=norm_X!=0), np.swapaxes(np.divide(Y, norm_Y, where=norm_Y!=0), -1, -2))

    # Semantic matching
    # shape X_hat: (s,n,d), Y_hat: (s,m,d)
    X_hat = semantic_match_word_to_sentence(X, Y, A, w=window)
    Y_hat = semantic_match_word_to_sentence(Y, X, np.swapaxes(A, -1, -2), w=window)
    # Decomposition (pos, neg)
    X_decomp = decompose(X, X_hat, method=method)
    Y_decomp = decompose(Y, Y_hat, method=method)

    return X_decomp, Y_decomp #lines 8 and 12



#### End of code from https://github.com/CubasMike/paraphrase_identification/blob/master/wang_lexdecomp_approach.py #######


def composition(input_shape, embeddings_dim, max_seq_length, filters): 

    #compose features vectors S_vect and T_vect from [S+ , S-] or [T+,T-]
    #embeddings_dim = 300
    #max_seq_length in {39, 50}
    #filter_size in {1, 2, 3}
    #returns S_vect or T_vect
      
    X_input_placeholder = Input(input_shape)
    
    conv_list = []
    for i, (filter_size, nf) in enumerate(filters):
        # Output shape: (batch_size, outdim_conv, number_of_filters)
        conv = Conv2D(nf, #nf = 500
                      kernel_size=(filter_size, embeddings_dim),
                      strides=1,
                      kernel_initializer='random_normal',
                      bias_initializer=tf.constant_initializer(0.1),
                      trainable=True,
                      activation = 'tanh',
                      name="conv"+str(i))(X_input_placeholder)
        
        #conv = LeakyReLU()(conv)
        #conv = ELU()(conv)
        
        outdim_conv = max_seq_length - filter_size + 1
        conv = MaxPooling2D(pool_size=(outdim_conv, 1),
                            name="maxpool"+str(i)
                            )(conv)
        
        conv = Flatten()(conv)

        conv_list.append(conv)

    X = Concatenate()(conv_list)

    model = Model(inputs=X_input_placeholder, outputs=X)
    return model



def main_model(input_shape, embeddings_dim, max_seq_length, filters, dropout=0.5):
    #model from composition to prediction

    S_input = Input((max_seq_length,embeddings_dim,2)) #input for S decomposition
    T_input = Input((max_seq_length,embeddings_dim,2)) #input for T decomposition

    #compute S_vect and T_vect
    S_vect = composition((max_seq_length,embeddings_dim,2) ,embeddings_dim, max_seq_length, filters)(S_input)
    T_vect = composition((max_seq_length,embeddings_dim,2) , embeddings_dim, max_seq_length, filters)(T_input)

    #concatenate features vectors 
    X_sim = Concatenate()([S_vect, T_vect])

    X_sim = BatchNormalization()(X_sim)
    X_sim = Dropout(dropout)(X_sim)

    nhiddens1 = 256
    #nhiddens2 = 32

    X_sim = Dense(nhiddens1, name = "Dense1", kernel_initializer='random_normal', \
        bias_initializer=tf.constant_initializer(0.01), kernel_regularizer=keras.regularizers.l2(0.01))(X_sim)

    X_sim = BatchNormalization()(X_sim)

    X_sim = Dropout(dropout)(X_sim)

    X_sim = LeakyReLU()(X_sim)

    X_sim = Dense(1, name = "Dense2", kernel_initializer='random_normal',activation='sigmoid', \
        bias_initializer=tf.constant_initializer(0.01), kernel_regularizer=keras.regularizers.l2(0.01))(X_sim)


    #X = Dense(1, name = "Dense3", activation="sigmoid",kernel_initializer='random_normal', bias_initializer='zeros', kernel_regularizer=keras.regularizers.l2(0.01))(X)

    model = Model(inputs=[S_input,T_input], outputs=X_sim, name="main_model")

    return model
