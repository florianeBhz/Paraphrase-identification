
configs = {
    "global": { #global settings
        "DATASET" : "msrpc",
        "MAX_SEQ_LENGTH" : 50,
        "VEC_DIM" : 300,
        "VOCABULARY_SIZE" : 200000,
        "MAX_SAMPLE_SIZE" : 3500,
        "EPOCHS" : 20,
        "BATCH_SIZE": 64,
        "WINDOW": 3, #window for local-l
        "METHOD" : "orthogonal", # method = linear or orthogonal
        "FILTERS" : [(1,500), (2,500), (3,500)] #500 filters of each size (unigram to trigram)
            },

    "preprocess": { #preprocessing parameters
        "lower" : True,
        "stemming":True,
        "removestopwords": False,
        "contract_expand": True,
        "remove_punctuation": True
    },

    "msrpc":{
        "TRAIN_FILE" : 'MSRPC/msr_paraphrase_train.txt',
        "TEST_FILE" : 'MSRPC/msr_paraphrase_test.txt',
        "ROOT_DIR" : 'MSRPC/'
    },
    "quora":{
        "TRAIN_FILE" : 'Quora/train.csv',
        "TEST_FILE" : 'Quora/test.csv',
        "ROOT_DIR" : "Quora/"

    }
}