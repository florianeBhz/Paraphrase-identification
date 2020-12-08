# Deep learning - PARAPHRASE IDENTIFICATION 
## Introduction
We present our deep learning project where the goal was to read and summarize the article : "Sentence Similarity Learning by Lexical Decomposition and Composition" from https://arxiv.org/pdf/1602.07019.pdf at first. Secondly, we reimplemented the proposed model in Keras which aims at identifying if two sentences are paraphrases or not. The model uses similarity and dissimilarity components to predict the class through a two channels CNN. Our implementation is adapted from https://github.com/CubasMike/paraphrase_identification. We also test our model on Quora question pairs dataset to analyze its ability to generalize. 

## Files description
* main_model: main file
* data_generation : preprocess data and build embedding matrix
* lexdecomp_model : contains the model implementation and associated functions
* configs : global, dataset and preprocessing parameters
* Quora_tests : tests on Quora dataset 

## Prerequisites 
To be able to train the model you need to have python3 and the following python packages installed : numpy, pickle, keras, gensim , re, codecs, nltk.

You also need to :
* download the word embedding file from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
* download the Microsoft Research Paraphrase Corpus dataset from https://www.microsoft.com/en-us/download/details.aspx?id=52398&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F607d14d9-20cd-47e3-85bc-a2f65cd28042%2F 
* donwload the Quora dataset from https://www.kaggle.com/c/quora-question-pairs/data 

## Training 

In order to reproduce the training:
* generate dataset according to configurations in configs file
* run main_model file
http://www.univ-rouen.fr/ https://www.insa-rouen.fr/ 

## Acknowledgments
http://www.univ-rouen.fr/

https://www.insa-rouen.fr/

https://www.litislab.fr/

## Special acknowledgement to our teachers : 
* Romain Hérault http://asi.insa-rouen.fr/enseignants/~rherault/pelican/index.html 
* Clément Chatelain http://pagesperso.litislab.fr/cchatelain/ 
* Rachel Blin http://pagesperso.litislab.fr/rblin/contacts/ 

## Authors 
Floriane BEHANZIN and  Antoine LUBOYA MBINGILA
