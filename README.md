# TweetClassifier

This project classifies tweets as happy or sad, based on the words used in the tweet. It expects training input as a n x p matrix where n is the number of tweets and p is the vocabulary size, and each entry is the number of times a word appeared in the tweet.

## Model Overview
We used a voted average approach of a combination of methods - generative, discriminative, dimensionality reduction, and ensemble methods.
- PCA
- Naive Bayes
- SVM
- K-nearest neighbors
- Logistic Regression
- Weak Learner Ensembles
- Neural Net

## Training
Code for training the models is in `prep_ensemble.m`. It assumes that the training data is in the current directory saved as `words_train.mat` with an n x p matrix of training features and an n x 1 matrix of classifications (happy or sad).

For our report, we determined accuracy by holding out 10% of the training data and using it to test our voted model. This code is in `proj_final_accuracy_testing.m`.

We used `DL_toolbox` functions for training the neural net.

## Testing
To test after models have been trained:
- run `predict_labels.m` providing the function inputs. The only input that matters is a matrix of word counts of the tweets to classify, provided in the same format as the training dataset.
- predict_labels will compute the label for each test instance for all of
the models, and take the voted majority
- the voted majority is returned as Y_hat

## Result
Training on 4500 tweets, we were able to achieve an accuracy of 0.8076 on the testing dataset and 0.80178 in the validation set.

## Credits
Built by Andrew Murphy, Jason Kim, and Lucy Chai: Machine Learning final project.

