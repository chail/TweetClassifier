function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 0 for sad)

load('./words_voted_ensemble.mat')
load('./words_voted_ensemble2.mat')
load('./words_voted_ensemble3.mat')
load('./words_voted_ensemble4.mat')
load('./ModelNNMult.mat');
load('./words_train.mat', 'X');

% labels test data based on word counts
[label_logitboost] = predict(ens_logitboost,full(word_counts));
[label_NB] = NB_test(word_counts, log_probs_X0, log_probs_X1, P_X0, P_X1);
[label_gentleboost] = predict(ens_gentleboost,full(word_counts));
[label_bag] = predict(ens_bag,full(word_counts));
[label_nn] = nnpredict(nn, full(word_counts)) - 1;
[label_SVM] = predict(SVM_Model, full(word_counts));

% PCA on training data
[coeff_X, scores_X] = pca(full(X), 'numcomponents', 450);
coeff_X = single(coeff_X);
mean_X = mean(X);

% label test data based on PCA scores
test_scores_X= bsxfun(@minus, word_counts, mean_X) * double(coeff_X);
[labels_PC] = predict(mdl_X, test_scores_X);
[label_SVM_PC] = predict(SVMModelPC, test_scores_X);
[label_KNN] = predict(mdl_KNN, single(test_scores_X));
scores_logistic = mnrval(logistic, test_scores_X);
[~,label_logistic]=max(scores_logistic');
label_logistic = label_logistic-1;
label_logistic = label_logistic';

labels = [label_logitboost, label_NB, label_gentleboost, label_bag, ...
    full(labels_PC), label_SVM_PC, label_logistic, full(label_nn), ...
    full(label_KNN), label_SVM];

Y_hat = mode(labels, 2);
end

