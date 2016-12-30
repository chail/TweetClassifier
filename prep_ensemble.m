%% train models
clear
load('./words_train.mat')

% Generative - Naive Bayes
[log_probs_X0, log_probs_X1, P_X0, P_X1] = NB_train(X, Y);

% Discriminative - SVM
SVM_Model = fitcsvm(single(full(X)),single(full(Y)));

% Semi-supervised dimensionality reduction - PCA
[coeff_X, scores_X] = pca(full(X), 'numcomponents', 450);
coeff_X = single(coeff_X);
mean_X = mean(X);

% SVM with PCA
SVMModelPC = fitcsvm(single(full(scores_X)),single(full(Y)));

% logistic Regression with PCA
logistic = mnrfit(single(full(scores_X)),categorical(Y));

% Instance Based -KNN on principle components
mdl_KNN = fitcknn(single(full(scores_X)),Y,'numneighbors', 80,'distance','spearman');

% Ensemble gentleboost
ens_gentleboost = fitensemble(single(full(X)),single(full(Y)),'GentleBoost',200,'Tree');

% Ensemble Bag
ens_bag= fitensemble(single(full(X)),single(full(Y)),'Bag',10,'Tree', 'Type', 'classification');

% Ensemble - logitboost
ens_logitboost = fitensemble(single(full(X)), single(full(Y)),'LogitBoost',150,'Tree','type','classification');

% Ensemble - logitboost on principle components
mdl_X = fitensemble(single(scores_X),Y,'LogitBoost',300,'Tree');

save words_voted_ensemble ens_logitboost log_probs_X1 log_probs_X0 P_X0 P_X1
save words_voted_ensemble2 mdl_X ens_gentleboost ens_bag
save words_voted_ensemble3 logistic SVMModelPC
save words_voted_ensemble4 SVM_Model mdl_KNN

%% generate and save NN model - make sure DL_toolbox is added
addpath(genpath('./DL_toolbox'))
% gen_NN
genNN_train



