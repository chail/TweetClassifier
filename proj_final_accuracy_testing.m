%% Initialize
clear; clc;

%% set up path
addpath(genpath('./DL_toolbox'))

%% Load Data
load ./words_train

%% Holdout on train data
nTest = 450;
nTrain = size(X,1) - nTest;
testInd = randperm(size(X,1), nTest);
trainInd = setdiff(1:size(X,1), testInd);

XMat = X;
xval_test_Y = full(Y(testInd));
xval_train_Y = full(Y(trainInd));
xval_test_X = XMat(testInd,:);
xval_train_X = XMat(trainInd,:);


%% Train
% Generative -- Naive Bayes Model
[log_probs_X0, log_probs_X1, P_X0, P_X1] = NB_train(xval_train_X, xval_train_Y);

% Discriminative - SVM
SVM_Model = fitcsvm(single(full(xval_train_X)),single(full(xval_train_Y)));

% Semisupervised Dimensionality Reduction PCA 
[coeff_X, scores_X] = pca(full(xval_train_X), 'numcomponents', 450);
coeff_X = single(coeff_X);
mean_X = mean(xval_train_X);

% Discriminative - SVM on principle components
SVM_Model_PC = fitcsvm(single(full(scores_X)),single(full(xval_train_Y)));

% Discriminative - Logistic on principle components
logistic = mnrfit(single(full(scores_X)),categorical(xval_train_Y));

% Instance Based -KNN on principle components
mdl_KNN = fitcknn(single(full(scores_X)),xval_train_Y,'numneighbors', 80,'distance','spearman');

% Ensemble - logitboost on principal components
mdl_X = fitensemble(single(scores_X),xval_train_Y,'LogitBoost',300,'Tree');

% Ensemble - logitboost
ens_logitboost = fitensemble(single(full(xval_train_X)),single(full(xval_train_Y)),'LogitBoost',200,'Tree');

% Ensemble - gentleboost
ens_gentleboost = fitensemble(single(full(xval_train_X)), single(full(xval_train_Y)),'GentleBoost',200,'Tree');

% Ensemble - bag
ens_bag= fitensemble(single(full(xval_train_X)),single(full(xval_train_Y)),'Bag',10,'Tree', 'Type', 'classification');

% Neural Net
YTest = full(xval_test_Y);
YTrain = full(xval_train_Y);
XTest = xval_test_X;
XTrain = xval_train_X;
nn = nnsetup([size(XTrain,2) 30 2]); 
nn.weightPenaltyL2 = 1e-5;  %  L2 weight decay
opts.numepochs =  25;       %  Number of full sweeps through data
opts.batchsize = 150;       %  Take a mean gradient step over this many samples
[nn loss] = nntrain(nn, XTrain, [YTrain==0 YTrain==1], opts);
%% test on held out data

% test NB
label_NB = NB_test(xval_test_X, log_probs_X0, log_probs_X1, P_X0, P_X1);
disp(sum(label_NB == xval_test_Y) / length(xval_test_Y))

% test SVM
[label_SVM] = predict(SVM_Model, full(xval_test_X));
disp(sum(label_SVM == xval_test_Y) / length(xval_test_Y))

% compute PCA scores
test_scores_X= bsxfun(@minus, xval_test_X, mean_X) * double(coeff_X);

% test SVM on PC
[label_SVM_PC] = predict(SVM_Model_PC, test_scores_X);
disp(sum(label_SVM_PC == xval_test_Y) / length(xval_test_Y))

% test logistic on PC
scores_logistic = mnrval(logistic, test_scores_X);
[~,label_logistic]=max(scores_logistic');
label_logistic = label_logistic-1;
label_logistic = label_logistic';
disp(sum(label_logistic == xval_test_Y) / length(xval_test_Y))

% test KNN on PC
label_KNN = predict(mdl_KNN, test_scores_X);
disp(sum(label_KNN == xval_test_Y) / length(xval_test_Y))

% test ensemble - logitboost on principal components
label_PC = predict(mdl_X, test_scores_X);
disp(sum(label_PC == xval_test_Y) / length(xval_test_Y))

% test ensemble - logitboost
label_logitboost = predict(ens_logitboost,full(xval_test_X));
disp(sum(label_logitboost == xval_test_Y) / length(xval_test_Y))

% test ensemble - gentleboost
label_gentleboost = predict(ens_gentleboost,full(xval_test_X));
disp(sum(label_gentleboost == xval_test_Y) / length(xval_test_Y))

% test ensemble - bag
label_bag= predict(ens_bag,full(xval_test_X));
disp(sum(label_bag == xval_test_Y) / length(xval_test_Y))

% test neural net
[label_nn] = nnpredict(nn, full(XTest)) - 1;
disp(sum(label_nn == xval_test_Y) / length(xval_test_Y))

% final voted prediction
labels = [label_logitboost, label_NB, label_gentleboost, label_bag, full(label_PC), label_SVM, label_SVM_PC, label_logistic, full(label_nn), label_KNN];
Y_hat = mode(labels, 2);
disp(sum(Y_hat == xval_test_Y) / length(xval_test_Y))

%save labels label_logitboost label_NB label_gentleboost label_bag label_PC label_SVM label_SVM_PC  label_logistic label_nn label_KNN xval_test_Y


