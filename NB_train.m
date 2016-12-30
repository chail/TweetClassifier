function [log_probs_X0, log_probs_X1, P_X0, P_X1] = NB_train(X, Y)

X0 = X((Y==0), :);
X1 = X((Y==1), :);

[n0, p] = size(X0);
alpha = 1;

sum_all_X0 = sum(sum(X0));
log_probs_X0 = log((sum(X0) + alpha)/(sum_all_X0 + alpha * p));

sum_all_X1 = sum(sum(X1));
log_probs_X1 = log((sum(X1) + alpha)/(sum_all_X1 + alpha * p));

P_X0 = 1 - sum(Y) / length(Y);
P_X1 = sum(Y) / length(Y);
