function [yhat, scores] = NB_test(X, log_probs_X0, log_probs_X1, P_X0, P_X1)

[n, p] = size(X);

yhat = zeros(n, 1);
scores = zeros(n, 2);

for ii = 1:n
    features = X(ii, :);
    log_X0_feature = P_X0;
    log_X1_feature = P_X1;
    for jj = find(features ~= 0)
        log_X0_feature = log_X0_feature + features(jj) * log_probs_X0(jj);
        log_X1_feature = log_X1_feature + features(jj) * log_probs_X1(jj);
    end
    
    scores(ii, 1) = exp(log_X0_feature) / (exp(log_X0_feature)+exp(log_X1_feature));
    scores(ii, 2) = exp(log_X1_feature) / (exp(log_X0_feature)+exp(log_X1_feature));
    
    if log_X0_feature > log_X1_feature
        yhat(ii) = 0;
    else
        yhat(ii) = 1;
    end
end
