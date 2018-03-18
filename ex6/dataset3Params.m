function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
Cvec=[0.01;0.03;0.1;0.3;1;3;10;30]
%Cvec=[0.01;0.03];
pred_errors=[];
for lxx = 1:8
  for lyy = 1:8
  	tmp_c=Cvec(lxx);
    tmp_sig=Cvec(lyy);
    model= svmTrain(X, y, tmp_c, @(x1, x2) gaussianKernel(x1, x2, tmp_sig));
    pred = svmPredict(model,Xval); 
    pred_error = mean(double(pred ~= yval));
    pred_errors =[pred_error tmp_c tmp_sig; pred_errors];
  endfor
endfor
[mm,idx]=min(pred_errors(:,1));
C = pred_errors(idx,2)
sigma = pred_errors(idx,3)

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
