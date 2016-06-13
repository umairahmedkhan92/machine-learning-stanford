function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

<<<<<<< HEAD


H = X * theta;
E = (H - y).^2;
T = [0; theta(2:end)]; % Ignoring theta for j = 0;
J = (1/(2*m)) * sum(E) + (lambda/(2*m)) * sum(T.^2); % Computed regularized cost function for liner regression

grad = (1/m) * (X' * (H - y)) + (lambda/m) * T; % Computed regularized gradient 
=======
H = X * theta;
E = (H - y) .^ 2;
T = [0; theta(2:end)];
J = (1/(2*m)) * sum(E) + (lambda/(2*m)) * sum(T.^2);

grad = (1/m) * (X' * (H - y)) + (lambda/m) * (T);
>>>>>>> 5fcbc5e322f43991569030e089774b831086481c
% =========================================================================

grad = grad(:);

end
