function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % Number of rows in X
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % Zero matrix of dimensions (hidden_layer_size * input_layer_size+1)
Theta2_grad = zeros(size(Theta2)); % Zero matrix of dimensions (num_labels * hidden_layer_size+1) 

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% For Part 1 - Cost Function Computation
% -------------------------------------------------------------

X = [ones(m,1) X]; % Adding the bias unit with all unit values equal to 1
A1 = X;
A2 = sigmoid(A1 * Theta1'); % Computing the second layer activation using a2 = g(z2), where z2 = a1 * Theta1
A2 = [ones(size(A2, 1),1) A2]; % Adding the bias uint
H = sigmoid(A2 * Theta2'); % Computing the output layer h(x) = g(z3);

recoded_y = eye(num_labels)(y, :); % To get a vector of 0s and 1s showing y's value

T = -recoded_y .* log(H) - (1 - recoded_y) .* log(1 - H);
J = 1/m * (sum(sum(T, 2))); % Computed the cost using the cost function for Neural Network 

REG = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2));
J = J + lambda/(2*m) * (REG); % Computed using regularized cost function for Neural Netwrok 

% For Part 2 - Backpropagation Algorithm to Compute the Gradients
% ------------------------------------------------------------------
G1 = zeros(size(Theta1));
G2 = zeros(size(Theta2));

for t = 1:m,
	A1 = X(t, :)';
	Z2 = Theta1 * A1;
	A2 = sigmoid(Z2);
	A2 = [1;A2];
	Z3 = Theta2 * A2;
	A3 = sigmoid(Z3);

	ER3 = A3 - recoded_y(t, :)';  % using δ(3) = (a(3) − yk)
	ER2 = (Theta2' * ER3)(2:end, 1) .* sigmoidGradient(Z2); % using δ(2) =  Θ(2)' δ(3). ∗ g′(z(2))

	G1 = G1 + ER2 * A1'; % using ∆(l) = ∆(l) + δ(l+1)(a(l))'
	G2 = G2 + ER3 * A2';
end

R_Theta1 = [zeros(hidden_layer_size, 1) Theta1(:, 2:end)]; % modified Theta1 for regularization 
R_Theta2 = [zeros(num_labels, 1) Theta2(:, 2:end)]; % modified Theta2 for regularization 

Theta1_grad = 1/m * G1 + lambda/m * (R_Theta1); % Computed regularized gradients
Theta2_grad = 1/m * G2 + lambda/m * (R_Theta2); 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
