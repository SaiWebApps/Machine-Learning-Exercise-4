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

m = size(X, 1); % Number of training examples
n = size(X, 2); % Number of features
hidden_layer_total_size = hidden_layer_size * (input_layer_size + 1);
addColumnOfOnes = @(matrix) [ones(size(matrix, 1), 1), matrix];
ignoreColumn1 = @(matrix) matrix(:, 2:size(matrix, 2));

% Reshape nn_params back into the parameters Theta1 and Theta2, 
% the weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_total_size), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + hidden_layer_total_size):end), ...
                 num_labels, (hidden_layer_size + 1));

% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% 5000x401 * 401x25 = 5000x25 (25 activation units per training example)
a_hidden = sigmoid(addColumnOfOnes(X) * Theta1');

% 5000x26 * 26x10 = 5000x10 (10 hypotheses/predictions per training example)
h = sigmoid(addColumnOfOnes(a_hidden) * Theta2');

% Recode y from 5000x1 matrix to 5000x10 matrix (num_labels = 10).
% Each y(i) is a digit from 0 to 9, representing the output class that
% the input x(i) belongs to.
y_recoded = zeros(m, num_labels);
for r = 1:m
	% Set the index y(r) to 1 to indicate the output class for row r.
	y_recoded(r, y(r)) = 1; 
end

% Calculate J using h and y_recoded, both of which are 5000x10 matrices.
J_part1 = sum(y_recoded .* log(h)); % 1x5000 - column sums
J_part2 = sum((1-y_recoded) .* log(1-h)); % 1x5000 - column sums
J_unreg = (-1/m) * sum(J_part1 + J_part2);

% Incorporate regularization into J.
Theta1_squared = sum(sum(ignoreColumn1(Theta1) .^ 2));
Theta2_squared = sum(sum(ignoreColumn1(Theta2) .^ 2));
J = J_unreg + (lambda/(2*m)) * (Theta1_squared + Theta2_squared);

% ====================== YOUR CODE HERE ======================
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
% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end