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

% 5000x401 * 401x25 = 5000x25 (25 activation units per training example)
z2 = addColumnOfOnes(X) * Theta1';
% Add bias units to hidden layer. (5000x25 -> 5000x26)
a_hidden = addColumnOfOnes(sigmoid(z2));

% 5000x26 * 26x10 = 5000x10 (10 hypotheses/predictions per training example)
h = sigmoid(a_hidden * Theta2');

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

% Backpropagation Algorithm
delta1 = zeros(size(Theta1)); % 25x401
delta2 = zeros(size(Theta2)); % 10x26

for t = 1:m
	x_t = [1 X(t,:)]; % 1x401
	y_t = y_recoded(t,:); % 1x10
	h_t = h(t,:); % 1x10

	output_layer_error = h_t - y_t; % 1x10
	sg = sigmoidGradient(z2(t,:)); % 1x25
	hidden_layer_error = ignoreColumn1(output_layer_error * Theta2) .* sg; % 1x25

	delta1 += hidden_layer_error' * x_t; % 25x1 * 1x401 = 25x401
	delta2 += output_layer_error' * a_hidden(t,:); % 10x1 * 1x26 = 10x26
end

Theta1_grad = delta1 ./ m;
Theta2_grad = delta2 ./ m;

% Regularize the gradients. Both vectors' column 1 is unaffected.
Theta1_grad(:,2:end) += (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) += (lambda/m) * Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:)];

end