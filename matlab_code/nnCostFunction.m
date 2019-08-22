function [J, grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

m = size(X, 1);

% pretvori oznake y v vektorje
vec_labels = eye(num_labels);
y_mat = vec_labels(:,y+1);

% feedforward, izracunaj h_theta(x)
a1 = [ones(1,m); X'];
z2 = Theta1 * a1;
a2 = [ones(1,m); sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

% izracunaj cost function
J = (1/m) * sum(sum((-y_mat .* log(a3) - (1-y_mat) .* log(1-a3))));

% cost funkciji dodaj regularizacijo
J = J + (lambda/(2*m)) * ...
    (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% backpropagation
d3 = a3 - y_mat;
d2 = (Theta2(:,2:end)' * d3) .* sigmoidGradient(z2);

Theta1_grad = d2 * a1';
Theta2_grad = d3 * a2';

% normalization and regularization
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ...
    (lambda/m) .* Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ...
    (lambda/m) .* Theta2(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
