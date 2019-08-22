%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;
hidden_layer_size = 25;
num_labels = 10;

% Load Training Data
fprintf('\nLoading training data ...\n');
X = dlmread('../MNIST/train-images-10k.dat');
y = dlmread('../MNIST/train-labels-10k.dat');
m = size(X, 1);

%% ================ Initializing Pameters ================
fprintf('\nInitializing Neural Network Parameters ...\n');

initial_nn_params = dlmread('data/param.dat');
param_len = length(initial_nn_params);

%% =================== Training NN ===================
fprintf('\nTraining Neural Network... \n');

lambda = 0.1;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

iterations = 100;
alpha = 1.0;

tic;
[nn_params, cost] = gradientDescent(costFunction, initial_nn_params, iterations, alpha);
toc;

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
