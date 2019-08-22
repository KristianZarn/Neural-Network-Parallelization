%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
% Load Training Data
load('data/ex4data1.mat');
m = size(X, 1);

%% ================= Predict =================
Theta1 = dlmread('data/Theta1.dat');
Theta2 = dlmread('data/Theta2.dat');

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);