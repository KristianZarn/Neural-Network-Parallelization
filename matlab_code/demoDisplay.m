clear ; close all; clc;

% testna mnozica
images_test = loadMNISTImages('data/test-images-10k');
labels_test = loadMNISTLabels('data/test-labels-10k');

% napovedi nevronske mreze
result_test = dlmread('data/result.dat');

% prikazi nekaj nakljucnih slik testne mnozice
subset_size = 7^2;
rand_idx = randperm(size(images_test, 1), subset_size);
images_subset = images_test(rand_idx, :);
labels_subset = labels_test(rand_idx);
result_subset = result_test(rand_idx);

example_width = 28;
displayData(images_subset, example_width);
font_size = 18;
displayLabels(labels_subset, result_subset, example_width, font_size);
