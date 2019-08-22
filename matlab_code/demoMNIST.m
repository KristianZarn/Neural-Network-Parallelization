clear ; close all; clc;

% ucna mnozica
images_train = loadMNISTImages('data/train-images-60k');
labels_train = loadMNISTLabels('data/train-labels-60k');

% testna mnozica
% images_test = loadMNISTImages('data/test-images-10k');
% labels_test = loadMNISTLabels('data/test-labels-10k');

% zapisi v datoteko
% testni primeri
% dlmwrite('MNIST/test-images-10k.dat',images_test,'delimiter','\t');
% dlmwrite('MNIST/test-labels-10k.dat',labels_test,'delimiter','\t');

% podmnozica ucnih primerov
subset_size = 10000;
rand_idx = randperm(size(images_train, 1), subset_size);
images_subset = images_train(rand_idx, :);
labels_subset = labels_train(rand_idx);

dlmwrite('MNIST/train-images-10k.dat',images_subset,'delimiter','\t');
dlmwrite('MNIST/train-labels-10k.dat',labels_subset,'delimiter','\t');