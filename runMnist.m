close all;
clear;
clc;
format compact;


here = pwd;
disp 'Run mnist experiment...'
addpath(fullfile(pwd, 'matconvnet/examples'));
cd experiments/mnist

[net_tt, info_tt] = cnn_mnist_tt('expDir', 'data/mnist-tt');

cd(here);

