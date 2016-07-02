
close all;
clear;
clc;
format compact

load('./experiments/mnist/mnist.mat')

% Do the TT mul
m2 = my_tt_mul(W, A);

% Check Error
s = sum(sum(m2-m));
if (s)
    error('Ref and model not same');
end

disp('== DONE ==')