close all;
clear;
clc;
format compact;


here = pwd;
disp 'Run vl_test_ttlayers...'

cd src/matlab

% CPU mode
run vl_test_ttlayers

% GPU mode
run vl_test_ttlayers(1)
cd(here);



