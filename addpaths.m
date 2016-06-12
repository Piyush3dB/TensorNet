
here = pwd;
disp 'Add matconvnet'
cd matconvnet/matlab
run vl_setupnn

cd(here);
addpath(fullfile(pwd, 'matconvnet/examples'));
addpath(fullfile(pwd, 'src/matlab'));

cd(here);

disp 'Add TT-Toolbox'
cd TT-Toolbox
run setup.m
cd(here);
