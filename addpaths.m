
here = pwd;
disp 'Add matconvnet'
cd matconvnet/matlab
run vl_setupnn
cd(here);

disp 'Add TT-Toolbox'
cd TT-Toolbox
run setup.m
cd(here);
