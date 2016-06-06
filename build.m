
here = pwd;

cd matconvnet/matlab
run vl_setupnn
run vl_compilenn('enableGpu', true)

run vl_test_nnlayers('gpu', true)

cd(here);
