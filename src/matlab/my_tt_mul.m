function [c] = my_tt_mul(a,b,varargin)
% a - weights
% b - input data for entire batch
%TT-matrix by full vector product

if ~( isa(a,'tt_matrix') && is_array(b) && nargin == 2 )
    error('error in fn');
end


n  = a.n;     % sizes of row indices
m  = a.m;     % sizes of col indices
tt = a.tt;    % TT-tensor of the vectorized TT-representation of the matrix
cra= tt.core; % cores of the TT-decomposition stored in one 'long' 1D array
d  = tt.d;    % dimension of the array
ps = tt.ps;   % markers for position of the k-the core in array tt.core
r  = tt.r;    % ranks of the decomposition

rb = size(b,2);      % number of batches
ns = [m',rb];        % new shape for input data
c  = reshape(b, ns); % reshape input data to new shape

for k=1:d
    %c is rk*jk...jd*(i1..ik-1) tensor, conv over
    %core is r(i)*n(i)*r(i+1)
    
    % Extract core and reshape
    cr=cra(ps(k):ps(k+1)-1);
    cs = [r(k),n(k),m(k),r(k+1)];
    cr=reshape(cr,cs);
    cr=permute(cr,[2,4,1,3]); 
    cr=reshape(cr,[n(k)*r(k+1),r(k)*m(k)]);
    
    M=numel(c);
    c=reshape(c,[r(k)*m(k),M/(r(k)*m(k))]);
    c=cr*c; 
    c=reshape(c,[n(k),numel(c)/n(k)]);
    c=permute(c,[2,1]);
end

c=c(:); 
c=reshape(c,[rb,numel(c)/rb]);
c=c.';
