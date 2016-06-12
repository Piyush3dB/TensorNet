function [c] = my_tt_mul(a,b,varargin)
if ~( isa(a,'tt_matrix') && is_array(b) && nargin == 2 )
    error('error in fn');
end

%TT-matrix by full vector product
n=a.n; 
m=a.m; 
tt=a.tt; 
cra=tt.core; 
d=tt.d; 
ps=tt.ps; 
r=tt.r;

rb=size(b,2);
c=reshape(b,[m',rb]);

for k=1:d
    %c is rk*jk...jd*(i1..ik-1) tensor, conv over
    %core is r(i)*n(i)*r(i+1)
    cr=cra(ps(k):ps(k+1)-1);
    cr=reshape(cr,[r(k),n(k),m(k),r(k+1)]);
    cr=permute(cr,[2,4,1,3]); cr=reshape(cr,[n(k)*r(k+1),r(k)*m(k)]);
    M=numel(c);
    c=reshape(c,[r(k)*m(k),M/(r(k)*m(k))]);
    c=cr*c; c=reshape(c,[n(k),numel(c)/n(k)]);
    c=permute(c,[2,1]);
end
c=c(:); c=reshape(c,[rb,numel(c)/rb]);
c=c.';
