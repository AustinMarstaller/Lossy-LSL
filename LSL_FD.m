%% Finite difference scheme to solve the BVP: -u'' + (p(x) + lambda)u = g(x), u'(0)=-1, u'(1)=0
% Suppose g(x) = delta(x)
% Suppose p(x) = exp(-(x-mu)^2/sigma^2)

function [u] = LSL_FD(M,p,x,h,lambda)
    A_diag = (2/h^2)*(eye(M+1,M+1).*p + eye(M+1,M+1).*lambda);
    A = spdiags([-1/h^2 0 -1/h^2],-1:1,M+1,M+1) + A_diag;
    A(1,2) = -2/h^2;
    A(M,M-1) = -2/h^2;
    % Construct right-hand-side Mx1 vector f =(2/h, 0, ..., 0)^T 
    f = zeros(M+1,1);
    f(1)=2/h;
    
    % Solve the linear system Au = f
    u = A\f;
end