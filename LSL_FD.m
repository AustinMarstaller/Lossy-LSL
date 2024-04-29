%% Finite difference scheme to solve the BVP: -u'' + (p(x) + lambda)u = 0, u'(0)=-1, u'(1)=0
% Suppose p(x) = exp(-x^2)


function [u] = LSL_FD(M,x,h,lambda)
    % Parameters
    h=1/M; % Grid point spacing
    x=(0:h:1)'; % Lattice in column vector
    
    lambda = 1:5; % Values of lambda to use
    
    % Store solution vectors per lambda in a matrix
    u_lambda = zeros(M+1,numel(lambda)); 
    
    for j = 1:numel(lambda)
        % Each column is a solution for a particular value of lambda
        u_lambda(:,j) = main(M,x,h,lambda(j));
    end




    % Construct the M-by-M matrix A
    A_diag = (2/h^2)*(eye(M+1,M+1).*exp(-x.^2) + eye(M+1,M+1).*lambda);
    A = spdiags([-1/h^2 0 -1/h^2],-1:1,M+1,M+1) + A_diag;
    A(1,2) = -2/h^2;
    A(M,M-1) = -2/h^2;
    % Construct right-hand-side Mx1 vector f =(-h, 0, ..., 0)^T 
    f = zeros(M+1,1);
    f(1)=h;
    
    % Solve the linear system Au = f
    u = A\f;
end