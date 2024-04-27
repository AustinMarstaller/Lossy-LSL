%% Lanzcos Iteration: following Trefethen
% alpha = main diagonal, beta = super/sub diagonal
% A = Q T_n Q*

n=10;
A = generateSPDmatrix(n);
b = rand(n,1); % vector of length n = 10
[Q,alpha,beta] = Lanczos(A,b,n);
full(Q' * A * Q) % Verify T = Q' A Q

function [Q, alpha, beta] = Lanczos(A,b,iter)
    %% Some initialization
    [row, col] = size(b);
    Q          = zeros(row, iter); % Q = [q_1 | q_2 | ... | q_n]
    Q(:,1)     = b/norm(b);  % q_1 = b/||b||
    alpha = zeros(iter,1);   % Main diagonal
    beta  = zeros(iter-1,1); % beta = (beta_1,..., beta_n). 
    
    %% Perform the Lanczos iteration: three-term recurrence relation
    for i = 1:iter
        %% Construct column vector q_(i+1) as Q = [... | q_(i+1) | ...]
        v        = A*Q(:,i);      % A * q_i 
        alpha(i) = (Q(:,i))' * v; % q_i^T M * A * q_i
    
        v = v - alpha(i)*Q(:,i);  
        if i > 1
            v = v - beta(i-1)*Q(:,i-1); % q_i = (A q_i - beta_(i-1) q_(i-1) - alpha_i q_i)
        end
        
        beta(i)  = norm(v); 
        v = v/beta(i); % Normalize 
        
        if i < iter
            % i = iter => Q is n x n+1
            Q(:,i+1) = v;
        end
    end
end

%% SPD generation code found on Math.SE
function A = generateSPDmatrix(n)
% Generate a dense n x n symmetric, positive definite matrix

A = rand(n,n); % generate a random n x n matrix

% construct a symmetric matrix using either
A = 0.5*(A+A'); 
%A = A*A';
% The first is significantly faster: O(n^2) compared to O(n^3)

% since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
%   is symmetric positive definite, which can be ensured by adding nI
A = A + n*eye(n);
end