clear all
close all
tic
%% Synthetic data construction
M=9; % M+1 total grid points
h=1/M; % Grid point spacing. Implement h^hats
x=(0:h:1)'; % Lattice in column vector

mu = 0.35;
sigma = 0.27;
p = exp(-(x-mu).^2 / sigma^2); % p = 0 for reference problem
p_reference = zeros(M+1,1);
lambda = [2,4,6,8,16,32,48];

%% Store solution vectors per lambda in a matrix
u_lambda           = zeros(M+1,numel(lambda)); % [u(x; lambda_1) | ... | u(x; lambda_2024)]
u_lambda_reference = zeros(M+1,numel(lambda)); 

for j = 1:numel(lambda)
    % Each column is a solution for a particular value of lambda
    [u_lambda(:,j),A] = LSL_FD(M,p,x,h,lambda(j)); % (u_j(0), u_j(1), ..., u_j(M))^T
    [u_lambda_reference(:,j),C] = LSL_FD(M,p_reference,x,h,lambda(j));
end

D = ones(1,M+1)*h;
D(1) = h/2;
D(end) = h/2;
D = diag(D);

%% Synthetic data F(lambda) = u(0,lambda), dF/dlambda = u^T u
F = u_lambda(1,:); % u(0, lambda_i)
F_reference = u_lambda_reference(1,:);

dF_dlambda = zeros(1,numel(lambda));
for i = 1:numel(lambda)
    % dF/dlambda = -u^T D u
    dF_dlambda(i) = -u_lambda(:,i)' * D * u_lambda(:,i);
end


%% Mass & Stiffness matrices
% M & S are symmetric w.r.t <-,->_D
Mass      = -diag(dF_dlambda);
Stiffness = (dF_dlambda)*diag(lambda) + dF_dlambda; % lambda dF/dlambda + dF/dlambda

for i = 1:numel(lambda)
    for j = 1:numel(lambda)
        if j ~= i
            Mass(i,j) = (F(i) - F(j))/(lambda(j) - lambda(i));
            Stiffness(i,j) = (F(j)*lambda(j) - F(i)*lambda(i))/(lambda(j) - lambda(i));
        end
    end
end


%% Benchmark test: D = diag(h_0^, ..., h_(M)^): 
% M(i,j) ?= u_i^T D u_j,
% S(i,j) ?= u_i^T D A u_j

benchmark_Mass = Mass*0;
benchmark_Stiffness = Stiffness*0;

for i = 1:numel(lambda)
    for j = 1:numel(lambda)
        benchmark_Mass(i,j) = u_lambda(:,i)' * D * u_lambda(:,j);
        benchmark_Stiffness(i,j) = u_lambda(:,i)' * D * A * u_lambda(:,j); 
    end
end

n=10;
b = rand(n,1); % vector of length n = 10
[Q,alpha,beta] = Lanczos(A,b,n);
%full(Q' * B * Q) % Verify T = Q' A Q
% Also check Q^T Q = Identity
%full(Q' * Q)

%% Lanczos algorithm
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
toc