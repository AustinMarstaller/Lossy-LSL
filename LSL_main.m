clear all
close all
tic
%% Synthetic data construction
M=9; % M+1 total grid points
h=1/M; % Grid point spacing. 
x=(0:h:1)'; % Lattice in column vector

mu = 0.35;
sigma = 0.27;
p = exp(-(x-mu).^2 / sigma^2); % p = 0 for reference problem
p_reference = zeros(M+1,1);
lambda = [2,4,6,8,16,32,48];

% Operator L = -Del + p
L_diag = 2/h^2 * eye(M+1,M+1) + diag(p);
L = spdiags([-1/h^2 0 -1/h^2],-1:1,M+1,M+1) + L_diag;
L(1,2) = -2/h^2;
L(M+1,M) = -2/h^2;

L_ref_diag = 2/h^2 * eye(M+1,M+1) + diag(p_reference);
L_ref = spdiags([-1/h^2 0 -1/h^2],-1:1,M+1,M+1) + L_diag;
L_ref(1,2) = -2/h^2;
L_ref(M+1,M) = -2/h^2;

%% Store solution vectors per lambda in a matrix
u_lambda           = zeros(M+1,numel(lambda)); % [u(x; lambda_1) | ... | u(x; lambda_2024)]
u_lambda_reference = zeros(M+1,numel(lambda)); 

for j = 1:numel(lambda)
    % Each column is a solution for a particular value of lambda
    [u_lambda(:,j)] = LSL_FD(M,L,h,lambda(j)); % (u_j(0), u_j(1), ..., u_j(M))^T
    [u_lambda_reference(:,j)] = LSL_FD(M,L,h,lambda(j));
end

D = ones(1,M+1)*h;
D(1) = h/2;
D(end) = h/2;
D = diag(D);

%% Synthetic data F(lambda) = u(0,lambda), dF/dlambda = u^T u
F = u_lambda(1,:); % u(0, lambda_i)
F_reference = u_lambda_reference(1,:);

dF_dlambda = zeros(1,numel(lambda));
dF_dlambda_reference = zeros(1,numel(lambda));

for i = 1:numel(lambda)
    % dF/dlambda = -u^T D u
    dF_dlambda(i) = -u_lambda(:,i)' * D * u_lambda(:,i);
    dF_dlambda_reference(i) = -u_lambda_reference(:,i)' * D * u_lambda_reference(:,i);
end

%% Mass & Stiffness matrices
% M & S are symmetric w.r.t <-,->_D
% M_ii = -u^T D u
% S_ii = lambda dF/dlambda + F
Mass      = -diag(dF_dlambda); 
Stiffness = diag((dF_dlambda)*diag(lambda) + F); % lambda dF/dlambda + F

Mass_reference =  -diag(dF_dlambda_reference);
Stiffness_reference = diag((dF_dlambda_reference)*diag(lambda) + F_reference);

% Mass(i,j) ?= u_i^T D u_j =: benchmark_Mass,
% Stiffness(i,j) ?= u_i^T D A u_j =: benchmark_Stiffness
benchmark_Mass = Mass*0;
benchmark_Stiffness = Stiffness*0;

benchmark_Mass_reference = Mass_reference*0;
benchmark_Stiffness_reference = Stiffness_reference*0;

for i = 1:numel(lambda)
    for j = 1:numel(lambda)
        if j ~= i
            Mass(i,j) = (F(i) - F(j))/(lambda(j) - lambda(i));
            Stiffness(i,j) = (F(j)*lambda(j) - F(i)*lambda(i))/(lambda(j) - lambda(i));

            Mass_reference(i,j) = (F_reference(i) - F_reference(j))/(lambda(j) - lambda(i));
            Stiffness_reference(i,j) = (F_reference(j)*lambda(j) - F_reference(i)*lambda(i))/(lambda(j) - lambda(i));
        end
            benchmark_Mass(i,j) = u_lambda(:,i)' * D * u_lambda(:,j);
            benchmark_Stiffness(i,j) = u_lambda(:,i)' * D * L * u_lambda(:,j); 

            benchmark_Mass_reference(i,j) = u_lambda_reference(:,i)' * D * u_lambda_reference(:,j);
            benchmark_Stiffness_reference(i,j) = u_lambda_reference(:,i)' * D * L_ref * u_lambda_reference(:,j); 
    end
end

%% Finite difference scheme to solve the BVP: -u'' + (p(x) + lambda)u = g(x), u'(0)=-1, u'(1)=0
function [u] = LSL_FD(M,L,h,lambda)
    % Coefficient matrix A is M+1 x M+1
    A = L + lambda * eye(M+1,M+1);
    % Construct right-hand-side Mx1 vector f =(2/h, 0, ..., 0)^T 
    f = zeros(M+1,1);
    f(1)=2/h;
    
    % Solve the linear system Au = f
    u = A\f;
end

n=10;
b = rand(n,1); % vector of length n = 10
%[Q,alpha,beta] = Lanczos(A,b,n); % Perform Lanczos for change of basis: QV

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