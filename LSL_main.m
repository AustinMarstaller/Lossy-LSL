clear all
close all
tic
%% Synthetic data construction
M=255; % M+1 total grid points
h=1/M; % Grid point spacing. 
x=(0:h:1)'; % Lattice in column vector

mu          = 0.3;
sigma       = 0.125;
p           = exp(-(x-mu).^2 / sigma^2); % p = 0 for reference problem
p_reference = zeros(M+1,1);
lambda      = [2,4,6,8,16,32,48];
%lambda = [2,6,16,48];
% Operator L = -Del + p

L_diag   = 2/h^2 * eye(M+1,M+1) + diag(p);
L        = spdiags([-1/h^2 0 -1/h^2],-1:1,M+1,M+1) + L_diag;
L(1,2)   = -2/h^2;
L(M+1,M) = -2/h^2;

L_ref_diag   = 2/h^2 * eye(M+1,M+1) + diag(p_reference);
L_ref        = spdiags([-1/h^2 0 -1/h^2],-1:1,M+1,M+1) + L_diag;
L_ref(1,2)   = -2/h^2;
L_ref(M+1,M) = -2/h^2;

%% Store solution vectors per lambda in a matrix
u_lambda           = zeros(M+1,numel(lambda)); % [u(x; lambda_1) | ... | u(x; lambda_2024)]
u_lambda_reference = zeros(M+1,numel(lambda)); 

for j = 1:numel(lambda)
    % Each column is a solution for a particular value of lambda
    [u_lambda(:,j)]           = LSL_FD(M,L,h,lambda(j)); % (u_j(0), u_j(1), ..., u_j(M))^T
    [u_lambda_reference(:,j)] = LSL_FD(M,L_ref,h,lambda(j));
end

D      = ones(1,M+1)*h;
D(1)   = h/2;
D(end) = h/2;
D      = diag(D);

%% Synthetic data F(lambda) = u(0,lambda), dF/dlambda = u^T u
F           = u_lambda(1,:); % u(0, lambda_i)
F_reference = u_lambda_reference(1,:);

dF_dlambda           = zeros(1,numel(lambda));
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

Mass_reference      =  -diag(dF_dlambda_reference);
Stiffness_reference = diag((dF_dlambda_reference)*diag(lambda) + F_reference);

% Mass(i,j) ?= u_i^T D u_j =: benchmark_Mass,
% Stiffness(i,j) ?= u_i^T D A u_j =: benchmark_Stiffness
benchmark_Mass      = Mass*0;
benchmark_Stiffness = Stiffness*0;

benchmark_Mass_reference      = Mass_reference*0;
benchmark_Stiffness_reference = Stiffness_reference*0;

for i = 1:numel(lambda)
    for j = 1:numel(lambda)

        if j ~= i
            Mass(i,j)      = (F(i)           -       F(j)    )/(lambda(j) - lambda(i));
            Stiffness(i,j) = (F(j)*lambda(j) - F(i)*lambda(i))/(lambda(j) - lambda(i));

            Mass_reference(i,j)      = (F_reference(i)           - F_reference(j)          )/(lambda(j) - lambda(i));
            Stiffness_reference(i,j) = (F_reference(j)*lambda(j) - F_reference(i)*lambda(i))/(lambda(j) - lambda(i));
        end

      benchmark_Mass(i,j)      = u_lambda(:,i)' * D * u_lambda(:,j);
      benchmark_Stiffness(i,j) = u_lambda(:,i)' * D * L * u_lambda(:,j); 

      benchmark_Mass_reference(i,j)      = u_lambda_reference(:,i)' * D * u_lambda_reference(:,j);
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


%% Lanczos algorithm & truncated ROM
V           = u_lambda;
V_ref       = u_lambda_reference;

threshold = 10^(-12);

[X,D]          = eig(Mass);
[X_ref, D_ref] = eig(Mass_reference);

% Enforce increasing order of eigenvalues & eigenvectors
if ~issorted(diag(D))
    [D,I] = sort(diag(D));
    X = X(:, I);
end

if ~issorted(diag(D_ref))
    [D_ref,I_ref] = sort(diag(D_ref));
    X_ref = X_ref(:, I_ref);
end

% Grab the dominant eigenvectors 
index_threshold = 0;
for i = 1:length(diag(D))
    if D(i,i) > D(end)*threshold
        index_threshold=i;
        break;
    end
end
Z = X(:,index_threshold:end);

index_threshold = 0;
for i = 1:length(diag(D_ref))
    if D_ref(i,i) > D_ref(end)*threshold
        index_threshold=i;
        break;
    end
end
Z_ref = X_ref(:,index_threshold:end);

V_tilde = V * Z;
M_tilde = V_tilde' * V_tilde;
S_tilde = V_tilde' * L * V_tilde; 

V_tilde_ref = V_ref * Z_ref;
M_tilde_ref = V_tilde_ref' * V_tilde_ref;
S_tilde_ref = V_tilde_ref' * L_ref * V_tilde_ref; 


%M_inverse = inv(Mass);
%A = M_inverse*Stiffness;
M_inverse = inv(M_tilde);
A         = M_inverse*S_tilde;
b         = Z'*u_lambda(1,:)';  
[row col] = size(M_tilde);
m         = col;

M_inverse_ref     = inv(M_tilde_ref);
A_ref             = M_inverse_ref*S_tilde_ref;
b_ref             = Z_ref'*u_lambda_reference(1,:)';  
[row_ref col_ref] = size(M_tilde_ref);
m_ref             = col_ref;

[Q_ref,alpha_ref,beta_ref] = Lanczos(A_ref,M_tilde_ref,M_inverse_ref,b_ref,m_ref); % Perform Lanczos for change of basis: QV

[Q,alpha,beta] = Lanczos(A,M_tilde,M_inverse,b,m); % Perform Lanczos for change of basis: QV

%% Lippman-Schwinger

% Eq (3.8): u ~= sqrt(b^T inv(M) b) V_0 Q_0 (T+ lambda * Id)^-1 e_1
T = diag(beta(1:end-1),-1) + diag(alpha,0) + diag(beta(1:end-1),1);
%u_approx = sqrt(b' * M_inverse * b) * V_tilde_ref * Q_ref * inv(T + lambda*);
e1 = ones(length(b),1);
e1(2:end) = 0;
u_approx = u_lambda*0;
for i = 1:numel(lambda)
    u_approx(:,i) = diag(sqrt(b' * M_inverse * b),M) * V_tilde_ref * Q_ref * inv(T + lambda(i)*eye(size(T))) * e1;
end

% Compare the exact sol. with u_approx
figure
subplot(2,1,1)
hold on;
plot(x,u_lambda(:,1), 'LineWidth',2.5);
plot(x,u_approx(:,1), 'LineWidth',2.5);
legend('$u(\lambda_1)$','$\mathbf{u}(\lambda_1)$','Interpreter','latex','FontSize',16)
xlabel("Grid");
ylabel("Data"), 

title('$u(\lambda_1)$ and $\mathbf{u}(\lambda_1)$','Interpreter','latex','FontSize',16)
hold off;
subplot(2,1,2)
plot(x,V_tilde * Q(:,1:3))
hold on
plot(x,V_tilde_ref * Q_ref(:,1:3), '--')
legend('$\widetilde{V}Q(:,1)$','$\widetilde{V}Q(:,2)$','$\widetilde{V}Q(:,3)$','$\widetilde{V}_0Q_0(:,1)$', ...
    '$\widetilde{V}_0Q_0(:,2)$', ...
    '$\widetilde{V}_0Q_0(:,3)$', ...
    'Interpreter','latex','FontSize',16)
xlabel("Grid");
ylabel("Basis vectors"), 

title('First three columns: $\widetilde{V}Q$ and $\widetilde{V}_0 Q_0$','Interpreter','latex','FontSize',16)
hold off;
function [Q, alpha, beta] = Lanczos(A,Mass,M_inverse,b,iter)
    %% Some initialization
    [row, col] = size(b);

    Q          = zeros(row, iter); % Q = [q_1 | q_2 | ... | q_m]
    Q(:,1)     = (M_inverse*b)/sqrt(b'*M_inverse*b);  % q_1 = M^-1 b /sqrt(b^T inv(M) b)

    alpha = zeros(iter,1);   % Main diagonal
    beta  = zeros(iter-1,1); % Sub/Super diagonal
    
    %% Perform the Lanczos iteration: three-term recurrence relation
    for i = 1:iter
        %% Construct column vector q_(i+1) as Q = [... | q_(i+1) | ...]
        v        = A*Q(:,i);      % A * q_i 
        alpha(i) = Q(:,i)' * Mass * v; % M inner product: q_i^T M * A * q_i
    
        v = v - alpha(i)*Q(:,i);  
        if i > 1
            v = v - beta(i-1)*Q(:,i-1); % q_i = (A q_i - beta_(i-1) q_(i-1) - alpha_i q_i)
        end

        % Enforce orthoganalization to overcome loss of orthognalization
        for k = 1:3
            for j = i:-1:1
                cf  = Q(:,j)' * Mass * v;
                v   = v - cf*Q(:,j);
            end
        end
        
        beta(i)  = sqrt(v' * Mass * v); % norm corresponding to M inner product
        v = v/beta(i); % Normalize w.r.t M inner-product
        
        if i < iter
            % i = iter => Q is m x m+1
            Q(:,i+1) = v;
        end
    end
end

toc