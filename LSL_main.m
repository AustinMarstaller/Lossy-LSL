clear all
close all
tic
%% Synthetic data construction
M=500; % M+1 total grid points previously 256
h=1/M; % Grid point spacing. 
x=(0:h:1)'; % Lattice in column vector

mu          = 0.22; 
sigma       = 0.05; 
gamma       = 0.74;
p           = gamma*exp(-(x-mu).^2 / sigma^2); % p = 0 for reference problem
p_reference = zeros(M+1,1);
lambda      = [2,4,8,16,32,48]; 

L_diag   = 2/h^2 * eye(M+1,M+1) + diag(p);
L        = spdiags([-1/h^2 0 -1/h^2],-1:1,M+1,M+1) + L_diag;
L(1,2)   = -2/h^2;
L(M+1,M) = -2/h^2;

L_ref_diag   = 2/h^2 * eye(M+1,M+1) + diag(p_reference);
L_ref        = spdiags([-1/h^2 0 -1/h^2],-1:1,M+1,M+1) + L_ref_diag;
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
F           = u_lambda(1,:)'; % u(0, lambda_i)
F_reference = u_lambda_reference(1,:)';

dF_dlambda           = zeros(numel(lambda),1);
dF_dlambda_reference = zeros(numel(lambda),1);
b                    = F;
b_ref                = F_reference;

for i = 1:numel(lambda)
    % dF/dlambda = -u^T D u
    dF_dlambda(i) = -u_lambda(:,i)' * D * u_lambda(:,i);
    dF_dlambda_reference(i) = -u_lambda_reference(:,i)' * D * u_lambda_reference(:,i);
end

%% Mass & Stiffness matrices & b column vector
% M & S are symmetric w.r.t <-,->_D
% M_ii = -u^T D u
% S_ii = lambda dF/dlambda + F
Mass      = -diag(dF_dlambda); 
Stiffness = diag(diag(lambda)*(dF_dlambda) + F); % lambda dF/dlambda + F

Mass_reference      =  -diag(dF_dlambda_reference);
Stiffness_reference = diag(diag(lambda) * (dF_dlambda_reference) + F_reference);

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

%% Lanczos algorithm & truncated ROM
V           = u_lambda;
V_ref       = u_lambda_reference;

threshold = 5.e-12; 

[X,eig_values]          = eig(Mass);
[X_ref, eig_values_ref] = eig(Mass_reference);

% Enforce increasing order of eigenvalues & eigenvectors
if ~issorted(diag(eig_values))
    [eig_values,I] = sort(diag(eig_values));
    X = X(:, I);
end

if ~issorted(diag(eig_values_ref))
    [eig_values_ref,I_ref] = sort(diag(eig_values_ref));
    X_ref = X_ref(:, I_ref);
end

% Grab the dominant eigenvectors 
index_threshold = 0;
for i = 1:length(diag(eig_values))
    if eig_values(i,i) > eig_values(end)*threshold
        index_threshold=i;
        break;
    end
end
Z = X(:,index_threshold:end);
[row_Z, l] = size(Z); % Eq. (5.8) Z in R^mxl
index_threshold = 0;
for i = 1:length(diag(eig_values_ref))
    if eig_values_ref(i,i) > eig_values_ref(end)*threshold
        index_threshold=i;
        break;
    end
end
Z_ref = X_ref(:,index_threshold:end);

V_tilde = V * Z;
M_tilde = Z' * Mass * Z;
S_tilde = Z' * Stiffness * Z; 

V_tilde_ref = V_ref * Z_ref;
M_tilde_ref = Z_ref' * Mass_reference * Z_ref;   
S_tilde_ref = Z_ref' * Stiffness_reference * Z_ref; 

M_inverse = inv(M_tilde);
A_tilde         = M_inverse*S_tilde;
b_tilde         = Z'*b;  
[row col] = size(M_tilde);
m         = col;

M_inverse_ref           = inv(M_tilde_ref);
A_tilde_ref             = M_inverse_ref*S_tilde_ref;
b_tilde_ref             = Z_ref'*b_ref;  
[row_ref col_ref] = size(M_tilde_ref);
m_ref             = col_ref;

[Q_tilde_ref,alpha_tilde_ref,beta_tilde_ref] = Lanczos(A_tilde_ref,M_tilde_ref,M_inverse_ref,b_tilde_ref,m_ref); % Perform Lanczos for change of basis: QV

[Q_tilde,alpha_tilde,beta_tilde] = Lanczos(A_tilde,M_tilde,M_inverse,b_tilde,m); % Perform Lanczos for change of basis: QV

%% Lippman-Schwinger formulation 

% Eq (3.8): u ~= sqrt(b^T inv(M) b) V_0 Q_0 (T+ lambda * Id)^-1 e_1
T_tilde = diag(beta_tilde(1:end-1),-1) + diag(alpha_tilde,0) + diag(beta_tilde(1:end-1),1);
T_tilde_ref = diag(beta_tilde_ref(1:end-1),-1) + diag(alpha_tilde_ref,0) + diag(beta_tilde_ref(1:end-1),1);

e1 = ones(length(b_tilde),1);
e1(2:end) = 0;
u_approx = zeros(length(x), l);

for i = 1:numel(lambda)
    u_approx(:,i) = sqrt(b_tilde' * M_inverse * b_tilde) * V_tilde_ref * Q_tilde_ref * inv(T_tilde + lambda(i)*eye(size(T_tilde))) * e1;
end

% Solve inverse problem F_0 - F_p = <u_approx, p u_ref>

W       = zeros(numel(lambda),length(x));
deltaF  = zeros(numel(lambda),1);

for j = 1:numel(lambda)
    W(j,:)    = (   u_approx(:,j)  .* diag(D) .*  u_lambda(:,j)   )'; % Each row corresponds to lambda = lambda_j
    deltaF(j) = (   F_reference(j) -            F(j)   );  % Real column vector with l components
end

%p_reconstructed = W\deltaF;
[U,S,V] = svd(W,'econ');
conditionNumbers = zeros(3,1);
conditionNumbers(1) = 6.e-5; 
conditionNumbers(2) = 6.e-4;
conditionNumbers(3) = 6.e-3;

err = zeros(1,4);
figure 
subplot(2,1,1)
hold on
plot(x,p);

invWtruncated   = V *  inv(S) * U';
p_reconstructed = invWtruncated * deltaF;
err(1)          = norm(p - p_reconstructed);

plot(x,p_reconstructed,'--');
for j = 1:3
    r               = max(    find( diag(S) > max(    S(:)     ) * conditionNumbers(j) )     );
    invWtruncated   = V(:,1:r) *  inv(S(1:r, 1:r)) * U(:,1:r)'; % Approximate 
    p_reconstructed = invWtruncated * deltaF; 
    err(j+1) = norm(p - p_reconstructed);
    plot(x,p_reconstructed,'--');
end
xlabel("Grid");
ylabel("Potentials"), 
title("True and reconstructed potentials",'FontSize',16);
subtitle("\gamma = "+gamma+", \sigma="+sigma+", \mu="+mu,'FontSize',16);
legend('True','Unmodified $\kappa$','$\kappa = 6e^{-5}$','$\kappa = 6e^{-4}$','$\kappa = 6e^{-3}$','Interpreter','latex','FontSize',16)
hold off


subplot(2,1,2)
plot(1:4,err,'-o', ...
    'MarkerSize',10, ...
     'MarkerEdgeColor','red' )
%legend('$\kappa = e^{-5}$','$\kappa = e^{-4}$','$\kappa = e^{-3}$','Interpreter','latex','FontSize',16)
xlabel('Unmodified $\kappa$ and modified $\kappa$: $6e^{-n}$, for $n=5,4,3$','Interpreter','latex','FontSize',16), ylabel('Error')
title('$L_2$ norm error for varying condition number $\kappa$','Interpreter','latex','FontSize',16)
subtitle('$||p - \widetilde{p}||_{2}$','Interpreter','latex','FontSize',16)


%{
figure

hold on
plot(x, V_tilde * Q_tilde(:,1), x, V_tilde_ref * Q_tilde_ref(:,1),'--');
plot(x, V_tilde * Q_tilde(:,2), x, V_tilde_ref * Q_tilde_ref(:,2),'--');
plot(x, V_tilde * Q_tilde(:,3), x, V_tilde_ref * Q_tilde_ref(:,3),'--');
xlabel("Grid");
ylabel("Basis vectors"), 

title('First three columns: $\widetilde{V}\widetilde{Q}$ and $\widetilde{V}_0 \widetilde{Q}_0$','Interpreter','latex','FontSize',16)
hold off
%% Plot singular values
figure
semilogy(diag(S),'-ok','LineWidth',1.5), hold on, grid on
semilogy(   diag(S(1:r,1:r)), 'or','LineWidth',1.5   )
%}


%% Plot background and internal solutions

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