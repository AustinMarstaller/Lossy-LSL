% MAIN ROM FILE
%
% 
%
close all
clear all
format long
%% PARAMETER GENERATION
 number_of_mu        = 10;
 condNumber_fineness = 20;
 number_of_lambda    = 1;
 alpha               = 0.5;
 sigma               = 0.05;
 gamma               = 0.75;
 K                   = 1;
[lambda,mu] = lambda_construction(number_of_lambda, number_of_mu);

wavelength = 2*pi / sqrt(mu(end)); 
h          = wavelength/20; % Grid point spacing is 20x smaller than wavelength  
x          = (0:h:1)'; % Lattice in column vector   

% potential term
p = gamma*exp(-(x-0.2).^2 / sigma^2); % vector
p=0*p;


%% SETUP
   v_per_frequency= zeros(length(x), length(lambda));
   trial_space = zeros(2*length(x), length(lambda));
   test_space = trial_space*0;
   b_per_frequency = trial_space*0;
    
   C                        = spdiags([-1/h^2, 2/h^2, -1/h^2],-1:1,length(x),length(x)) + diag(p); 
   C(1,2)                   = -2/h^2;
   C(length(x),length(x)-1) = -2/h^2; 

   D      = zeros(length(x),length(x));
   D(1,1) = alpha*(2/h);

%% Forward solver: obtain numerical solution per frequency 
for j = 1:length(lambda)
    % For every λᵢ ∈ λ::Vector, generate the forward solution via the Finite Difference scheme
   
   omega  = sqrt(-lambda(j));
   c2     = -(exp(1i*omega))/(1i*omega*( (1-alpha)*exp(1i*omega) - (1+alpha)*exp(-1i*omega) ));
   c1     = (1 + 1i*omega*(1 - alpha)*c2) / (1i*omega*(1+alpha));

   % solution corresponding to p=0 scenario
   exact_solution = c1*exp(1i*omega*x) + c2*exp(-1i*omega*x);

   f     = zeros(length(x),1);
   f(1)  = 2/h;

   % Solve the linear system Av = f
   v_per_frequency(:,j) = (-C + 1i*omega*D - lambda(j)*eye(length(x)) )\f;

   b_per_frequency(end,j) = (2/h);

   %{
   % visualize forward solver
   figure
   hold on
   plot(x,abs(exact_solution).^2,'LineWidth',3,'Color',"blue")
   plot(x,abs(v).^2,'--','LineWidth',3,'Color',"red")

   title("Benchmark with \lambda = "+lambda(j))
   legend('Solid: exact','Dashed: approximate')
   %}
end

   % C and D are symmetrix w.r.t the finite-difference operator X = diag(2/h^2, 1/h^2, ..., 1/h^2, 2/h^2)
   X                      = eye(length(x));
   X                      = (1/h^(2)) * X;
   X(1,1)                 = 2/h^2;
   X(length(x),length(x)) = 2/h^2;

   % Similiarity transfrom for C: C = X^{-1/2} C X^{1/2}. It's SPD now
   C = inv(sqrtm(X)) * C * sqrtm(X);
   
   L = chol(round(C,5)); % C = L L^T

   for j=1:width(v_per_frequency)
        w          = (1i/omega) * L * v_per_frequency(:,j);
        trial_space(:,j)     = [w; v_per_frequency(:,j)];
        test_space(:,j) = [-w; v_per_frequency(:,j)];
   end

   % A is not SPD
   A = zeros(2*length(x),2*length(x));
   A( 1:length(x), (length(x)+1):2*length(x) )             = -L;
   A( (length(x)+1):2*length(x), 1:length(x) )             = L';
   A((length(x)+1):2*length(x), (length(x)+1):2*length(x)) = D;

   % A is symmetric (not PD) w.r.t the weighted inner-product induced by I
   I = eye(length(A),length(A));
   I(length(A)+1:end, length(A)+1:end) = -1;

%% Mass & Stiffness matrices
   % MASS
   M = (test_space)' * I * trial_space; % WARNING: NOT PD, but symmetric 

   % STIFFNESS
   S = (test_space)' * I * A * trial_space; % WARNING: NOT PD, but symmetric





