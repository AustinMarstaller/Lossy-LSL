% MAIN ROM FILE
%
% 
%
close all
%% PARAMETER GENERATION
 number_of_mu        = 10;
 condNumber_fineness = 20;
 number_of_lambda    = 1;
 alpha               = 0.5;
 sigma               = 0.05;
 gamma               = 0.75;

[lambda,mu] = lambda_construction(number_of_lambda, number_of_mu);

wavelength = 2*pi / sqrt(mu(end)); 
h          = wavelength/20; % Grid point spacing is 20x smaller than wavelength  
x          = (0:h:1)'; % Lattice in column vector   

% potential terms
p           = gamma*exp(-(x-0.2).^2 / sigma^2); % p = 0 for reference problem
p_reference = 0*p;

%% SYNTHETIC DATA GENERATION
solutions_per_frequency.gaussian_potential = zeros(length(x), length(lambda));
solutions_per_frequency.zero_potential     = zeros(length(x), length(lambda));

for j = 1:length(lambda)
    % For every λᵢ ∈ λ::Vector, generate the synthetic data via the Finite Difference scheme
    solutions_per_frequency.gaussian_potential(:,j) = Lossy_synthetic_data(x,h,alpha,p,lambda(j));
    solutions_per_frequency.zero_potential(:,j)     = Lossy_synthetic_data(x,h,alpha,p_reference,lambda(j));
end

%% ROM 