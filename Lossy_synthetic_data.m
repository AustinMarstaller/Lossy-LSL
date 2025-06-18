% SYNTHETIC DATA GENERATION
% 
% INPUT: x - discrete spatial grid,
% h - grid spacing,
% p - potential function
% lambda := -omega² < 0 from the Fourier transform of ∂ₜₜ
% OUTPUT: v(x,lambda)::vector - numerical solution of the 1-d1iensional frequency domain wave equation w/ potential function. 

function [v] = Lossy_synthetic_data(x,h,alpha,p,lambda)
   omega  = sqrt(-lambda);
   c2     = -(exp(1i*omega))/(1i*omega*( (1-alpha)*exp(1i*omega) - (1+alpha)*exp(-1i*omega) ));
   c1     = (1 + 1i*omega*(1 - alpha)*c2) / (1i*omega*(1+alpha));

   % solution corresponding to p=0 scenario
   exact_solution = c1*exp(1i*omega*x) + c2*exp(-1i*omega*x);

   C                        = spdiags([1/h^2, 2/h^2, 1/h^2],-1:1,length(x),length(x)) + diag(p); 
   C(1,2)                   = 2/h^2;
   C(length(x),length(x)-1) = 2/h^2;   

   D      = 0*diag(x);
   D(1,1) = alpha*(2/h);

   A = -C + 1i*omega*D - lambda*eye(length(x));

   f     = zeros(length(x),1);
   f(1)  = 2/h;
        
   % Solve the linear system Av = f
   v = A\f;
end
