%{
λ = -ω² value construction. 

number_of_lambda = # of distinct λ values between two adjacent μ values
number_of_mu     = # of values of μ to calculate

μ₁ λ₁ λ₂ .. μ₂ ... μ₃ ... 
%}

function [lambda, mu] = lambda_construction(number_of_lambda, number_of_mu)
        mu = zeros(1,number_of_mu);

        for j = 1:number_of_mu
            mu(j) = (j-1)^2 * (2 * pi)^2; % Weyl's eigenvalue dist. law   
        end
        
        % calculate the values of lambda and construct the lambda vector
        for k = 1:number_of_mu-1
            temp = linspace(mu(k),mu(k+1), number_of_lambda+2);
            
            if k==1
                lambda = temp(2:width( temp ) - 1 );
            end
            if k>1
                lambda = cat(2,lambda,temp(2:width( temp ) - 1 ));
            end
        end

   lambda     = -1*flip(lambda);
end