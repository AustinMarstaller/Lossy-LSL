function [lambda, mu] = lambda_construction(number_of_lambda, mu, number_of_mu)
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