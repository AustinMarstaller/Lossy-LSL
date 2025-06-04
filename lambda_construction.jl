#=
λ = -ω² value construction. 

number_of_lambda = # of distinct λ values between two adjacent μ values
number_of_mu     = # of values of μ to calculate
μ₁ λ₁ λ₂ .. μ₂ ... μ₃ ... 
=#
function lambda_construction(number_of_lambda::Int64, number_of_mu::Int64)
    # calculate the values of lambda and construct the lambda vector

    μ = zeros(1,number_of_mu)

    for j = 1:number_of_mu
        μ[j] = (j-1)^2 * (2 * pi)^2; # Weyl's eigenvalue dist. law   
    end
    
    temp    = LinRange(μ[1],μ[1+1], number_of_lambda+2);
    λ       = temp[2:length( temp ) - 1] 
    for k = 2:number_of_mu-1       
        temp = LinRange(μ[k],μ[k+1], number_of_lambda+2);
        λ = hcat(λ,temp[2:length( temp ) - 1 ]);
    end

return vec(-λ),vec(μ) 
end
