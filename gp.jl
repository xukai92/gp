using Distributions

# Define kernel

function κ(x1::Real, x2::Real; method="se", param=Dict())
  if method == "se"
    σ = haskey(param, "σ") ? param["σ"] : 1
    l = haskey(param, "l") ? param["l"] : 1
    σ^2 * exp(-1 / (2 * l^2) * (x1 - x2)^2)
  end
end

function κ(X1::Vector, X2::Vector, method="se", param=Dict())
  N1 = size(X1)[1]
  N2 = size(X2)[1]
  K = Matrix{Float64}((N1, N2))
  for n1 = 1:N1, n2 = 1:N2
    K[n1, n2] = κ(X1[n1], X2[n2]; method=method, param=param)
  end
  K
end

function predictdist(X, f, Xn)
  K = κ(X, X)
  Kn = κ(X, Xn)
  Knn = κ(Xn, Xn)
  μn = mean(Xn) + Kn' * inv(K) * (f - mean(X))
  Σn = Knn - Kn' * inv(K) * Kn
  MvNormal(μn, Σn)
end

d = predictdist([1, 2, 3, 4, 5], [1, 2, 1, 2, 1], [2.001])
