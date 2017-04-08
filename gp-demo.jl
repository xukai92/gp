# Noisy GP

k(a::Real, b::Real) = begin
  sig = 1.0
  l = 1.0
  sig^2 * exp(-1 / (2 * l^2) * (a - b)^2)
end

k(a::Vector{Real}, b::Vector{Real}) = begin
  la, lb = length(a), length(b)
  res = Matrix{Real}(la, lb)
  for i = 1:la, j = 1:lb
    res[i, j] = k(a[i], b[j])
  end
  res
end

gp(X_star, X, f) = begin
  sig_y = 0.1

  K       = k(X, X)
  K_star  = k(X, X_star)
  K_star2 = k(X_star, X_star)
  K_y = K = K + sig_y^2 * eye(size(K)...)

  mu_star    = K_star' * inv(K_y) * f
  Sigma_star = K_star2 - K_star' * inv(K_y) * K_star

  mu_star, Sigma_star
end

using Distributions, Gadfly

make_plot(N) = begin

  X = Real[0, 1, 4, 5]
  f = map(x -> x + sqrt(x) + 1, X)

  X_star = Vector{Real}(collect(0:0.1:8))
  mu_star, Sigma_star = gp(X_star, X, f)

  posterior = MvNormal(mu_star, diag(Sigma_star))


  layer_training = layer(x=X, y=f, Geom.point, Theme(default_color=colorant"#8e44ad"))

  layer_predict_means = []
  for _ = 1:N
    f_star = rand(posterior)
    layer_predict_mean = layer(x=X_star, y=f_star, Geom.line, Theme(default_color=colorant"#bdc3c7"))
    push!(layer_predict_means, layer_predict_mean)
  end

  plot(layer_training, layer_predict_means..., Guide.title("GP demo: $N samples from posterior"))
end

p = make_plot(20)
draw(PNG("gp-demo.png", 8inch, 4.5inch), p)
