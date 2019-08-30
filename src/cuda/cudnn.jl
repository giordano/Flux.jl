import ..Flux: data
import CuArrays.CUDNN: batchnorm, ∇batchnorm

(BN::Flux.BatchNorm)(x::Union{CuParam{T,2},CuParam{T,4},CuParam{T,5}}, cache = nothing) where T<:Union{Float32, Float64} =
  BN.λ.(batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ², BN.momentum; cache = cache, alpha = 1, beta = 0, eps = BN.ϵ, training = BN.active))

batchnorm(g::TrackedArray, b::TrackedArray, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:Union{Float32, Float64} =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::TrackedArray, b::TrackedArray, x::CuArray{T}, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:Union{Float32, Float64} =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::TrackedArray, b::CuArray{T}, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:Union{Float32, Float64} =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::CuArray{T}, b::TrackedArray, x::CuArray{T}, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:Union{Float32, Float64} =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::CuArray{T}, b::TrackedArray, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:Union{Float32, Float64} =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::TrackedArray, b::CuArray{T}, x::CuArray{T}, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:Union{Float32, Float64} =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

batchnorm(g::CuArray{T}, b::CuArray{T}, x::TrackedArray, running_mean::CuArray{T},
          running_var::CuArray{T}, momentum; kw...) where T<:Union{Float32, Float64} =
  track(batchnorm, g, b, x, running_mean, running_var, momentum; kw...)

@grad batchnorm(g, b, x, running_mean, running_var, momentum; kw...) =
  batchnorm(data.((g, b, x))..., running_mean, running_var, momentum; kw...), Δ -> (nobacksies(:batchnorm, ∇batchnorm(data.((g, b, x, Δ))..., running_mean, running_var, momentum; kw...))..., nothing, nothing, nothing)
