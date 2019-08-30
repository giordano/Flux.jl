import ..Flux: Flux, relu
import ..Tracker: TrackedArray
using CuArrays.CUDAnative
using CuArrays: @cuindex, cudims
using LinearAlgebra

function LinearAlgebra.copy_transpose!(dst::CuArray, src::CuArray)
  function kernel(dst, src)
    I = @cuindex dst
    dst[I...] = src[reverse(I)...]
    return
  end
  blk, thr = cudims(dst)
  @cuda blocks=blk threads=thr kernel(dst, src)
  return dst
end

CuParam{T,N} = Union{CuArray{T,N},TrackedArray{T,N,CuArray{T,N}}}
CuRNN{T} = Flux.RNNCell{<:Union{typeof(tanh),typeof(relu)},<:CuParam{T,2},<:CuParam{T,1}}
CuGRU{T} = Flux.GRUCell{<:CuParam{T,2},<:CuParam{T,1}}
CuLSTM{T} = Flux.LSTMCell{<:CuParam{T,2},<:CuParam{T,1}}
CuRNNs{T} = Union{CuRNN{T},CuGRU{T},CuLSTM{T}}

function copyparams!(m::CuRNNs, d::CUDNN.RNNDesc)
  Wi, Wh = d.weights
  copy_transpose!(Wi, Flux.data(m.Wi))
  copy_transpose!(Wh, Flux.data(m.Wh))
  copy_transpose!(d.bias, Flux.data(m.b))
  return
end

function CUDNN.RNNDesc(m::CuRNNs{T}) where T
  h, i = length(m.h), size(m.Wi, 2)
  mode = m isa CuRNN ?
    (m.σ == tanh ? CUDNN.CUDNN_RNN_TANH : CUDNN.CUDNN_RNN_RELU) :
    m isa CuGRU ? CUDNN.CUDNN_GRU : CUDNN.CUDNN_LSTM
  r = CUDNN.RNNDesc{T}(mode, i, h)
  return r
end

const descs = WeakKeyDict()

function desc(rnn)
  d = haskey(descs, rnn) ? descs[rnn] : (descs[rnn] = CUDNN.RNNDesc(rnn))
  copyparams!(rnn, d)
  return d
end

import Flux.Tracker
import Flux.Tracker: data, istracked, track, unbroadcast, @grad, nobacksies

istrain(m::CuRNNs, args...) = any(x -> x isa TrackedArray, (m.Wi, m.Wh, m.b, args...))

function (m::CuRNN{T})(h::CuParam{T}, x::CuParam{T}) where T <: Union{Float32,Float64}
  result = istrain(m, h, x) ?
    track(m, x, h, m.Wi, m.Wh, m.b) :
    forward(desc(m), x, h)
  return result[2], result[1]
end

function (m::CuGRU{T})(h::CuParam{T}, x::CuParam{T}) where T <: Union{Float32,Float64}
  result = istrain(m, h, x) ?
    track(m, x, h, m.Wi, m.Wh, m.b) :
    forward(desc(m), x, h)
  return result[2], result[1]
end

function (m::CuLSTM{T})(h::NTuple{2,CuParam{T}}, x::CuParam{T}) where T <: Union{Float32,Float64}
  result = istrain(m, h, x) ?
    track(m, x, h[1], h[2], m.Wi, m.Wh, m.b) :
    forward(desc(m), x, h[1], h[2])
  return (result[2], result[3]), result[1]
end

(m::CuRNN{T})(h::CuParam{T}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))
(m::CuGRU{T})(h::CuParam{T}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))
(m::CuLSTM{T})(h::NTuple{2,CuParam{T}}, x) where T <: Union{Float32,Float64} = m(h, CuArray{T}(x))

@grad function (m::Union{CuRNN,CuGRU})(x, h, Wi, Wh, b)
  reserve, result = CUDNN.forwardTrain(desc(m), data(x), data(h))
  result, function (Δ)
    y, ho = result
    dy, dho = Δ
    h_ = CUDNN.hBatch(x, data(h))
    dx, dh = CUDNN.backwardData(descs[m], y, dy, dho, h_, reserve)
    (dWi, dWh), db = CUDNN.backwardWeights(descs[m], data(x), h_, y, reserve)
    nobacksies(:RNN, (dx, unbroadcast(h, dh), transpose(dWi), transpose(dWh), db))
  end
end

@grad function (m::CuLSTM)(x, h, c, Wi, Wh, b)
  reserve, result = CUDNN.forwardTrain(desc(m), data.((x, h, c))...)
  result, function (Δ)
    y, ho = result
    dy, dho, dco = Δ
    h_ = CUDNN.hBatch(x, data(h))
    c_ = CUDNN.hBatch(x, data(c))
    dx, dh, dc = CUDNN.backwardData(descs[m], y, dy, dho, dco, h_, c_, reserve)
    (dWi, dWh), db = CUDNN.backwardWeights(descs[m], data(x), h_, y, reserve)
    nobacksies(:RNN,
      (dx, unbroadcast(h, dh), unbroadcast(c, dc),
       transpose(dWi), transpose(dWh), db))
  end
end
