export LSR1Operator, diag  #, InverseLSR1Operator

"A data type to hold information relative to LSR1 operators."
mutable struct LSR1Data{T,I<:Integer}
  mem::I
  scaling::Bool
  scaling_factor::T
  s::Vector{Vector{T}}
  y::Vector{Vector{T}}
  ys::Vector{T}
  a::Vector{Vector{T}}
  as::Vector{T}
  insert::I
  Ax::Vector{T}
end

function LSR1Data(T::DataType, n::I; mem::I = 5, scaling::Bool = true, inverse::Bool = false) where {I<:Integer}
  inverse && @warn "inverse LSR1 operator not yet implemented"
  LSR1Data{T,I}(
    max(mem, 1),
    scaling,
    convert(T, 1),
    [zeros(T, n) for _ = 1:mem],
    [zeros(T, n) for _ = 1:mem],
    zeros(T, mem),
    [zeros(T, n) for _ = 1:mem],
    zeros(T, mem),
    1,
    Vector{T}(undef, n),
  )
end

LSR1Data(n::I; kwargs...) where {I<:Integer} = LSR1Data(Float64, n; kwargs...)

"A type for limited-memory SR1 approximations."
mutable struct LSR1Operator{T,I<:Integer,F,Ft,Fct} <: AbstractLinearOperator{T}
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!::F     # apply the operator to a vector
  tprod!::Ft    # apply the transpose operator to a vector
  ctprod!::Fct   # apply the transpose conjugate operator to a vector
  inverse::Bool
  data::LSR1Data{T,I}
  nprod::I
  ntprod::I
  nctprod::I
end

LSR1Operator{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  inverse::Bool,
  data::LSR1Data{T,I},
) where {T,I<:Integer,F,Ft,Fct} =
  LSR1Operator{T,I,F,Ft,Fct}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, inverse, data, 0, 0, 0)

has_args5(op::LSR1Operator) = true

"""
    LSR1Operator(T, n; [mem=5, scaling=false)
    LSR1Operator(n; [mem=5, scaling=false)
Construct a limited-memory SR1 approximation in forward form. If the type `T` is
omitted, then `Float64` is used.
"""
function LSR1Operator(T::DataType, n::I; kwargs...) where {I<:Integer}
  lsr1_data = LSR1Data(T, n; kwargs...)

  function lsr1_multiply(q::AbstractVector, data::LSR1Data, x::AbstractArray, α, β :: T2) where T2
    # Multiply operator with a vector.

    if β == zero(T2)
      q .= α .* x ./ data.scaling_factor 
    else
      q .= α .* x ./ data.scaling_factor .+ β .* q
    end

    for i = 1:(data.mem)
      k = mod(data.insert + i - 2, data.mem) + 1
      if data.ys[k] != 0
        ax = α * dot(data.a[k], x) / data.as[k]
        for j ∈ eachindex(q)
          q[j] += ax * data.a[k][j]
        end
      end
    end
  end

  prod! = @closure (res, x, α, β) -> lsr1_multiply(res, lsr1_data, x, α, β)
  return LSR1Operator{T}(n, n, true, true, prod!, nothing, nothing, false, lsr1_data)
end

LSR1Operator(n::I; kwargs...) where {I<:Integer} = LSR1Operator(Float64, n; kwargs...)

"""
    push!(op, s, y)
Push a new {s,y} pair into a L-SR1 operator.
"""
function push!(op::LSR1Operator, s::AbstractVector, y::AbstractVector)

  # op.counters.updates += 1
  data = op.data
  Bs = op * s
  ymBs = y - Bs
  ys = dot(y, s)
  sNorm = norm(s)
  yy = dot(y, y)

  ϵ = eps(eltype(op))
  well_defined = abs(dot(ymBs, s)) ≥ ϵ + ϵ * norm(ymBs) * sNorm

  sufficient_curvature = true
  scaling_condition = true
  if data.scaling
    yNorm = √yy
    sufficient_curvature = abs(ys) ≥ ϵ * yNorm * sNorm
    if sufficient_curvature
      scaling_factor = ys / yy
      scaling_condition = norm(y - s / scaling_factor) >= ϵ * yNorm * sNorm
    end
  end

  if !(well_defined && sufficient_curvature && scaling_condition)
    @debug "LSR1 update rejected" well_defined sufficient_curvature scaling_condition
    # op.counters.rejects += 1
    return op
  end

  data.s[data.insert] .= s
  data.y[data.insert] .= y
  data.ys[data.insert] = ys

  # update scaling factor
  data.scaling && (data.scaling_factor = ys / yy)

  # update next insertion position
  data.insert = mod(data.insert, data.mem) + 1

  # update rank-1 terms
  for i = 1:(data.mem)
    k = mod(data.insert + i - 2, data.mem) + 1
    if data.ys[k] != 0
      data.a[k] .= data.y[k] - data.s[k] / data.scaling_factor  # = y - B₀ * s
      for j = 1:(i - 1)
        l = mod(data.insert + j - 2, data.mem) + 1
        if data.ys[l] != 0
          as = dot(data.a[l], data.s[k]) / data.as[l]
          data.a[k] .-= as * data.a[l]
        end
      end
      data.as[k] = dot(data.a[k], data.s[k])
    end
  end

  return op
end

"""
    diag(op)
    diag!(op, d)
Extract the diagonal of a L-SR1 operator in forward mode.
"""
function diag(op::LSR1Operator{T}) where {T}
  d = Vector{T}(undef, op.nrow)
  diag!(op, d)
end

function diag!(op::LSR1Operator{T}, d) where {T}
  op.inverse && throw("only the diagonal of a forward L-SR1 approximation is available")
  data = op.data

  fill!(d, 1)
  data.scaling && (d ./= data.scaling_factor)

  for i = 1:(data.mem)
    k = mod(data.insert + i - 2, data.mem) + 1
    if data.ys[k] != 0.0
      for j = 1:(op.nrow)
        d[j] += data.a[k][j]^2 / data.as[k]
      end
    end
  end
  return d
end

"""
    reset!(data)
Reset the given LSR1 data.
"""
function reset!(data::LSR1Data{T,I}) where {T,I<:Integer}
  for i = 1:(data.mem)
    fill!(data.s[i], 0)
    fill!(data.y[i], 0)
    fill!(data.a[i], 0)
  end
  fill!(data.ys, 0)
  fill!(data.as, 0)
  data.scaling_factor = T(1)
  data.insert = 1
  return data
end

"""
    reset!(op)
Resets the LSR1 data of the given operator.
"""
function reset!(op::LSR1Operator)
  reset!(op.data)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end
