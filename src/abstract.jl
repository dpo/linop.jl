export AbstractLinearOperator,
  LinearOperator,
  LinearOperatorException,
  hermitian,
  ishermitian,
  symmetric,
  issymmetric,
  nprod,
  ntprod,
  nctprod,
  reset!,
  shape

mutable struct LinearOperatorException <: Exception
  msg::AbstractString
end

# when indexing, Colon() is treated separately
const LinearOperatorIndexType{I} = Union{UnitRange{I}, StepRange{I, I}, AbstractVector{I}} where {I<:Integer}

# import methods we overload
import Base.eltype, Base.isreal, Base.size, Base.show
import LinearAlgebra.issymmetric, LinearAlgebra.ishermitian

abstract type AbstractLinearOperator{T} end
OperatorOrMatrix = Union{AbstractLinearOperator, AbstractMatrix}

eltype(A::AbstractLinearOperator{T}) where {T} = T
isreal(A::AbstractLinearOperator{T}) where {T} = T <: Real

"""
Base type to represent a linear operator.
The usual arithmetic operations may be applied to operators
to combine or otherwise alter them. They can be combined with
other operators, with matrices and with scalars. Operators may
be transposed and conjugate-transposed using the usual Julia syntax.
"""
mutable struct LinearOperator{T, I<:Integer, F, Ft, Fct} <: AbstractLinearOperator{T}
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!::F
  tprod!::Ft
  ctprod!::Fct
  nprod::I
  ntprod::I
  nctprod::I
  args5::Bool
end


LinearOperator{T}(nrow::I, ncol::I, symmetric::Bool, hermitian::Bool, 
                  prod!::F, tprod!::Ft, ctprod!::Fct,
                  nprod::I, ntprod::I, nctprod::I, args5::Bool
                  ) where {T,I<:Integer,F,Ft,Fct} = LinearOperator{T,I,F,Ft,Fct}(nrow, ncol, symmetric, hermitian, 
                                                                                     prod!, tprod!, ctprod!,
                                                                                     nprod, ntprod, nctprod, args5)

LinearOperator{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!,
  tprod!,
  ctprod!,
) where {T,I<:Integer} = LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!, 0, 0, 0, true)

nprod(op::AbstractLinearOperator) = op.nprod
ntprod(op::AbstractLinearOperator) = op.ntprod
nctprod(op::AbstractLinearOperator) = op.nctprod

increase_nprod(op::AbstractLinearOperator) = (op.nprod += 1)
increase_ntprod(op::AbstractLinearOperator) = (op.ntprod += 1)
increase_nctprod(op::AbstractLinearOperator) = (op.nctprod += 1)

"""
  reset!(op)

Reset the product counters of a linear operator.
"""
function reset!(op::AbstractLinearOperator)
  op.nprod = 0
  op.ntprod = 0
  op.nctprod = 0
  return op
end

"""
    m, n = size(op)

Return the size of a linear operator as a tuple.
"""
size(op::AbstractLinearOperator) = (op.nrow, op.ncol)

"""
    m = size(op, d)

Return the size of a linear operator along dimension `d`.
"""
function size(op::AbstractLinearOperator, d::Integer)
  nrow, ncol = size(op)
  if d == 1
    return nrow
  end
  if d == 2
    return ncol
  end
  throw(LinearOperatorException("Linear operators only have 2 dimensions for now"))
end

"""
    m, n = shape(op)

An alias for size.
"""
shape(op::AbstractLinearOperator) = size(op)

"""
    hermitian(op)
    ishermitian(op)

Determine whether the operator is Hermitian.
"""
hermitian(op::AbstractLinearOperator) = op.hermitian
ishermitian(op::AbstractLinearOperator) = op.hermitian

"""
    symmetric(op)
    issymmetric(op)

Determine whether the operator is symmetric.
"""
symmetric(op::AbstractLinearOperator) = op.symmetric
issymmetric(op::AbstractLinearOperator) = op.symmetric

"""
    show(io, op)

Display basic information about a linear operator.
"""
function show(io::IO, op::AbstractLinearOperator)
  s = "Linear operator\n"
  nrow, ncol = size(op)
  s *= @sprintf("  nrow: %s\n", nrow)
  s *= @sprintf("  ncol: %d\n", ncol)
  s *= @sprintf("  eltype: %s\n", eltype(op))
  s *= @sprintf("  symmetric: %s\n", symmetric(op))
  s *= @sprintf("  hermitian: %s\n", hermitian(op))
  s *= @sprintf("  nprod:   %d\n", nprod(op))
  s *= @sprintf("  ntprod:  %d\n", ntprod(op))
  s *= @sprintf("  nctprod: %d\n", nctprod(op))
  s *= "\n"
  print(io, s)
end

"""
    A = Matrix(op)

Materialize an operator as a dense array using `op.ncol` products.
"""
function Base.Matrix(op::AbstractLinearOperator{T}) where {T}
  (m, n) = size(op)
  A = Array{T}(undef, m, n)
  ei = zeros(T, n)
  for i = 1:n
    ei[i] = one(T)
    A[:, i] = op * ei
    ei[i] = zero(T)
  end
  return A
end
