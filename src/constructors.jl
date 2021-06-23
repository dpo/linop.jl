# Constructors.
"""
    LinearOperator(M; symmetric=false, hermitian=false)
Construct a linear operator from a dense or sparse matrix.
Use the optional keyword arguments to indicate whether the operator
is symmetric and/or hermitian.
"""
function LinearOperator(
  M::AbstractMatrix{T};
  symmetric = false,
  hermitian = false,
) where {T}
  nrow, ncol = size(M)
  prod! = @closure (res, v, α, β) -> mul!(res, M, v, α, β)
  tprod! = @closure (res, u, α, β) -> mul!(res, transpose(M), u, α, β)
  ctprod! = @closure (res, w, α, β) -> mul!(res, adjoint(M), w, α, β)
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!)
end

  """
  LinearOperator(Mv, M :: Symmetric{<:Real})

Construct a linear operator from a symmetric real square matrix `M`.
"""
LinearOperator(
  M::Symmetric{T},
  ) where {T <: Real} = LinearOperator(M, symmetric = true, hermitian = true)

"""
    LinearOperator(M)

Constructs a linear operator from a symmetric tridiagonal matrix. If
its elements are real, it is also Hermitian, otherwise complex
symmetric.
"""
function LinearOperator(M::SymTridiagonal{T}) where {T}
  hermitian = eltype(M) <: Real
  LinearOperator(M, symmetric = true, hermitian = hermitian)
end

"""
    LinearOperator(M)
    
Constructs a linear operator from a Hermitian matrix. If
its elements are real, it is also symmetric.
"""
function LinearOperator(M::Hermitian{T}) where {T}
  symmetric = eltype(M) <: Real
  LinearOperator(M, symmetric = symmetric, hermitian = true)
end

"""
    LinearOperator(type, nrow, ncol, symmetric, hermitian, prod!,
                    [tprod!=nothing,
                    ctprod!=nothing])
                    
Construct a linear operator from functions where the type is specified as the first argument.
Notice that the linear operator does not enforce the type, so using a wrong type can
result in errors. For instance,
```
A = [im 1.0; 0.0 1.0] # Complex matrix
function mulOp!(res, M, v, α, β)
  mul!(res, M, v, α, β)
end
op = LinearOperator(Float64, 2, 2, false, false, 
                    (res, v, α, β) -> mulOp!(res, A, v, α, β), 
                    (res, u, α, β) -> mulOp!(res, transpose(A), u, α, β), 
                    (res, w, α, β) -> mulOp!(res, A', w, α, β))
Matrix(op) # InexactError
```
The error is caused because `Matrix(op)` tries to create a Float64 matrix with the
contents of the complex matrix `A`.

Using `*` may generate a vector that contains `NaN` values.
This can also happen if you use the 3-args `mul!` function with a preallocated vector such as 
`Vector{Float64}(undef, n)`.
To fix this issue you will have to deal with the cases `β == 0` and `β != 0` separately:
```
d1 = [2.0; 3.0]
function mulSquareOpDiagonal!(res, d, v, α, β::T) where T
  if β == zero(T)
    res .= α .* d .* v
  else 
    res .= α .* d .* v .+ β .* res
  end
end
op = LinearOperator(Float64, 2, 2, true, true, 
                    (res, v, α, β) -> mulSquareOpDiagonal!(res, d, v, α, β))
```
"""
function LinearOperator(
  ::Type{T},
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!,
  tprod! = nothing,
  ctprod! = nothing,
) where {T, I<:Integer}
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod!, tprod!, ctprod!)
end

export LinearOperator_3args
function LinearOperator_3args(
  ::Type{T},
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!,
  tprod! = nothing,
  ctprod! = nothing,
) where {T, I<:Integer}
  prod5! = (res, v, α, β) -> prod!(res, v)
  tprod5! = (tprod! == nothing) ? nothing : (res, u, α, β) -> tprod!(res, u)
  ctprod5! = (ctprod! == nothing) ? nothing : (res, w, α, β) -> ctprod!(res, w)
  LinearOperator{T}(nrow, ncol, symmetric, hermitian, prod5!, tprod5!, ctprod5!, 0, 0, 0, false)
end
