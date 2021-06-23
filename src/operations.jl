import Base.+, Base.-, Base.*, LinearAlgebra.mul!

function mul!(res::AbstractVector, op::AbstractLinearOperator{T}, v::AbstractVector, α, β) where T
  if typeof(op) <: LBFGSOperator || typeof(op) <: LSR1Operator
    args5 = false
  else
    args5 = (typeof(op) <: TimedLinearOperator) ? op.op.args5 : op.args5
  end
  args5 || (α == 1 && β == 0) || throw(LinearOperatorException("5-args product not defined"))
  (size(v, 1) == size(op, 2) && size(res, 1) == size(op, 1)) || throw(LinearOperatorException("shape mismatch"))
  increase_nprod(op)
  op.prod!(res, v, α, β)
end

function mul!(res::AbstractVector, op::AbstractLinearOperator, v::AbstractVector{T}) where T
  mul!(res, op, v, one(T), zero(T))
end

# Apply an operator to a vector.
function *(op::AbstractLinearOperator{T}, v::AbstractVector{S}) where {T, S}
  nrow, ncol = size(op)
  res = similar(v, promote_type(T, S), nrow)
  mul!(res, op, v)
  return res
end

# Unary operations.
+(op::AbstractLinearOperator) = op

function -(op::AbstractLinearOperator{T}) where T
  prod! = @closure (res, v, α, β) -> mul!(res, op, v, -α, β)
  tprod! = @closure (res, u, α, β) -> mul!(res, transpose(op), u, -α, β)
  ctprod! = @closure (res, w, α, β) -> mul!(res, adjoint(op), w, -α, β)
  LinearOperator{T}(op.nrow, op.ncol, op.symmetric, op.hermitian, prod!, tprod!, ctprod!)
end

function prod_op!(res::AbstractVector, op1::AbstractLinearOperator, op2::AbstractLinearOperator, 
                  vtmp::AbstractVector, v::AbstractVector, α, β)
  mul!(vtmp, op2, v)
  mul!(res, op1, vtmp, α, β)
end

## Operator times operator.
function *(op1::AbstractLinearOperator, op2::AbstractLinearOperator)
  T = promote_type(eltype(op1), eltype(op2))
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if m2 != n1
    throw(LinearOperatorException("shape mismatch"))
  end
  #tmp vector for products
  vtmp = zeros(T, m2)
  utmp = zeros(T, n1)
  wtmp = zeros(T, n1)
  prod! = @closure (res, v, α, β) -> prod_op!(res, op1, op2, vtmp, v, α, β)
  tprod! = @closure (res, u, α, β) -> prod_op!(res, transpose(op2), transpose(op1), utmp, u, α, β)
  ctprod! = @closure (res, w, α, β) -> prod_op!(res, adjoint(op2), adjoint(op1), wtmp, w, α, β)
  LinearOperator{T}(m1, n2, false, false, prod!, tprod!, ctprod!)
end

## Matrix times operator.
*(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) * op
*(op::AbstractLinearOperator, M::AbstractMatrix) = op * LinearOperator(M)

## Scalar times operator. (# commutation α*v ???)
function *(op::AbstractLinearOperator, x::Number)
  S = promote_type(eltype(op), typeof(x))
  prod! = @closure (res, v, α, β) -> mul!(res, op, v, x * α, β)
  tprod! = @closure (res, u, α, β) -> mul!(res, transpose(op), u, x * α, β)
  ctprod! = @closure (res, w, α, β) -> mul!(res, adjoint(op), w, x' * α, β)
  LinearOperator{S}(op.nrow, op.ncol, op.symmetric, op.hermitian && isreal(x), prod!, tprod!, ctprod!)
end

function *(x::Number, op::AbstractLinearOperator)
  return op * x
end

# Operator + operator.

function sum_prod!(res::AbstractVector, op1::AbstractLinearOperator, op2::AbstractLinearOperator{T}, v::AbstractVector, α, β) where T
  mul!(res, op1, v, α, β)
  mul!(res, op2, v, α, one(T))
end

function +(op1::AbstractLinearOperator, op2::AbstractLinearOperator)
  (m1, n1) = size(op1)
  (m2, n2) = size(op2)
  if (m1 != m2) || (n1 != n2)
    throw(LinearOperatorException("shape mismatch"))
  end
  S = promote_type(eltype(op1), eltype(op2))
  prod! = @closure (res, v, α, β) -> sum_prod!(res, op1, op2, v, α, β)
  tprod! = @closure (res, u, α, β) -> sum_prod!(res, transpose(op1), transpose(op2), u, α, β)
  ctprod! = @closure (res, w, α, β) -> sum_prod!(res, adjoint(op1), adjoint(op2), w, α, β)
  symm = (symmetric(op1) && symmetric(op2))
  herm = (hermitian(op1) && hermitian(op2))
  return LinearOperator{S}(m1, n1, symm, herm, prod!, tprod!, ctprod!)
end

# Operator + matrix.
+(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) + op
+(op::AbstractLinearOperator, M::AbstractMatrix) = op + LinearOperator(M)

# Operator .+ scalar.
+(op::AbstractLinearOperator, x::Number) = op + x * opOnes(op.nrow, op.ncol)
+(x::Number, op::AbstractLinearOperator) = x * opOnes(op.nrow, op.ncol) + op

# Operator - operator
-(op1::AbstractLinearOperator, op2::AbstractLinearOperator) = op1 + (-op2)

# Operator - matrix.
-(M::AbstractMatrix, op::AbstractLinearOperator) = LinearOperator(M) - op
-(op::AbstractLinearOperator, M::AbstractMatrix) = op - LinearOperator(M)

# Operator - scalar.
-(op::AbstractLinearOperator, x::Number) = op + (-x)
-(x::Number, op::AbstractLinearOperator) = x + (-op)
