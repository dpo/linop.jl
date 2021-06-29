function test_linop()
  (nrow, ncol) = (10, 6)
  ϵ = eps(Float64)
  rtol = sqrt(ϵ)
  A1 = simple_matrix(ComplexF64, nrow, ncol)

  @testset ExtendedTestSet "Basic operations" begin
    for op in (
      LinearOperator(A1),
      LinearOperator(A1')',
      transpose(LinearOperator(transpose(A1))),
      conj(LinearOperator(conj(A1))),
    )
      show(op)

      @testset "Data type" begin
        @test eltype(op) == eltype(A1)
        @test !isreal(op)
      end

      @testset "Size" begin
        @test(size(op) == (nrow, ncol))
        @test(shape(op) == (nrow, ncol))
        @test(size(op, 1) == nrow)
        @test(size(op, 2) == ncol)
        @test_throws LinearOperatorException size(op, 3)
        @test_throws LinearOperatorException op * ones(ncol + 1)
      end

      @testset "Boolean operators" begin
        @test(symmetric(op) == false)
        @test(hermitian(op) == false)
      end

      @testset "Full" begin
        @test(norm(A1 - Matrix(op)) <= ϵ * norm(A1))
      end

      @testset "Unary +." begin
        @test(norm(A1 - Matrix(+op)) <= ϵ * norm(A1))
      end
    end

    @testset "LinearOperator(Matrix)" begin
      A2 = simple_sparse_matrix(ComplexF64, nrow, ncol)
      for A in (A1, A2)
        op = LinearOperator(A)
        @test(op.nrow == nrow)
        @test(op.ncol == ncol)

        @test(norm(A - Matrix(op)) <= rtol * norm(A))
        @test(norm(transpose(A) - Matrix(transpose(op))) <= rtol * norm(A))
        @test(norm(A' - Matrix(op')) <= rtol * norm(A))

        v = simple_vector(ComplexF64, ncol)
        @test(norm(A * v - op * v) <= rtol * norm(v))

        u = simple_vector(ComplexF64, nrow)
        @test(norm(transpose(A) * u - transpose(op) * u) <= rtol * norm(u))
        @test(norm(A' * u - op' * u) <= rtol * norm(u))
      end

      A3 = Hermitian(A2' * A2)
      op3 = LinearOperator(A3)
      nrow3, ncol3 = size(A3)
      @test(op3.nrow == op3.ncol == nrow3)
      v3 = simple_vector(ComplexF64, ncol3)
      @test(norm(A3 * v3 - op3 * v3) <= rtol * norm(v3))

      A3 = Symmetric(A2' * A2)
      op3 = LinearOperator(A3)
      nrow3, ncol3 = size(A3)
      @test(op3.nrow == op3.ncol == nrow3)
      v3 = simple_vector(ComplexF64, ncol3)
      @test(norm(A3 * v3 - op3 * v3) <= rtol * norm(v3))

      A4 = SymTridiagonal(A3)
      op4 = LinearOperator(A4)
      nrow4, ncol4 = size(A4)
      @test(op4.nrow == op4.ncol == nrow4)
      v4 = simple_vector(ComplexF64, ncol4)
      @test(norm(A4 * v4 - op4 * v4) <= rtol * norm(v4))
    end

    @testset "Constructor with specified structure" begin
      v = simple_vector(Float64, nrow)
      A = simple_matrix(ComplexF64, nrow, nrow)
      A = A + A'
      A = A + transpose(A)
      op = LinearOperator(A, symmetric = true, hermitian = true)
      @test norm(op * v - A * v) <= rtol * norm(A)
      @test norm(transpose(op) * v - transpose(A) * v) <= rtol * norm(A)
      @test norm(op' * v - A' * v) <= rtol * norm(A)

      A = simple_matrix(ComplexF64, nrow, nrow)
      A = A + A' # Hermitian
      op = LinearOperator(A, symmetric = false, hermitian = true)
      @test norm(op * v - A * v) <= rtol * norm(A)
      @test norm(transpose(op) * v - transpose(A) * v) <= rtol * norm(A)
      @test norm(op' * v - A' * v) <= rtol * norm(A)

      A = simple_matrix(ComplexF64, nrow, nrow)
      A = A + transpose(A) # Symmetric
      op = LinearOperator(A, symmetric = true, hermitian = false)
      @test norm(op * v - A * v) <= rtol * norm(A)
      @test norm(transpose(op) * v - transpose(A) * v) <= rtol * norm(A)
      @test norm(op' * v - A' * v) <= rtol * norm(A)
    end

    @testset "Basic arithmetic operations" begin
      B1 = simple_matrix(ComplexF64, nrow, ncol)

      for q in (+, -)
        C = q(A1, B1)
        opC = q(LinearOperator(A1), LinearOperator(B1))
        v = simple_vector(ComplexF64, ncol)
        @test(norm(opC * v - C * v) <= rtol * norm(v))
        u = simple_vector(ComplexF64, nrow)
        @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u))
        @test(norm(opC' * u - C' * u) <= rtol * norm(u))

        opC = q(A1, LinearOperator(B1))
        @test(norm(opC * v - C * v) <= rtol * norm(v))
        @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u))
        @test(norm(opC' * u - C' * u) <= rtol * norm(u))

        opC = q(LinearOperator(A1), B1)
        @test(norm(opC * v - C * v) <= rtol * norm(v))
        @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u))
        @test(norm(opC' * u - C' * u) <= rtol * norm(u))
      end
    end

    B2 = simple_matrix(ComplexF64, ncol, ncol + 1)
    @testset "Operator ± scalar" begin
      opC = LinearOperator(A1) + 2.12345
      @test(norm(A1 .+ 2.12345 - Matrix(opC)) <= rtol * norm(A1 .+ 2.12345))

      opC = 2.12345 + LinearOperator(A1)
      @test(norm(A1 .+ 2.12345 - Matrix(opC)) <= rtol * norm(A1 .+ 2.12345))

      opC = LinearOperator(A1) - 2.12345
      @test(norm((A1 .- 2.12345) - Matrix(opC)) <= rtol * norm(A1 .- 2.12345))

      opC = 2.12345 - LinearOperator(A1)
      @test(norm((2.12345 .- A1) - Matrix(opC)) <= rtol * norm(2.12345 .- A1))
    end

    @testset "Operator × Operator" begin
      A4 = simple_matrix(ComplexF64, ncol, ncol)
      B4 = simple_matrix(ComplexF64, ncol, ncol)
      for Ai in (A4, transpose(A4), adjoint(A4), conj(A4)) 
        for Bi in (B4, transpose(B4), adjoint(B4), conj(B4))
          C = Ai * Bi
          opC = LinearOperator(Ai) * LinearOperator(Bi)
          v = simple_vector(ComplexF64, ncol)
          @test(norm(opC * v - C * v) <= rtol * norm(v))
          u = simple_vector(ComplexF64, ncol)
          @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u))
          @test(norm(opC' * u - C' * u) <= rtol * norm(u))
        end
      end

      C = A1 * B2
      opC = LinearOperator(A1) * LinearOperator(B2)
      v = simple_vector(ComplexF64, ncol + 1)
      @test(norm(opC * v - C * v) <= rtol * norm(v))
      u = simple_vector(ComplexF64, nrow)
      @test(norm(transpose(opC) * u - transpose(C) * u) <= rtol * norm(u))
      @test(norm(opC' * u - C' * u) <= rtol * norm(u))

      @test_throws LinearOperatorException LinearOperator(A1) + LinearOperator(B2)
      @test_throws LinearOperatorException LinearOperator(B2) * LinearOperator(A1)
    end

    A1B2 = A1 * B2
    @testset "Matrix × operator" begin
      opC = A1 * LinearOperator(B2)
      @test(norm(A1B2 - Matrix(opC)) <= rtol * norm(A1B2))
    end

    @testset "Operator × matrix" begin
      opC = LinearOperator(A1) * B2
      @test(norm(A1B2 - Matrix(opC)) <= rtol * norm(A1B2))
    end

    AA1 = 2.12345 * A1
    @testset "Scalar × operator" begin
      opC = 2.12345 * LinearOperator(A1)
      @test(norm(AA1 - Matrix(opC)) <= rtol * norm(AA1))
    end

    @testset "Operator × scalar" begin
      opC = LinearOperator(A1) * 2.12345
      @test(norm(AA1 - Matrix(opC)) <= rtol * norm(AA1))
    end
  end

  @testset ExtendedTestSet "Basic operators" begin
    @testset "Identity" begin
      opI = opEye(nrow)
      opI2 = opEye(nrow, nrow) 
      v = simple_vector(ComplexF64, nrow)
      @test(abs(norm(opI * v - v)) <= ϵ * norm(v))
      @test(abs(norm(opI2 * v - v)) <= ϵ * norm(v))
      @test(abs(norm(transpose(opI) * v - v)) <= ϵ * norm(v))
      @test(abs(norm(opI' * v - v)) <= ϵ * norm(v))
      @test(norm(Matrix(opI) - Matrix(1.0I, nrow, nrow)) <= ϵ * norm(Matrix(1.0I, nrow, nrow)))

      w = opI * v
      w[1] = -1.0
      @test v[1] != w[1]

      opI = opEye(nrow, ncol)
      v = simple_vector(ComplexF64, ncol)
      v0 = [v; zeros(nrow - ncol)]
      vu = [v; ones(nrow - ncol)]
      @test(abs(norm(opI * v - v0)) <= ϵ * norm(v))
      @test(abs(norm(transpose(opI) * vu - v)) <= ϵ * norm(v))
      @test(abs(norm(opI' * vu - v)) <= ϵ * norm(v))
      @test(norm(Matrix(opI) - Matrix(1.0I, nrow, ncol)) <= ϵ * norm(Matrix(1.0I, nrow, ncol)))

      opI = opEye(ncol, nrow)
      @test(abs(norm(opI * vu - v)) <= ϵ * norm(v))
      @test(abs(norm(transpose(opI) * v - v0)) <= ϵ * norm(v))
      @test(abs(norm(opI' * v - v0)) <= ϵ * norm(v))
      @test(norm(Matrix(opI) - Matrix(1.0I, ncol, nrow)) <= ϵ * norm(Matrix(1.0I, ncol, nrow)))
    end

    @testset "Identity (non-convertible to matrix)" begin
      op = opEye()
      show(op)

      v = simple_vector(Float64, 5)
      w = op * v
      @test w === v
      w = v * op

      @test w === v
      A2 = op * A1
      @test A2 === A1
      A2 = A1 * op
      @test A2 === A1

      T1 = LinearOperator(A1)
      T2 = op * T1
      @test T2 === T1
      T2 = T1 * op
      @test T2 === T1

      op2 = opEye()
      @test op === op2
      @test op === op * op2 === op2 * op
    end

    @testset "Ones" begin
      E = opOnes(nrow, ncol)
      v = simple_vector(ComplexF64, nrow)
      u = simple_vector(ComplexF64, ncol)
      @test(norm(E * u - sum(u) * ones(nrow)) <= rtol * norm(u))
      @test(norm(transpose(E) * v - sum(v) * ones(ncol)) <= rtol * norm(v))
      @test(norm(E' * v - sum(v) * ones(ncol)) <= rtol * norm(v))
    end

    @testset "Zeros" begin
      O = opZeros(nrow, ncol)
      v = simple_vector(ComplexF64, nrow)
      u = simple_vector(ComplexF64, ncol)
      @test(norm(O * u) <= ϵ)
      @test(norm(transpose(O) * v) <= ϵ)
      @test(norm(O' * v) <= ϵ)
    end

    @testset "Diagonal" begin
      v = simple_vector(ComplexF64, nrow)
      D = opDiagonal(v)
      u = simple_vector(ComplexF64, nrow)
      @test(norm(D * u - v .* u) <= ϵ * norm(u))
      @test(norm(transpose(D) * u - v .* u) <= ϵ * norm(u))
      @test(norm(D' * u - conj(v) .* u) <= ϵ * norm(u))
      res = simple_vector(ComplexF64, nrow)
      res2 = v .* u .* 2.0 + 2.0 .* res
      mul!(res, D, u, 2.0, 2.0)
      @test(norm(res - res2) <= ϵ * norm(u))
    end

    @testset "Rectangular diagonal" begin
      nmin = min(nrow, ncol)
      nmax = max(nrow, ncol)
      A = zeros(ComplexF64, nmax, nmin)
      v = simple_vector(ComplexF64, nmin)
      for i = 1:nmin
        A[i, i] = v[i]
      end
      D = opDiagonal(nmax, nmin, v)
      u = simple_vector(ComplexF64, nmin)
      @test(norm(A * u - D * u) <= ϵ * norm(u))
      w = simple_vector(ComplexF64, nmax)
      @test(norm(transpose(A) * w - transpose(D) * w) <= ϵ * norm(w))
      @test(norm(A' * w - D' * w) <= ϵ * norm(w))

      A = zeros(ComplexF64, nmin, nmax)
      for i = 1:nmin
        A[i, i] = v[i]
      end
      D = opDiagonal(nmin, nmax, v)
      @test(norm(A * w - D * w) <= ϵ * norm(w))
      @test(norm(transpose(A) * u - transpose(D) * u) <= ϵ * norm(u))
      @test(norm(A' * u - D' * u) <= ϵ * norm(u))
    end

    @testset "posdef" begin
      v = simple_vector(ComplexF64, nrow)
      H = opHouseholder(v)
      Λ = collect(eltype(v), 1:nrow)
      op = H * opDiagonal(Λ) * H'
      @test check_positive_definite(op)
      @test check_positive_definite(op, semi = true)
      Λ = collect(eltype(v), 0:(nrow - 1))
      op = H * opDiagonal(Λ) * H'
      @test check_positive_definite(op, semi = true)
      A = Matrix(op)
      @test check_positive_definite(A, semi = true)
    end

    @testset "Hermitian" begin
      A = simple_matrix(ComplexF64, nrow, nrow)
      d = real.(diag(A))
      A = tril(A, -1)
      C = A + A' + diagm(0 => d)
      H = opHermitian(d, A)
      v = simple_vector(ComplexF64, nrow)
      @test(norm(H * v - C * v) <= rtol * norm(v))
      @test(norm(transpose(H) * v - transpose(C) * v) <= rtol * norm(v))
      @test(norm(H' * v - C * v) <= rtol * norm(v))

      @test(!check_hermitian(LinearOperator(A - A')))
      @test(!check_positive_definite(LinearOperator(-A' * A)))

      C = simple_matrix(ComplexF64, nrow, nrow, symmetric = true)
      H = opHermitian(C)
      v = simple_vector(ComplexF64, nrow)
      @test(norm(H * v - C * v) <= rtol * norm(v))
      @test(norm(transpose(H) * v - transpose(C) * v) <= rtol * norm(v))
      @test(norm(H' * v - C * v) <= rtol * norm(v))
    end

    @testset "Transpose and adjoint" begin
      A = simple_matrix(ComplexF64, nrow, nrow)
      v = simple_vector(ComplexF64, nrow)

      op = LinearOperator(ComplexF64, nrow, nrow, false, false, 
        (res, v, α, β) -> mul!(res, A, v, α, β), 
        (res, v, α, β) -> mul!(res, transpose(A), v, α, β), 
        nothing)
      @test(norm(transpose(A) * v - transpose(op) * v) <= rtol * norm(v))
      @test(norm(adjoint(A) * v - adjoint(op) * v) <= rtol * norm(v))
      @test(norm(A * v - transpose(transpose(op)) * v) <= rtol * norm(v))
      @test(norm(A * v - adjoint(adjoint(op)) * v) <= rtol * norm(v))
      @test(norm(conj.(A) * v - transpose(adjoint(op)) * v) <= rtol * norm(v))
      @test(norm(conj.(A) * v - adjoint(transpose(op)) * v) <= rtol * norm(v))

      op = LinearOperator(ComplexF64, nrow, nrow, false, false, 
        (res, v, α, β) -> mul!(res, A, v, α, β), 
        nothing, 
        (res, v, α, β) -> mul!(res, adjoint(A), v, α, β))
      @test(norm(transpose(A) * v - transpose(op) * v) <= rtol * norm(v))
      @test(norm(adjoint(A) * v - adjoint(op) * v) <= rtol * norm(v))
      @test(norm(A * v - transpose(transpose(op)) * v) <= rtol * norm(v))
      @test(norm(A * v - adjoint(adjoint(op)) * v) <= rtol * norm(v))
      @test(norm(conj.(A) * v - transpose(adjoint(op)) * v) <= rtol * norm(v))
      @test(norm(conj.(A) * v - adjoint(transpose(op)) * v) <= rtol * norm(v))
    end

    @testset "Integer" begin
      A = round.(Int, simple_matrix(Float64, nrow, nrow))
      op = LinearOperator(A)
      @test check_ctranspose(op)
      @test check_hermitian(op + op')
      @test check_positive_definite(op * op')
    end

    @testset "Restriction and Extension" begin
      n = 10
      J = [1; 2; 4; 7]
      r = 3:6
      s = 1:2:7
      k = 4
      v = simple_vector(Float64, nrow)

      for idx in (J, r, s, Colon(), k)
        P = opRestriction(idx, n)
        Z = opExtension(idx, n)

        # 1d slices are different; in Julia, v[idx] is a scalar
        w = v[idx]
        typeof(idx) <: Number && (w = [w])
        vz = zeros(n)
        vz[idx] = v[idx]

        @test P * v == w
        @test P' * w == vz
        @test Z * w == vz
        @test Z' * v == w
        @test (P * Z) * w == w
        @test (Z * P) * v == vz
      end

      A = rand(n, n)
      opA = LinearOperator(A)
      v = simple_vector(Float64, 2)
      @test norm(A[[3,4], [5,6]] * v - opA[[3,4], [5,6]] * v) ≤ rtol * norm(v)
    end
  end

  @testset ExtendedTestSet "Linear system operators" begin
    A = simple_matrix(ComplexF64, nrow, nrow)
    v = simple_vector(Float64, nrow)

    @testset "Inverse" begin
      Ainv = opInverse(A)
      @test(norm(A \ v - Ainv * v) <= rtol * norm(v))
      @test(norm(transpose(A) \ v - transpose(Ainv) * v) <= rtol * norm(v))
      @test(norm(A' \ v - Ainv' * v) <= rtol * norm(v))
    end

    @testset "Cholesky and LDL" begin
      B = A' * A
      Binv = opCholesky(B)  #, check=true);
      @test(norm(B \ v - Binv * v) <= rtol * norm(v))
      @test(norm(transpose(B) \ v - transpose(Binv) * v) <= rtol * norm(v))
      @test(norm(B' \ v - Binv' * v) <= rtol * norm(v))

      @test_throws LinearOperatorException opCholesky(simple_matrix(Float64, 3, 5))
      @test_throws LinearOperatorException opCholesky(simple_matrix(Float64, 5, 5), check = true)

      # Test Cholesky operator on SQD matrix.
      A = simple_matrix(Float64, 3, 3, symmetric = true)
      B = simple_matrix(Float64, 2, 3)
      C = simple_matrix(Float64, 2, 2, symmetric = true)
      K = Symmetric([A B'; B -C])

      # Dense Cholesky should throw an exception.
      @test_throws LinearAlgebra.PosDefException opCholesky(K)

      # Compute the LDL' factorization.
      LDL = opLDL(sparse(K))
      e = ones(size(K, 1))
      @test(norm(LDL * (K * e) - e) < rtol * norm(e))
    end

    @testset "Householder" begin
      v = simple_vector(ComplexF64, nrow)
      H = opHouseholder(v)
      u = simple_vector(ComplexF64, nrow)
      @test(norm(H * u - (u - 2 * dot(v, u) * v)) <= rtol * norm(u))
      @test(norm(transpose(H) * u - (u - 2 * dot(conj(v), u) * conj(v))) <= rtol * norm(u))
      @test(norm(H' * u - (u - 2 * dot(v, u) * v)) <= rtol * norm(u))
    end
  end

  @testset ExtendedTestSet "Inference" begin
    function test_func(res)
      res .= 1.0 .+ im * 1.0
    end
    op = LinearOperator(ComplexF64, 5, 3, false, false, (res, p, α, β) -> test_func(res))
    @test eltype(op) == ComplexF64
    v = rand(5)
    @test_throws LinearOperatorException transpose(op) * v  # cannot be inferred
    @test_throws LinearOperatorException op' * v            # cannot be inferred

    op2 = conj(op)
    @test(norm(Matrix(op2) - conj(Matrix(op))) <= ϵ * norm(Matrix(op)))

    A = simple_matrix(ComplexF64, 5, 3)
    op = LinearOperator(A)
    @test(check_ctranspose(A))
    @test(check_ctranspose(op))
    @test_throws LinearOperatorException opCholesky(A)  # Shape mismatch

    A = simple_matrix(ComplexF64, 5, 5)
    @test_throws LinearOperatorException opCholesky(A, check = true)  # Not Hermitian / positive definite
    @test_throws LinearOperatorException opCholesky(-A' * A, check = true)  # Not positive definite

    # Adjoint of a symmetric non-hermitian
    A = simple_matrix(ComplexF64, 3, 3)
    A = A + transpose(A)
    op = LinearOperator(ComplexF64, 3, 3, true, false, (res, v, α, β) -> mul!(res, A, v))
    v = rand(3)
    @test op' * v ≈ A' * v
  end

  @testset ExtendedTestSet "Type specific operator" begin
    function prod!(res, v, α, β) 
      res[1] = v[1] + v[2] 
      res[2] = v[2]
    end
    function ctprod!(res, v, α, β) 
      res[1] = v[1] 
      res[2] = v[1] + v[2]
    end
    for T in (Complex{Float64}, Complex{Float32}, BigFloat, Float64, Float32, Float16, Int32)
      op = LinearOperator(T, 2, 2, false, false, prod!, nothing, ctprod!)
      w = ones(T, 2)
      @test eltype(op) == T
      @test op * w == T[2; 1]
      @test eltype(op * w) == T
    end

    A = [im 1.0; 0.0 1.0]
    function prod2!(res, v, α, β) 
      mul!(res, A, v)
    end
    function tprod2!(res, u, α, β)
      mul!(res, transpose(A), u)
    end
    function ctprod2!(res, w, α, β)
      mul!(res, A', w)
    end
    opC = LinearOperator(ComplexF64, 2, 2, false, false, prod2!, tprod2!, ctprod2!)
    v = simple_vector(ComplexF64, 2)
    @test A == Matrix(opC)
    opF = LinearOperator(Float64, 2, 2, false, false, prod2!, tprod2!, ctprod2!) # The type is a lie
    @test eltype(opF) == Float64
    @test_throws InexactError Matrix(opF) # changed here TypeError to InexactError
  end

  # Issue #80
  @testset ExtendedTestSet "Test mul!" begin
    A = [1.0 1.0; 1.0 0.0]
    op = LinearOperator(A)
    y = zeros(2)
    x = ones(2)
    mul!(y, op, x)
    @test y == [2.0; 1.0]
  end

  # Issue #107
  @testset ExtendedTestSet "Unary and scalar operations on Adjoint and Transpose operators" begin
    op = LinearOperator(rand(5, 3))
    for adjtrans in [adjoint, transpose]
      @test Matrix(adjtrans(-op)) == Matrix(-adjtrans(op))
      @test Matrix(adjtrans(2 * op)) == Matrix(2 * adjtrans(op))
    end
  end

  # Issue #109
  @testset ExtendedTestSet "Sum with Adjoint and Transpose" begin
    A = rand(3, 3) + im * rand(3, 3)
    opA = LinearOperator(A)
    for adjtrans in [adjoint, transpose]
      @test Matrix(adjtrans(opA) + opA) == Matrix(opA + adjtrans(opA)) == A + adjtrans(A)
      @test Matrix(adjtrans(opA) + A) == Matrix(A + adjtrans(opA)) == A + adjtrans(A)
    end
  end

  # Issue #109
  @testset ExtendedTestSet "Cat with Adjoint and Transpose" begin
    A = rand(3, 3) + im * rand(3, 3)
    opA = LinearOperator(A)
    for adjtrans in [adjoint, transpose]
      @test Matrix([adjtrans(opA) opA]) == Matrix([adjtrans(A) A])
      @test Matrix([adjtrans(opA); opA]) == Matrix([adjtrans(A); A])
      @test Matrix([adjtrans(opA); adjtrans(opA)]) == Matrix([adjtrans(A); adjtrans(A)])
      @test Matrix([opA adjtrans(opA)]) == Matrix([A adjtrans(A)])
      @test Matrix([opA; adjtrans(opA)]) == Matrix([A; adjtrans(A)])
      @test Matrix([adjtrans(opA) A]) == Matrix([adjtrans(A) A])
      @test Matrix([adjtrans(opA); A]) == Matrix([adjtrans(A); A])
      @test Matrix([A adjtrans(opA)]) == Matrix([A adjtrans(A)])
      @test Matrix([A; adjtrans(opA)]) == Matrix([A; adjtrans(A)])
      @test Matrix([adjtrans(opA) opA; opA adjtrans(opA)]) == Matrix([adjtrans(A) A; A adjtrans(A)])
    end
  end

  @testset ExtendedTestSet "Counters" begin
    op = LinearOperator(rand(3, 4) + im * rand(3, 4))
    @test nprod(op) == 0
    @test ntprod(op) == 0
    @test nctprod(op) == 0
    nprods = 5
    ntprods = 4
    nctprods = 7
    for _ = 1:nprods
      op * rand(4)
    end
    for _ = 1:ntprods
      transpose(op) * rand(3)
    end
    for _ = 1:nctprods
      op' * rand(3)
    end
    @test nprod(op) == nprods
    @test ntprod(op) == ntprods
    @test nctprod(op) == nctprods
    for _ = 1:nprods
      conj(op) * rand(4)
    end
    @test nprod(op) == 2 * nprods

    opᵀ = transpose(op)
    @test nprod(opᵀ) == ntprod(op)
    @test ntprod(opᵀ) == nprod(op)
    @test nctprod(opᵀ) == nprod(op)

    opᴴ = op'
    @test nprod(opᴴ) == nctprod(op)
    @test ntprod(opᴴ) == nprod(op)
    @test nctprod(opᴴ) == nprod(op)

    reset!(op)
    @test nprod(op) == 0
    @test ntprod(op) == 0
    @test nctprod(op) == 0
  end

  @testset ExtendedTestSet "Timers" begin
    op = LinearOperator(rand(3, 4) + im * rand(3, 4))
    top = TimedLinearOperator(op)
    show(top)
    nprods = 5
    ntprods = 4
    nctprods = 7
    for _ = 1:nprods
      op * rand(4)
    end
    for _ = 1:nprods
      conj(op) * rand(4)
    end
    for _ = 1:ntprods
      transpose(op) * rand(3)
    end
    for _ = 1:nctprods
      op' * rand(3)
    end
    for fn ∈ (
      :size,
      :shape,
      :symmetric,
      :issymmetric,
      :hermitian,
      :ishermitian,
      :nprod,
      :ntprod,
      :nctprod,
    )
      @eval begin
        @test $fn($top) == $fn($top.op)
      end
    end

    reset!(op)
    reset!(top)

    top2 = TimedLinearOperator(op')  # the same as top'
    nrow, ncol = size(op)
    u = rand(nrow) + im * rand(nrow)
    @test all(top' * u .== top2 * u)
    v = rand(ncol) + im * rand(ncol)
    @test all(top * v .== top2' * v)

    top3 = TimedLinearOperator(transpose(op))  # the same as transpose(top)
    nrow, ncol = size(op)
    u = rand(nrow) + im * rand(nrow)
    @test all(transpose(top) * u .== top3 * u)
    v = rand(ncol) + im * rand(ncol)
    @test all(top * v .== transpose(top3) * v)
  end

  @testset ExtendedTestSet "BlockDiagonal" begin
    A = rand(3, 4) + im * rand(3, 4)
    B = rand(3, 3) + im * rand(3, 3)
    C = rand(4, 2) + im * rand(4, 2)
    D = [
      A zeros(3, 3) zeros(3, 2)
      zeros(3, 4) B zeros(3, 2)
      zeros(4, 4) zeros(4, 3) C
    ]
    M = BlockDiagonalOperator(LinearOperator.((A, B, C))...)
    @test size(M, 1) == size(A, 1) + size(B, 1) + size(C, 1)
    @test size(M, 2) == size(A, 2) + size(B, 2) + size(C, 2)
    @test norm(Matrix(M) - D) ≤ sqrt(eps()) * norm(D)
    @test norm(Matrix(transpose(M)) - transpose(D)) ≤ sqrt(eps()) * norm(D)
    @test norm(Matrix(M') - D') ≤ sqrt(eps()) * norm(D)
  end

  # Issue #139
  @testset ExtendedTestSet "Matrix-vector products with SparseMatrix and SparseVector" begin
    A = sprand(10, 10, 0.2)
    b = sprand(10, 0.2)
    opA = LinearOperator(A)
    @test opA * b == A * b
    @test transpose(opA) * b == transpose(A) * b
    @test adjoint(opA) * b == adjoint(A) * b
  end

  @testset ExtendedTestSet "3-args" begin
    A = rand(10, 10)
    b = rand(10)
    prod! = (res, v) -> mul!(res, A, v)
    tprod! = (res, v) -> mul!(res, A', v)
    opA = LinearOperator_args3(Float64, 10, 10, false, false, prod!, tprod!)
    @test opA * b == A * b
    res = rand(10)
    mul!(res, opA, b)
    @test res == A * b
    α, β = 1.0, 3.0
    @test_throws LinearOperatorException mul!(res, opA, b, α, β)
    mul!(res, opA', b)
    @test res == A' * b
  end
end

test_linop()
