using RobustLeastSquares
using BaseTestNext

Logging.configure(level=Logging.DEBUG) # Useful to check convergence

# Test solve()
@testset "RobustLeastSqures" begin

    @testset "Solvers" begin
        s1 = 20
        s2 = 10

        rng = srand(0)
        A = rand(rng,s1,s2)
        b = rand(rng,s1)
        weights = 1 + 0.2(rand(rng,s1)-0.5)

        sol_qr = RobustLeastSquares.solve(A,b,weights,:qr)
        sol_normal = RobustLeastSquares.solve(A,b,weights,:normal)
        sol_cg = RobustLeastSquares.solve(A,b,weights,:cg)

        @test norm(sol_qr - sol_normal) < 1e-10
        @test norm(sol_qr - sol_cg) < 1e-10
    end

    # Test the MEstimators
    @testset "MEstimators" begin
        width = 2.0
        r = [5.0]

        s1 = 20
        s2 = 10

        rng = srand(0)
        A = rand(rng,s1,s2)
        b = rand(rng,s1)
        weights = 1 + 0.2(rand(rng,s1)-0.5)

        @test RobustLeastSquares.estimator_rho(r,L2Estimator())        ≈ [12.5]
        @test RobustLeastSquares.estimator_psi(r,L2Estimator())        ≈ [5.0]
        @test RobustLeastSquares.estimator_weight(r,L2Estimator())     ≈ [1.0]
        @test RobustLeastSquares.estimator_sqrtweight(r,L2Estimator()) ≈ [1.0]

        @test RobustLeastSquares.estimator_rho(r,L1Estimator())        ≈ [5.0]
        @test RobustLeastSquares.estimator_psi(r,L1Estimator())        ≈ [1.0]
        @test RobustLeastSquares.estimator_weight(r,L1Estimator())     ≈ [0.20]
        @test RobustLeastSquares.estimator_sqrtweight(r,L1Estimator()) ≈ [0.4472135954999579]

        # problem here: width not implemented correctly for L1L2Estimator
        @test RobustLeastSquares.estimator_rho(r,L1L2Estimator(1.0))        ≈ [5.3484692283495345]
        @test RobustLeastSquares.estimator_psi(r,L1L2Estimator(1.0))        ≈ [1.3608276348795434]
        @test RobustLeastSquares.estimator_weight(r,L1L2Estimator(1.0))     ≈ [0.2721655269759087]
        @test RobustLeastSquares.estimator_sqrtweight(r,L1L2Estimator(1.0)) ≈ [0.5216948600244291]

        @test RobustLeastSquares.estimator_rho(r,HuberEstimator(width))        ≈ [8.0]
        @test RobustLeastSquares.estimator_psi(r,HuberEstimator(width))        ≈ [2.0]
        @test RobustLeastSquares.estimator_weight(r,HuberEstimator(width))     ≈ [0.4]
        @test RobustLeastSquares.estimator_sqrtweight(r,HuberEstimator(width)) ≈ [0.6324555320336759]

        @test RobustLeastSquares.estimator_rho(r,FairEstimator(width))        ≈ [4.988948126018528]
        @test RobustLeastSquares.estimator_psi(r,FairEstimator(width))        ≈ [1.4285714285714286]
        @test RobustLeastSquares.estimator_weight(r,FairEstimator(width))     ≈ [0.2857142857142857]
        @test RobustLeastSquares.estimator_sqrtweight(r,FairEstimator(width)) ≈ [0.5345224838248488]

        @test RobustLeastSquares.estimator_rho(r,CauchyEstimator(width))        ≈ [3.9620029377331667]
        @test RobustLeastSquares.estimator_psi(r,CauchyEstimator(width))        ≈ [0.6896551724137931]
        @test RobustLeastSquares.estimator_weight(r,CauchyEstimator(width))     ≈ [0.13793103448275862]
        @test RobustLeastSquares.estimator_sqrtweight(r,CauchyEstimator(width)) ≈ [0.3713906763541037]

        @test RobustLeastSquares.estimator_rho(r,GemanEstimator())        ≈ [0.4807692307692308]
        @test RobustLeastSquares.estimator_psi(r,GemanEstimator())        ≈ [0.0073964497041420114]
        @test RobustLeastSquares.estimator_weight(r,GemanEstimator())     ≈ [0.0014792899408284023]
        @test RobustLeastSquares.estimator_sqrtweight(r,GemanEstimator()) ≈ [0.038461538461538464]

        @test RobustLeastSquares.estimator_rho(r,WelschEstimator(width))        ≈ [1.9961390917275446]
        @test RobustLeastSquares.estimator_psi(r,WelschEstimator(width))        ≈ [0.009652270681138546]
        @test RobustLeastSquares.estimator_weight(r,WelschEstimator(width))     ≈ [0.0019304541362277093]
        @test RobustLeastSquares.estimator_sqrtweight(r,WelschEstimator(width)) ≈ [0.04393693362340742]

        @test RobustLeastSquares.estimator_rho(r,TukeyEstimator(width))        ≈ [0.6666666666666666]
        @test RobustLeastSquares.estimator_psi(r,TukeyEstimator(width))        ≈ [0.0]
        @test RobustLeastSquares.estimator_weight(r,TukeyEstimator(width))     ≈ [0.0]
        @test RobustLeastSquares.estimator_sqrtweight(r,TukeyEstimator(width)) ≈ [0.0]

        @test RobustLeastSquares.estimator_sqrtweight(b,MultiEstimator(CauchyEstimator(width)=>1:10))[1:10] ≈ RobustLeastSquares.estimator_sqrtweight(b[1:10],CauchyEstimator(width))
        @test RobustLeastSquares.estimator_sqrtweight(b,MultiEstimator(CauchyEstimator(width)=>1:10))[11:20] ≈ RobustLeastSquares.estimator_sqrtweight(b[11:20],L2Estimator(width))
    end

    @testset "Reweighted least squares" begin
        width = 2.0
        r = [5.0]

        s1 = 20
        s2 = 10

        rng = srand(0)
        A = rand(rng,s1,s2)
        b = rand(rng,s1)
        weights = 1 + 0.2(rand(rng,s1)-0.5)

        # Test the overall solver works...
        est = MultiEstimator(CauchyEstimator(width)=>1:10)
        sol2_qr, res, weights = reweighted_lsqr(A,b,est;method=:qr, n_iter=30, refit = false)
        sol2_normal, res, weights = reweighted_lsqr(A,b,est;method=:normal, n_iter=30, refit = false)
        sol2_cg, res, weights = reweighted_lsqr(A,b,est;method=:cg, n_iter=30, refit = false)

        @test norm(sol2_qr - sol2_normal) < 1e-7
        @test norm(sol2_qr - sol2_cg) < 1e-7

        sol3_qr, res, weights = reweighted_lsqr(A,b,est;method=:qr, n_iter=30, refit = true)
        sol3_normal, res, weights = reweighted_lsqr(A,b,est;method=:normal, n_iter=30, refit = true)
        sol3_cg, res, weights = reweighted_lsqr(A,b,est;method=:cg, n_iter=30, refit = true)

        @test norm(sol3_qr - sol3_normal) < 1e-7
        @test norm(sol3_qr - sol3_cg) < 1e-7
    end
end
