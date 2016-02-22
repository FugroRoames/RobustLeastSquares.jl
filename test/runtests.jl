using RobustLeastSquares
using Base.Test

s1 = 20
s2 = 10

rng = srand(0)
A = rand(rng,s1,s2)
b = rand(rng,s1)
weights = 1 + 0.2(rand(rng,s1)-0.5)

sol_qr = RobustLeastSquares.solve(A,b,weights,:qr)
sol_normal = RobustLeastSquares.solve(A,b,weights,:normal)
sol_cg = RobustLeastSquares.solve(A,b,weights,:cg)

println(sol_qr)
println(sol_normal)
println(sol_cg)

@test norm(sol_qr - sol_normal) < 1e-10
@test norm(sol_qr - sol_cg) < 1e-10
