using Symbolics

@variables t x y z

ex0 = Differential(t)(x) ~ y + z
Symbolics.solve_for(ex0, Differential(t)(x))
Symbolics.solve_for(ex0, y)

ex1 = -x + 1//2 * y - z ~ 0
ex2 = -y + 1//5 * z - x ~ 60
ex3 = -z + 1//2 * x - y ~ 10
Symbolics.solve_for([ex1, ex2, ex3], [x, y, z])


