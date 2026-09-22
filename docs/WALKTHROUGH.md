# ODRBINDy.jl — implementation walkthrough

How the method in [the paper](https://doi.org/10.1145/3831701) becomes the code
in `src/`. Part 1 derives the algorithm from scratch; Part 2 goes through each
source file and shows which equation it implements.

Read it with the source open alongside. If you only want to *use* the package,
the [README](../README.md) is enough — this document is for understanding or
modifying the internals.

For the correspondence with the reference MATLAB implementation, see
[PORTING.md](PORTING.md).

---

## Part 1 — The theory, from scratch

### 1.1 The problem

You have measured a system over time. You believe it obeys some differential
equation you don't know. You want to find that equation.

SINDy's idea: assume the equation is a **sparse combination of simple candidate
functions**. Write

$$\dot{\mathbf{x}} = \Theta(\mathbf{x})\,\Xi \tag{paper eq. 1}$$

where:

- $\mathbf{x}(t) \in \mathbb{R}^D$ is the state ($D=3$ for Lorenz: $x, y, z$);
- $\Theta(\mathbf{x}) \in \mathbb{R}^{1 \times M}$ is a **dictionary** of $M$
  candidate functions evaluated at $\mathbf{x}$ — for us, all monomials up to
  order 2: $[1, x, y, z, x^2, xy, xz, y^2, yz, z^2]$, so $M = 10$;
- $\Xi \in \mathbb{R}^{M \times D}$ holds the coefficients.

The Lorenz system $\dot{x} = \sigma(y-x)$, $\dot{y} = x(\rho - z) - y$,
$\dot{z} = xy - \beta z$ in this form is a $10 \times 3$ matrix with only 7
non-zero entries. *Finding the equation = finding which entries of $\Xi$ are
non-zero, and what they equal.*

### 1.2 Why plain SINDy struggles

Stack up all $N$ measurements into $\hat{X} \in \mathbb{R}^{N\times D}$, estimate
the derivatives $\dot{X}$ somehow, and solve $\dot{X} = \Theta(\hat X)\Xi$ by
least squares. Least squares assumes all the error is in the *left-hand side*
(the thing you're predicting, the **regressand**).

But your noise is in the measurements $\hat X$, which appear on the *right-hand
side*, inside $\Theta$ (the **regressors**). This is the **error-in-variables
problem**. Fig. 1(a) in the paper draws it: SINDy puts the error bars in the
$\dot{x}$ direction, which is the wrong direction.

Worse, $\Theta$ is nonlinear. Noise passed through $x \mapsto x^2$ doesn't stay
Gaussian, so the least-squares assumptions break down further as noise grows.

### 1.3 The ODR-BINDy fix

Treat the true, denoised states $X$ as **unknowns to be solved for**, alongside
$\Xi$. That's the "orthogonal distance regression" idea: instead of measuring
error vertically down to the curve, measure it perpendicular-ish — allowing the
fitted point to move in *both* directions. Fig. 1(c).

Now discretise. To compare $\dot X$ with $\Theta(X)\Xi$ we need two linear
operators:

- $L_{\partial t}$ — takes states, returns estimated time derivatives;
- $L_I$ — takes anything defined at the sample points, returns it at the
  points where we're evaluating the equation.

Both map $N$ sample points to $N_{eq}$ *collocation points*. With a 6th-order
central finite difference stencil, you lose 3 points at each end (no data
beyond the boundary to build the stencil from), so $N_{eq} = N - 6$. $L_I$ is
then just "pick the middle point of the stencil".

The exact equation would be a **hard constraint**:

$$L_{\partial t}X - L_I\Theta(X)\Xi = 0 \tag{paper eq. 5}$$

This is what modified-SINDy and DySMHO impose. The paper's key argument is that
this is too rigid — it assumes the discretisation is *exact*, which it never is,
and the resulting stiffness stops you from fitting trajectories longer than a
Lyapunov time in a chaotic system.

So instead, relax it into a penalty — a **soft constraint**:

$$\mathcal{L}(X,\Xi)=\underbrace{\frac{1}{2\sigma_x^2}\|\hat{X}-X\|_F^2}_{\text{data loss}}+\underbrace{\frac{1}{2\sigma_{\partial t}^2}\|L_{\partial t}X-L_I\Theta(X)\Xi\|_F^2}_{\text{model loss}}+\underbrace{\frac{1}{2\sigma_p^2}\|\Xi\|_F^2}_{\text{prior}} \tag{paper eq. 6}$$

$\|\cdot\|_F$ is the **Frobenius norm** — square every element, add them up,
square root. So $\|A\|_F^2 = \sum_{ij}A_{ij}^2$.

**This single equation is the whole algorithm.** Read the three terms:

- *Data loss*: don't wander far from what you measured. Weighted by
  $1/\sigma_x^2$, where $\sigma_x$ is the measurement noise.
- *Model loss*: nearly satisfy the differential equation. Weighted by
  $1/\sigma_{\partial t}^2$, where $\sigma_{\partial t}$ is how much truncation
  error you'll tolerate.
- *Prior*: prefer small coefficients. Weighted by $1/\sigma_p^2$.

Each weight is $1/\text{variance}$. That's not a coincidence — it's what you get
from taking $-\log$ of a product of Gaussian probability densities. Minimising
$\mathcal{L}$ finds the most probable $(X,\Xi)$ given the data, the **MAP**
(maximum a posteriori) estimate.

### 1.4 Model selection: why Bayesian evidence

Minimising $\mathcal{L}$ gives the best coefficients for a *given set of
candidate terms*. It does not tell you which terms to keep. If you keep all 30,
you'll fit the noise.

SINDy thresholds: delete any coefficient smaller than $\lambda$. This is crude —
a physically real term with a genuinely small coefficient gets deleted, and
$\lambda$ is a knob you have to tune.

Bayesian model selection instead asks: **how probable is my data under this
model?** That quantity is the *evidence*,

$$p(\text{data} \mid \text{model}) = \int p(\text{data}\mid\Xi)\,p(\Xi)\,\mathrm{d}\Xi$$

Note the integral over *all* possible $\Xi$, not just the best one. This is what
makes it automatically favour parsimony. A model with many parameters can fit
anything, but it spreads its probability thinly over a huge parameter space, so
for any *particular* dataset the integral is small. A simple model concentrates
its bets. This is the **Occam factor**, and it means you get sparsity for free
with no threshold to tune. (MacKay ch. 28 is the classic explanation; Faul §8.13 covers
the same ground.)

The integral is intractable, so use **Laplace's approximation**: near the
minimum, approximate $\mathcal{L}$ by a quadratic (its Taylor expansion), which
makes the integrand a Gaussian, which integrates in closed form. A Gaussian's
integral depends on the determinant of its covariance — hence the Hessian
$H = \mathrm{d}^2\mathcal{L}/\mathrm{d}\Xi^2$ appearing in the formula. The
result, paper §A.2.1:

$$-\log\mathcal{E} = \mathcal{L}(\Xi^*,X^*) + \frac{N_\Xi}{2}\left(\log 2\pi + 2\log\sigma_p\right) + \frac{1}{2}\log\left|\frac{H}{2\pi}\right|$$

We minimise this (minimising the negative = maximising the evidence).

### 1.5 Greedy search

You can't evaluate all $2^{30}$ subsets. So: start with everything, try deleting
each term in turn, keep whichever deletion gave the best evidence, repeat. Stop
when the evidence stops improving. Cost $O(M^2D^2/2)$ instead of $O(2^{MD})$.

### 1.6 The shape table

Every symbol in the maths, the variable that holds it, and its shape. The code
follows this throughout.

| symbol | code | shape | meaning |
|---|---|---|---|
| $\hat X$ | `prob.Xdata` | $N \times D$ | noisy measurements |
| $X$ | `X` | $N \times D$ | denoised states (unknown) |
| $\Xi$ | `Xi` | $M \times D$ | coefficients (unknown) |
| $\xi$ | `xi` | $N_p$ | the *active* entries of $\Xi$, flattened |
| $\Theta(X)$ | `theta(lib, X)` | $N \times M$ | dictionary evaluated at every time |
| $\partial\Theta/\partial x$ | `dtheta(lib, X)` | $N \times M \times D$ | its derivatives |
| $L_I$ | `prob.IMat` | $N_{eq} \times N$ | collocation operator |
| $L_{\partial t}$ | `prob.DMat` | $N_{eq} \times N$ | derivative operator |
| $\eta$ | `eta` | $N_{eq} \times D$ | model error |
| $\zeta$ | `zeta` | $N \times D$ | data error |
| $\sigma_x$ | `hyper.sigma_x` | $N \times D$ | measurement noise std |
| $\sigma_{\partial t}$ | `hyper.sigma_y` | $N_{eq} \times D$ | model error std |
| $\sigma_p$ | `hyper.sigma_p` | $M \times D$ | prior std |
| — | `mask` | $M \times D$ (Bool) | which terms are currently active |
| — | `z` | $ND + N_p$ | everything unknown, in one flat vector |

For Lorenz with $N=500$: $D=3$, $M=10$, $N_{eq}=494$, and $z$ has
$500 \times 3 + 30 = 1530$ entries. That's the size of the optimisation problem.

---

## Part 2 — The code, file by file

### 2.1 `libraries.jl` — the dictionary $\Theta$

**Replaces:** `Polynomial3D2O.m`, `Polynomial3D2Od.m`, `Polynomial3D2Odd.m`,
`Polynomial3D2Oddd_f.m` and their 2D/3rd-order variants — eight hand-written
MATLAB files, each hard-coding one library.

**Implements:** the $\Theta$ of paper eq. (1).

The MATLAB package hard-codes each library, which means adding a new one means
writing new files by hand, with derivatives, correctly. We instead store the
library *symbolically* as a list of exponents:

```julia
struct PolynomialLibrary <: AbstractLibrary
    D::Int
    order::Int
    powers::Vector{Vector{Int}}   # powers[n][e] = exponent of x_e in term n
    names::Vector{String}
end
```

For `PolynomialLibrary(3, 2)`, `powers` is

```
[0,0,0]  ->  1
[1,0,0]  ->  x
[0,1,0]  ->  y
[0,0,1]  ->  z
[2,0,0]  ->  x^2
[1,1,0]  ->  x*y
[1,0,1]  ->  x*z
[0,2,0]  ->  y^2
[0,1,1]  ->  y*z
[0,0,2]  ->  z^2
```

**The ordering matters and is deliberate.** It's produced by `_multisets`, which
generates non-decreasing index tuples — `(1,1), (1,2), (1,3), (2,2), (2,3),
(3,3)` for degree 2. This reproduces exactly the column ordering of SINDy's
`poolData.m`, which means the ground-truth matrices in `Lorenz.m` transfer
without reindexing. You can check: `Xi_truth(7,:) = [0, -1, 0]` in MATLAB is the
$xz$ term in $\dot y$, and index 7 above is indeed $xz$. If I'd ordered the
monomials differently, every comparison against the MATLAB results would have
been silently wrong.

Then $\Theta$ is a product of powers:

$$\Theta_{in} = \prod_{e=1}^{D} X_{ie}^{\,p^{(n)}_e}$$

```julia
Th = ones(T, Nx, M)
for n in 1:M
    p = lib.powers[n]
    for e in 1:lib.D
        p[e] == 0 && continue          # skip: x^0 = 1
        for i in 1:Nx
            Th[i, n] *= X[i, e]^p[e]
        end
    end
end
```

Start from all ones (the constant term needs no work), then multiply in each
variable's power. `p[e] == 0 && continue` is Julia's terse `if`: "if the
exponent is zero, skip to the next iteration".

The derivative is the product rule on a monomial:

$$\frac{\partial \Theta_{in}}{\partial x_e} = p_e \cdot X_{ie}^{\,p_e - 1}\prod_{f \neq e} X_{if}^{\,p_f}$$

```julia
fill!(col, T(p[e]))                    # bring down the exponent
for f in 1:lib.D
    q = (f == e) ? p[f] - 1 : p[f]     # reduce this variable's power by one
    q == 0 && continue
    for i in 1:Nx
        col[i] *= X[i, f]^q
    end
end
```

**What's missing, and why it matters.** The MATLAB library supplies second
(`ddTheta`) and third (`dddTheta`) derivatives. We supply neither. That falls
out of a simplification in `evidence.jl` explained in section 2.7 — and it's the single
biggest reduction in this port. Two functions to implement instead of four,
which is why plugging in `DataDrivenDiffEq.Basis` later will be easy.

**The abstract interface.** At the top of the file:

```julia
abstract type AbstractLibrary end
function nterms end
function theta end
function dtheta end
```

An abstract type is a promise, not an implementation: "anything calling itself
an `AbstractLibrary` must provide `nterms`, `theta` and `dtheta`". No other file
in the package mentions `PolynomialLibrary` by name — they all say
`AbstractLibrary`. That is the point of the interface: the rest of the code cannot tell, and does not care, which
library it's using. Adding Fourier terms, or wrapping a `Basis`, means writing
one new struct and three methods, and touching nothing else.

### 2.2 `differentiation.jl` — the operators $L_I$ and $L_{\partial t}$

**Replaces:** `FD.m`, which hard-codes stencil coefficients for orders 2, 4, 6,
8 and 10 as literal number arrays — around 60 lines of magic constants.

We derive them instead. A finite-difference formula is a weighted sum of nearby
values approximating a derivative:

$$f'(t_i) \approx \frac{1}{\Delta t}\sum_{k=-n/2}^{n/2} c_k\, f(t_{i+k})$$

To find the weights $c_k$, demand that the formula be *exact* for every
polynomial up to degree $n$. Testing on $f(t)=t^i$ for $i = 0,\dots,n$ gives
$n+1$ linear equations in the $n+1$ unknowns $c_k$:

$$\sum_k c_k\, s_k^{\,i} = \frac{\mathrm{d}}{\mathrm{d}t}t^i\Big|_{t=0} = \delta_{i1}, \qquad s_k = -n/2,\dots,n/2$$

That matrix, $V_{ik} = s_k^i$, is a **Vandermonde matrix**. In code:

```julia
s = collect(-n ÷ 2:n ÷ 2)
V = [float(sk)^i for i in 0:n, sk in s]    # V[i+1, k] = s_k^i
rhs = zeros(n + 1)
rhs[2] = 1.0                               # only the t^1 row has a nonzero derivative at 0
c = V \ rhs                                # backslash solves the linear system
```

That is a direct transcription of `CentralFD` at the bottom of `FD.m`
(`MAT=power(-n/2:n/2,(0:n)')`, `CoeffD1=MAT\[0;1;zeros(n-1,1)]`) — but we use it
for *all* orders, rather than only as a fallback, so the 60 lines of constants
disappear.

Then both operators are banded matrices:

```julia
cD = central_fd_coefficients(n) ./ dt      # derivative weights
cI = zeros(n + 1); cI[n ÷ 2 + 1] = 1.0     # "pick the centre point"
Neq = Nx - n
return _banded(cI, Neq, Nx), _banded(cD, Neq, Nx)
```

$L_I$ is a matrix of zeros with a single 1 per row, offset one step each row —
it just extracts the middle of each stencil. $N_{eq} = N - n$ because the first
and last $n/2$ points have no complete stencil. Paper §2.1 notes this loss.

**Why this file is a good abstraction boundary.** Everything downstream only
uses `IMat` and `DMat` as matrices. Swap in the weak-form (integral) operators
the BINDy paper describes and *nothing else changes*. The discretisation is a
swappable component, not a fixed choice.

### 2.3 `problem.jl` — bookkeeping

No physics here. Three jobs.

**`ODRHyperParameters`** holds $\sigma_x, \sigma_{\partial t}, \sigma_p$ as full
matrices, not scalars, exactly mirroring `HyperObj` in the MATLAB. Why matrices?
Because the noise can genuinely vary. `NonlinearOscillator.m` sets
`SigmaY = 1e-1*((IMat*t).^-6)` — a truncation error that decays as the dynamics
slow down. Scalars would forbid that. The convenience constructor broadcasts a
scalar when you don't need the generality.

**`ODRProblem`** bundles data, library, operators and hyperparameters, and —
importantly — *validates the shapes in its constructor*:

```julia
size(hyper.sigma_y) == (Neq, D) || throw(DimensionMismatch("sigma_y must be $Neq x $D"))
```

Do this. A mismatched $\sigma$ would otherwise broadcast silently into something
plausible-looking and wrong, and you'd never find it.

**The packing convention.** This is the fiddliest idea in the package, so slowly:

The optimiser wants *one flat vector of unknowns*. We have a matrix $X$
($N\times D$) and a matrix $\Xi$ ($M\times D$) of which only some entries are
active. So we define

$$z = \begin{bmatrix}\mathrm{vec}(X)\\ \xi\end{bmatrix}$$

`vec(X)` flattens column-major: all of $X[:,1]$, then all of $X[:,2]$, etc.
And `xi` is `Xi[mask]`, which in Julia *also* extracts in column-major order —
so it's all of dimension 1's active coefficients, then dimension 2's, and so on.

Two helpers keep this consistent:

```julia
function param_ranges(mask)          # which slice of xi belongs to state dim d
    for d in 1:D
        k = count(view(mask, :, d))
        rngs[d] = (offset + 1):(offset + k)
        offset += k
    end
end

function unpack(prob, z)             # split z back into X and xi
    nx = prob.Nx * prob.D
    X = reshape(view(z, 1:nx), prob.Nx, prob.D)
    xi = view(z, (nx + 1):length(z))
end
```

`reshape(view(...))` gives a *matrix-shaped window* onto part of the flat vector
— no copying at all. The MATLAB does the same thing with
`reshape(Xp(1:data.Nx*data.ND), data.Nx, data.ND)`, but copies.

The reason this all works out is that Julia and MATLAB are both column-major.
In Python/NumPy, which is row-major by default, you'd have to be far more
careful here.

### 2.4 `residual.jl` — the heart

**Replaces:** `Jsq_xp.m` (and `Jsq_x.m`, which is the same thing with the
parameter columns removed).

**Implements:** paper eq. (6) and the derivative table in §A.2.

#### The residual

Optimisers for problems of the form "minimise a sum of squares" want you to
hand them not the loss but the **residual vector** $r$ whose squared norm is
twice the loss. Compare eq. (6) to $\mathcal{L} = \frac{1}{2}\|r\|^2$ and read
off:

$$r=\begin{bmatrix}\mathrm{vec}(\eta/\sigma_{\partial t})\\ \mathrm{vec}(\zeta/\sigma_x)\\ \xi/\sigma_p\end{bmatrix},\qquad \eta = L_{\partial t}X-L_I\Theta(X)\Xi,\qquad \zeta = X-\hat{X}$$

Each of the three loss terms becomes one stacked block. Squaring and summing
gives exactly eq. (6), with the $\tfrac12$ out front.

```julia
ThI  = prob.IMat * theta(prob.lib, X)          # L_I Theta(X),  Neq x M
Xdot = prob.DMat * X                           # L_dt X,        Neq x D

for d in 1:prob.D
    pred = ThI[:, idx[d]] * xi[rngs[d]]        # the model's prediction for dim d
    eta[:, d] .= (Xdot[:, d] .- pred) ./ view(prob.hyper.sigma_y, :, d)
end
zeta = (X .- prob.Xdata) ./ prob.hyper.sigma_x

return vcat(vec(eta), vec(zeta), xi ./ prob.hyper.sigma_p[mask])
```

The loop over `d` is because each state dimension has its own set of active
terms. `idx[d]` are the active library rows for dimension `d`; `rngs[d]` is the
matching slice of the flat `xi`. This is precisely the MATLAB

```matlab
startd = sum(mask(:,1:d-1),'all')+1;
endd   = startd-1+sum(mask(:,d));
eta(:,d) = (dx(:,d)-Theta(:,mask(:,d))*p(startd:endd))./Data.SigmaY(:,d);
```

written with named helpers instead of inline arithmetic.

#### The Jacobian

The **Jacobian** $J$ is the matrix of every partial derivative of every residual
component with respect to every unknown: $J_{ij} = \partial r_i/\partial z_j$.
The optimiser needs it to know which way is downhill. You could let the computer
estimate it by finite differences, but with 1530 unknowns that's 1530 extra
residual evaluations *per iteration* — hopeless. So we derive it by hand.

It has natural block structure, one row-block per residual block, one column-
block per group of unknowns:

$$J=\begin{bmatrix}A & B\\ C & 0\\ 0 & P\end{bmatrix}$$

- $A = \partial(\eta/\sigma_{\partial t})/\partial X$
- $B = \partial(\eta/\sigma_{\partial t})/\partial \xi$
- $C = \partial(\zeta/\sigma_x)/\partial X = \mathrm{diag}(1/\sigma_x)$
- $P = \partial(\xi/\sigma_p)/\partial \xi = \mathrm{diag}(1/\sigma_p)$

The two zeros: $\zeta = X - \hat X$ doesn't involve $\xi$, and the prior term
doesn't involve $X$.

$C$ and $P$ are trivial. $B$ is nearly so — differentiate
$\eta_{:,d} = L_{\partial t}X_{:,d} - L_I\Theta(X)\,\Xi_{:,d}$ with respect to
$\Xi_{:,d}$ and the $\Xi$ just drops out:

$$\frac{\partial \eta_{:,d}}{\partial \xi_d} = -\left[L_I\Theta(X)\right]_{:,\,\text{active}_d}$$

And since $\xi_d$ only affects $\eta_{:,d}$, $B$ is **block diagonal**:

```julia
Bd = [sparse(Diagonal(1 ./ sigma_y[:, d]) * (-ThI[:, idx[d]])) for d in 1:D]
B  = blockdiag(Bd...)
```

$A$ is the interesting one, and it's paper eq. (A2)'s $\partial\eta_{id}/\partial X_{je}$:

$$\frac{\partial \eta_{:,d}}{\partial X_{:,e}} = \delta_{de}L_{\partial t} - L_I \,\mathrm{diag}(g_{de}), \qquad g_{de}[i] = \sum_{n \in \text{active}_d}\frac{\partial\Theta_{in}}{\partial x_e}\,\xi_{d}[n]$$

Where does this come from? Two terms:

1. $L_{\partial t}X_{:,d}$ depends on $X_{:,e}$ only when $d = e$ — hence the
   Kronecker delta $\delta_{de}$ (1 if $d=e$, 0 otherwise).
2. $\Theta(X)$ depends on *every* state, so the chain rule brings down
   $\partial\Theta/\partial x_e$, contracted with the coefficients $\xi_d$. The
   result is diagonal in the time index $i$, because $\Theta$ at time $i$
   depends only on $X$ at time $i$ — hence `Diagonal(g)`.

```julia
for d in 1:D
    Wd = Diagonal(1 ./ sigma_y[:, d])
    for e in 1:D
        g   = dTh[:, idx[d], e] * xi[rngs[d]]      # contract dTheta with the coefficients
        blk = prob.IMat * Diagonal(g)
        blk = (d == e) ? (prob.DMat - blk) : -blk  # the Kronecker delta
        Ablocks[d, e] = sparse(Wd * blk)
    end
end
A = reduce(vcat, [reduce(hcat, Ablocks[d, :]) for d in 1:D])
```

The MATLAB builds the same thing as one enormous `kron`/`sparse` expression:

```matlab
detadx = (kron(speye(ND),DMat) - kron(speye(ND),IMat)*sparse(...)) ./ reshape(SigmaY,[],1);
```

Mathematically identical. The block form is used here because you can point at
`(d == e) ? (prob.DMat - blk) : -blk` and see the $\delta_{de}$ in the paper.
That readability is worth more than the small allocation cost, especially in a
package someone else will maintain.

Finally the assembly, which reads like the block matrix above:

```julia
return [A                  B;
        C                  spzeros(T, nx, Np);
        spzeros(T, Np, nx) P]
```

**The sparsity is the whole reason this is tractable.** $L_I$ and
$L_{\partial t}$ are banded — 7 non-zeros per row for a 6th-order stencil. So
$J$ is banded too. Paper §2.3.1 gives the complexity as $O(N D b^2)$ with $b$
the bandwidth, rather than $O((ND)^3)$ for a dense solve. For Lorenz that's the
difference between milliseconds and minutes per iteration.

### 2.5 `initialguess.jl` — where to start

**Replaces:** `EnsembleLinRegress_Wprior.m`.

Eq. (6) is not convex — it has multiple local minima, and a gradient-based
optimiser finds whichever one it's nearest. So the starting point matters. Paper
§2.3: "The data $\hat X$ functions as the initial guess for $X$, whereas
multiple initial guesses for $\Xi$ is provided by bootstrap linear regression."

Pretend for a moment the states are noise-free. Then the problem is *linear* in
$\Xi$ and solvable in closed form. With the Gaussian prior acting as a ridge
penalty:

$$\xi_d = \left(\Theta^\top W\Theta + \mathrm{diag}(\sigma_p^{-2})\right)^{-1}\Theta^\top W\,\dot{x}_d, \qquad W = \mathrm{diag}(1/\sigma_{\partial t}^2)$$

```julia
rows = rand(rng, 1:prob.Neq, prob.Neq)    # resample rows WITH replacement
Ts   = Td[rows, :]
Amat = Ts' * (Ts .* ws) + Diagonal(prior)
bvec = Ts' * (y[rows] .* ws)
coeffs[:, s] = Amat \ bvec
```

**What bootstrapping is.** Draw $N_{eq}$ rows at random *with replacement* from
your $N_{eq}$ rows. Some appear twice, some not at all. Solve. Repeat 100 times.
You get 100 slightly different estimates, and their spread tells you how
sensitive the answer is to the data.

Taking the **median** across the 100 is called *bragging* (bootstrap robust
aggregating) — the MATLAB's `options.Bragging`. The median ignores the
occasional wild estimate from an unlucky resample, so it's more robust than the
mean.

The randomness also buys something else, and this is the clever part: each
bootstrap run is a *different* starting point. Running the optimiser from
several and keeping the best is **multi-start global optimisation** — a cheap
way to escape local minima. That's `fit_model_multistart` in section 2.8.

### 2.6 `lm.jl` — the optimiser

**Replaces:** MATLAB's `lsqnonlin` with `'Algorithm','trust-region-reflective'`.

MATLAB ships a good nonlinear-least-squares solver. Julia's equivalents live in
external packages (`NonlinearSolve.jl`, `LeastSquaresOptim.jl`). A compact
Levenberg–Marquardt is implemented here instead, for three reasons: the package has zero
dependencies so it always runs; LM needs $J$, which we need for the evidence
anyway, so nothing is wasted; and having *one* correct reference implementation
makes it far easier to check a swapped-in optimiser later.

**How LM works.** At the current point $z$, approximate each residual as linear:
$r(z+\delta) \approx r + J\delta$. Minimising $\|r+J\delta\|^2$ over $\delta$ gives
the **normal equations**

$$J^\top J\,\delta = -J^\top r$$

This is the Gauss–Newton step. It's fast near a minimum but can overshoot
badly far from one. LM adds damping:

$$\left(J^\top J + \lambda\,\mathrm{diag}(J^\top J)\right)\delta = -J^\top r$$

- $\lambda \to 0$: pure Gauss–Newton — big, confident steps.
- $\lambda \to \infty$: the step shrinks and rotates toward $-J^\top r$, the
  gradient — small, cautious steps.

$\lambda$ adapts: shrink it when a step succeeds, grow it when one fails. The
naive version of that is a fixed factor — divide by 3 on success, multiply by 3
on failure. It is one line, and on this problem it *oscillates*: accept, reject,
accept, reject, with $\lambda$ pinned and the cost crawling down by a fraction
of a percent per iteration. The residual blocks are weighted by $1/\sigma_x$ and
$1/\sigma_{\partial t}$, which differ by orders of magnitude, so a single fixed
factor is right for neither.

So we use the **Nielsen–Madsen gain ratio** instead: compare the reduction the
step actually achieved against the one the damped linear model *predicted*,

$$
ho=rac{C-C_{	ext{new}}}{	frac12\,\delta^	op(\lambda D\delta-g)}$$

and let that set the new $\lambda$. $
hopprox1$ means the quadratic model is
trustworthy here, so drop the damping sharply; $
ho$ barely positive means it is
not, so drop it barely. Failures grow $\lambda$ geometrically ($	imes2$,
$	imes4$, $	imes8$, ...) so a bad region is escaped quickly.

```julia
J = fJ(z)
g = J' * r
H = J' * J
F = cholesky(Symmetric(H + lambda * Diagonal(dH)))
delta = -(F \ g)

pred = dot(delta, lambda .* dH .* delta .- g) / 2   # predicted reduction
rho  = pred > 0 ? (C - Cnew) / pred : -1.0

if isfinite(Cnew) && rho > 0
    z, r, C = znew, rnew, Cnew
    lambda = max(lambda * max(1/3, 1 - (2rho - 1)^3), 1e-12)   # trust it more
    nu = 2.0
else
    lambda *= nu; nu *= 2                                      # back off, faster
end
```

`cholesky` is **Cholesky decomposition**: factor a symmetric positive-definite
matrix as $H = LL^\top$ with $L$ lower triangular. Solving then means two
triangular solves, which are cheap. It's about twice as fast as a general solve
and it *fails* if the matrix isn't positive definite — which here is a useful
signal, caught by the `try`/`catch` that bumps $\lambda$.

**Convergence** is declared on any of: tiny gradient (`gtol`), tiny relative cost
decrease (`ftol`), tiny relative step (`xtol`). These mirror MATLAB's
`FunctionTolerance` and `StepTolerance` settings.

**Non-convergence is information, not an error.** Paper §2.3 point 3: "if the
trial model lacks the required term, the optimisation frequently fails to
converge. This prompts us to restrict the maximum number of steps during model
trials and to retain the term whose elimination would result in
non-convergence." So `LMResult.converged = false` propagates upward as
"evidence = $\infty$", and the greedy search simply won't remove that term. The
tight `lm_maxiter = 100` during trials is deliberate, not a corner cut.

### 2.7 `evidence.jl` — the payoff

**Replaces:** `d2Jdp2_GaussNewtonEst.m` (~200 lines of nested tensor
permutations) and `d2Jdp2.m`.

**Implements:** paper §A.2.1.

This is where the port gets genuinely shorter, so it's worth going slowly.

#### The problem

The evidence needs $\mathrm{d}^2\mathcal{L}/\mathrm{d}\Xi^2$ — the Hessian with
respect to the coefficients **only**. But $\mathcal{L}$ also depends on $X$,
and $X$ is not a free variable: for each $\Xi$, it sits at the value $X^*(\Xi)$
that minimises the loss. So when you nudge $\Xi$, $X^*$ moves too, and the
chain rule has to account for it. That's what makes eq. (A3)–(A5) so
intimidating: they carry $\mathrm{d}X^*/\mathrm{d}\Xi$ and
$\mathrm{d}^2X^*/\mathrm{d}\Xi^2$ through every term.

#### The Gauss–Newton approximation

Paper eq. (A6) drops all second-derivative terms. This is exactly the same
approximation LM makes: replace the true Hessian by $J^\top J$. It's good when
the residuals at the optimum are small, which they are for a well-fitting model,
and §A.5 shows it barely affects the model ranking.

#### The simplification

Block $J^\top J$ over $z = [\mathrm{vec}(X); \xi]$:

$$J^\top J = \begin{bmatrix} H_{xx} & H_{x\xi}\\ H_{x\xi}^\top & H_{\xi\xi}\end{bmatrix},\qquad H_{xx}=A^\top A + C^\top C,\quad H_{x\xi}=A^\top B,\quad H_{\xi\xi}=B^\top B + P^\top P$$

Because $X^*$ minimises $\mathcal{L}$ for every $\Xi$, we have
$\partial\mathcal{L}/\partial X = 0$ there — that's paper eq. (A1), the
orthogonality condition of ODR. Differentiating that identity with respect to
$\Xi$ (the **implicit function theorem**) gives

$$\frac{\mathrm{d}X^*}{\mathrm{d}\Xi} = -H_{xx}^{-1}H_{x\xi}$$

Now substitute into (A6). Writing $G = \mathrm{d}X^*/\mathrm{d}\Xi$, eq. (A6) is

$$H \approx (AG+B)^\top(AG+B) + G^\top C^\top C\,G + P^\top P$$

Expand:

$$= G^\top\!\left(A^\top A + C^\top C\right)G + G^\top A^\top B + B^\top A G + B^\top B + P^\top P$$
$$= G^\top H_{xx}G + G^\top H_{x\xi} + H_{x\xi}^\top G + H_{\xi\xi}$$

Substituting $G = -H_{xx}^{-1}H_{x\xi}$, the first three terms become
$+H_{x\xi}^\top H_{xx}^{-1}H_{x\xi} - H_{x\xi}^\top H_{xx}^{-1}H_{x\xi}
- H_{x\xi}^\top H_{xx}^{-1}H_{x\xi}$, i.e. exactly one negative copy. So

$$\boxed{\;H_{\text{red}} = H_{\xi\xi} - H_{x\xi}^\top H_{xx}^{-1}H_{x\xi}\;}$$

This is the **Schur complement** of $J^\top J$ — a standard construction meaning
"the effective Hessian in one block after optimally eliminating the other". In
code:

```julia
H   = J' * J
Hxx = H[1:nx, 1:nx]
Hxp = H[1:nx, (nx+1):end]
Hpp = H[(nx+1):end, (nx+1):end]
F   = cholesky(Symmetric(Hxx))
S   = Matrix(Hpp) - Matrix(Hxp)' * (F \ Matrix(Hxp))
```

Ten lines replacing two hundred. And we never need $\partial^2\Theta$ or
$\partial^3\Theta$, because they only appeared in the terms Gauss–Newton drops
— which is why `libraries.jl` needs half of what the MATLAB libraries need.

`F \ Matrix(Hxp)` solves $H_{xx}Y = H_{x\xi}$ for $Y$, rather than forming
$H_{xx}^{-1}$ explicitly. Always do this: explicit inverses are slower and less
accurate. $H_{x\xi}$ has only $N_p \le 30$ columns, so it's 30 cheap sparse
solves.

#### Two free results

**$H_{\text{red}}$ is automatically positive definite.** $H_{xx}$ contains
$C^\top C = \mathrm{diag}(1/\sigma_x^2)$ on its diagonal, so it's positive
definite no matter what. That means $J$ has full column rank, $J^\top J$ is
positive definite, and the Schur complement of a positive definite matrix is
positive definite. The paper says as much in §2.3 point 4 ("which also ensures
the positive definiteness of the covariance") — the derivation shows *why*. The
MATLAB `while (any(eig(Hessian_p)<0) && exitflag>0)` retry loop has no
counterpart in our port.

**The $\log 2\pi$ terms cancel.** The paper's formula is

$$-\log\mathcal{E} = \mathcal{L}(\Xi^*,X^*) + \frac{N_\Xi}{2}\left(\log 2\pi + 2\log\sigma_p\right)+\frac{1}{2}\log\left|\frac{H_{\text{red}}}{2\pi}\right|$$

Since $\log|H/2\pi| = \log|H| - N_\Xi\log 2\pi$, the $+\tfrac{N_\Xi}{2}\log 2\pi$
and $-\tfrac{N_\Xi}{2}\log 2\pi$ cancel:

$$-\log\mathcal{E} = \mathcal{L}(\Xi^*,X^*) + \sum_{\text{active}}\log\sigma_p + \tfrac{1}{2}\log\det H_{\text{red}}$$

```julia
nle = cost(r) + sum(log, prob.hyper.sigma_p[mask]) + logdet(ch) / 2
```

Read the three terms — they're the Bayesian story in miniature:

- $\mathcal{L}$: **how well does it fit?** Lower is better. More terms always
  helps here.
- $\sum\log\sigma_p$: **how many terms?** Grows with $N_\Xi$. A direct penalty.
- $\tfrac12\log\det H_{\text{red}}$: the **Occam factor**. $H_{\text{red}}$ is
  the inverse posterior covariance, so a large determinant means a sharply
  peaked posterior — a model finely tuned to this exact dataset. Penalised.

Terms 2 and 3 are what produce sparsity, without any threshold. Paper §2.2:
"the Bayesian method does not discard coefficients according to their magnitude.
Instead, it eliminates coefficients [that] are too sensitive to minor
perturbations." That sensitivity is precisely $\log\det H_{\text{red}}$.

### 2.8 `regression.jl` — fit one model

**Replaces:** `ODR_BINDy_Regression.m` and `ODR_BINDy_Regression_MultiStart.m`.

`fit_model` does one thing: given a `mask`, find the best $(X, \Xi)$ and score it.

```julia
xi_start = bootstrap_ridge(prob, mask, X0; ...)      # section 2.5
z0  = vcat(vec(Matrix{T}(X0)), T.(xi_start))         # pack
res = levenberg_marquardt(fr, fJ, z0; maxiter, ...)  # section 2.6
X, xi = unpack(prob, res.z)                          # unpack
Xi = zeros(T, prob.M, prob.D); Xi[mask] .= xi        # scatter back into M x D
res.converged || return ODRFit(Xi, Matrix(X), T(Inf), ...)
nle, _ = neg_log_evidence(prob, mask, fJ(res.z), res.r)   # section 2.7
```

`Xi[mask] .= xi` is the inverse of `Xi[mask]` — it writes the flat vector back
into the masked positions of the $M\times D$ matrix, in the same column-major
order. That symmetry is why the packing convention in section 2.3 had to be nailed down
carefully.

Note `res.converged || return ...`: reading as "converged, *or else* return".
Non-convergence yields evidence $\infty$, which the greedy loop reads as "don't
remove that term" (section 2.6).

The `X0` and `xi0` arguments are **warm starting** — paper §2.3 point 2: "the
outcome $(X,\Xi)$ from the preceding iteration with more terms is typically a
reliable initial guess for the next nonlinear optimisation." Note the paper says
$(X,\Xi)$, *both*: neither the denoised states nor the surviving coefficients
change much when you delete one term. Passing both means a trial converges in a
handful of iterations instead of hundreds. This is the single biggest speedup in
the algorithm, and it is what makes the tight `lm_maxiter` of section 2.6 a
meaningful test rather than a guaranteed failure — with a cold `xi0` every trial
exhausts its budget, scores $\infty$, and no term is ever removed.

`xi0` defaults to `nothing`, which falls back to the bootstrap of section 2.5.
That is the right behaviour for the initial full-library fit, which has no
preceding model to inherit from.

`fit_model_multistart` runs several independent bootstrap starts and keeps the
best. Used only for the initial full-library fit, where there's no previous
model to warm-start from.

### 2.9 `greedy.jl` — the search

**Replaces:** `ODR_BINDy_Greedy.m`.

**Implements:** paper §2.3.

```julia
mask = trues(M, D)                                 # start with everything
current = fit_model_multistart(prob, mask, opts)

while count(mask) > D && rises < opts.stop_after_rises
    for i in findall(vec(mask))                    # every currently active term
        trial = copy(mask); trial[i] = false       # try deleting it
        all(any(trial; dims = 1)) || continue      # each state keeps >= 1 term
        f = fit_model(prob, trial, opts; X0 = current.X,        # warm start: X
                      xi0 = current.Xi[trial],                 # and Xi
                      maxiter = opts.lm_maxiter)
        if f.nlevidence < cand_fit.nlevidence      # best deletion so far?
            cand_fit = f; cand_mask = trial
        end
    end
    isfinite(cand_fit.nlevidence) || break         # nothing removable
    ...
    rises = cand_fit.nlevidence >= current.nlevidence ? rises + 1 : 0
end
```

Line by line:

- `trues(M, D)` — a matrix of `true`, the full library, $MD = 30$ coefficients.
- `findall(vec(mask))` — the linear indices of active terms, matching MATLAB's
  `m = (1:NpAll)'; m = m(mask);`.
- `all(any(trial; dims = 1))` — `any(...; dims=1)` collapses each column to
  "does this state have any term left?"; `all` demands yes for every state.
  MATLAB: `all(any(reshape(mask_loc,M,ND),1))`. Without it you could end up with
  $\dot{z} = 0$, which is degenerate.
- `rises` counts *consecutive* failures to improve. Paper §2.3: "The algorithm
  terminates and selects the global maximum evidence model when it observes a
  successive decline in evidence following the elimination of several terms."
  It's a patience counter — evidence can dip once and recover, so you don't stop
  at the first bad step. MATLAB's `ContRun_flag`/`ContRunStop`.
- `best` is tracked separately from `current`, so even if we overshoot past the
  optimum before stopping, we return the best model *ever seen*, not the last.

The `refine_after_removal` block re-fits the winning model with a much larger
iteration cap. During the trials we deliberately cap iterations low (it's a
ranking problem — good enough is good enough), but the model we carry forward as
the next warm start should be properly converged. MATLAB does this with two
separate `optimoptions` objects, `lsqnonlinopt` and `lsqnonlinopt2`.

`print_model` just formats $\Xi$ into readable equations, replacing MATLAB's
`poolDataLIST`.

### 2.10 `ODRBINDy.jl` — the module

```julia
module ODRBINDy
using LinearAlgebra, SparseArrays, Statistics, Random, Printf
include("libraries.jl")
...
export PolynomialLibrary, odr_bindy, ...
end
```

`include` literally pastes each file's contents in — the split into files is for
humans, not the compiler, so the order matters only where one file's code runs
at load time (none here). `export` lists the names visible after
`using ODRBINDy`; anything not exported is still reachable as
`ODRBINDy.something`.

The `"""..."""` blocks above definitions are **docstrings**. At the REPL, `?`
then a name shows them. `Documenter.jl` later turns them into a website. Write
them as you go; retrofitting documentation is miserable.

### 2.11 `examples/check_derivatives.jl` — the most important file

Bottom-up verification of every derivative in the package. Four checks:

1. **Library derivatives** vs central differences:
   $f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}$. Expect $\sim 10^{-9}$ — central
   differences are accurate to $O(h^2)$, and floating-point noise sets the
   floor around $\sqrt{\epsilon} \approx 10^{-8}$.
2. **FD operators** against $\sin(3t)$, whose derivative you know exactly.
3. **Residual Jacobian** — every column, against central differences of the
   residual. The deliberately uneven mask (`mask[5,1] = false;
   mask[9,2] = false`) is important: a uniform mask wouldn't catch an
   off-by-one in the `param_ranges` indexing, which is exactly the kind of bug
   that's easiest to introduce and hardest to spot.
4. **Reduced Hessian** against a numerically profiled loss: for each pair
   $(a,b)$, re-minimise over $X$ at four perturbed $\Xi$ values and form the
   second difference. A few percent error is *expected* — Gauss–Newton genuinely
   drops real terms. A factor of 2, or a sign flip, is a bug.

These checks are intended to become `test/runtests.jl`, by wrapping each one in
`@test` and `@testset`.

---

## Glossary

| term | meaning |
|---|---|
| **regressand / regressor** | the thing predicted / the things you predict from |
| **error-in-variables** | noise in the regressors, which ordinary least squares mishandles |
| **Frobenius norm** $\|A\|_F$ | $\sqrt{\sum_{ij}A_{ij}^2}$ — treat the matrix as one long vector |
| **collocation point** | a time at which the differential equation is enforced |
| **residual** | a vector whose squared norm is (twice) the loss |
| **Jacobian** | matrix of all first partial derivatives |
| **Hessian** | matrix of all second partial derivatives; measures curvature |
| **MAP** | maximum a posteriori — the most probable parameters given the data |
| **prior** | what you believed before seeing the data |
| **evidence** | $p(\text{data}\mid\text{model})$, integrated over all parameters |
| **Occam factor** | the part of the evidence that penalises complexity |
| **Laplace approximation** | approximate a peaked distribution by a Gaussian at its peak |
| **Gauss–Newton** | approximate the Hessian by $J^\top J$, dropping second derivatives |
| **Schur complement** | the effective block of a matrix after eliminating another block |
| **Cholesky** | factor a positive-definite matrix as $LL^\top$ for fast, stable solves |
| **bootstrap** | resample your data with replacement to gauge uncertainty |
| **bagging / bragging** | average / take the median over bootstrap estimates |
| **warm start** | begin an optimisation from a nearby known solution |
| **sparse matrix** | one stored as only its non-zero entries |
| **column-major** | matrices stored column by column (Julia, MATLAB, Fortran) |
