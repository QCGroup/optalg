using LinearAlgebra, Random, Statistics
using MatrixEquations
using JuMP
const MOI = JuMP.MOI
using JLD2

# can be changed to use different default solver
using Clarabel
DEFAULT_SOLVER = Clarabel

"""
Load every top-level name from the JLD2 file `fname` and define it
as a global in `mod` (default = Main), overwriting any previous binding.
"""
function loadall(fname::AbstractString; mod::Module = Main)
    data = load(fname)               # a Dict{String,Any}
    for (name, val) in data
        # build the assignment expression `foo = <val>`...
        ex = Expr(:(=), Symbol(name), val)
        # ...and evaluate it at module‐scope, creating or overwriting a global
        Base.eval(mod, ex)
    end
    return nothing
end

"""
simulate gradient descent on random quadratic functions
  m: strong convexity parameter (minimum eigenvalue of Hessian)
  L: Lipschitz smoothness parameter (maximum eigenvalue of Hessian)
  d: dimension of the quadratic
  α: stepsize (learning rate) used by Gradient Descent
  T: time horizon (number of timesteps)
  N: total number of trials
Returns: mean and standard deviation of the trials as a function of time
"""
function simulate_GD_quadratics(m::Real, L::Real, d::Integer, α::Real, T::Integer, N::Integer)
    
    e = zeros(T+1,N)    # quantity we will plot (error)

    for trial = 1:N
        # generate random quadratic
        A = rand(d,d)
        X = eigvecs(A+A')
        Λ = [m*ones(Int(d/2)); L*ones(Int(d/2))]
        P = X*diagm(Λ)*X'

        x = zeros(d,T+1)  # state
        x[:,1] = 1000*[1; zeros(d-1,1)]

        w = randn(d,T)  # random noise
        for t = 1:T
            g = P*x[:,t] + w[:,t]
            x[:,t+1] = x[:,t] - α*g
            e[t,trial] = norm(x[:,t])
        end
        e[T+1,trial] = norm(x[:,T])
    end
    
    ebar = mean(e, dims=2)[:]
    sig = std(e, dims=2, corrected=false)[:]
    
    return (ebar,sig)
end

"""
simple binary search
  f: function that evaluates to true or false
  a: lower bound
  b: upper bound
  tol: tolerance
Assumes f(a)==false and f(b)==true and f is monotone (only one cross-over point)
Returns the smallest x in [a,b] (within tol) such that f(x)==true.
"""
function bsmin(f, a, b; tol=1e-5, verbose=false, T=Float64)
    # swap if out of order
    tol = T(tol)
    a, b = a > b ? (b, a) : (a, b)
    i=0
    while (b - a) > tol
        i += 1
        c = (a + b) / T(2)
        f(c) ? (b = c) : (a = c)
        if verbose            
            println("iteration $i: ($(Double64(a)), $(Double64(b))), gap = $(Float64(b-a))")
        end
    end
    return b
end

"""
Matlab nostalgia functions (linspace, logspace, eye)
"""
function linspace(a,b,n)
    range(a, stop=b, length=n)
end

# slightly different from the matlab version, specify start and end
function logspace(a,b,n)
    exp10.(range(log10(a), stop=log10(b), length=n))
end

function eye(n)
    return Matrix{Float64}(I,n,n)
end

"""
Algorithm parameters in canonical form for various algorithms
  m: strong convexity parameter (minimum eigenvalue of Hessian)
  L: Lipschitz smoothness parameter (maximum eigenvalue of Hessian)
  ρ: additional parameter (needed for algorithms such as RAM, GD, etc.)
Returns parameters for standard form (α,β,η)
"""
function algo_params(algo,m,L,ρ)
    κ = L/m
    if algo == "GD"
        return (1-ρ)/m, 0, 0 # assuming low-alpha branch. Otherwise it's (1+ρ)/L
    elseif algo == "GD 1/L"
        return 1/L, 0, 0
    elseif algo == "GD 2/(L+m)" || algo == "GD 2/(m+L)"
        return 2/(L+m), 0, 0
    elseif algo == "RHB"
        return 1/m*(1-ρ)^2, ρ^2, 0
    elseif algo == "HB"
        return 4/(√L+√m)^2, ((√κ-1)/(√κ+1))^2, 0

    elseif algo == "RAM"
        return 1/m*(1+ρ)*(1-ρ)^2, ρ*(L*(1-ρ+2*ρ^2)-m*(1+ρ))/((L-m)*(3-ρ)), ρ*(L*(1-ρ^2)-m*(1+2*ρ-ρ^2))/((L-m)*(3-ρ)*(1-ρ^2))
    elseif algo == "RM"
        return 1/m*(1+ρ)*(1-ρ)^2, L*ρ^3/(L-m), m*ρ^3/((L-m)*(1-ρ)^2*(1+ρ))
    elseif algo == "TM"
        ρ = 1-1/√κ
        return (1+ρ)/L, ρ^2/(2-ρ), ρ^2/((1+ρ)*(2-ρ))
    elseif algo == "FG"
        return 1/L, (√κ-1)/(√κ+1), (√κ-1)/(√κ+1)
    else
        @error "unknown algorithm"
    end
end

function algo_params(algo,m,L)
    algo_params(algo,m,L,nothing)
end

"""
QUADRATIC FUNCTION CLASS
get gamma or rho for a given algorithm applied to a quadratic function
"""
function get_rho_Q(α,β,η,m,L)
    max( get_rho_help_Q(α,β,η,m), get_rho_help_Q(α,β,η,L) )
end

function get_gam_Q(α,β,η,m,L)
    max( get_gam_help_Q(α,β,η,m), get_gam_help_Q(α,β,η,L) )
end

function get_rho_help_Q(α,β,η,q)
    b = β + 1 - α*q*(η + 1)
    c = β - α*η*q
    Δ = b^2 - 4c
    return Δ < 0 ? √c : 1/2*(abs(b) + √Δ)
end

function get_gam_help_Q(α,β,η,q)
    g2 = α*(1+β+(1+2η)*α*η*q) / ( q*(1-β+α*η*q)*(2+2β-(1+2η)*α*q) )
    if g2 < 0
        return Inf
    elseif isnan(g2)
        return 0
    else
        return √g2
    end
end

"""
SMOOTH AND STRONGLY CONVEX
get gamma or rho for a given algorithm applied to FmL
"""
function get_rho_F(α,β,η,m,L; ℓ=1, tol=1e-6, returnsol=false, solver=DEFAULT_SOLVER, T=Float64, verbose=false)
    rhval = try
        bsmin( ρ -> solver_helper(α,β,η,m,L,ρ,ℓ, norm="rho", returnsol=false, solver=solver, T=T, tol=tol), T(1)-T(1)/sqrt(L/m), T(1.1); tol=tol, verbose=verbose, T=T)
    catch
        NaN
    end
    return solver_helper(α,β,η,m,L,rhval,ℓ, norm="rho", returnsol=returnsol, solver=solver, T=T, tol=tol)
end

function get_gam_F(α,β,η,m,L; ℓ=6, returnsol=false, solver=DEFAULT_SOLVER, T=Float64, tol=1e-6)
    gamval = try
        solver_helper(α,β,η,m,L,1,ℓ, norm="H2", returnsol=returnsol, solver=solver, T=T, tol=tol)
    catch
        NaN
    end
end

# Solver helper. This is where all the magic happens. norm ∈ {"H2", "rho"}
# rho: returns true/false if the given rho is feasible
# H2: returns the sensitivity gamma (or NaN if infeasible)
# returnsol: also return solved model
function solver_helper(α,β,η,m,L,ρ,ℓ; returnsol=false, norm="H2", solver=DEFAULT_SOLVER, T=Float64, tol=1e-6)

    α, β, η, m, L, ρ = T(α), T(β), T(η), T(m), T(L), T(ρ)
    
    A = [T(1)+β -β; T(1) T(0)]
    B = [-α; T(0)]
    C = [T(1)+η -η]
    n = size(A,1)
    
    Zp = [T.(I(ℓ)) zeros(T,ℓ,1)]  # ℓ × (ℓ+1)
    Z = [zeros(T,ℓ,1) T.(I(ℓ))]   # ℓ × (ℓ+1)
    Ib = [T.(I(ℓ+1)) zeros(T,ℓ+1,1)]
    e1 = Ib[:,1]       # (ℓ+1) × 1
    eℓ1 = Ib[:,ℓ+1]
    
    if norm=="rho"
        
        if ℓ == 0
            Abr = A
            Bbr = B
            Cbr = [C; zeros(T,1,n)]
            Dbr = [T(0); T(1)]
            Xbr = [ T.(I(n)) zeros(T,n,1) ]
            Ybr = [ C zeros(T,1,1) ]
        else
            # reduced dynamics
            Ψ11 = A^ℓ
            Ψ12 = hcat( [A^k*B for k in 0:ℓ-1]... )  # controllability matrix
            Ψ21 = vcat( [C*A^(ℓ-1-k) for k in 0:ℓ-1]... )  # flipped observability matrix
            if ℓ == 1
                Ψ22 = T(0)
            else
                Ψ22 = diagm( [ k => (C*A^(k-1)*B)[1]*ones(T,ℓ-k) for k in 1:ℓ-1 ]... )  # CA^kB Toeplitz matrix
            end
            Abr = [A B*eℓ1'*Z'
                   zeros(T,ℓ,n) Zp*Z']
            Bbr = [zeros(T,n,1); Zp*e1]
            Cbr = [e1*C*Ψ11+Z'*Ψ21 e1*C*Ψ12+Z'*Ψ22; zeros(T,ℓ+1,n) Z']
            Dbr = [zeros(T,ℓ+1,1); e1]

            Xbr = [ Ψ11 Ψ12 zeros(T,n,1) ]
            Ybr = [ C*Ψ11 C*Ψ12 T(0) ]
        end
        
    elseif norm=="H2"
        
        if ℓ == 0
            Ab = A
            Bb = Hb = B
            Cb = [C; zeros(T,1,n)]
            Db = [T(0); T(1)]      
            Xb = [ T.(I(n)) zeros(T,n,1) ]
            Yb = [ C zeros(T,1,1) ]
        else
            # augmented dynamics
            Ab = [ A          zeros(T,n,ℓ) zeros(T,n,ℓ)
                   Zp*e1*C    Zp*Z'        zeros(T,ℓ,ℓ)
                   zeros(T,ℓ,n) zeros(T,ℓ,ℓ) Zp*Z'      ]
            Bb = [ B; zeros(T,ℓ,1); Zp*e1 ]
            Hb = [ B; zeros(T,ℓ,1); zeros(T,ℓ,1) ]
            Cb = [ e1*C           Z'             zeros(T,ℓ+1,ℓ)
                   zeros(T,ℓ+1,n) zeros(T,ℓ+1,ℓ) Z'           ]
            Db = [ zeros(T,ℓ+1,1); e1 ]

            Xb = [ T.(I(n)) zeros(T,n,2ℓ+1) ]
            Yb = [ C zeros(T,1,2ℓ+1) ]
        end        
    else
        @assert(false,"unrecognized norm")
    end

    # Create JuMP model
    model = GenericModel{T}(solver.Optimizer{T})
    set_silent(model)
    
    # Define variables
    ndim = ( norm == "rho" ? n+ℓ : n+2ℓ )
    @variable(model, P[1:ndim, 1:ndim], Symmetric)
    @variable(model, p[1:ℓ])
    @variable(model, Λ1[1:ℓ+2, 1:ℓ+2] >= 0)
    @variable(model, Λ2[1:ℓ+2, 1:ℓ+2] >= 0)
    
    Π1 = zeros(T,2ℓ+2,2ℓ+2)
    π1 = zeros(T,ℓ+1,1)
    for i ∈ range(1,stop=ℓ+2)
        ei = Ib[:,i]
        for j ∈ range(1,stop=ℓ+2)
            ej = Ib[:,j]
            Π1 = Π1 + Λ1[i,j] * [ -m*L*(ei-ej)*(ei-ej)'  (ei-ej)*(m*ei-L*ej)'; (m*ei-L*ej)*(ei-ej)'  -(ei-ej)*(ei-ej)' ]
            π1 = π1 + T(2)*(L-m)*Λ1[i,j] * (ei-ej)
        end
    end

    Π2 = zeros(T,2ℓ+2,2ℓ+2)
    π2 = zeros(T,ℓ+1,1)
    for i ∈ range(1,stop=ℓ+2)
        ei = Ib[:,i]
        for j ∈ range(1,stop=ℓ+2)
            ej = Ib[:,j]
            Π2 = Π2 + Λ2[i,j] * [ -m*L*(ei-ej)*(ei-ej)'  (ei-ej)*(m*ei-L*ej)'; (m*ei-L*ej)*(ei-ej)'  -(ei-ej)*(ei-ej)' ]
            π2 = π2 + T(2)*(L-m)*Λ2[i,j] * (ei-ej)
        end
    end
    
    if norm == "rho"
        X1 = [Abr Bbr; T.(I(n+ℓ)) zeros(T,n+ℓ,1)]'*[P zeros(T,n+ℓ,n+ℓ); zeros(T,n+ℓ,n+ℓ) -ρ^2*P]*[Abr Bbr; T.(I(n+ℓ)) zeros(T,n+ℓ,1)] + [Cbr Dbr]'*Π1*[Cbr Dbr]
        Y1 = (Zp-ρ^2*Z)'*p + π1
        X2 = Xbr'*Xbr - [T.(I(n+ℓ)) zeros(T,n+ℓ,1)]'*P*[T.(I(n+ℓ)) zeros(T,n+ℓ,1)] + [Cbr Dbr]'*Π2*[Cbr Dbr]
        Y2 = -Z'*p + π2
    elseif norm == "H2"
        X1 = [Ab Bb; T.(I(n+2ℓ)) zeros(T,n+2ℓ,1)]'*[P zeros(T,n+2ℓ,n+2ℓ); zeros(T,n+2ℓ,n+2ℓ) -ρ^2*P]*[Ab Bb; T.(I(n+2ℓ)) zeros(T,n+2ℓ,1)] + [Cb Db]'*Π1*[Cb Db] + Yb'*Yb
        Y1 = (Zp-ρ^2*Z)'*p + π1
        X2 = -[T.(I(n+2ℓ)) zeros(T,n+2ℓ,1)]'*P*[T.(I(n+2ℓ)) zeros(T,n+2ℓ,1)] + [Cb Db]'*Π2*[Cb Db]
        Y2 = -Z'*p + π2
    end

    # Add constraints
    # LMI constraints: X ⪯ 0 is equivalent to -(X + X') is PSD
    @constraint(model, -(X1 + X1') in PSDCone())
    @constraint(model, -(X2 + X2') in PSDCone())
    @constraint(model, Y1 .<= 0)
    @constraint(model, Y2 .<= 0)

    # Set objective
    if norm == "rho"
        # Feasibility problem - JuMP doesn't have "satisfy" so use dummy objective
        @objective(model, Min, 0)
    elseif norm == "H2"
        @objective(model, Min, tr(Hb'*P*Hb))
    end

    # Solve
    optimize!(model)
    status = (termination_status(model) == MOI.OPTIMAL)

    if norm == "rho"
        return returnsol ? model : status
    elseif norm == "H2"
        if returnsol
            return status ? sqrt(objective_value(model)) : T(NaN)
        else
            return model
        end
    else
        @assert(false,"unrecognized norm")
    end
end

