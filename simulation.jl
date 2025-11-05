using LinearAlgebra
using Statistics
using Random
using Roots: find_zero
using JLD2


struct Algorithm{T}
    α::T
    β::T
    η::T

    function Algorithm{T}( α::T, β::T, η::T ) where {T}
        return new(α,β,η)
    end
end

function ss( alg::Algorithm{T} ) where {T}
    α, β, η = alg.α, alg.β, alg.η
    A = reshape( T[ 1+β -β; 1 0 ], 2, 2 )
    B = reshape( T[ -α; 0 ],       2, 1 )
    C = reshape( T[ 1+η -η ],      1, 2 )
    D = reshape( T[ 0 ],           1, 1 )
    return A, B, C, D
end

function GD( α::T ) where {T}
    Algorithm{T}( α, T(0), T(0) )
end

function FG( m::T, L::T ) where {T}
    α = 1/L
    β = (√L-√m)/(√L+√m)
    Algorithm{T}( α, β, β )
end

function HB( m::T, L::T ) where {T}
    α = T(4)/(√L+√m)^2
    β = ((√L-√m)/(√L+√m))^2
    Algorithm{T}( α, β, T(0) )
end

function RHB( m::T, L::T, ρ::T ) where {T}
    κ = L/m
    if !( (√κ-1)/(√κ+1) ≤ ρ ≤ 1)
        error("Parameter ρ not in [(√κ-1)/(√κ+1),1] for RHB")
    end
    α = 1/m*(1-ρ)^2
    β = ρ^2
    Algorithm{T}( α, β, T(0) )
end

function random_matrix(d,m,L)
    U = svd(randn(d,d)).U
    if d == 1
        Λ = [ m ]
    elseif d == 2
        Λ = [ m, L ]
    else
#         Λ = [ m; m .+ (L-m)*rand(d-2); L ]
        Λ = range(m, stop=L, length=d)
    end
    Q = U'*diagm(Λ)*U
end

function allzero( A )
    return all( x -> x==0, A )
end

# return the Q matrix for Nesterov's lower-bound worst-case function
function nesterov_lower_bound_fun(d,m,L)
    Q     = diagm(0 => (L+m)/2 * ones(d), 1 => (L-m)/4 * ones(d-1), -1 => (L-m)/4 * ones(d-1))
    e1    = zeros(d)
    e1[1] = 1
    xopt  = (Q + 4/(L/m-1)*I) \ e1
    
    return Q, xopt
end
    
function simulation( alg::Algorithm{T}, m::T, L::T, σ::T;
    function_class::Symbol = :Q,
    trials::Int            = 100,
    iter::Int              = 100,
    d::Int                 = 2,
    name                   = "") where {T}

    if function_class != :Q
        error("Simulation not implemented for function class $function_class")
    end

    A,B,C,D = ss(alg)

    if !allzero(D)
        error("Need D=0 for simulation")
    end

    n = size(A,1)

    err = zeros(iter,trials)

    Q, xopt = nesterov_lower_bound_fun(d,m,L)
    yopt = xopt'
    ∇f = y -> (y-yopt)*Q + σ*randn(1,d)
    
    for t = 1:trials
        Random.seed!(t)
        
        x = zeros(n,d)
        
        for k = 1:iter
            y = C*x
            u = ∇f(y)
            x = A*x + B*u
            err[k,t] = norm(y-yopt)
        end
    end
    return vec(mean(err,dims=2)), vec(std(err,dims=2)), name
end

function simRHBdiminishing(m,L,σ,k0; d,trials,iter, name="RHB diminishing")
    
    err = zeros(iter,trials)
    
    Q, xopt = nesterov_lower_bound_fun(d,m,L)
    yopt = xopt'
    ∇f = y -> (y-yopt)*Q + σ*randn(1,d)
    
    for t = 1:trials
        Random.seed!(t)      
        x = zeros(2,d)
        
        for k = 1:iter
            ρ = k+k0>(sqrt(L/m)-1)/2 ? (k+k0)/(k+k0+2) : (sqrt(L)-sqrt(m))/(sqrt(L)+sqrt(m))
            α = 1/m*(1-ρ)^2
            β = ρ^2
            
            y = [1 0]*x
            u = ∇f(y)
            x = [1+β -β; 1 0]*x + [-α; 0]*u
            err[k,t] = norm(y-yopt)
        end
    end
    return vec(mean(err,dims=2)), vec(std(err,dims=2)), name
end 

function simGDdiminishing(m,L,σ,k0; d,trials,iter,init_scale, name="GD diminishing")
    
    err = zeros(iter,trials)
    
    Q, xopt = nesterov_lower_bound_fun(d,m,L)
    yopt = xopt'
    ∇f = y -> (y-yopt)*Q + σ*randn(1,d)
    
    for t = 1:trials
        Random.seed!(t)      
        x = zeros(1,d)
        
        for k = 1:iter
            ρ = k+k0>(L-m)/(2m) ? (k+k0)/(k+k0+1) : (L-m)/(L+m)
            α = 1/m*(1-ρ)
            y = x
            u = ∇f(x)
            x = x -α*u
            err[k,t] = norm(y-yopt)
        end
    end
    return vec(mean(err,dims=2)), vec(std(err,dims=2)), name
end 
    

function simulationCG( m::T, L::T, σ::T;
    trials::Int            = 100,
    iter::Int              = 100,
    d::Int                 = 2,
    name                   = "") where {T}

    err = zeros(iter,trials)

    for t = 1:trials
        Random.seed!(t)
        A = random_matrix(d,m,L)
        xopt = randn(d)
        xopt = init_scale * xopt/norm(xopt)
                
        ∇f = x -> A*(xopt-x)
        x = zeros(d)
        
        r = ∇f(x)
        p = ∇f(x)

        for k = 1:iter

            w  = σ*randn(d)
            Ap = A*(p+w)  # noisy matrix-vector evaluation
            α  = (r'*r)/(p'*Ap)
            x⁺ = x + α*p
            r⁺ = r - α*Ap  # r⁺ = ∇f(x⁺)+w
            β  = (r⁺'*r⁺)/(r'*r)
            p⁺ = r⁺ + β*p

            err[k,t] = norm(x-xopt)

            x,r,p = x⁺,r⁺,p⁺
        end
    end

    return vec(mean(err,dims=2)), vec(std(err,dims=2)), name
end

function simulationNLCG( m::T, L::T, σ::T;
    trials::Int            = 100,
    iter::Int              = 100,
    d::Int                 = 2,
    name                   = "",
    opt                    = "") where {T}

    err = zeros(iter,trials)

    A, xopt = nesterov_lower_bound_fun(d,m,L)
    ∇f = x -> A*(x-xopt) + σ*randn(d)
    
    for t = 1:trials
        Random.seed!(t)
                
        x = zeros(d)
        err[1,t] = norm(x-xopt)  # initial error
        
        Δx = -∇f(x)                 # Δx0
        s = Δx                      # s0
        α = (s'*Δx) / (s'*A*s)      # α0  (exact linesearch)
        x = x + α*s                 # x1
        err[2,t] = norm(x-xopt)     # first iteration
        
        for k = 3:iter
            Δx⁺ = -∇f(x)                 # Δxn

            β = if "FR" in opt
                    (Δx⁺'*Δx⁺) / (Δx'*Δx)
                elseif "PR" in opt
                    (Δx⁺'*(Δx⁺-Δx)) / (Δx'*Δx)
                elseif "HS" in opt
                    (Δx⁺'*(Δx⁺-Δx)) / (-s'*(Δx⁺-Δx))
                elseif "DY" in opt
                    (Δx⁺'*Δx⁺) / (-s'*(Δx⁺-Δx))
                else
                    @assert(false,"no β specification in opt")
                end
            s = Δx⁺ + β * s            # sn
            
            α = if "inexact" in opt
                    (s'*Δx⁺) / (s'*A*s)              # inexact linesearch
                elseif "exact" in opt
                    -(s'*A*(x-xopt)) / (s'*A*s)      # exact linesearch
                else
                    @assert(false,"no linesearch specification in opt")
                end
            x = x + α*s                  # x{n+1}
            Δx = Δx⁺
            err[k,t] = norm(x-xopt)
        end
    end

    return vec(mean(err,dims=2)), vec(std(err,dims=2)), name
end

function simulationQuasiNewton( m::T, L::T, σ::T;
    trials::Int            = 100,
    iter::Int              = 100,
    d::Int                 = 2,
    name                   = "",
    opt                    = "") where {T}

    err = zeros(iter,trials)

    A, xopt = nesterov_lower_bound_fun(d,m,L)
    ∇f = x -> A*(x-xopt) + σ*randn(d)
        
    for t = 1:trials
        Random.seed!(t)
        
        x = zeros(d)
        err[1,t] = norm(x-xopt)
        
        g = ∇f(x)
        H = Matrix{T}(I,d,d)

        # not done yet
        for k = 2:iter

            p  = -H*g
            α  = -(p'*g)/(p'*A*p)
            s  = α*p
            x⁺ = x + s
            g⁺ = ∇f(x⁺)
            y  = g⁺ - g
            
            H⁺ = if "BFGS" in opt
                H + (s'*y + y'*H*y)*(s*s')/(s'*y)^2 - (H*y*s' + s*y'*H)/(s'*y)
            elseif "Broyden" in opt
                H + (s-H*y)*s'*H / (s'*H*y)
            elseif "DFP" in opt
                H + (s*s')/(s'*y) - (H*y)*(H*y)'/(y'*H*y)
            elseif "SR1" in opt
                H + (s-H*y)*(s-H*y)' / ((s-H*y)'*y)
            else
                @assert(false,"no quasi-Newton type specified")
            end

            err[k,t] = norm(x-xopt)
            x,g,H = x⁺,g⁺,H⁺
        end
    end

    return vec(mean(err,dims=2)), vec(std(err,dims=2)), name
end

function simulationHB_GD( m::T, L::T, σ::T;
    function_class::Symbol = :Q,
    trials::Int            = 100,
    iter::Int              = 100,
    d::Int                 = 2,
    name                   = "") where {T}

    if function_class != :Q
        error("Simulation not implemented for function class $function_class")
    end

    err = zeros(iter,trials)

    Q, xopt = nesterov_lower_bound_fun(d,m,L)
    yopt = xopt'
    ∇f = y -> (y-yopt)*Q + σ*randn(1,d)
    
    for t = 1:trials
        Random.seed!(t)

        A,B,C,D = ss(HB(m,L))
        
        α0 = -B[1]
        
        n = size(A,1)
        
        x = zeros(n,d)
        s = []
        y = C*x
        u = ∇f(y)
        
        # heavy ball parameters
        ρ = (√L-√m)/(√L+√m)
        γ = σ*√d/m*√((1-ρ^4)/(1+ρ)^4)

        k0 = ceil(log(γ)/log(ρ))
        
        for k = 1:iter
            y = C*x
            u = ∇f(y)
            x = A*x + B*u
            err[k,t] = norm(y-yopt)
            
            if k > k0
                ρ = find_zero( r -> d*σ^2*(1-r)-m^2*γ^2*r^2*(1+r), ρ )
                γ *= ρ
                α = 1/m*(1-ρ)
                A,B,C,D = ss(GD(α))
            end
        end
    end
    return vec(mean(err,dims=2)), vec(std(err,dims=2)), name
end

function simulationDiminishing( alg, m::T, L::T, σ::T;
    function_class::Symbol = :Q,
    trials::Int            = 100,
    iter::Int              = 100,
    d::Int                 = 2,
    θ::T                   = T(0.5),
    Δk::Int                = 10,
    kvals::Vector{Int}     = Int[],
    ρvals::Vector{T}       = T[],
    name                   = "") where {T}

    if function_class != :Q
        error("Simulation not implemented for function class $function_class")
    end

    err = zeros(iter,trials)

    Q, xopt = nesterov_lower_bound_fun(d,m,L)
    yopt = xopt'
    ∇f = y -> (y-yopt)*Q + σ*randn(1,d)
    
    for t = 1:trials
        Random.seed!(t)
        
        # heavy ball parameters
        ρ = (√L-√m)/(√L+√m)
        γ1 = σ*√d/m*√((1-ρ^4)/(1+ρ)^4)
        γ0 = 1
        k0 = ceil(log(γ1/γ0)/log(ρ))

        A,B,C,D = ss(alg(ρ))
        
        n = size(A,1)
        
        x = zeros(n,d)
        s = []
        y = C*x
        u = ∇f(y)
        
        for k = 1:iter
            um = u
            ym = y
            y = C*x
            u = ∇f(y)
            x = A*x + B*u
            err[k,t] = norm(y-yopt)
            
#             if k > k0
#                 ρ = find_zero( r -> d*σ^2*(1-r)*(1+r^2)-m^2*γ1^2*r^2*(1+r)^3, ρ )
#                 γ1 *= ρ
#                 A,B,C,D = ss(RHB(m,L,ρ))
#             end
            
            # update tuning parameter
            if k in kvals
                
                ind = findfirst(val -> val==k, kvals)
                
                ρ = ρvals[ind]
                
                # update algorithm
                A,B,C,D = ss(alg(ρ))
                
                # reset momentum
                x[2,:] = x[1,:]
            end
            
#             if k ≥ k0 && mod(k-k0,Δk) == 0
                
#                 # update ρ
# #                 a = -γ1^2/(2*log(ρ))
# #                 b = a/γ1^2 - k0
# # #                 ρ = 1 - 2/(1+(k+b)*σ^2*d/(a*m^2))
# #                 ρ = find_zero( r -> (k+b)*d*σ^2*(1-r)*(1+r^2)-a*m^2*(1+r)^3, ρ )
                
#                 ρ = find_zero( r -> d*σ^2*(1-r)*(1+r^2)-m^2*γ1^2*r^(2Δk)*(1+r)^3, ρ )
#                 ρ0 = θ + (1-θ)*ρ
#                 γ1 *= ρ0^(2Δk)
                
#                 # update γ
# #                 γ0 = γ1
# #                 γ1 = σ*√d/m*√((1-ρ^4)/(1+ρ)^4)
                
#                 # compute number of iterations to reach next noise level
# #                 k0 += round(log(γ1/γ0)/log(ρ))
                
# #                 display( (k,γ1) )
                
#                 # update algorithm
#                 A,B,C,D = ss(alg(ρ))
                
#                 # reset momentum
#                 x[2,:] = x[1,:]
#             end
        end
    end
    return vec(mean(err,dims=2)), vec(std(err,dims=2)), name
end