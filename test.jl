using BitBasis
using Catalyst
using Combinatorics
using Distances
using Distributions
using HomotopyContinuation
using IterTools
using LinearAlgebra
using NLsolve
using OrderedCollections
using OrdinaryDiffEq
using Phosphorylation
using UnPack

using CairoMakie
using Makie

using Infiltrator

subscripts = Dict(
    0 => "₀",
    1 => "₁",
    2 => "₂",
    3 => "₃",
    4 => "₄",
    5 => "₅",
    6 => "₆",
    7 => "₇",
    8 => "₈",
    9 => "₉"
)

function level(x::BitStr)
    return length(ons(x))
end

function symbolify(c)
    return Symbol.(collect(c))
end

# function to_subscript(i::Integer)
#     @assert i >= 0
#     sub = string(i)
#     for (k, v) in subscripts
#         sub = replace(sub, string(k) => v)
#     end
#     return sub
# end

function to_subscript(i::Integer)
    @assert i >= 0
    sub = string(i)
    return sub
end

function to_subscript(x::BitStr{N}) where {N}
    sub = string(x)
    for (k, v) in subscripts
        sub = replace(sub, string(k) => v)
    end
    return sub
end

struct SpeciesSymbols
    substrates::Vector{Symbol}
    kinase_complexes::Vector{Symbol}
    phosphatase_complexes::Vector{Symbol}
end

struct ParameterSymbols
    k⁺s::Vector{Symbol}
    k⁻s::Vector{Symbol}
    p⁺s::Vector{Symbol}
    p⁻s::Vector{Symbol}
    αs::Vector{Symbol}
    βs::Vector{Symbol}
end

function full_phosphorylation_network(n)
    rate_constants = String[]
    rxns = String[]

    # subgroups of species, so we can order them later
    substrates = String[]
    kinase_complexes = String[]
    phosphatase_complexes = String[]

    # subgroups of parameters, so we can order them later
    k⁺s = OrderedSet{String}()
    k⁻s = OrderedSet{String}()
    p⁺s = OrderedSet{String}()
    p⁻s = OrderedSet{String}()
    αs = OrderedSet{String}()
    βs = OrderedSet{String}()
    
    for i in 0:2^n-1
        bitstr = BitStr{n, Int}(i)
        sub = to_subscript(i)     
        A = "A$sub"
        XA = "XA$sub"
        YA = "YA$sub"

        ℓ = level(bitstr)
        ℓ_sub = to_subscript(ℓ)
        k⁺ = "k⁺$ℓ_sub"
        k⁻ = "k⁻$ℓ_sub"
        p⁺ = "p⁺$ℓ_sub"
        p⁻ = "p⁻$ℓ_sub"

        push!(substrates, A)

        # binding/unbinding of kinase
        if ℓ < n
            push!(kinase_complexes, XA)
            push!(k⁺s, k⁺)
            push!(k⁻s, k⁻)
            push!(rate_constants, "($k⁺, $k⁻)")
            push!(rxns, "X + $A <--> $XA")
        end

        # binding/unbinding of phosphatase
        if ℓ > 0
            push!(phosphatase_complexes, YA)
            push!(p⁺s, p⁺)
            push!(p⁻s, p⁻)
            push!(rate_constants, "($p⁺, $p⁻)")
            push!(rxns, "Y + $A <--> $YA")
        end

        # adding phosphates
        for idx in offs(bitstr)
            new_bitstr = Phosphorylation.flip(bitstr, idx)
            new_sub = to_subscript(Int(new_bitstr))
            new_ℓ_sub = to_subscript(level(new_bitstr))

            A_new = "A$new_sub"
            α = "α$(ℓ_sub)to$(new_ℓ_sub)"

            # addition
            push!(αs, α)
            push!(rate_constants, α)
            push!(rxns, "$XA --> X + $(A_new)")
        end

        # removing phosphates
        for idx in ons(bitstr)
            new_bitstr = Phosphorylation.flip(bitstr, idx)
            new_sub = to_subscript(Int(new_bitstr))
            new_ℓ_sub = to_subscript(level(new_bitstr))
            
            A_new = "A$new_sub"
            β = "β$(ℓ_sub)to$(new_ℓ_sub)"

            # addition
            push!(βs, β)
            push!(rate_constants, β)
            push!(rxns, "$YA --> Y + $(A_new)")
        end

    end
    species = vcat(substrates, "X", kinase_complexes, "Y", phosphatase_complexes)
    parameters = vcat(collect.([k⁺s, k⁻s, p⁺s, p⁻s, αs, βs])...)
    net = reaction_network(rate_constants, rxns, species=species, parameters=parameters)
    species_symbols = SpeciesSymbols(symbolify.([substrates, kinase_complexes, phosphatase_complexes])...)
    parameter_symbols = ParameterSymbols(symbolify.([
        k⁺s, k⁻s, p⁺s, p⁻s, αs, βs
    ])...)
    return net, species_symbols, parameter_symbols
end

function generate_ic(X0, Y0, species_symbols::SpeciesSymbols)
    ic = Pair{Symbol, Float64}[:X => X0, :Y => Y0]

    for s in species_symbols.substrates[2:end-1]
        push!(ic, s => 0.0)
    end
    a = rand(Uniform(0, 1))

    push!(ic, species_symbols.substrates[1] => a)
    push!(ic, species_symbols.substrates[end] => 1.0-a)

    for s in chain(species_symbols.kinase_complexes, species_symbols.phosphatase_complexes)
        push!(ic, s => 0.0)
    end

    return ic
end

function generate_params(min_rate, max_rate, parameter_symbols::ParameterSymbols)
    d = LogUniform(min_rate, max_rate)

    @unpack k⁺s, k⁻s, p⁺s, p⁻s, αs, βs = parameter_symbols

    params = Pair{Symbol, Float64}[]
    for p in IterTools.chain(k⁺s, k⁻s, p⁺s, p⁻s, αs, βs)
        rate = rand(d)
        while rate < min_rate
            rate = rand(d)
        end
        push!(params, p => rate)
    end

    return params
end


X0 = 1e-3
Y0 = 1e-3
μ = 10.0
σ = 100.0
t_max = 1000000.0
min_rate = 1.0e-3
max_rate = 100.0

model, species_symbols, parameter_symbols = full_phosphorylation_network(2)

function states_good(states, p; dist_tol=0.1, eig_tol=1e-6)
    length(states) > 2 || return false
    for (s1, s2) in combinations(states, 2)
        euclidean(s1, s2) > dist_tol || return false
    end
    try
        sum(steady_state_stability(s, model, p; tol=eig_tol) for s in states) > 1 || return false
    catch
        return false
    end
    return true
end

p = nothing
u0 = nothing
prob = nothing
states = nothing
while true
    global p = generate_params(min_rate, max_rate, parameter_symbols)
    tspan = (0.0, t_max)
    global u0 = generate_ic(X0, Y0, species_symbols)
    global states = hc_steady_states(model, p; u0=u0)
    if states_good(states, p)
        break
    end
end
