using Turing, Distributions, StatsBase, Random, BenchmarkTools, MCMCChains, DataFrames
import StatsFuns: log1mexp


# Log file

log_file = open("neg_bin_jl.log", "w")

function log(msg)
    println(log_file, msg)
    println(msg)
end

# Simulate negative binomial data
function simulate_data(mu, alpha, num)
    p = alpha / (mu + alpha)
    r = alpha
    return rand(NegativeBinomial(r, p), num)
end

# Zero-truncated negative binomial RNG
function rng_ztnb(mu, alpha, num)
    #p = alpha / (mu + alpha)
    p = clamp(alpha / (mu + alpha), 1e-6, 1 - 1e-6)  # clamp to (0,1)
    r = alpha
    samples = rand(NegativeBinomial(r, p), num)
    while any(samples .== 0)
        idx = findall(==(0), samples)
        samples[idx] .= rand(NegativeBinomial(r, p), length(idx))
    end
    return samples
end

# Test data
mu_true = 2.7
alpha_true = 0.6
data_small = simulate_data(mu_true, alpha_true, 10_000)
data_large = simulate_data(mu_true, alpha_true, 200_000)

# Define Negative Binomial Model
@model function negbin_model(data)
    mu ~ truncated(Normal(2, 5), 1e-3, Inf)
    alpha ~ truncated(Gamma(5, 0.5),1e-3, Inf)
    r = alpha
    #p = alpha / (mu + alpha)
    p = clamp(alpha / (mu + alpha), 1e-6, 1 - 1e-6) 
    data .~ NegativeBinomial(r, p)
end

# Sampling the model
model_small = negbin_model(data_small)
model_large = negbin_model(data_large)

# Benchmark sampling small dataset
log("Sampling small dataset:")
time_small = @elapsed chain_small = sample(model_small, NUTS(0.65), MCMCThreads(), 1000, 4)
log("Elapsed time (small dataset): $(time_small) seconds")

# Benchmark sampling large dataset
log("\nSampling large dataset:")
time_large = @elapsed chain_large = sample(model_large, NUTS(0.65), MCMCThreads(), 1000, 4)
log("Elapsed time (large dataset): $(time_large) seconds")

# Zero-truncated Negative Binomial Model

function logp_ztnb(y, mu, alpha)
    p = alpha / (mu + alpha)
    r = alpha
    nb = NegativeBinomial(r, p)

    # Zero-truncated adjustment
    logpdf(nb, y) - log1mexp(logpdf(nb, 0))
end


@model function zt_negbin_model(data)
    mu ~ truncated(Normal(2, 5), 0, Inf)
    alpha ~ Gamma(5, 0.5)

    for i in eachindex(data)
        Turing.@addlogprob! logp_ztnb(data[i], mu, alpha)
    end
end

# Generate zero-truncated data
data_zt = rng_ztnb(mu_true, alpha_true, 20_000)

# Zero-truncated model sampling
model_zt = zt_negbin_model(data_zt)


# Zero-truncated model sampling
log("\nSampling zero-truncated dataset:")
time_zt = @elapsed chain_zt = sample(model_zt, NUTS(0.65), MCMCThreads(), 1000, 4)
log("Elapsed time (zero-truncated dataset): $(time_zt) seconds")


log("\nChain summary (small dataset):")
log(DataFrame(summarystats(chain_small)))

log("\nChain summary (large dataset):")
log(DataFrame(summarystats(chain_large)))

log("\nChain summary (zero-truncated dataset):")
log(DataFrame(summarystats(chain_zt)))

# Close log file
close(log_file)