# =============================================================================
# Summarise benchmarks/results/lorenz_paper_trials.csv against the paper.
#
#     julia --project=. benchmarks/summarise_lorenz_paper.jl
#
# For every (T, noise, sigma_y) cell: success rate with a 95% Wilson confidence
# interval, next to the rate read off the paper's Fig. 4
# (benchmarks/data/paper_fig4_lorenz_odrbindy.csv). A cell "agrees" when the
# paper's rate lies inside our interval, widened by the +-0.02 read-off error.
# Writes benchmarks/results/lorenz_paper_summary.csv and prints a Markdown table.
# =============================================================================

using Printf, Statistics

const HERE = @__DIR__
const TRIALS = joinpath(HERE, "results", "lorenz_paper_trials.csv")
const PAPER = joinpath(HERE, "data", "paper_fig4_lorenz_odrbindy.csv")
const SUMMARY = joinpath(HERE, "results", "lorenz_paper_summary.csv")
const READOFF = 0.02

"95% Wilson score interval for `k` successes out of `n`."
function wilson(k, n; z = 1.959963984540054)
    n == 0 && return (NaN, NaN)
    p = k / n
    c = (p + z^2 / (2n)) / (1 + z^2 / n)
    h = z / (1 + z^2 / n) * sqrt(p * (1 - p) / n + z^2 / (4n^2))
    return (max(c - h, 0.0), min(c + h, 1.0))
end

function read_csv(path)
    lines = filter(l -> !isempty(l) && !startswith(l, '#'), readlines(path))
    names = Symbol.(split(lines[1], ','))
    return [Dict(zip(names, split(l, ','))) for l in lines[2:end]]
end

function main()
    paper = Dict((parse(Int, r[:T]), round(parse(Float64, r[:noise]); digits = 3)) =>
                 parse(Float64, r[:paper_success_rate]) for r in read_csv(PAPER))

    cells = Dict{Tuple{Int,Float64,Float64},Vector{Dict{Symbol,SubString{String}}}}()
    for r in read_csv(TRIALS)
        key = (parse(Int, r[:T]), round(parse(Float64, r[:noise]); digits = 3),
               parse(Float64, r[:sigma_y]))
        push!(get!(cells, key, []), r)
    end

    open(SUMMARY, "w") do io
        println(io, "T,noise,sigma_y,trials,successes,success_rate,ci_low,ci_high," *
                    "paper_success_rate,agrees,median_model_error,median_rms_data," *
                    "median_rms_denoised,median_runtime_s")
        println("| T | N | noise | trials | success | 95% CI | paper | agrees? | " *
                "median coef. error | RMS data -> denoised | median runtime |")
        println("|---|---|---|---|---|---|---|---|---|---|---|")
        for key in sort(collect(keys(cells)))
            T, noise, sy = key
            rs = cells[key]
            n = length(rs)
            k = count(r -> r[:success] == "1", rs)
            lo, hi = wilson(k, n)
            p = get(paper, (T, noise), NaN)
            agrees = isnan(p) ? "" : (lo - READOFF <= p <= hi + READOFF ? "yes" : "no")
            med(s) = median(parse.(Float64, getindex.(rs, s)))
            @printf(io, "%d,%.3f,%g,%d,%d,%.4f,%.4f,%.4f,%s,%s,%.4g,%.4g,%.4g,%.1f\n",
                    T, noise, sy, n, k, k / n, lo, hi, isnan(p) ? "" : @sprintf("%.2f", p),
                    agrees, med(:model_error), med(:rms_data), med(:rms_denoised),
                    med(:runtime_s))
            @printf("| %d | %d | %.1f%% | %d | %.0f%% (%d/%d) | %.0f–%.0f%% | %s | %s | %.2g | %.2f -> %.2f | %.0f s |\n",
                    T, round(Int, T / 0.01), 100noise, n, 100k / n, k, n, 100lo, 100hi,
                    isnan(p) ? "–" : @sprintf("%.0f%%", 100p), agrees,
                    med(:model_error), med(:rms_data), med(:rms_denoised), med(:runtime_s))
        end
    end
    println("\nwritten: ", relpath(SUMMARY, pwd()))
end

main()
