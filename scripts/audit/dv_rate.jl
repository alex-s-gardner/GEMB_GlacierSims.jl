# Finding: the sweep's record-mean rate divides the cumulative change after N output intervals by the
# span from the first to the last output time, which is N - 1 intervals, so the rate is N/(N-1) too
# large.
#
# Weekly output stamped at the last hour of each week, hourly forcing from 2000-01-03 (a Monday, so
# GEMB's calendar weeks are whole), and dv rising by r = 0.25 km³ per week: the true rate is
# r·365.25/7 = 13.045 km³/yr.
#
# Run from the repository root:  julia --project=<env> scripts/audit/dv_rate.jl
# Expected on feat/native-resolution-run:  sweep formula 15.654 km³/yr (6/5 of the true rate);
#                                          record_years is not defined
# Expected on fix/pipeline-audit:          sweep formula as above for comparison, and
#                                          record_years rate 13.045 km³/yr
include(joinpath(@__DIR__, "common.jl"))

r, N = 0.25, 6
start = DateTime(2000, 1, 3)
time = [start + Day(7i) - Hour(1) for i in 1:N]
dv = [r * i for i in 1:N]
truth = r * 365.25 / 7
sweep_years = (last(time) - first(time)).value / (365.25 * 86_400_000)    # the formula on the old branch
println("true rate                 = ", round(truth; digits = 3), " km3/yr")
println("first-to-last-time rate   = ", round(dv[end] / sweep_years; digits = 3), " km3/yr")
if isdefined(GS, :record_years)
    run = minimal_tile_run(time; dv, provenance = Dict{String,Any}("time_coverage_start" => start,
                                                                    "forcing_time_step_seconds" => 3600.0))
    println("record_years rate         = ", round(run.totals[:dv][end, 1, 1] / GS.record_years(run); digits = 3),
            " km3/yr")
else
    println("record_years              = not defined on this branch")
end
