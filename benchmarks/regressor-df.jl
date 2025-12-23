using Statistics
using EvoTrees
using DataFrames
using Random: seed!
import CUDA

# Tunables (via ENV to make quick repros easy)
nobs = parse(Int, get(ENV, "EVOTREES_NOBS", "100000"))
num_feat = parse(Int, get(ENV, "EVOTREES_NFEATS", "10"))
nrounds = parse(Int, get(ENV, "EVOTREES_NROUNDS", "200"))
colsample = parse(Float64, get(ENV, "EVOTREES_COLSAMPLE", "0.5"))
rowsample = parse(Float64, get(ENV, "EVOTREES_ROWSAMPLE", "0.5"))
T = Float64
nthread = Base.Threads.nthreads()
@info "testing with: $nobs observations | $num_feat features. nthread: $nthread"
@info "rowsample=$rowsample colsample=$colsample nrounds=$nrounds"
seed!(123)
x_train = rand(T, nobs, num_feat)
y_train = rand(T, size(x_train, 1))

@info nthread
loss = "mse"
if loss == "mse"
    loss_evo = :mse
    metric_evo = :mae
elseif loss == "logloss"
    loss_evo = :logloss
    metric_evo = :logloss
end

@info "EvoTrees"
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train
target_name = "y"
verbosity = 0

params_evo = EvoTreeRegressor(;
    loss=loss_evo,
    metric=metric_evo,
    nrounds=nrounds,
    alpha=0.5,
    lambda=0.0,
    gamma=0.0,
    eta=0.05,
    max_depth=6,
    min_weight=1.0,
    rowsample=rowsample,
    colsample=colsample,
    nbins=64,
    rng=123,
)
@info "EvoTrees CPU"
params_evo.device = :cpu
# @info "init"
# @time m_df, cache_df = EvoTrees.init(params_evo, dtrain; target_name);
# @time m_df, cache_df = EvoTrees.init(params_evo, dtrain; target_name);

# @info "train - no eval"
# @time m_evo_df = fit(params_evo, dtrain; target_name, device, verbosity, print_every_n=100);
# @time m_evo_df = fit(params_evo, dtrain; target_name, device, verbosity, print_every_n=100);

@info "train - eval"
@time m_cpu = EvoTrees.fit(params_evo, dtrain; target_name, deval=dtrain, verbosity, print_every_n=100);
# @time m_cpu = fit(params_evo, dtrain; target_name, device);
# @btime fit($params_evo, $dtrain; target_name, deval=dtrain, metric=metric_evo, device, verbosity, print_every_n=100);
@info "predict"
@time pred_cpu = m_cpu(dtrain);
# @btime m_evo($dtrain);

@info "EvoTrees GPU"
_device = :gpu
params_evo.device = _device
@info "train"
@time m_gpu = EvoTrees.fit(params_evo, dtrain; target_name, deval=dtrain, verbosity, print_every_n=100);
@info "predict"
@time pred_gpu = m_gpu(dtrain; device=_device);
# @btime m_gpu($dtrain; _device);

avg_splits(m) = sum(sum(t.split) for t in m.trees) / max(1, (length(m.trees) - 1))
cpu_splits = avg_splits(m_cpu)
gpu_splits = avg_splits(m_gpu)
@info "avg splits per tree (CPU) = $cpu_splits"
@info "avg splits per tree (GPU) = $gpu_splits"
@info "cor(pred_cpu, pred_gpu) = $(cor(pred_cpu, pred_gpu))"

