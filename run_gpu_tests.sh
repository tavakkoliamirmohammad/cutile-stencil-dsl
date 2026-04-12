#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=cutile-gpu-tests
#SBATCH --output=gpu_test_output.log

cd /uufs/chpc.utah.edu/common/home/u1419116/projects/cutile-stencil-dsl
source venv/bin/activate
module load cuda/13.1.0

echo "============================================================"
echo "Running new cuTile architecture tests (non-GPU)"
echo "============================================================"
python -m pytest tests/test_cutile_new.py -v -k "not GPU and not Benchmark" --tb=short 2>&1

echo ""
echo "============================================================"
echo "Running GPU convergence tests"
echo "============================================================"
python -m pytest tests/test_cutile_new.py -v -k "GPU" --tb=short 2>&1

echo ""
echo "============================================================"
echo "Running benchmarks (cuTile only)"
echo "============================================================"
python -c "
from cutile.benchmarks.bench_cutile import run_all_benchmarks
report = run_all_benchmarks()
print()
print(report.to_table())
report.to_json('benchmark_results.json')
" 2>&1

echo ""
echo "============================================================"
echo "Multi-GPU scaling test (2, 4, 5 GPUs)"
echo "============================================================"
python -c "
import cupy as cp
n_gpus = cp.cuda.runtime.getDeviceCount()
print(f'Available GPUs: {n_gpus}')
for i in range(n_gpus):
    props = cp.cuda.runtime.getDeviceProperties(i)
    print(f'  GPU {i}: {props[\"name\"].decode()}')
" 2>&1

echo ""
echo "Done!"
