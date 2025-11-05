# Speed-Robustness Trade-Off for First-Order Methods
This code reproduces all figures and tables from the paper

> Bryan Van Scoy and Laurent Lessard. "The Speed-Robustness Trade-Off for First-Order Methods with Additive Gradient Noise.", 2025 ([arXiv](https://arxiv.org/abs/2109.05059))


## Description of files

- `main.ipynb` : Main Jupyter notebook. This file contains code that:
  - Optionally run simulations and save results to `.jld2` files in `/data` subdirectory
  - Extract information from data files and produce Figs. 1-5 from the paper
  - Optionally save the figures as `.pdf` in the `/figures` subdirectory
- `high_precision.ipynb` : Notebook containing high-precision example
  - Solves analysis instance using arbitrary-precision arithmetic solver
  - Outputs data for Table 2 from the paper
- `optalg.jl` and `simulation.jl`: helper files used by the notebooks above
- `Project.toml` and `Manifest.toml`: package versions and dependencies (for reproducibility)

## How to run the code

1. Open `main.ipynb`
   
2. Put this in a cell and run it:
```
import Pkg
Pkg.activate("/path/to/project")   # path that contains Project.toml. Can write Pkg.activate(".") if in current directory.
Pkg.instantiate()                  # installs all missing packages in the project (only need to do once)
```
3. Set parameters in the preamble block:
   - `GENERATE_DATA`: whether you want to re-generate `.jld2` data files (may take awhile), or use existing files in the `/data` directory.
   - `SAVE_FIGS`: whether you want to save figures as `.pdf` files or simply display them.
   - `SOLVER`: which solver to use. Default is `Clarabel` (open-source). For the paper, we used `Mosek` (academic or professional license required). 

4. To reproduce a figure from the paper, run the corresponding cell block in `main.ipynb`

5. To reproduce Table 2, run `high_precision.ipynb`




## Instructions for enabling multithreading capability

The code that generates the brute-force "cloud plots" (Figs. 2-4) is _embarrassingly parallel_ so can easily exploit multithreading capability. We used this for running the long data-generation batches. It is not required, but will greatly accelerate the code if used.

By default, Julia is single-threaded, so if you want to use multiple threads, this must be enabled. You can check the number of available threads in Julia by running at the REPL:

```
Threads.nthreads()
```

I will use 8 cores as an example, but if your CPU has more cores, you can use a larger number. To open julia with multithreaded capability:

```
julia -t 8
```

To create multithreaded-enabled kernel for ease of use later in Jupyter Lab:
```
using IJulia
installkernel("Julia (8 threads)", env = Dict("JULIA_NUM_THREADS" => "8"))
```
Then you can select the new kernel once you open the notebook.

Alternatively, you can change environment variables: `JULIA_NUM_THREADS`.
This can be done in VSCode in the Julia extension, or you can set it as an environment variable before running VSCode or Jupyter Lab.

