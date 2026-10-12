# NEAT CartPole Swing-Up — Experimental Framework

A reproducible research framework for evolving **minimal neural controllers**
using [NEAT](https://neat-python.readthedocs.io/) (NeuroEvolution of
Augmenting Topologies) on a custom CartPole Swing-Up task.

## Motivation

Standard CartPole only requires balancing a pole that starts nearly upright.
**Swing-Up** starts the pole hanging downward (θ = π) and demands the agent
learn an energy-pumping strategy before it can even begin to balance — a
fundamentally harder control problem that tests whether NEAT can discover
compact, effective controllers.

## Three Experimental Modes

| Mode | Fitness Function |
|------|-----------------|
| **Performance** | Maximise raw episode reward |
| **Complexity-aware** | Maximise reward − λ · (active connections + hidden neurons) |
| **Hardware-aware** | Maximise reward − λ · estimated inference operations |

### Annealed Complexity Penalty

> **Critical design choice:** complexity and hardware penalties are *only*
> applied once a genome's base reward exceeds a **survival threshold**
> (default: pole above the horizontal axis on average). This prevents
> the population from collapsing into trivial zero-connection networks
> in early generations when no genome has learned to swing up yet.

## Project Structure

```
neat/
├── main.py                        # CLI entry point
├── ui_server.py                   # Real-time 60 Hz visualizer & sandbox server
├── requirements.txt
├── controller.h                   # Standalone exported C controller header
│
├── web/
│   └── index.html                 # Single-file Dieter Rams aesthetic web UI
│
├── config/
│   ├── experiment_config.py       # Centralised hyperparameters
│   └── neat_config.ini            # NEAT-python configuration
│
├── env/
│   └── cartpole_swingup.py        # Custom gym.Env (swing-up physics)
│
├── evolution/
│   ├── fitness.py                 # Fitness functions & annealed penalty
│   └── neat_evolve.py             # NEAT evolution engine
│
├── metrics/
│   └── complexity.py              # Node count, connection count, MACs
│
├── utils/
│   ├── serialization.py           # Genome pickle & JSON topology
│   └── c_exporter.py              # Standalone ANSI C header compiler
│
├── experiments/
│   └── runner.py                  # Multi-seed orchestration & CSV writer
│
└── analysis/
    ├── compare.py                 # Statistical comparison & Markdown report
    └── stress_test.py             # Environmental shift robustness evaluation
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full experiment sweep (3 modes × 5 seeds = 15 evolution runs)
python main.py

# 3. Re-generate the comparison report from existing results
python main.py --compare

# 4. Run robustness stress-test across 4 perturbation scenarios
python main.py --stress-test

# 5. Compile evolved champion topology to standalone C header
python main.py --export-c

# 6. Launch real-time web visualizer & interactive sandbox
python ui_server.py
```

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --modes {performance,complexity,hardware} [...]
                        Modes to run (default: all three)
  --seeds SEED [SEED ...]
                        Random seeds (default: 42 123 456 789 1024)
  --generations N       NEAT generations per run (default: 100)
  --pop-size N          Population size (default: 150)
  --results-dir DIR     Output directory (default: results/)
  --compare             Only generate report from existing CSV
  --stress-test         Evaluate robustness under environmental shifts
  --export-c [TOPOLOGY] Export network topology to standalone C header controller.h
```

### Examples

```bash
# Quick test with fewer generations
python main.py --modes performance --seeds 42 --generations 20

# Compare only complexity and hardware with 7 seeds
python main.py --modes complexity hardware --seeds 1 2 3 4 5 6 7

# High-fidelity run
python main.py --generations 200 --pop-size 300
```

## Environment: CartPoleSwingUpEnv

| Property | Value |
|----------|-------|
| Initial pole angle | π (hanging down) |
| Observation | `[x, ẋ, cos θ, sin θ, θ̇]` — 5 dimensions |
| Action | Continuous ∈ [−1, 1] → force ∈ [−10, 10] N |
| Reward | `cos(θ) − 0.01·(x/2.4)²` |
| Termination | Cart exits track (\|x\| > 2.4) |
| Truncation | 200 steps |

## Configuration

All tuneable parameters live in
[`config/experiment_config.py`](config/experiment_config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 150 | NEAT population |
| `generations` | 100 | Generations per run |
| `max_episode_steps` | 200 | Env steps per episode |
| `num_eval_episodes` | 3 | Episodes per genome evaluation |
| `survival_threshold` | 0.0 | Reward gate for penalties |
| `complexity_penalty_weight` | 0.5 | λ for complexity mode |
| `hardware_penalty_weight` | 0.3 | λ for hardware mode |
| `success_threshold` | 100.0 | Reward for success flag |

## Outputs

### `results/results.csv`

Machine-readable results — one row per (seed, mode):

```
seed,mode,best_reward,mean_reward,hidden_neurons,active_connections,estimated_operations,success_flag
42,performance,145.23,87.61,3,12,16,True
42,complexity,138.91,92.10,1,7,9,True
42,hardware,130.55,85.44,1,6,8,True
```

### `results/comparison_report.md`

Markdown table averaging across seeds with mean ± std.

### `results/stress_test_results.csv`

Robustness evaluation records across 4 perturbation scenarios:
`mode, seed, scenario, mean_reward, baseline_reward, retention_pct`

### `results/robustness_comparison.png`

Publication-quality bar chart displaying retention percentage across shift scenarios.

### `controller.h`

Standalone, zero-dependency ANSI C header file with `run_controller(const float state[5])`.

### `results/<mode>/seed_<N>/`

- `best_genome.pkl` — pickled best genome + NEAT config
- `topology.json` — JSON graph (nodes, connections, types, weights)

## Metrics

| Metric | Computation |
|--------|-------------|
| Hidden neurons | `len(genome.nodes) − num_outputs` |
| Active connections | Count of enabled connection genes |
| Estimated operations | `active_connections + hidden_neurons + num_outputs` |

## Reproducibility

- Python's `random` and NumPy's `np.random` are seeded before each run.
- All genomes in a generation share the same environment seeds.
- The final champion is re-evaluated with a *separate* set of env seeds.

## License

MIT
