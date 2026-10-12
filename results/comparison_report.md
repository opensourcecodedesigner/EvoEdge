# Experimental Results – NEAT CartPole Swing-Up

## Statistical Comparison Across Modes

| Mode | Best Reward | Mean Reward | Hidden Neurons | Active Conns | Est. Ops | Success Rate |
|------|-------------|-------------|----------------|--------------|----------|--------------|
| performance | -21.23 ± 0.00 | -158.11 ± 0.00 | 2.0 ± 0.0 | 4.0 ± 0.0 | 7.0 ± 0.0 | 0% (0/1) |
| hardware | -24.11 ± 0.00 | -136.02 ± 0.00 | 2.0 ± 0.0 | 6.0 ± 0.0 | 9.0 ± 0.0 | 0% (0/1) |

### Column Definitions

| Column | Meaning |
|--------|---------|
| **Best Reward** | Raw (unpenalised) reward of the best evolved genome |
| **Mean Reward** | Population mean fitness in the final generation (includes any penalty) |
| **Hidden Neurons** | Hidden nodes in the best genome's topology |
| **Active Conns** | Enabled synaptic connections in the best genome |
| **Est. Ops** | Estimated single-sample inference cost (connections + nodes) |
| **Success Rate** | Fraction of seeds where best reward exceeded the success threshold |
