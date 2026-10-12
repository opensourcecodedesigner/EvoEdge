# Benchmark Suite: Controller Comparison Report

## Task A: Inverted Balancing (theta = 0 +/- 0.2 rad)

| Controller | Mean Reward | Std Reward | Boundary Violation Rate (%) | Steady-State Jitter RMS (deg) | Active Parameters | Theoretical MACs |
|------------|-------------|------------|-----------------------------|-------------------------------|-------------------|------------------|
| **Classical LQR** | 499.66 | 0.27 | 0.0% | 0.0024° | 4 | 4 |
| **Hybrid ES-LQR** | 499.69 | 0.25 | 0.0% | 0.3911° | 7 | 7 |
| **TinyML-Neural** | 495.31 | 35.27 | 1.0% | 2.4888° | 22 | 22 |

## Task B: Full Swing-Up (theta = pi +/- 0.2 rad)

| Controller | Mean Reward | Std Reward | Boundary Violation Rate (%) | Steady-State Jitter RMS (deg) | Active Parameters | Theoretical MACs |
|------------|-------------|------------|-----------------------------|-------------------------------|-------------------|------------------|
| **Classical LQR** | -498.23 | 2.40 | 0.0% | 175.3654° | 4 | 4 |
| **Hybrid ES-LQR** | -418.61 | 13.86 | 0.0% | 148.8594° | 7 | 7 |
| **TinyML-Neural** | -25.70 | 95.07 | 94.0% | 4.2776° | 22 | 22 |