# Independent Review & Handoff Report (Phase 1: Verification & Audit)

## Executive Summary

This handoff report documents the empirical audit of the CartPole swing-up and balance repository (`d:/neat`). All code, baseline metrics, and discrepancy measurements presented below were collected directly from the active files and executed commands without modification to code or evaluation definitions.

---

## 1. Verbatim Environment Implementation & Parameter Analysis

### File: `env/cartpole_swingup.py`

```python
"""CartPole Swing-Up environment.

A variant of the classic CartPole where the pole starts hanging downward
(θ = π rad) and the agent must swing it up to the inverted position (θ = 0)
while keeping the cart near the centre of the track.

Implements gymnasium.Env so it can be used with any standard RL or
neuroevolution loop.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CartPoleSwingUpEnv(gym.Env):
    """CartPole Swing-Up with continuous or discrete force control.

    Observation (5-dim continuous):
        Index  Meaning
        0      Cart position              x       ∈ [−4.8, 4.8]
        1      Cart velocity              ẋ
        2      cos(pole angle)            cos θ   ∈ [−1, 1]
        3      sin(pole angle)            sin θ   ∈ [−1, 1]
        4      Pole angular velocity      θ̇

    Action:
        continuous  →  scalar ∈ [−1, 1], scaled to [−force_mag, force_mag]
        discrete    →  {0, 1} mapped to {−force_mag, +force_mag}

    Reward per step:
        cos(θ) − 0.01·(x / x_threshold)²
        +1 when the pole is perfectly upright and the cart is centred;
        −1 when the pole hangs straight down.

    Termination:
        |x| > x_threshold  (cart fell off the track).

    Truncation:
        step count ≥ max_steps.
    """

    metadata: dict = {"render_modes": []}

    # ── Physics constants ────────────────────────────────────
    GRAVITY: float = 9.8
    MASS_CART: float = 1.0
    MASS_POLE: float = 0.1
    LENGTH: float = 0.5        # half-length of the pole
    FORCE_MAG: float = 10.0
    TAU: float = 0.02          # integration timestep (s)
    X_THRESHOLD: float = 2.4   # track half-width

    def __init__(
        self,
        max_steps: int = 200,
        continuous: bool = True,
    ) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.continuous = continuous

        # Derived constants (cached for speed)
        self._total_mass = self.MASS_CART + self.MASS_POLE
        self._polemass_length = self.MASS_POLE * self.LENGTH

        # Action space
        if continuous:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(2)

        # Observation space: [x, x_dot, cos_theta, sin_theta, theta_dot]
        high = np.array(
            [
                self.X_THRESHOLD * 2.0,
                np.finfo(np.float32).max,
                1.0,
                1.0,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Internal state: [x, x_dot, theta, theta_dot]
        self._state: np.ndarray | None = None
        self._step_count: int = 0

    # ── Gymnasium API ────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pole starts hanging down (θ = π) with a small random perturbation
        self._state = np.array(
            [0.0, 0.0, np.pi, 0.0], dtype=np.float64
        )
        self._state += self.np_random.uniform(
            low=-0.05, high=0.05, size=(4,)
        )
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "Call reset() before step()."

        x, x_dot, theta, theta_dot = self._state

        # ── Compute applied force ────────────────────────────
        if self.continuous:
            if isinstance(action, (list, np.ndarray)):
                action = action[0]
            force = float(np.clip(action, -1.0, 1.0)) * self.FORCE_MAG
        else:
            force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        # ── Cart-pole dynamics (semi-implicit Euler) ─────────
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        temp = (
            force + self._polemass_length * theta_dot ** 2 * sin_theta
        ) / self._total_mass

        theta_acc = (
            self.GRAVITY * sin_theta - cos_theta * temp
        ) / (
            self.LENGTH
            * (4.0 / 3.0 - self.MASS_POLE * cos_theta ** 2 / self._total_mass)
        )

        x_acc = (
            temp
            - self._polemass_length * theta_acc * cos_theta / self._total_mass
        )

        # Euler integration
        x_dot = x_dot + self.TAU * x_acc
        x = x + self.TAU * x_dot
        theta_dot = theta_dot + self.TAU * theta_acc
        theta = theta + self.TAU * theta_dot

        # Wrap θ to [−π, π]
        theta = ((theta + np.pi) % (2.0 * np.pi)) - np.pi

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self._step_count += 1

        # ── Reward ───────────────────────────────────────────
        reward = float(np.cos(theta)) - 0.01 * (x / self.X_THRESHOLD) ** 2

        # ── Done conditions ──────────────────────────────────
        terminated = bool(abs(x) > self.X_THRESHOLD)
        truncated = self._step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    # ── Helpers ──────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return the 5-dim observation from the internal 4-dim state."""
        x, x_dot, theta, theta_dot = self._state
        return np.array(
            [x, x_dot, np.cos(theta), np.sin(theta), theta_dot],
            dtype=np.float32,
        )
```

### Analysis of `env/cartpole_swingup.py`:
- **Reward Formula:** `reward = float(np.cos(theta)) - 0.01 * (x / self.X_THRESHOLD) ** 2` (where `X_THRESHOLD = 2.4`). Range is approximately $[-1.01, +1.0]$.
- **Max Steps:** Default `max_steps = 200` (in `__init__`). Truncation flag is `truncated = self._step_count >= self.max_steps`.
- **Termination:** `terminated = bool(abs(x) > self.X_THRESHOLD)` (cart exits track $|x| > 2.4\text{ m}$).
- **Initial State:** Hanging downward ($\theta = \pi$) centered on track: `[0.0, 0.0, np.pi, 0.0] + self.np_random.uniform(low=-0.05, high=0.05, size=(4,))`.
- **Seeding Method:** `super().reset(seed=seed)` seeds Gym's internal PRNG generator (`self.np_random`).
- **How Perturbations are Applied:** `CartPoleSwingUpEnv` contains **no perturbation parameters**; physical constants (`MASS_CART = 1.0`, `MASS_POLE = 0.1`, `LENGTH = 0.5`) are fixed class variables. Perturbations were implemented by subclassing in `BenchmarkCartPoleEnv`.

---

## 2. Verbatim TinyML Evolution Script & Audit

### File: `evolution/evolve_tinyml.py`

```python
"""Fast evolutionary optimization for a compact feedforward TinyML neural controller.

Topology:
  - Inputs (5): [x / 2.4, clip(x_dot / 10, -1, 1), cos(theta), sin(theta), clip(theta_dot / 10, -1, 1)]
  - Hidden Layer (3): 3 neurons, Tanh activation (15 weights + 3 biases = 18 MACs)
  - Output Layer (1): 1 neuron, Tanh activation (3 weights + 1 bias = 4 MACs)
  - Total Parameters: 22
  - Total MAC Count: 22 MACs (within ~20–25 MAC range)

Targets Task A (Balancing) and Task B (Swing-Up) with fixed environment reward.
Saves champion as results/champion_neural_genome.pkl and results/neural_controller.h.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import numpy as np

# Ensure project root in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import neat
from env.cartpole_swingup import CartPoleSwingUpEnv
from experiments.benchmark_suite import generate_initial_condition_grid, BenchmarkCartPoleEnv


NUM_INPUTS = 5
NUM_HIDDEN = 3
NUM_OUTPUTS = 1
NUM_PARAMS = NUM_INPUTS * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_OUTPUTS + NUM_OUTPUTS  # 15 + 3 + 3 + 1 = 22


def unpack_params(params: np.ndarray):
    """Unpack flat 22-element parameter vector into weights and biases."""
    idx = 0
    W1 = params[idx: idx + NUM_INPUTS * NUM_HIDDEN].reshape(NUM_HIDDEN, NUM_INPUTS)
    idx += NUM_INPUTS * NUM_HIDDEN
    b1 = params[idx: idx + NUM_HIDDEN]
    idx += NUM_HIDDEN
    W2 = params[idx: idx + NUM_HIDDEN * NUM_OUTPUTS].reshape(NUM_OUTPUTS, NUM_HIDDEN)
    idx += NUM_HIDDEN * NUM_OUTPUTS
    b2 = params[idx: idx + NUM_OUTPUTS]
    return W1, b1, W2, b2


def _act_tanh(z: np.ndarray) -> np.ndarray:
    """NEAT-Python standard scaled hyperbolic tangent."""
    scaled = np.clip(2.5 * z, -60.0, 60.0)
    return np.tanh(scaled)


def forward_step(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, obs: np.ndarray) -> float:
    """Execute single feedforward inference step."""
    norm_obs = np.array([
        obs[0] / 2.4,
        np.clip(obs[1] / 10.0, -1.0, 1.0),
        obs[2],
        obs[3],
        np.clip(obs[4] / 10.0, -1.0, 1.0),
    ], dtype=np.float32)

    hidden = _act_tanh(W1 @ norm_obs + b1)
    out = float(_act_tanh(W2 @ hidden + b2)[0])
    return out


def evaluate_candidate(
    params: np.ndarray,
    task_a_conditions: list,
    task_b_conditions: list,
    max_steps: int = 400,
) -> float:
    """Evaluate candidate parameter vector across Task A and Task B."""
    W1, b1, W2, b2 = unpack_params(params)
    env = BenchmarkCartPoleEnv(max_steps=max_steps, continuous=True)

    total_fitness = 0.0
    episodes = 0

    # Task A (Balancing)
    for seed, (x0, th0) in task_a_conditions:
        obs, _ = env.reset_with_initial_state(x0, th0, seed=seed)
        tot_rew = 0.0
        for _ in range(max_steps):
            u = forward_step(W1, b1, W2, b2, obs)
            obs, r, term, trunc, _ = env.step(u)
            tot_rew += r
            if term:
                steps_remaining = max_steps - env._step_count
                tot_rew -= steps_remaining * 2.0
                break
        total_fitness += tot_rew
        episodes += 1

    # Task B (Swing-Up)
    for seed, (x0, th0) in task_b_conditions:
        obs, _ = env.reset_with_initial_state(x0, th0, seed=seed)
        tot_rew = 0.0
        for _ in range(max_steps):
            u = forward_step(W1, b1, W2, b2, obs)
            obs, r, term, trunc, _ = env.step(u)
            tot_rew += r
            if term:
                steps_remaining = max_steps - env._step_count
                tot_rew -= steps_remaining * 2.0
                break
        total_fitness += tot_rew
        episodes += 1

    return total_fitness / max(episodes, 1)


def evolve_champion(
    pop_size: int = 60,
    generations: int = 35,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Run genetic optimization / evolution strategy to find champion parameters."""
    rng = np.random.default_rng(seed)

    # Fast evaluation seeds during generations (3 Task A + 3 Task B)
    train_seeds_a = list(generate_initial_condition_grid(range(1000, 1003), theta_center=0.0).items())
    train_seeds_b = list(generate_initial_condition_grid(range(1000, 1003), theta_center=np.pi).items())

    # Population initialization
    population = rng.normal(0.0, 0.4, size=(pop_size, NUM_PARAMS))

    # Prior 1: High balance gains in unit 0
    pop0 = np.zeros(NUM_PARAMS)
    W1_p0, b1_p0, W2_p0, b2_p0 = unpack_params(pop0)
    W1_p0[0, 0] = 0.24
    W1_p0[0, 1] = 1.28
    W1_p0[0, 2] = 0.50
    W1_p0[0, 3] = 3.50
    W1_p0[0, 4] = 2.32
    W2_p0[0, 0] = 2.00
    population[0] = np.concatenate([W1_p0.flatten(), b1_p0, W2_p0.flatten(), b2_p0])

    # Prior 2: Pumping unit + balance unit
    pop1 = np.zeros(NUM_PARAMS)
    W1_p1, b1_p1, W2_p1, b2_p1 = unpack_params(pop1)
    W1_p1[0, 0] = 0.20
    W1_p1[0, 1] = 1.20
    W1_p1[0, 3] = 3.50
    W1_p1[0, 4] = 2.30
    W2_p1[0, 0] = 2.50
    W1_p1[1, 2] = -2.0  # active when cos_th < 0
    W1_p1[1, 4] = 3.0   # angular velocity
    W2_p1[0, 1] = 1.50
    W1_p1[2, 0] = -1.50
    W1_p1[2, 1] = -1.00
    W2_p1[0, 2] = 0.80
    population[1] = np.concatenate([W1_p1.flatten(), b1_p1, W2_p1.flatten(), b2_p1])

    # Initial evaluation
    fitnesses = np.array([
        evaluate_candidate(ind, train_seeds_a, train_seeds_b) for ind in population
    ])

    best_idx = np.argmax(fitnesses)
    best_fitness = fitnesses[best_idx]
    best_params = population[best_idx].copy()

    if verbose:
        print(f"[Init] Best Fitness: {best_fitness:.2f} | Mean: {np.mean(fitnesses):.2f}", flush=True)

    start_time = time.time()

    for gen in range(generations):
        # Elitism: retain top 4
        elite_indices = np.argsort(fitnesses)[-4:]
        new_population = [population[i].copy() for i in elite_indices]

        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection (size 3)
            tourn_a = rng.choice(pop_size, size=3, replace=False)
            parent_a = population[tourn_a[np.argmax(fitnesses[tourn_a])]]

            tourn_b = rng.choice(pop_size, size=3, replace=False)
            parent_b = population[tourn_b[np.argmax(fitnesses[tourn_b])]]

            # Uniform crossover
            mask = rng.random(NUM_PARAMS) < 0.5
            child = np.where(mask, parent_a, parent_b)

            # Gaussian mutation with decaying variance
            mut_prob = 0.25
            mut_scale = 0.20 * (0.97 ** gen)
            mut_mask = rng.random(NUM_PARAMS) < mut_prob
            child[mut_mask] += rng.normal(0.0, mut_scale, size=np.sum(mut_mask))

            new_population.append(child)

        population = np.array(new_population)

        # Evaluate
        fitnesses = np.array([
            evaluate_candidate(ind, train_seeds_a, train_seeds_b) for ind in population
        ])

        cur_best_idx = np.argmax(fitnesses)
        cur_best_fitness = fitnesses[cur_best_idx]

        if cur_best_fitness > best_fitness:
            best_fitness = cur_best_fitness
            best_params = population[cur_best_idx].copy()

        if verbose and ((gen + 1) % 5 == 0 or gen == generations - 1):
            elapsed = time.time() - start_time
            print(f"Gen {gen+1:2d}/{generations} | Best Fitness: {best_fitness:>7.2f} | "
                  f"Pop Mean: {np.mean(fitnesses):>7.2f} | Time: {elapsed:.1f}s", flush=True)

    # Final selection over broader evaluation
    full_seeds_a = list(generate_initial_condition_grid(range(1000, 1020), theta_center=0.0).items())
    full_seeds_b = list(generate_initial_condition_grid(range(1000, 1020), theta_center=np.pi).items())
    
    top_candidates = [population[i] for i in np.argsort(fitnesses)[-5:]]
    top_candidates.append(best_params)
    
    full_scores = [
        evaluate_candidate(cand, full_seeds_a, full_seeds_b, max_steps=500)
        for cand in top_candidates
    ]
    champion = top_candidates[np.argmax(full_scores)]
    if verbose:
        print(f"[Champion Validation] 40-Episode Score: {np.max(full_scores):.2f}", flush=True)

    return champion


def export_to_neat_genome(
    params: np.ndarray,
    config_path: str = "config/neat_config.ini",
    output_pkl: str = "results/champion_neural_genome.pkl",
) -> None:
    """Save parameter vector as a standard neat.DefaultGenome pickle dictionary."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    W1, b1, W2, b2 = unpack_params(params)

    genome = neat.DefaultGenome(1)
    genome.fitness = 450.0  # high fitness

    # Output node 0
    node_0 = config.genome_config.node_gene_type(0)
    node_0.bias = float(b2[0])
    node_0.response = 1.0
    node_0.activation = "tanh"
    node_0.aggregation = "sum"
    genome.nodes[0] = node_0

    # Hidden nodes 1, 2, 3
    for h in range(NUM_HIDDEN):
        node_h = config.genome_config.node_gene_type(h + 1)
        node_h.bias = float(b1[h])
        node_h.response = 1.0
        node_h.activation = "tanh"
        node_h.aggregation = "sum"
        genome.nodes[h + 1] = node_h

    # Connections: inputs (-1..-5) to hidden (1..3)
    inno = 1
    for h in range(NUM_HIDDEN):
        for i in range(NUM_INPUTS):
            input_id = -(i + 1)
            hidden_id = h + 1
            cg = config.genome_config.connection_gene_type((input_id, hidden_id), innovation=inno)
            cg.weight = float(W1[h, i])
            cg.enabled = True
            genome.connections[(input_id, hidden_id)] = cg
            inno += 1

    # Connections: hidden (1..3) to output (0)
    for h in range(NUM_HIDDEN):
        hidden_id = h + 1
        cg = config.genome_config.connection_gene_type((hidden_id, 0), innovation=inno)
        cg.weight = float(W2[0, h])
        cg.enabled = True
        genome.connections[(hidden_id, 0)] = cg
        inno += 1

    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    with open(output_pkl, "wb") as fh:
        pickle.dump({"genome": genome, "config": config}, fh)

    print(f"[OK] Saved NEAT champion genome -> {output_pkl}", flush=True)


def export_to_c_header(
    params: np.ndarray,
    output_header: str = "results/neural_controller.h",
) -> None:
    """Export parameters to standalone TinyML C header with float compute_neural_action(obs)."""
    W1, b1, W2, b2 = unpack_params(params)

    # Documented MAC count
    # Input to hidden: 5 * 3 = 15 mults + 3 bias adds = 18 MACs
    # Hidden to output: 3 * 1 = 3 mults + 1 bias add = 4 MACs
    # Total MACs = 22 MACs
    mac_count = 22

    lines = [
        "/* ========================================================================",
        " * TinyML Neural Controller — Compact Evolved Feedforward Architecture",
        " * ========================================================================",
        " * Architecture: 5 Inputs -> 3 Hidden (Tanh) -> 1 Output (Tanh)",
        " *",
        " * Theoretical Complexity & Embedded Microcontroller Metrics:",
        " *   - Total Parameters:          22 (18 weights + 4 biases)",
        f" *   - Multiply-Accumulate (MAC): {mac_count} MACs per inference step",
        " *   - RAM Footprint:             < 64 bytes (pure stack, 0 bytes heap)",
        " *   - MCU Execution Time:        ~0.85 us on ESP32 / Xtensa LX6 @ 240 MHz",
        " *   - Hardware Target:           ESP32 / ARM Cortex-M4 / TinyML Embedded",
        " *",
        " * Observation Vector Format (const float obs[5]):",
        " *   obs[0] : Cart Position (m)             [normalized internally by / 2.4]",
        " *   obs[1] : Cart Velocity (m/s)           [normalized internally by clip(/ 10, -1, 1)]",
        " *   obs[2] : Pole Cosine Angle (cos theta) [-1.0, 1.0]",
        " *   obs[3] : Pole Sine Angle (sin theta)   [-1.0, 1.0]",
        " *   obs[4] : Pole Angular Velocity (rad/s) [normalized internally by clip(/ 10, -1, 1)]",
        " *",
        " * Return Value:",
        " *   float  : Continuous control force in [-1.0, 1.0]",
        " * ======================================================================== */",
        "",
        "#ifndef NEURAL_CONTROLLER_H",
        "#define NEURAL_CONTROLLER_H",
        "",
        "#include <math.h>",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        f"#define NEURAL_NUM_PARAMS 22",
        f"#define NEURAL_NUM_MACS   {mac_count}",
        "",
        "/* Fast clamp helper */",
        "static inline float _neural_clamp(float v, float min_v, float max_v) {",
        "    if (v < min_v) return min_v;",
        "    if (v > max_v) return max_v;",
        "    return v;",
        "}",
        "",
        "/* Scaled hyperbolic tangent matching NEAT-Python specification: tanh(2.5 * z) */",
        "static inline float _neat_act_tanh(float z) {",
        "    float scaled = 2.5f * z;",
        "    if (scaled > 60.0f) scaled = 60.0f;",
        "    else if (scaled < -60.0f) scaled = -60.0f;",
        "    return tanhf(scaled);",
        "}",
        "",
        "/**",
        " * Compute continuous control action using the evolved TinyML neural network.",
        " *",
        " * @param obs  Raw observation array [x, x_dot, cos_theta, sin_theta, theta_dot]",
        " * @return     Control action in [-1.0, 1.0]",
        " */",
        "static inline float compute_neural_action(const float obs[5]) {",
        "    /* 1. Feature normalization matching training inputs */",
        "    const float x_norm      = obs[0] / 2.4f;",
        "    const float x_dot_norm  = _neural_clamp(obs[1] / 10.0f, -1.0f, 1.0f);",
        "    const float cos_th      = obs[2];",
        "    const float sin_th      = obs[3];",
        "    const float th_dot_norm = _neural_clamp(obs[4] / 10.0f, -1.0f, 1.0f);",
        "",
        "    /* 2. Hidden Layer (3 Neurons, Tanh activations) — 18 MACs */",
    ]

    for h in range(NUM_HIDDEN):
        w = W1[h]
        b = b1[h]
        lines.append(f"    /* Hidden Neuron {h} */")
        lines.append(f"    float h{h}_sum = ({b:+.6f}f);")
        lines.append(f"    h{h}_sum += x_norm      * ({w[0]:+.6f}f);")
        lines.append(f"    h{h}_sum += x_dot_norm  * ({w[1]:+.6f}f);")
        lines.append(f"    h{h}_sum += cos_th      * ({w[2]:+.6f}f);")
        lines.append(f"    h{h}_sum += sin_th      * ({w[3]:+.6f}f);")
        lines.append(f"    h{h}_sum += th_dot_norm * ({w[4]:+.6f}f);")
        lines.append(f"    const float h{h} = _neat_act_tanh(h{h}_sum);")
        lines.append("")

    lines.extend([
        "    /* 3. Output Layer (1 Neuron, Tanh activation) — 4 MACs */",
        f"    float out_sum = ({b2[0]:+.6f}f);",
    ])
    for h in range(NUM_HIDDEN):
        lines.append(f"    out_sum += h{h} * ({W2[0, h]:+.6f}f);")

    lines.extend([
        "    const float action = _neat_act_tanh(out_sum);",
        "",
        "    /* Saturate to [-1.0, 1.0] */",
        "    return _neural_clamp(action, -1.0f, 1.0f);",
        "}",
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        "#endif /* NEURAL_CONTROLLER_H */",
        "",
    ])

    header_code = "\n".join(lines)
    os.makedirs(os.path.dirname(output_header) or ".", exist_ok=True)
    with open(output_header, "w", encoding="utf-8") as fh:
        fh.write(header_code)

    print(f"[OK] Saved TinyML C header -> {output_header} (documented {mac_count} MACs)", flush=True)


def main():
    print("=" * 70, flush=True)
    print("  Evolutionary Optimization for TinyML Neural Controller (Task A & B)", flush=True)
    print("=" * 70, flush=True)
    champion_params = evolve_champion(pop_size=60, generations=35, seed=42, verbose=True)

    # Save pickle
    export_to_neat_genome(champion_params, output_pkl="results/champion_neural_genome.pkl")

    # Save C header
    export_to_c_header(champion_params, output_header="results/neural_controller.h")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
```

### Analysis of `evolution/evolve_tinyml.py`:
- **Start State Used in Training:** Mixed: 3 conditions from Task A (balancing, $\theta_0 \approx 0 \pm 0.2$ rad) and 3 conditions from Task B (hanging, $\theta_0 \approx \pi \pm 0.2$ rad).
- **Episodes Per Genome:** 6 episodes during generational loop (3 Task A + 3 Task B). Final selection on top 5 candidates uses 40 episodes (20 Task A + 20 Task B).
- **Training Seeds:** Generational loop uses seeds `[1000, 1001, 1002]`; final candidate validation uses seeds `[1000..1019]`.
- **Termination Penalty:** Yes: `tot_rew -= (max_steps - env._step_count) * 2.0` on track boundary exit.
- **Optimizer:** Genetic algorithm / Real-valued evolution strategy with size-3 tournament selection, uniform crossover, Gaussian mutation with exponential variance decay ($0.20 \times 0.97^{\text{gen}}$), and top-4 elitism.
- **Population Size:** 60 (`pop_size = 60`).
- **Generations:** 35 (`generations = 35`).
- **Hidden Size:** Fixed 3 hidden neurons (`NUM_HIDDEN = 3`), fully connected feedforward topology ($5 \to 3 \to 1$).
- **Overlap with Evaluation Seeds (1000–1099):** **YES (100% overlap)**. The training seeds `[1000, 1001, 1002]` and validation seeds `[1000..1019]` directly overlap the evaluation seeds 1000..1099.

---

## 3. Verbatim Benchmark Suite Implementation & Audit

### File: `experiments/benchmark_suite.py`

*(See lines 1 to 853 of `experiments/benchmark_suite.py`)*

### Analysis of `experiments/benchmark_suite.py`:
- **Tasks Listed:**
  - **Task A (Balancing):** `theta_center = 0.0`, `theta_spread = 0.2`, `x_range = (-1.5, 1.5)` m.
  - **Task B (Swing-Up):** `theta_center = np.pi`, `theta_spread = 0.2`, `x_range = (-1.5, 1.5)` m.
- **How Mass and Noise Perturbations are Defined:**
  - **Mass Perturbation:** Implemented in `BenchmarkCartPoleEnv.__init__` via `self.MASS_CART = 1.0 * self.cart_mass_scale`. For $+30\%$ perturbation, `cart_mass_scale = 1.30`, giving $M_{\text{cart}} = 1.30\text{ kg}$ and total system mass $M_{\text{total}} = 1.40\text{ kg}$. Pole mass is unperturbed ($m = 0.1\text{ kg}$).
  - **Noise Perturbation:** Implemented in `BenchmarkCartPoleEnv.reset_with_initial_state` and `BenchmarkCartPoleEnv.step` via:
    ```python
    if self.obs_noise_sigma > 0.0:
        noise = self._rng.normal(0.0, self.obs_noise_sigma, size=obs.shape).astype(np.float32)
        obs = obs + noise
    ```
- **Which Observation Elements Receive Noise:**
  - Because `size=obs.shape` is used on the 5-dimensional observation vector, **ALL 5 elements** ($x, \dot{x}, \cos\theta, \sin\theta, \dot{\theta}$) receive additive i.i.d. Gaussian noise $\mathcal{N}(0, \sigma^2)$. Notice this independently corrupts $\cos\theta$ and $\sin\theta$, breaking the unit circle constraint $\cos^2\theta + \sin^2\theta = 1$.

---

## 4. Empirical Swing-Up Run (500 Steps, Seeds 1000–1099)

### Evaluation Protocol
- **Episode Length:** 500 steps.
- **Evaluation Seeds:** 1000 to 1099 (100 seeds).
- **Strict SUCCESS Definition:** `no rail hit (|x| <= 2.4 throughout) AND cos(theta) > 0.9 for at least 95 of the final 100 steps of the 500-step episode`.
- **Controllers Evaluated:**
  1. Classical LQR (`LQRController(method="continuous")`)
  2. Hybrid Teacher (`HybridController()`, gains $k_E=0.4, k_x=1.2, k_v=0.8$)
  3. TinyML Neural (`load_tinyml_champion()`, 5-3-1 net)

### Raw Execution Output

```text
================================================================================
EVALUATION 1: Standard CartPoleSwingUpEnv reset(seed=seed) [theta ~ pi, x ~ 0]
Episode Length: 500 steps, Seeds: 1000-1099
Success: no rail hit AND cos(theta) > 0.9 for >= 95 of final 100 steps
================================================================================
Controller: Classical LQR
  Success Rate:        0.0% (0/100)
  Rail Collision Rate: 0.0% (0/100)
  Median Steps to Upright: Never / None (Reached in 0/100 episodes)

Controller: Hybrid Teacher
  Success Rate:        0.0% (0/100)
  Rail Collision Rate: 0.0% (0/100)
  Median Steps to Upright: Never / None (Reached in 0/100 episodes)

Controller: TinyML Neural
  Success Rate:        0.0% (0/100)
  Rail Collision Rate: 100.0% (100/100)
  Median Steps to Upright: Never / None (Reached in 0/100 episodes)

================================================================================
EVALUATION 2: BenchmarkCartPoleEnv Grid Task B [theta_0 ~ pi +/- 0.2, x_0 in [-1.5, 1.5]]
Episode Length: 500 steps, Seeds: 1000-1099
Success: no rail hit AND cos(theta) > 0.9 for >= 95 of final 100 steps
================================================================================
Controller: Classical LQR
  Success Rate:        0.0% (0/100)
  Rail Collision Rate: 0.0% (0/100)
  Median Steps to Upright: Never / None (Reached in 0/100 episodes)

Controller: Hybrid Teacher
  Success Rate:        0.0% (0/100)
  Rail Collision Rate: 0.0% (0/100)
  Median Steps to Upright: Never / None (Reached in 0/100 episodes)

Controller: TinyML Neural
  Success Rate:        6.0% (6/100)
  Rail Collision Rate: 94.0% (94/100)
  Median Steps to Upright: 110.0 (Reached in 6/100 episodes)
```

### Empirical Findings:
1. **Classical LQR:** 0% success, 0% collisions. Because LQR is a linear controller designed for small perturbations around $\theta=0$, it cannot swing up from $\theta \approx \pi$; it stays hanging down indefinitely.
2. **Hybrid Teacher:** 0% success, 0% collisions, 0 episodes reaching upright. With the previous gains ($k_E=0.4, k_x=1.2, k_v=0.8$), the cart centering and velocity damping completely over-damp the system, starving the pendulum of mechanical energy so it never reaches the $\theta < 0.35$ rad capture basin.
3. **TinyML Neural:**
   - Standard hanging start ($x_0 \approx 0$): **0% success, 100% rail collisions**. The cart immediately accelerates toward the rail and crashes.
   - Grid Task B ($x_0 \in [-1.5, 1.5]$): **6% success, 94% rail collisions**.

---

## 5. Teacher Gains Verification

Comparing gains in `baselines/hybrid_controller.py` and `results/controller.h`:

| Parameter | `baselines/hybrid_controller.py` | `results/controller.h` | Match Confirmed? |
|---|---|---|---|
| Balancing Gain $K_x$ | `-0.8` | `HYBRID_K_X (-0.80f)` | **Yes** |
| Balancing Gain $K_{\dot{x}}$ | `-1.2` | `HYBRID_K_X_DOT (-1.20f)` | **Yes** |
| Balancing Gain $K_\theta$ | `-35.0` | `HYBRID_K_THETA (-35.00f)` | **Yes** |
| Balancing Gain $K_{\dot{\theta}}$ | `-6.5` | `HYBRID_K_THETA_DOT (-6.50f)` | **Yes** |
| Energy Pumping Gain $k_E$ | `0.4` | `HYBRID_K_E (0.40f)` | **Yes** |
| Centering Gain $k_x$ | `1.2` | `HYBRID_K_CENT_X (1.20f)` | **Yes** |
| Velocity Damping Gain $k_v$ | `0.8` | `HYBRID_K_CENT_V (0.80f)` | **Yes** |
| Hand-off Angle $|\theta|$ Threshold | `0.35` rad | `HYBRID_THETA_THRESHOLD_RAD (0.35f)` | **Yes** |
| Hand-off Velocity $|\dot{\theta}|$ Threshold | `2.5` rad/s | `HYBRID_THETA_DOT_THRESHOLD_RAD_S (2.50f)` | **Yes** |
| Kinetic Coefficient $c_{\text{kin}}$ | $0.5 \cdot m \cdot L^2 = 0.0125$ | $0.0125\text{f}$ (in `controller.c`) | **Yes** |
| Potential Coefficient $c_{\text{pot}}$ | $m \cdot g \cdot L = 0.4900$ | $0.4900\text{f}$ (in `controller.c`) | **Yes** |

**Conclusion:** The parameter definitions between Python and C are in exact synchronization.

---

## 6. Test Suite and Git Status Output

### Output of `python -m unittest test_baselines.py`
```text
..............
----------------------------------------------------------------------
Ran 14 tests in 0.700s

OK
```

### Output of `git log --oneline -10`
```text
fatal: not a git repository (or any of the parent directories): .git
```

### Output of `git status --short`
```text
fatal: not a git repository (or any of the parent directories): .git
```

*(Note: The workspace directory `d:\neat` had not yet been initialized as a git repository; git initialization will occur in Phase 2).*

---

## 7. Status of the NEAT-Python Framework (`main.py`, `results/results.csv`)

### Verbatim Contents of `results/results.csv`:
```csv
seed,mode,best_reward,mean_reward,hidden_neurons,active_connections,estimated_operations,success_flag
42,performance,-21.2267,-158.1052,2,4,7,False
42,hardware,-24.1133,-136.021,2,6,9,False
```

### Summary of NEAT-Python Execution:
- The standard experiment sweep specified in `README.md` (3 modes $\times$ 5 seeds = 15 runs across 100 generations) **was not run**.
- Only 2 exploratory runs were recorded for seed 42 in `performance` and `hardware` modes.
- Neither run achieved swing-up success (`success_flag = False` for both, with best rewards of $-21.23$ and $-24.11$).
- The `complexity` mode was not executed in `results.csv`.
- The `results/champion_neural_genome.pkl` was produced by `evolution/evolve_tinyml.py` (a custom genetic optimizer for a fixed 5-3-1 MLP architecture), rather than by unconstrained topological NEAT evolution from `main.py`.

---

## 8. Python vs. C Neural Controller Parity (10,000 Rollout Observations)

### Measurement Protocol
- 10,000 real 5-dimensional observation vectors collected from CartPole simulation rollouts across diverse trajectories (including random exploratory forces to thoroughly exercise saturated regimes).
- Python evaluator: `neat.nn.FeedForwardNetwork` with feature normalization and `tanh(2.5 * z)` scaling.
- C evaluator: `compute_neural_action(obs)` compiled via GCC with `-O3` into a shared library and executed on identical inputs via `ctypes`.

### Raw Comparison Results:
```text
Total observations evaluated: 10000
Max absolute action difference: 1.4305710401e-05
Mean absolute action difference: 2.2700919707e-07
Observations with diff > 1e-6: 398
Observations with diff > 1e-5: 33
Min Python action: -0.999993, Max Python action: 0.999998
Min C action:      -0.999993, Max C action:      0.999998
```

### Conclusion:
The actual maximum absolute discrepancy between the Python model and the exported C code is **$1.43 \times 10^{-5}$** (single-precision float vs. double-precision intermediate roundoff in `tanhf`). The previous claim of "parity to 8 decimal places" is unsupported; real parity is within **$1.5 \times 10^{-5}$** (5 decimal places).

---

## 9. Inventory of ESP32 Latency and MCU Cycle Claims

| File | Line | Verbatim Claim / Text | Measured on Physical Hardware? | Reality / Source |
|---|---|---|---|---|
| `results/neural_controller.h` | 10 | `* - MCU Execution Time: ~0.85 us on ESP32 / Xtensa LX6 @ 240 MHz` | **NO** | Theoretical heuristic ($22\text{ MACs} \times \text{cycles} / 240\text{ MHz}$) |
| `results/controller.h` | 17 | `* - Runtime Execution Time: ~0.20 microseconds on ESP32 @ 240 MHz` | **NO** | Theoretical heuristic ($7\text{ MACs} / 240\text{ MHz}$) |
| `results/controller.h` | 32 | `#define HYBRID_EST_TIME_US_ESP32 0.20f` | **NO** | Hardcoded macro estimate |
| `evolution/evolve_tinyml.py` | 325 | `* - MCU Execution Time: ~0.85 us on ESP32 / Xtensa LX6 @ 240 MHz` | **NO** | Template string copied into exported C header |
| `results/final_evaluation_report.md` | 16 | `Estimated cycle bounds on Xtensa LX6 @ 240MHz: ~120–180 cycles` | **NO** | Analytical cycle bound estimate |
| `results/final_evaluation_report.md` | 17 | `Estimated cycle bounds on Xtensa LX6 @ 240MHz: ~200–280 cycles` | **NO** | Analytical cycle bound estimate |
| `results/final_evaluation_report.md` | 18 | `Estimated cycle bounds on Xtensa LX6 @ 240MHz: ~450–600 cycles including trig library` | **NO** | Analytical cycle bound estimate |
| `results/final_evaluation_report.md` | 77 | `execution bounds remain well under 600 CPU cycles (< 2.5 microseconds)` | **NO** | Analytical cycle bound estimate |
| `experiments/benchmark_suite.py` | 21, 328, 470, 481, 497, 716, 718, 807 | Various instances of Xtensa cycle bounds (~120–180, ~200–280, ~450–600) | **NO** | Analytical cycle bound estimate |

**Plain Statement:** **Zero hardware measurements were performed on physical ESP32 or Xtensa silicon.** All execution times and cycle ranges are analytical approximations based on operation counts and instruction cycle estimates.
