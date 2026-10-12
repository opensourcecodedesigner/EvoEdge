"""Baselines & Pruning Experiment Suite (Fix 5).

Trains a 64x64 MLP controller with the identical evaluation protocol and budget
as the tiny net. Evaluates magnitude-pruned versions at 50%, 75%, and 90% (both
prune-only and prune-plus-finetune) on Task B and the perturbation suite.
Produces success vs parameters and success vs estimated ops plots with the Pareto frontier.
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.cartpole_swingup import CartPoleSwingUpEnv
from baselines.hybrid_controller import HybridController


class MLP64(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(2.5 * self.fc1(x))
        h2 = torch.tanh(2.5 * self.fc2(h1))
        out = torch.tanh(2.5 * self.fc3(h2))
        return out


def norm_obs(obs: np.ndarray) -> np.ndarray:
    return np.array([
        obs[0] / 2.4,
        np.clip(obs[1] / 10.0, -1.0, 1.0),
        obs[2],
        obs[3],
        np.clip(obs[4] / 10.0, -1.0, 1.0)
    ], dtype=np.float32)


class PerturbedEnv(CartPoleSwingUpEnv):
    def __init__(
        self,
        pole_mass_scale: float = 1.0,
        cart_mass_scale: float = 1.0,
        b_cart: float = 0.0,
        b_pole: float = 0.0,
        obs_noise_sigma: float = 0.0,
        delay_steps: int = 0,
        seed: int | None = None,
    ):
        super().__init__(max_steps=500, continuous=True)
        self.MASS_POLE = 0.1 * pole_mass_scale
        self.MASS_CART = 1.0 * cart_mass_scale
        self._total_mass = self.MASS_CART + self.MASS_POLE
        self._polemass_length = self.MASS_POLE * self.LENGTH
        self.b_cart = b_cart
        self.b_pole = b_pole
        self.obs_noise_sigma = obs_noise_sigma
        self.delay_steps = delay_steps
        self.action_queue = [0.0] * delay_steps
        self._rng = np.random.default_rng(seed if seed is not None else 42)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = super().reset(seed=seed, options=options)
        self.action_queue = [0.0] * self.delay_steps
        if self.obs_noise_sigma > 0.0:
            obs = obs + self._rng.normal(0.0, self.obs_noise_sigma, size=obs.shape).astype(np.float32)
        return obs, info

    def step(self, action):
        if self.delay_steps > 0:
            self.action_queue.append(action)
            act_applied = self.action_queue.pop(0)
        else:
            act_applied = action

        x, x_dot, theta, theta_dot = self._state
        force = float(np.clip(act_applied, -1.0, 1.0)) * self.FORCE_MAG

        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        temp = (force - self.b_cart * x_dot + self._polemass_length * theta_dot**2 * sin_th) / self._total_mass
        fric_pole = (self.b_pole / max(self._polemass_length, 1e-6)) * theta_dot
        theta_acc = (self.GRAVITY * sin_th - cos_th * temp - fric_pole) / (
            self.LENGTH * (4.0 / 3.0 - (self.MASS_POLE * cos_th**2) / self._total_mass)
        )
        x_acc = temp - (self._polemass_length * theta_acc * cos_th) / self._total_mass

        x_dot = x_dot + self.TAU * x_acc
        x = x + self.TAU * x_dot
        theta_dot = theta_dot + self.TAU * theta_acc
        theta = theta + self.TAU * theta_dot
        theta = ((theta + np.pi) % (2.0 * np.pi)) - np.pi

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self._step_count += 1
        reward = float(np.cos(theta)) - 0.01 * (x / self.X_THRESHOLD) ** 2
        terminated = bool(abs(x) > self.X_THRESHOLD)
        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()
        if self.obs_noise_sigma > 0.0:
            obs = obs + self._rng.normal(0.0, self.obs_noise_sigma, size=obs.shape).astype(np.float32)
        return obs, reward, terminated, truncated, {}


def evaluate_model_on_env(act_fn, seeds: range = range(1000, 1100), **env_kwargs) -> Tuple[int, int]:
    """Evaluate an action function on given seeds and environment perturbations."""
    succ = 0
    rail = 0
    for s in seeds:
        env = PerturbedEnv(seed=s, **env_kwargs)
        obs, _ = env.reset(seed=s)
        cos_thetas = []
        hit = False
        for _ in range(500):
            a = act_fn(obs)
            obs, r, term, trunc, _ = env.step(a)
            cos_thetas.append(obs[2])
            if term:
                hit = True
                break
        if not hit and len(cos_thetas) == 500 and sum(1 for c in cos_thetas[-100:] if c > 0.9) >= 95:
            succ += 1
        if hit:
            rail += 1
    return succ, rail


def prune_model(model: nn.Module, prune_ratio: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """Prune weights by absolute magnitude and return pruned model and masks."""
    pruned = copy.deepcopy(model)
    masks = {}
    with torch.no_grad():
        all_weights = []
        for name, p in pruned.named_parameters():
            if "weight" in name:
                all_weights.append(p.view(-1).abs())
        all_weights = torch.cat(all_weights)
        k = int(prune_ratio * len(all_weights))
        if k > 0:
            threshold = torch.kthvalue(all_weights, k).values.item()
            for name, p in pruned.named_parameters():
                if "weight" in name:
                    mask = (p.abs() > threshold).float()
                    p.mul_(mask)
                    masks[name] = mask
        else:
            for name, p in pruned.named_parameters():
                if "weight" in name:
                    masks[name] = torch.ones_like(p)
    return pruned, masks


def finetune_pruned(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    train_dataset: torch.utils.data.TensorDataset,
    epochs: int = 15,
    lr: float = 0.001,
) -> nn.Module:
    """Finetune non-zero weights of a pruned model while enforcing masks."""
    ft_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    ft_model.train()
    for _ in range(epochs):
        for bx, by in loader:
            pred = ft_model(bx)
            loss = nn.functional.mse_loss(pred, by)
            optimizer.zero_grad()
            loss.backward()
            # Zero out gradients for pruned weights
            with torch.no_grad():
                for name, p in ft_model.named_parameters():
                    if name in masks and p.grad is not None:
                        p.grad.mul_(masks[name])
            optimizer.step()
            # Re-apply masks
            with torch.no_grad():
                for name, p in ft_model.named_parameters():
                    if name in masks:
                        p.mul_(masks[name])
    return ft_model


def count_active_params_and_ops(model: nn.Module) -> Tuple[int, int]:
    """Count non-zero parameters and ops for feedforward inference."""
    non_zero_params = 0
    non_zero_macs = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            nz = int((p != 0).sum().item())
            non_zero_params += nz
            if "weight" in name:
                non_zero_macs += nz
    # Ops = 2 * MACs (multiply + add) + non-linear activations
    ops = non_zero_macs * 2 + 64 + 64 + 1
    return non_zero_params, ops


def main():
    print("=== Fix 5: Baselines & Pruning Suite Execution ===")
    torch.manual_seed(42)
    np.random.seed(42)

    teacher = HybridController()
    model = MLP64()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    all_X = []
    all_Y = []
    for s in range(20):
        env = CartPoleSwingUpEnv(max_steps=500, continuous=True)
        obs, _ = env.reset(seed=s)
        for _ in range(500):
            a = teacher.compute_action(obs)
            all_X.append(norm_obs(obs))
            all_Y.append(a)
            obs, _, term, trunc, _ = env.step(a)
            if term:
                break

    for dagger_iter in range(6):
        X_t = torch.tensor(np.array(all_X), dtype=torch.float32)
        Y_t = torch.tensor(np.array(all_Y).reshape(-1, 1), dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        model.train()
        for epoch in range(15):
            for bx, by in loader:
                pred = model(bx)
                loss = nn.functional.mse_loss(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        new_X = []
        new_Y = []
        batch_train_seeds = list(range(20 + dagger_iter * 20, 40 + dagger_iter * 20))
        for s in batch_train_seeds:
            env = CartPoleSwingUpEnv(max_steps=500, continuous=True)
            obs, _ = env.reset(seed=s)
            for _ in range(500):
                t_a = teacher.compute_action(obs)
                new_X.append(norm_obs(obs))
                new_Y.append(t_a)
                with torch.no_grad():
                    tin = torch.tensor(norm_obs(obs), dtype=torch.float32).unsqueeze(0)
                    st_a = float(model(tin).item())
                obs, _, term, trunc, _ = env.step(st_a)
                if term:
                    break
        all_X.extend(new_X)
        all_Y.extend(new_Y)

    # Save trained baseline MLP64
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/mlp64_baseline.pt")
    final_dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.array(all_X), dtype=torch.float32),
        torch.tensor(np.array(all_Y).reshape(-1, 1), dtype=torch.float32),
    )

    # Pruned variants
    models_dict = {
        "MLP 64x64 (Dense)": model,
    }

    for pr in [0.50, 0.75, 0.90]:
        pct = int(pr * 100)
        pruned, masks = prune_model(model, pr)
        models_dict[f"MLP 64x64 ({pct}% Pruned)"] = pruned

        ft_model = finetune_pruned(pruned, masks, final_dataset, epochs=15, lr=0.001)
        models_dict[f"MLP 64x64 ({pct}% Pruned + Fine-Tuned)"] = ft_model

    # Load TinyNet champions for plotting comparison
    with open("results/champion_neural_h8.pkl", "rb") as f:
        h8_data = pickle.load(f)
    w1_8, b1_8, w2_8, b2_8 = h8_data["w1"], h8_data["b1"], h8_data["w2"], h8_data["b2"]
    def act_tiny_h8(obs):
        norm = np.array([obs[0]/2.4, np.clip(obs[1]/10.0, -1.0, 1.0), obs[2], obs[3], np.clip(obs[4]/10.0, -1.0, 1.0)], dtype=np.float32)
        h = np.tanh(2.5 * (w1_8 @ norm + b1_8))
        out = np.tanh(2.5 * (w2_8 @ h + b2_8))
        return float(out[0])

    with open("results/champion_neural_h6.pkl", "rb") as f:
        h6_data = pickle.load(f)
    w1_6, b1_6, w2_6, b2_6 = h6_data["w1"], h6_data["b1"], h6_data["w2"], h6_data["b2"]
    def act_tiny_h6(obs):
        norm = np.array([obs[0]/2.4, np.clip(obs[1]/10.0, -1.0, 1.0), obs[2], obs[3], np.clip(obs[4]/10.0, -1.0, 1.0)], dtype=np.float32)
        h = np.tanh(2.5 * (w1_6 @ norm + b1_6))
        out = np.tanh(2.5 * (w2_6 @ h + b2_6))
        return float(out[0])

    with open("results/champion_neural_h3.pkl", "rb") as f:
        h3_data = pickle.load(f)
    w1_3, b1_3, w2_3, b2_3 = h3_data["w1"], h3_data["b1"], h3_data["w2"], h3_data["b2"]
    def act_tiny_h3(obs):
        norm = np.array([obs[0]/2.4, np.clip(obs[1]/10.0, -1.0, 1.0), obs[2], obs[3], np.clip(obs[4]/10.0, -1.0, 1.0)], dtype=np.float32)
        h = np.tanh(2.5 * (w1_3 @ norm + b1_3))
        out = np.tanh(2.5 * (w2_3 @ h + b2_3))
        return float(out[0])

    # Define perturbations
    perturbations = {
        "Nominal Task B": {},
        "Pole Mass +30%": {"pole_mass_scale": 1.3},
        "Cart Mass +30%": {"cart_mass_scale": 1.3},
        "Friction": {"b_cart": 0.5, "b_pole": 0.01},
        "Sensor Noise": {"obs_noise_sigma": 0.05},
        "1-Step Delay": {"delay_steps": 1},
        "2-Step Delay": {"delay_steps": 2},
        "Combined": {"pole_mass_scale": 1.3, "cart_mass_scale": 1.3, "b_cart": 0.5, "b_pole": 0.01, "obs_noise_sigma": 0.05, "delay_steps": 1},
    }

    # Evaluate all MLP variants on Task B and perturbations
    results_summary = []
    print("\n--- Evaluating Models Across Perturbation Suite (Seeds 1000..1099) ---")

    for model_name, m_variant in models_dict.items():
        m_variant.eval()
        n_params, ops = count_active_params_and_ops(m_variant)
        act_fn = lambda obs, m=m_variant: float(m(torch.tensor(norm_obs(obs), dtype=torch.float32).unsqueeze(0)).item())

        print(f"\nModel: {model_name} (Active Params: {n_params}, Est Ops: {ops})")
        variant_eval = {
            "name": model_name,
            "params": n_params,
            "ops": ops,
            "scores": {},
        }
        for cond_name, kwargs in perturbations.items():
            succ, rail = evaluate_model_on_env(act_fn, seeds=range(1000, 1100), **kwargs)
            variant_eval["scores"][cond_name] = (succ, rail)
            print(f"  {cond_name:<18}: Success = {succ:>3}/100, Rail = {rail:>3}%")
        results_summary.append(variant_eval)

    # Save evaluation summary to pickle
    with open("results/baselines_pruning_results.pkl", "wb") as f:
        pickle.dump(results_summary, f)

    # Prepare data for plotting
    # Combine TinyML models + MLP models
    plot_points = []
    # TinyML H=3: 22 params, 44 ops (MAC*2) + 4 act = 48 ops
    plot_points.append({
        "name": "TinyML (H=3)",
        "params": 22,
        "ops": 22 * 2 + 4,
        "nominal_succ": h3_data["eval_success"],
        "category": "TinyML",
    })
    # TinyML H=6: 43 params, 86 ops + 7 act = 93 ops
    plot_points.append({
        "name": "TinyML (H=6)",
        "params": 43,
        "ops": 43 * 2 + 7,
        "nominal_succ": h6_data["eval_success"],
        "category": "TinyML",
    })
    # TinyML H=8: 57 params, 114 ops + 9 act = 123 ops
    plot_points.append({
        "name": "TinyML (H=8)",
        "params": 57,
        "ops": 57 * 2 + 9,
        "nominal_succ": h8_data["eval_success"],
        "category": "TinyML",
    })

    for v in results_summary:
        plot_points.append({
            "name": v["name"],
            "params": v["params"],
            "ops": v["ops"],
            "nominal_succ": v["scores"]["Nominal Task B"][0],
            "category": "MLP-Pruned" if "Pruned" in v["name"] else "MLP-Dense",
        })

    # Plot 1: Success vs Parameters
    # Plot 2: Success vs Estimated Ops
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Identify Pareto frontier for (params vs success) and (ops vs success)
    # Objective: Minimize cost (params/ops), Maximize success
    def get_pareto_frontier(points, x_key, y_key):
        sorted_pts = sorted(points, key=lambda p: (p[x_key], -p[y_key]))
        frontier = []
        max_y = -1
        for p in sorted_pts:
            if p[y_key] > max_y:
                frontier.append(p)
                max_y = p[y_key]
        return frontier

    frontier_params = get_pareto_frontier(plot_points, "params", "nominal_succ")
    frontier_ops = get_pareto_frontier(plot_points, "ops", "nominal_succ")

    colors = {
        "TinyML": "#1f77b4",
        "MLP-Dense": "#d62728",
        "MLP-Pruned": "#2ca02c",
    }

    # Plot on ax1 (Success vs Parameters)
    for p in plot_points:
        ax1.scatter(
            p["params"],
            p["nominal_succ"],
            color=colors[p["category"]],
            s=120,
            alpha=0.85,
            edgecolors="black",
            zorder=3,
        )
        ax1.annotate(
            p["name"],
            (p["params"], p["nominal_succ"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="bold" if p in frontier_params else "normal",
        )

    # Draw Pareto line
    sorted_fp = sorted(frontier_params, key=lambda x: x["params"])
    ax1.step(
        [p["params"] for p in sorted_fp],
        [p["nominal_succ"] for p in sorted_fp],
        where="post",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label="Pareto Frontier (Non-dominated)",
        zorder=2,
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Parameters (log scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Task B Swing-Up Success Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Success vs. Parameters", fontsize=13, fontweight="bold")
    ax1.set_ylim(-5, 108)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Plot on ax2 (Success vs Estimated Ops)
    for p in plot_points:
        ax2.scatter(
            p["ops"],
            p["nominal_succ"],
            color=colors[p["category"]],
            s=120,
            alpha=0.85,
            edgecolors="black",
            zorder=3,
        )
        ax2.annotate(
            p["name"],
            (p["ops"], p["nominal_succ"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="bold" if p in frontier_ops else "normal",
        )

    sorted_fo = sorted(frontier_ops, key=lambda x: x["ops"])
    ax2.step(
        [p["ops"] for p in sorted_fo],
        [p["nominal_succ"] for p in sorted_fo],
        where="post",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label="Pareto Frontier (Non-dominated)",
        zorder=2,
    )

    ax2.set_xscale("log")
    ax2.set_xlabel("Estimated Compute Ops (log scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Task B Swing-Up Success Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Success vs. Estimated Compute Operations", fontsize=13, fontweight="bold")
    ax2.set_ylim(-5, 108)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.suptitle("Pareto Trade-Off Analysis: Task B Swing-Up Success vs Complexity", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plot_path = "results/pareto_frontier.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[OK] Successfully generated Pareto frontier plot -> {plot_path}")


if __name__ == "__main__":
    main()
