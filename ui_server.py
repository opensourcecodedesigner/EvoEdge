"""Real-Time Web Visualizer and Interactive Sandbox Server.

Runs the NEAT CartPole Swing-Up simulation in lockstep with the evolved
neural controller, streaming 60 Hz telemetry over Server-Sent Events (SSE)
and accepting real-time physical disturbance commands from the UI.

Usage:
    python ui_server.py [--port 8000] [--genome path/to/best_genome.pkl]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set

# Ensure project root is in sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import neat  # type: ignore

from env.cartpole_swingup import CartPoleSwingUpEnv
from baselines.hybrid_controller import HybridController


# ── Observation Normalization & Input Definitions ───────────────────────────


INPUT_DEFS: Dict[int, Dict[str, Any]] = {
    -1: {
        "name": "x",
        "label": "Cart Pos",
        "math_symbol": "x",
        "latex_symbol": "x",
        "code_symbol": "x",
        "unit": "m",
        "y": 35,
    },
    -2: {
        "name": "ẋ",
        "label": "Cart Vel",
        "math_symbol": "ẋ",
        "latex_symbol": "\\dot{x}",
        "code_symbol": "x_dot",
        "unit": "m/s",
        "y": 75,
    },
    -3: {
        "name": "cos θ",
        "label": "Pole Cos",
        "math_symbol": "cosθ",
        "latex_symbol": "\\cos\\theta",
        "code_symbol": "cos(theta)",
        "unit": "",
        "y": 115,
    },
    -4: {
        "name": "sin θ",
        "label": "Pole Sin",
        "math_symbol": "sinθ",
        "latex_symbol": "\\sin\\theta",
        "code_symbol": "sin(theta)",
        "unit": "",
        "y": 155,
    },
    -5: {
        "name": "θ̇",
        "label": "Pole Vel",
        "math_symbol": "θ̇",
        "latex_symbol": "\\dot{\\theta}",
        "code_symbol": "theta_dot",
        "unit": "rad/s",
        "y": 195,
    },
}


def _normalize_obs(obs: np.ndarray) -> List[float]:
    """Scale the 5-dim observation into standard NN input range."""
    x, x_dot, cos_theta, sin_theta, theta_dot = obs
    return [
        float(x / 2.4),
        float(np.clip(x_dot / 10.0, -1.0, 1.0)),
        float(cos_theta),
        float(sin_theta),
        float(np.clip(theta_dot / 10.0, -1.0, 1.0)),
    ]


# ── Interactive Physical Simulation Manager ──────────────────────────────────


class SimulationManager:
    """Thread-safe continuous physics and NEAT or Hybrid controller inference runner."""

    def __init__(
        self,
        genome_path: Optional[str] = None,
        topology_path: Optional[str] = None,
        controller_type: str = "neat",
    ) -> None:
        self.lock = threading.Lock()
        self.controller_type = controller_type.lower()
        self.hybrid_ctrl: Optional[HybridController] = None
        self.genome: Optional[Any] = None
        self.neat_config: Optional[Any] = None
        self.net: Optional[Any] = None
        self.genome_path = genome_path or ""
        self.topology_path = topology_path

        if self.controller_type == "hybrid":
            # Instantiate deterministic dual-regime hybrid controller
            self.hybrid_ctrl = HybridController()
            self.topology = {}
        else:
            # Load evolved champion genome and config
            if not genome_path or not os.path.exists(genome_path):
                raise FileNotFoundError(f"Champion genome not found at: {genome_path}")

            with open(genome_path, "rb") as fh:
                data = pickle.load(fh)
                self.genome = data["genome"]
                self.neat_config = data["config"]

            # Instantiate feedforward neural controller
            self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.neat_config)

            # Load topology metadata
            self.topology = {}
            if topology_path and os.path.exists(topology_path):
                with open(topology_path, "r", encoding="utf-8") as fh:
                    self.topology = json.load(fh)
            else:
                cand = os.path.join(os.path.dirname(genome_path), "topology.json")
                if os.path.exists(cand):
                    with open(cand, "r", encoding="utf-8") as fh:
                        self.topology = json.load(fh)

        # Environment parameters
        self.env = CartPoleSwingUpEnv(max_steps=100000, continuous=True)
        self.mass_cart_nominal = 1.0
        self.mass_perturbed = False
        self.obs_noise_sigma = 0.0
        self.noise_rng = np.random.default_rng(42)

        # Simulation loop state
        self.paused = False
        self.step_count = 0
        self.start_time = time.time()
        self.disturbance_impulse: float = 0.0  # temporary external force kick
        self.last_impulse: float = 0.0
        self.peak_impulse: float = 0.0

        # Reset environment with downward pole (pi)
        self.reset_env(mode="down")

        # Snapshot cache for streaming
        self.current_snapshot: Dict[str, Any] = self._build_snapshot(0.0, 0.0)

        # SSE subscriber queues
        self.subscribers: List[threading.Event] = []

    def reset_env(self, mode: str = "down") -> None:
        """Reset state: 'down' (hanging pi) or 'up' (inverted 0)."""
        with self.lock:
            self.env.reset(seed=int(time.time() * 1000) % (2 ** 31))
            if mode == "up":
                # Small random upright perturbation
                noise = self.noise_rng.uniform(-0.04, 0.04, size=(4,))
                self.env._state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64) + noise
            else:
                # Downward hanging with small perturbation
                noise = self.noise_rng.uniform(-0.04, 0.04, size=(4,))
                self.env._state = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float64) + noise
            self.step_count = 0
            self.last_impulse = 0.0

    def apply_impulse(self, newton_seconds: float) -> None:
        """Inject instantaneous velocity disturbance to the cart."""
        with self.lock:
            if self.env._state is not None:
                # Delta velocity = Impulse / Total mass
                dv = newton_seconds / self.env._total_mass
                self.env._state[1] += dv
                self.last_impulse = abs(float(newton_seconds))
                self.peak_impulse = max(self.peak_impulse, self.last_impulse)

    def toggle_mass_perturb(self) -> None:
        """Toggle cart mass between nominal (1.0 kg) and perturbed (+40% = 1.4 kg)."""
        with self.lock:
            self.mass_perturbed = not self.mass_perturbed
            target_mass = 1.4 if self.mass_perturbed else 1.0
            self.env.MASS_CART = target_mass
            self.env._total_mass = self.env.MASS_CART + self.env.MASS_POLE
            self.env._polemass_length = self.env.MASS_POLE * self.env.LENGTH

    def set_mass(self, mass: float) -> None:
        with self.lock:
            self.env.MASS_CART = max(0.2, min(5.0, mass))
            self.env._total_mass = self.env.MASS_CART + self.env.MASS_POLE
            self.env._polemass_length = self.env.MASS_POLE * self.env.LENGTH
            self.mass_perturbed = abs(self.env.MASS_CART - 1.4) < 0.05

    def set_noise(self, sigma: float) -> None:
        with self.lock:
            self.obs_noise_sigma = max(0.0, min(0.5, sigma))

    def toggle_pause(self) -> None:
        with self.lock:
            self.paused = not self.paused

    def step_physics(self) -> Dict[str, Any]:
        """Perform one lockstep physics integration and network inference."""
        with self.lock:
            if self.paused or self.env._state is None:
                return self.current_snapshot

            raw_obs = self.env._get_obs()

            # Add sensor noise if enabled
            if self.obs_noise_sigma > 0.0:
                noise = self.noise_rng.normal(
                    0.0, self.obs_noise_sigma, size=raw_obs.shape
                ).astype(np.float32)
                perceived_obs = raw_obs + noise
            else:
                perceived_obs = raw_obs

            # Controller inference (NEAT vs Hybrid)
            norm_inputs = _normalize_obs(perceived_obs)
            if self.controller_type == "hybrid" and self.hybrid_ctrl is not None:
                action = self.hybrid_ctrl.compute_action(perceived_obs)
            else:
                action_out = self.net.activate(norm_inputs)
                action = float(action_out[0])

            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.step_count += 1

            # Soft bumper bounce on rail stops (x = +/-2.35m) to maintain continuous sandbox
            x, x_dot, theta, theta_dot = self.env._state
            if abs(x) > 2.34:
                # Bounce with restitution damping
                sign = 1.0 if x > 0 else -1.0
                self.env._state[0] = sign * 2.34
                if (x > 0 and x_dot > 0) or (x < 0 and x_dot < 0):
                    self.env._state[1] = -0.55 * x_dot

            applied_force = float(np.clip(action, -1.0, 1.0)) * self.env.FORCE_MAG
            regime = self.hybrid_ctrl.current_regime if self.hybrid_ctrl else "NEAT"
            self.current_snapshot = self._build_snapshot(action, applied_force, reward, norm_inputs, regime=regime)

        # Notify waiting SSE streams
        for ev in self.subscribers:
            ev.set()

        return self.current_snapshot

    def _build_snapshot(
        self,
        action: float,
        force: float,
        reward: float = 0.0,
        inputs: Optional[List[float]] = None,
        regime: str = "NEAT",
    ) -> Dict[str, Any]:
        """Build serialization payload for 60 Hz UI stream."""
        if self.env._state is None:
            x, x_dot, theta, theta_dot = 0.0, 0.0, np.pi, 0.0
        else:
            x, x_dot, theta, theta_dot = self.env._state

        pole_l = self.env.LENGTH * 2.0  # 1.0 m total tip reach
        tip_x = float(x + pole_l * np.sin(theta))
        tip_y = float(pole_l * np.cos(theta))

        return {
            "step": self.step_count,
            "x": round(float(x), 4),
            "x_dot": round(float(x_dot), 4),
            "theta": round(float(theta), 4),
            "theta_dot": round(float(theta_dot), 4),
            "tip_x": round(tip_x, 4),
            "tip_y": round(tip_y, 4),
            "action": round(float(action), 4),
            "force": round(float(force), 2),
            "reward": round(float(reward), 3),
            "cart_mass": round(float(self.env.MASS_CART), 2),
            "pole_length": round(float(self.env.LENGTH), 2),
            "mass_perturbed": bool(self.mass_perturbed),
            "noise_sigma": round(float(self.obs_noise_sigma), 3),
            "paused": bool(self.paused),
            "last_impulse": round(float(self.last_impulse), 2),
            "peak_impulse": round(float(self.peak_impulse), 2),
            "inputs": [round(v, 4) for v in (inputs or [0.0]*5)],
            "controller_type": self.controller_type,
            "regime": regime,
        }

    def get_model_spec(self) -> Dict[str, Any]:
        """Extract and structure the active NEAT or Hybrid model specification dynamically."""
        with self.lock:
            if self.controller_type == "hybrid" and self.hybrid_ctrl is not None:
                # Return explicit piecewise control law for Hybrid controller
                K = self.hybrid_ctrl.K
                k_E = self.hybrid_ctrl.k_E
                k_x = self.hybrid_ctrl.k_x
                th_th = self.hybrid_ctrl.theta_threshold

                # Formulate connections for visualization table (gain matrix)
                # 4 balance gains + 2 pumping gains
                active_conns = [
                    {"from": -1, "to": 0, "source_name": "x (Cart Pos)", "target_name": "u (Balance)", "weight": -float(K[0]), "gain_name": "K_x", "regime": "BALANCE", "enabled": True},
                    {"from": -2, "to": 0, "source_name": "ẋ (Cart Vel)", "target_name": "u (Balance)", "weight": -float(K[1]), "gain_name": "K_x_dot", "regime": "BALANCE", "enabled": True},
                    {"from": -4, "to": 0, "source_name": "θ (Pole Angle)", "target_name": "u (Balance)", "weight": -float(K[2]), "gain_name": "K_theta", "regime": "BALANCE", "enabled": True},
                    {"from": -5, "to": 0, "source_name": "θ̇ (Pole Vel)", "target_name": "u (Balance)", "weight": -float(K[3]), "gain_name": "K_theta_dot", "regime": "BALANCE", "enabled": True},
                    {"from": -3, "to": 0, "source_name": "E (Energy Error)", "target_name": "u (Pump)", "weight": float(k_E), "gain_name": "k_E", "regime": "PUMP", "enabled": True},
                    {"from": -1, "to": 0, "source_name": "x (Cart Center)", "target_name": "u (Pump)", "weight": -float(k_x), "gain_name": "k_x", "regime": "PUMP", "enabled": True},
                ]

                all_inputs = [
                    {"id": -1, "name": "x", "label": "Cart Pos", "math_symbol": "x", "latex_symbol": "x", "expressed": True, "y": 35},
                    {"id": -2, "name": "ẋ", "label": "Cart Vel", "math_symbol": "ẋ", "latex_symbol": "\\dot{x}", "expressed": True, "y": 75},
                    {"id": -3, "name": "cos θ", "label": "Pole Cos", "math_symbol": "cosθ", "latex_symbol": "\\cos\\theta", "expressed": True, "y": 115},
                    {"id": -4, "name": "sin θ", "label": "Pole Sin", "math_symbol": "sinθ", "latex_symbol": "\\sin\\theta", "expressed": True, "y": 155},
                    {"id": -5, "name": "θ̇", "label": "Pole Vel", "math_symbol": "θ̇", "latex_symbol": "\\dot{\\theta}", "expressed": True, "y": 195},
                ]

                formula_display = (
                    f"BALANCE (|θ| < {th_th} rad): u = -({K[0]:+.2f}·x {K[1]:+.2f}·ẋ {K[2]:+.2f}·θ {K[3]:+.2f}·θ̇)\n"
                    f"PUMP (|θ| ≥ {th_th} rad): u = {k_E:.2f}·E·sign(θ̇·cosθ) - {k_x:.2f}·x"
                )

                return {
                    "mode": "hybrid",
                    "controller_type": "hybrid",
                    "name": "Hybrid Energy-Shaping & LQR Controller",
                    "seed": 0,
                    "genome_path": "baselines/hybrid_controller.py",
                    "nodes": [{"id": 0, "type": "output", "name": "u", "label": "Force Output", "bias": 0.0, "activation": "linear"}],
                    "inputs": all_inputs,
                    "active_inputs": all_inputs,
                    "pruned_inputs": [],
                    "connections": active_conns,
                    "active_connections": active_conns,
                    "disabled_connections": [],
                    "output_bias": 0.0,
                    "output_activation": "linear",
                    "metrics": {
                        "num_inputs": 5,
                        "num_outputs": 1,
                        "active_inputs_count": 5,
                        "pruned_inputs_count": 0,
                        "active_synapses": 6,
                        "pruned_synapses": 0,
                        "hidden_neurons": 0,
                        "total_parameters": 6,
                        "theoretical_macs": 6,
                    },
                    "formula": {
                        "display": formula_display,
                        "latex": "u = \\text{piecewise}(E, \\text{LQR})",
                        "code": "compute_cartpole_action(x, x_dot, cos_th, sin_th, th_dot)",
                        "pruned_note": "Deterministic Dual-Regime: 4 LQR State-Feedback Gains + 2 Energy-Shaping Pumping Gains (6 MACs).",
                        "terms": [],
                    },
                    "hybrid_spec": self.hybrid_ctrl.get_spec(),
                }

            nodes_list: List[Dict[str, Any]] = []
            node_genes = self.genome.nodes

            hidden_nodes: List[Dict[str, Any]] = []
            output_nodes: List[Dict[str, Any]] = []

            for nid, ngene in sorted(node_genes.items()):
                act = getattr(ngene, "activation", "tanh")
                bias = float(getattr(ngene, "bias", 0.0))
                is_output = (nid == 0) or (
                    hasattr(self.neat_config, "genome_config")
                    and nid in getattr(self.neat_config.genome_config, "output_keys", [0])
                )
                ntype = "output" if is_output else "hidden"
                node_info = {
                    "id": int(nid),
                    "type": ntype,
                    "bias": round(bias, 6),
                    "activation": str(act),
                    "name": "u" if is_output else f"h{nid}",
                    "label": "Force Output" if is_output else f"Hidden {nid}",
                }
                nodes_list.append(node_info)
                if is_output:
                    output_nodes.append(node_info)
                else:
                    hidden_nodes.append(node_info)

            # Connections
            conns_list: List[Dict[str, Any]] = []
            conn_genes = self.genome.connections
            active_srcs: Set[int] = set()

            for (src, dst), cgene in sorted(conn_genes.items(), key=lambda item: (item[0][1], item[0][0])):
                enabled = bool(cgene.enabled)
                w = float(cgene.weight)
                if enabled:
                    active_srcs.add(src)
                src_name = INPUT_DEFS.get(src, {}).get("name", f"n{src}")
                dst_name = "u" if dst == 0 else f"h{dst}"
                from_y = INPUT_DEFS.get(src, {}).get("y", 115)
                conns_list.append({
                    "from": int(src),
                    "to": int(dst),
                    "source_name": src_name,
                    "target_name": dst_name,
                    "weight": round(w, 6),
                    "enabled": enabled,
                    "fromY": from_y,
                })

            active_conns = [c for c in conns_list if c["enabled"]]
            disabled_conns = [c for c in conns_list if not c["enabled"]]

            # Inputs (ordered -1, -2, -3, -4, -5)
            all_inputs: List[Dict[str, Any]] = []
            active_inputs: List[Dict[str, Any]] = []
            pruned_inputs: List[Dict[str, Any]] = []

            for in_id in [-1, -2, -3, -4, -5]:
                idef = INPUT_DEFS.get(in_id, {
                    "name": f"in_{abs(in_id)}",
                    "label": f"Input {abs(in_id)}",
                    "math_symbol": f"in_{abs(in_id)}",
                    "latex_symbol": f"in_{{{abs(in_id)}}}",
                    "code_symbol": f"in_{abs(in_id)}",
                    "y": 35 + abs(in_id + 1) * 40,
                })
                is_expr = in_id in active_srcs
                in_info = {
                    "id": in_id,
                    "name": idef["name"],
                    "label": idef["label"],
                    "math_symbol": idef["math_symbol"],
                    "latex_symbol": idef["latex_symbol"],
                    "expressed": is_expr,
                    "y": idef["y"],
                }
                all_inputs.append(in_info)
                if is_expr:
                    active_inputs.append(in_info)
                else:
                    pruned_inputs.append(in_info)

            # Complexity metrics
            total_params = len(active_conns) + len(node_genes)
            theoretical_macs = len(active_conns) + len(hidden_nodes) + len(output_nodes)

            # Formula string derivation for output node 0
            out_node = output_nodes[0] if output_nodes else {"bias": 0.0, "activation": "tanh"}
            out_bias = float(out_node["bias"])
            out_act = str(out_node["activation"])

            # Sort incoming active connections: largest absolute weight first
            incoming_active = [c for c in active_conns if c["to"] == 0]
            incoming_active.sort(key=lambda c: abs(c["weight"]), reverse=True)

            disp_terms: List[str] = []
            latex_terms: List[str] = []
            code_terms: List[str] = []

            for c in incoming_active:
                src = c["from"]
                w = c["weight"]
                src_math = INPUT_DEFS.get(src, {}).get("name", f"n{src}")
                src_latex = INPUT_DEFS.get(src, {}).get("latex_symbol", f"n_{{{src}}}")
                src_code = INPUT_DEFS.get(src, {}).get("code_symbol", src_math)

                sign = "+" if w >= 0 else "-"
                disp_terms.append(f"{sign} {abs(w):.2f}·{src_math}")
                latex_terms.append(f"{sign} {abs(w):.6f} \\cdot {src_latex}")
                code_terms.append(f"{sign} {abs(w):.6f}*{src_code}")

            # Assemble display formula
            if disp_terms:
                inner_disp = f"{out_bias:.3f} " + " ".join(disp_terms) if out_bias != 0 else " ".join(disp_terms).lstrip("+ ")
                inner_latex = f"{out_bias:.6f} " + " ".join(latex_terms) if out_bias != 0 else " ".join(latex_terms).lstrip("+ ")
                inner_code = f"{out_bias:.6f} " + " ".join(code_terms) if out_bias != 0 else " ".join(code_terms).lstrip("+ ")
            else:
                inner_disp = f"{out_bias:.3f}"
                inner_latex = f"{out_bias:.6f}"
                inner_code = f"{out_bias:.6f}"

            formula_display = f"u = {out_act}({inner_disp.strip()})"
            formula_latex = f"u = \\{out_act}({inner_latex.strip()})"
            formula_code = f"u = {out_act}({inner_code.strip()})"

            if pruned_inputs:
                pruned_names = ", ".join([f"{p['label']} ({p['name']})" for p in pruned_inputs])
                pruned_note = f"{pruned_names} successfully pruned by NEAT selection pressure."
            else:
                pruned_note = "All sensory input channels actively expressed."

            # Mode and seed detection from genome_path
            mode = "performance"
            seed = 42
            gpath = getattr(self, "genome_path", "")
            if gpath:
                parts = os.path.normpath(gpath).split(os.sep)
                for p in parts:
                    if p in ("performance", "interpretability", "complexity", "hardware"):
                        mode = p
                    if "seed_" in p:
                        try:
                            seed = int(p.replace("seed_", ""))
                        except ValueError:
                            pass

            return {
                "mode": mode,
                "seed": seed,
                "genome_path": gpath,
                "nodes": nodes_list,
                "inputs": all_inputs,
                "active_inputs": active_inputs,
                "pruned_inputs": pruned_inputs,
                "connections": conns_list,
                "active_connections": active_conns,
                "disabled_connections": disabled_conns,
                "output_bias": out_bias,
                "output_activation": out_act,
                "metrics": {
                    "num_inputs": len(all_inputs),
                    "num_outputs": len(output_nodes),
                    "active_inputs_count": len(active_inputs),
                    "pruned_inputs_count": len(pruned_inputs),
                    "active_synapses": len(active_conns),
                    "pruned_synapses": len(disabled_conns),
                    "hidden_neurons": len(hidden_nodes),
                    "total_parameters": total_params,
                    "theoretical_macs": theoretical_macs,
                },
                "formula": {
                    "display": formula_display,
                    "latex": formula_latex,
                    "code": formula_code,
                    "pruned_note": pruned_note,
                    "bias": out_bias,
                    "activation": out_act,
                    "terms": [
                        {
                            "source": c["from"],
                            "name": INPUT_DEFS.get(c["from"], {}).get("name", f"n{c['from']}"),
                            "weight": c["weight"],
                        }
                        for c in incoming_active
                    ],
                },
            }


# ── HTTP & SSE Request Handler ───────────────────────────────────────────────


class VisualizerRequestHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP server handling UI assets, SSE stream, and controls."""

    sim: SimulationManager  # injected at server instantiation

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP request logging for clean terminal output."""
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(_PROJECT_ROOT, "web", "index.html"), "text/html; charset=utf-8")
        elif path == "/api/topology":
            self._serve_json(self.sim.topology)
        elif path == "/api/model":
            self._serve_json(self.sim.get_model_spec())
        elif path == "/api/stream":
            self._serve_sse_stream()
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/control":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8")
                payload = json.loads(body)
                action = payload.get("action")
                val = payload.get("value")

                if action == "impulse":
                    self.sim.apply_impulse(float(val if val is not None else 10.0))
                elif action == "toggle_mass_perturb":
                    self.sim.toggle_mass_perturb()
                elif action == "set_mass":
                    self.sim.set_mass(float(val))
                elif action == "set_noise":
                    self.sim.set_noise(float(val))
                elif action == "reset":
                    self.sim.reset_env(mode=str(val or "down"))
                elif action == "toggle_pause":
                    self.sim.toggle_pause()

                self._serve_json({"status": "ok", "action": action})
            except Exception as e:
                self.send_error(400, f"Control Error: {e}")
        else:
            self.send_error(404, "Unknown API route")

    def _serve_file(self, filepath: str, content_type: str) -> None:
        if not os.path.exists(filepath):
            self.send_error(404, f"File not found: {filepath}")
            return
        with open(filepath, "rb") as fh:
            content = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(content)

    def _serve_json(self, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse_stream(self) -> None:
        """Stream real-time state events over HTTP at 60 Hz."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        event = threading.Event()
        with self.sim.lock:
            self.sim.subscribers.append(event)

        try:
            while True:
                # Wait for next physics step
                event.wait(timeout=0.03)
                event.clear()

                with self.sim.lock:
                    snapshot = self.sim.current_snapshot

                data_str = f"data: {json.dumps(snapshot)}\n\n"
                self.wfile.write(data_str.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with self.sim.lock:
                if event in self.sim.subscribers:
                    self.sim.subscribers.remove(event)


# ── Physics Simulation Worker Thread ─────────────────────────────────────────


def _run_physics_loop(sim: SimulationManager, target_hz: float = 60.0) -> None:
    """Independent 60 Hz physics integration thread."""
    dt = 1.0 / target_hz
    next_tick = time.perf_counter()

    while True:
        sim.step_physics()
        next_tick += dt
        sleep_dur = next_tick - time.perf_counter()
        if sleep_dur > 0:
            time.sleep(sleep_dur)
        else:
            # Running behind; reset next_tick to prevent spiral
            next_tick = time.perf_counter()


# ── Main Entry Point ─────────────────────────────────────────────────────────


class VisualizerServer(ThreadingHTTPServer):
    """Custom server that cleanly ignores client aborts during SSE streaming."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc_type, exc_val, _ = sys.exc_info()
        if exc_type in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return  # Clean client disconnect, no traceback needed
        super().handle_error(request, client_address)


def _find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Find the first available TCP port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start_port


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NEAT CartPole Swing-Up Real-Time Web Visualizer",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP server port (default: auto-detect from 8000).",
    )
    parser.add_argument(
        "--genome",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "results", "performance", "seed_42", "best_genome.pkl"),
        help="Path to evolved champion genome pickle.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "results", "performance", "seed_42", "topology.json"),
        help="Path to topology.json file.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="neat",
        choices=["neat", "hybrid"],
        help="Controller type: 'neat' (default) or 'hybrid' (Energy-Shaping + LQR State-Feedback).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web visualizer in the default browser.",
    )
    args = parser.parse_args()

    is_hybrid = args.controller.lower() == "hybrid"

    # Fallback to any champion found if default path does not exist and in NEAT mode
    if not is_hybrid and not os.path.exists(args.genome):
        for root, _, files in os.walk(os.path.join(_PROJECT_ROOT, "results")):
            if "best_genome.pkl" in files:
                args.genome = os.path.join(root, "best_genome.pkl")
                args.topology = os.path.join(root, "topology.json")
                break

    print("\n" + "=" * 70)
    print("  CartPole Swing-Up — Autonomous Controller Visualizer")
    print("=" * 70)
    print(f"  Controller Mode: {args.controller.upper()}")
    if is_hybrid:
        print("  Architecture   : Deterministic Hybrid (Energy-Shaping + LQR State-Feedback)")
        print("  Parameters     : 6 Gains (4 Balance + 2 Pumping) | 6 MACs")
    else:
        print(f"  Genome Path    : {args.genome}")
        print(f"  Topology Path  : {args.topology}")

    sim_manager = SimulationManager(
        genome_path=args.genome if not is_hybrid else None,
        topology_path=args.topology if not is_hybrid else None,
        controller_type=args.controller,
    )

    # Inject simulation manager into HTTP handler
    VisualizerRequestHandler.sim = sim_manager

    port = args.port or _find_free_port(8000)
    server = VisualizerServer(("127.0.0.1", port), VisualizerRequestHandler)

    # Launch physics thread
    physics_thread = threading.Thread(
        target=_run_physics_loop, args=(sim_manager, 60.0), daemon=True
    )
    physics_thread.start()

    url = f"http://127.0.0.1:{port}"
    print(f"  Local Server   : {url}")
    print(f"  Physics Stream : 60 Hz SSE (Server-Sent Events)")
    print("=" * 70)
    print("  Press Ctrl+C to terminate server.\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Shutting down web visualizer server...]")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
