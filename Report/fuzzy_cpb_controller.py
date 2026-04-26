"""Mamdani fuzzy controller for a CPB rotary blood pump.

The script evaluates the required operating point:
    Delta pressure = 84 mmHg
    Blood flow     = 6 L/min

It also generates the figures used by the LaTeX report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


MembershipParams = Tuple[float, float, float]


@dataclass(frozen=True)
class Rule:
    pressure: str
    flow: str
    speed: str


# Each triangular set is stored as (left foot, peak, right foot). These
# vertices come from the membership-function diagrams in the assignment.
FLOW_MFS: Dict[str, MembershipParams] = {
    "low": (0.0, 3.5, 6.0),
    "medium": (3.5, 6.0, 8.5),
    "high": (6.0, 8.5, 10.0),
}

PRESSURE_MFS: Dict[str, MembershipParams] = {
    "low": (50.0, 72.0, 100.0),
    "medium": (72.0, 100.0, 120.0),
    "high": (100.0, 120.0, 142.0),
}

SPEED_MFS: Dict[str, MembershipParams] = {
    "slow": (2800.0, 2900.0, 3000.0),
    "medium": (2900.0, 3000.0, 3100.0),
    "fast": (3000.0, 3100.0, 3200.0),
}

# Rule order follows Table 2 row by row: pressure term first, flow term second,
# and the pump-speed linguistic output as the consequent.
RULES: Tuple[Rule, ...] = (
    Rule("low", "low", "fast"),
    Rule("low", "medium", "medium"),
    Rule("low", "high", "slow"),
    Rule("medium", "low", "medium"),
    Rule("medium", "medium", "medium"),
    Rule("medium", "high", "slow"),
    Rule("high", "low", "medium"),
    Rule("high", "medium", "medium"),
    Rule("high", "high", "slow"),
)

PALETTE = {
    "low": "#6CC7D8",
    "medium": "#86C76A",
    "high": "#F3B04F",
    "slow": "#6CC7D8",
    "fast": "#F3B04F",
    "aggregate": "#8FB8DE",
    "ink": "#263342",
    "grid": "#C8D6DF",
}

FONT_DIR = Path(__file__).resolve().parent / "fonts"
for font_file in list(FONT_DIR.glob("NotoSerif-*.ttf")) + list(FONT_DIR.glob("NotoSans-*.ttf")):
    font_manager.fontManager.addfont(str(font_file))

PLOT_FONT = {
    "font.family": "serif",
    "font.serif": ["Noto Serif", "DejaVu Serif", "Times New Roman"],
}


def trimf(x: np.ndarray | float, params: MembershipParams) -> np.ndarray | float:
    """Triangular membership function."""
    a, b, c = params
    x_arr = np.asarray(x, dtype=float)
    y = np.zeros_like(x_arr)

    left = (a < x_arr) & (x_arr <= b)
    right = (b < x_arr) & (x_arr < c)
    y[left] = (x_arr[left] - a) / (b - a)
    y[right] = (c - x_arr[right]) / (c - b)
    y[x_arr == b] = 1.0

    if np.isscalar(x):
        return float(y)
    return y


def fuzzify(value: float, membership_functions: Dict[str, MembershipParams]) -> Dict[str, float]:
    return {name: float(trimf(value, params)) for name, params in membership_functions.items()}


def evaluate_controller(
    delta_pressure: float,
    flow: float,
    speed_domain: np.ndarray | None = None,
) -> Dict[str, object]:
    """Evaluate the Mamdani controller and return all intermediate results."""
    if speed_domain is None:
        speed_domain = np.linspace(2800.0, 3200.0, 4001)

    pressure_mu = fuzzify(delta_pressure, PRESSURE_MFS)
    flow_mu = fuzzify(flow, FLOW_MFS)
    firing_strengths = []
    output_strengths = {term: 0.0 for term in SPEED_MFS}

    # Mamdani inference: AND is modeled by min(), and rules with the same
    # consequent are combined by max() before clipping the output sets.
    for rule in RULES:
        strength = min(pressure_mu[rule.pressure], flow_mu[rule.flow])
        firing_strengths.append((rule, strength))
        output_strengths[rule.speed] = max(output_strengths[rule.speed], strength)

    aggregated = np.zeros_like(speed_domain)
    clipped_terms = {}
    for term, params in SPEED_MFS.items():
        # Implication clips each output membership function at its strongest
        # rule activation, then aggregation takes the pointwise maximum.
        mf = trimf(speed_domain, params)
        clipped = np.minimum(output_strengths[term], mf)
        clipped_terms[term] = clipped
        aggregated = np.maximum(aggregated, clipped)

    # The centroid is evaluated numerically over a dense rpm grid so the result
    # is reproducible without relying on a closed-form geometry derivation.
    area = np.trapezoid(aggregated, speed_domain)
    crisp_speed = np.nan
    if area > 1.0e-12:
        crisp_speed = np.trapezoid(speed_domain * aggregated, speed_domain) / area

    return {
        "pressure_mu": pressure_mu,
        "flow_mu": flow_mu,
        "firing_strengths": firing_strengths,
        "output_strengths": output_strengths,
        "speed_domain": speed_domain,
        "clipped_terms": clipped_terms,
        "aggregated": aggregated,
        "crisp_speed": float(crisp_speed),
    }


def crisp_speed_for_surface(delta_pressure: float, flow: float) -> float:
    return float(evaluate_controller(delta_pressure, flow)["crisp_speed"])


def configure_axes(ax: plt.Axes, xlabel: str, ylabel: str = "Membership") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.06)
    ax.set_axisbelow(True)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_membership_panel(
    ax: plt.Axes,
    domain: np.ndarray,
    membership_functions: Dict[str, MembershipParams],
    xlabel: str,
) -> None:
    for name, params in membership_functions.items():
        y = trimf(domain, params)
        ax.fill_between(domain, y, color=PALETTE[name], alpha=0.24)
        ax.plot(domain, y, color=PALETTE[name], linewidth=2.3, label=name.title())
    configure_axes(ax, xlabel)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=9,
    )


def save_single_membership_panel(
    output_path: Path,
    domain: np.ndarray,
    membership_functions: Dict[str, MembershipParams],
    xlabel: str,
) -> None:
    plt.rcParams.update({**PLOT_FONT, "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.25, 2.85), constrained_layout=True)
    plot_membership_panel(ax, domain, membership_functions, xlabel)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_membership_figure(output_path: Path) -> None:
    plt.rcParams.update({**PLOT_FONT, "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(8.1, 8.15), constrained_layout=True)

    plot_membership_panel(
        axes[0],
        np.linspace(0.0, 10.0, 1001),
        FLOW_MFS,
        "Blood flow Q (L/min)",
    )
    plot_membership_panel(
        axes[1],
        np.linspace(50.0, 150.0, 1001),
        PRESSURE_MFS,
        "Delta pressure $\\Delta P$ (mmHg)",
    )
    plot_membership_panel(
        axes[2],
        np.linspace(2800.0, 3200.0, 1001),
        SPEED_MFS,
        "Pump speed N (rpm)",
    )

    fig.suptitle("Input and Output Membership Functions", fontsize=14, color=PALETTE["ink"])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_aggregation_figure(output_path: Path, result: Dict[str, object]) -> None:
    plt.rcParams.update({**PLOT_FONT, "font.size": 10})
    speed_domain = result["speed_domain"]
    aggregated = result["aggregated"]
    crisp_speed = result["crisp_speed"]

    fig, ax = plt.subplots(figsize=(8.05, 4.0), constrained_layout=True)
    for term, params in SPEED_MFS.items():
        ax.plot(
            speed_domain,
            trimf(speed_domain, params),
            color=PALETTE[term],
            linewidth=1.5,
            alpha=0.38,
            label=f"{term.title()} MF",
        )

    ax.fill_between(
        speed_domain,
        aggregated,
        color=PALETTE["aggregate"],
        alpha=0.55,
        label="Aggregated output",
    )
    ax.plot(speed_domain, aggregated, color="#3F79A8", linewidth=2.4)
    ax.axvline(crisp_speed, color=PALETTE["ink"], linestyle="--", linewidth=2.0)
    ax.text(
        crisp_speed + 6,
        0.83,
        f"Centroid = {crisp_speed:.1f} rpm",
        color=PALETTE["ink"],
        fontsize=10,
    )
    configure_axes(ax, "Pump speed N (rpm)")
    ax.set_title("Aggregated Output Fuzzy Set for $\\Delta P=84$ mmHg, $Q=6$ L/min")
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        fontsize=8,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rule_activation_figure(output_path: Path, result: Dict[str, object]) -> None:
    plt.rcParams.update({**PLOT_FONT, "font.size": 10})
    flow_terms = ["low", "medium", "high"]
    pressure_terms = ["low", "medium", "high"]
    speed_labels = {
        (rule.pressure, rule.flow): rule.speed.title()
        for rule, _strength in result["firing_strengths"]
    }
    strengths = {
        (rule.pressure, rule.flow): strength
        for rule, strength in result["firing_strengths"]
    }

    matrix = np.array(
        [[strengths[(p, q)] for q in flow_terms] for p in pressure_terms],
        dtype=float,
    )

    # The heatmap labels show both the consequent and the firing strength, which
    # makes it easy to verify which rules affect the operating point.
    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    im = ax.imshow(matrix, cmap="GnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(flow_terms)), [term.title() for term in flow_terms])
    ax.set_yticks(range(len(pressure_terms)), [term.title() for term in pressure_terms])
    ax.set_xlabel("Blood flow Q")
    ax.set_ylabel("Delta pressure $\\Delta P$")
    ax.set_title("Rule Firing Strengths at the Given Operating Point")

    for i, p in enumerate(pressure_terms):
        for j, q in enumerate(flow_terms):
            ax.text(
                j,
                i,
                f"{speed_labels[(p, q)]}\n$\\alpha={matrix[i, j]:.3f}$",
                ha="center",
                va="center",
                color=PALETTE["ink"],
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Firing strength")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_control_surface(output_path: Path) -> None:
    plt.rcParams.update({**PLOT_FONT, "font.size": 10})
    pressure_values = np.linspace(50.1, 141.9, 61)
    flow_values = np.linspace(0.1, 9.9, 61)
    pressure_grid, flow_grid = np.meshgrid(pressure_values, flow_values)
    surface = np.zeros_like(pressure_grid)

    # Sweep the two-input space with the same controller used for the required
    # operating point; the highlighted point is not computed by a separate path.
    for row in range(surface.shape[0]):
        for col in range(surface.shape[1]):
            surface[row, col] = crisp_speed_for_surface(
                pressure_grid[row, col],
                flow_grid[row, col],
            )

    fig, ax = plt.subplots(figsize=(7.1, 4.0), constrained_layout=True)
    contour = ax.contourf(
        pressure_grid,
        flow_grid,
        surface,
        levels=16,
        cmap="GnBu",
    )
    ax.contour(pressure_grid, flow_grid, surface, colors="#263342", alpha=0.28, linewidths=0.6)
    ax.scatter([84.0], [6.0], color="#F3B04F", edgecolor=PALETTE["ink"], s=70, zorder=3)
    ax.set_xlabel("Delta pressure $\\Delta P$ (mmHg)")
    ax.set_ylabel("Blood flow Q (L/min)")
    ax.set_title("Defuzzified Pump-Speed Control Surface")
    cbar = fig.colorbar(contour, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Crisp speed N (rpm)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_result(result: Dict[str, object]) -> None:
    print("Input memberships")
    print("  Delta pressure:", result["pressure_mu"])
    print("  Blood flow:", result["flow_mu"])
    print("\nRule firing strengths")
    for index, (rule, strength) in enumerate(result["firing_strengths"], start=1):
        print(
            f"  R{index}: IF dP is {rule.pressure} AND Q is {rule.flow} "
            f"THEN speed is {rule.speed}: alpha={strength:.6f}"
        )
    print("\nAggregated output term strengths")
    for term, strength in result["output_strengths"].items():
        print(f"  {term}: {strength:.6f}")
    print(f"\nCentroid speed: {result['crisp_speed']:.6f} rpm")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    figure_dir = base_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_controller(delta_pressure=84.0, flow=6.0)
    print_result(result)

    save_membership_figure(figure_dir / "membership_functions.png")
    save_single_membership_panel(
        figure_dir / "membership_flow.png",
        np.linspace(0.0, 10.0, 1001),
        FLOW_MFS,
        "Blood flow Q (L/min)",
    )
    save_single_membership_panel(
        figure_dir / "membership_pressure.png",
        np.linspace(50.0, 150.0, 1001),
        PRESSURE_MFS,
        "Delta pressure $\\Delta P$ (mmHg)",
    )
    save_single_membership_panel(
        figure_dir / "membership_speed.png",
        np.linspace(2800.0, 3200.0, 1001),
        SPEED_MFS,
        "Pump speed N (rpm)",
    )
    save_rule_activation_figure(figure_dir / "rule_activation.png", result)
    save_aggregation_figure(figure_dir / "aggregated_output.png", result)
    save_control_surface(figure_dir / "control_surface.png")


if __name__ == "__main__":
    main()
