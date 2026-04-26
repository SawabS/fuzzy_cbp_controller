# Fuzzy CPB Controller

An elegant, reproducible Mamdani fuzzy-logic controller for a cardiopulmonary bypass
(CPB) rotary blood pump. The controller maps blood-flow and pressure measurements to
a safe pump-speed recommendation, then generates the figures used in the accompanying
technical report.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Fuzzy Logic](https://img.shields.io/badge/Method-Mamdani%20Fuzzy%20Inference-5B8DEF?style=for-the-badge)
![Status](https://img.shields.io/badge/Result-3000%20rpm-2F855A?style=for-the-badge)

## Overview

This project implements an AI-based smart blood-flow control system for a CPB rotary
pump. Given the operating point:

- Delta pressure: `84 mmHg`
- Blood flow: `6 L/min`

the fuzzy controller evaluates the active rules, aggregates the output membership
functions, and applies centroid defuzzification. For the supplied operating point,
the recommended pump speed is:

```text
3000 rpm
```

The repository includes both the Python implementation and a polished LaTeX report
with generated diagrams.

## Highlights

- Triangular membership functions for flow, pressure, and pump speed
- Nine-rule Mamdani fuzzy inference system
- `min` operator for fuzzy AND
- `max` aggregation for rule consequents
- Centroid defuzzification over a dense rpm grid
- Reproducible figures for membership functions, rule activation, aggregation, and
  the defuzzified control surface

## Repository Structure

```text
.
├── README.md
├── AI-Project-Final-Exam-Sp-26.pdf
└── LaTeX/
    ├── fuzzy_cpb_controller.py
    ├── main.tex
    ├── fuzzy_cpb_controller_report.pdf
    ├── figures/
    │   ├── membership_functions.png
    │   ├── membership_flow.png
    │   ├── membership_pressure.png
    │   ├── membership_speed.png
    │   ├── rule_activation.png
    │   ├── aggregated_output.png
    │   └── control_surface.png
    └── fonts/
```

## Controller Design

The controller uses two inputs and one output:

| Variable | Linguistic sets | Universe |
| --- | --- | --- |
| Blood flow `Q` | Low, Medium, High | `0-10 L/min` |
| Delta pressure `dP` | Low, Medium, High | `50-150 mmHg` |
| Pump speed `N` | Slow, Medium, Fast | `2800-3200 rpm` |

The rule base follows a compact clinical intuition: when flow is high, the pump tends
toward slower speeds; when flow is low under low pressure, the pump responds faster;
and the nominal medium-flow operating region settles around medium speed.

## Visual Results

### Membership Functions

![Membership functions](LaTeX/figures/membership_functions.png)

### Rule Activation

![Rule activation heatmap](LaTeX/figures/rule_activation.png)

### Aggregated Output

![Aggregated fuzzy output](LaTeX/figures/aggregated_output.png)

### Control Surface

![Defuzzified control surface](LaTeX/figures/control_surface.png)

## Quick Start

Install the Python dependencies:

```bash
pip install numpy matplotlib
```

Run the controller and regenerate all figures:

```bash
python LaTeX/fuzzy_cpb_controller.py
```

The script prints the fuzzified inputs, rule firing strengths, aggregated output
strengths, and final centroid speed. It also writes refreshed figures to
`LaTeX/figures/`.

## Expected Output

For `dP = 84 mmHg` and `Q = 6 L/min`, the important intermediate values are:

```text
Pressure low membership:    0.571
Pressure medium membership: 0.429
Flow medium membership:     1.000
Dominant output:            Medium speed
Centroid speed:             3000 rpm
```

Because both nonzero rules activate the medium-speed consequent, the aggregated
output is a clipped medium-speed triangle centered at `3000 rpm`.

## Report

The full derivation, rule table, membership plots, centroid calculation, and final
conclusion are available in:

- [`LaTeX/fuzzy_cpb_controller_report.pdf`](LaTeX/fuzzy_cpb_controller_report.pdf)
- [`LaTeX/main.tex`](LaTeX/main.tex)

## Built With

- Python
- NumPy
- Matplotlib
- LaTeX

## Author

Sawab Hussein

