from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Baseline": "#D97745",
    "BB + backtracking": "#167D8D",
}


def save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    return figure


def plot_objective_history(results, path):
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for method, history in results.items():
        iterations = np.arange(len(history["values"]))
        display_values = np.maximum(
            history["values"], np.finfo(float).tiny
        )
        axis.semilogy(
            iterations,
            display_values,
            marker="o" if len(iterations) < 15 else None,
            markersize=4,
            linewidth=2,
            color=COLORS[method],
            label=method,
        )

    axis.set(
        xlabel="Итерация",
        ylabel=r"Значение $L(w_k)$",
        title="Сходимость значения функции",
    )
    axis.grid(alpha=0.3, which="both")
    axis.legend()
    return save_figure(figure, path)


def plot_step_history(results, path):
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.8),
        sharey=True,
        layout="constrained",
    )
    baseline = results["Baseline"]
    adaptive = results["BB + backtracking"]
    baseline_iterations = np.arange(1, len(baseline["step_sizes"]) + 1)
    adaptive_iterations = np.arange(1, len(adaptive["step_sizes"]) + 1)
    total_iterations = max(len(baseline_iterations), len(adaptive_iterations))

    axes[0].plot(
        baseline_iterations,
        baseline["step_sizes"],
        color=COLORS["Baseline"],
        linewidth=2.5,
        label=rf"Baseline: $\alpha_k={baseline['step_sizes'][0]:g}$",
    )
    axes[0].plot(
        adaptive_iterations,
        adaptive["step_sizes"],
        color=COLORS["BB + backtracking"],
        marker="o",
        markersize=4,
        linewidth=2,
        label="BB + backtracking",
    )
    axes[0].text(
        0.97,
        0.08,
        f"Baseline сохраняет шаг {baseline['step_sizes'][0]:g}\n"
        f"на всех {len(baseline_iterations)} итерациях",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=COLORS["Baseline"],
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": COLORS["Baseline"],
            "alpha": 0.9,
        },
    )

    axes[1].plot(
        adaptive_iterations,
        adaptive["candidate_steps"],
        marker="x",
        linestyle="--",
        linewidth=1.6,
        color="#725AC1",
        label="Предложение BB",
        zorder=2,
    )
    axes[1].plot(
        adaptive_iterations,
        adaptive["step_sizes"],
        marker="o",
        markersize=4,
        linewidth=2,
        color=COLORS["BB + backtracking"],
        label="Принятый шаг",
        zorder=3,
    )
    reduced = adaptive["candidate_steps"] > adaptive["step_sizes"]
    axes[1].vlines(
        adaptive_iterations[reduced],
        adaptive["step_sizes"][reduced],
        adaptive["candidate_steps"][reduced],
        color="#C43D3D",
        linewidth=2,
        alpha=0.55,
        label="Уменьшение backtracking",
        zorder=1,
    )
    axes[1].scatter(
        adaptive_iterations[reduced],
        adaptive["step_sizes"][reduced],
        s=45,
        marker="v",
        color="#C43D3D",
        zorder=4,
    )

    axes[0].set(
        xlabel="Итерация",
        ylabel=r"Длина шага $\alpha_k$",
        title="Полная история принятых шагов",
        xlim=(0.5, total_iterations + 0.5),
    )
    axes[1].set(
        xlabel="Итерация",
        title="Проверка предложения BB",
        xlim=(0.5, len(adaptive_iterations) + 0.5),
    )
    figure.suptitle("Как методы выбирают длину шага", fontsize=15)

    for axis in axes:
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=9, loc="upper right")

    return save_figure(figure, path)


def plot_trajectories(results, minimizer, path):
    figure, axis = plt.subplots(figsize=(7, 6), layout="constrained")

    for method, history in results.items():
        points = history["points"]
        axis.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            markersize=4,
            markevery=max(1, len(points) // 15),
            linewidth=2,
            color=COLORS[method],
            label=method,
        )

    start = results["Baseline"]["points"][0]
    axis.scatter(
        [start[0]],
        [start[1]],
        marker="D",
        s=65,
        facecolors="white",
        edgecolors="#222222",
        linewidths=1.5,
        label="начальная точка",
        zorder=5,
    )

    axis.scatter(
        [minimizer[0]],
        [minimizer[1]],
        marker="*",
        s=160,
        color="#222222",
        label="минимум",
        zorder=5,
    )

    all_points = np.vstack(
        [history["points"] for history in results.values()] + [minimizer]
    )
    margin = 0.15 * max(np.ptp(all_points[:, 0]), np.ptp(all_points[:, 1]))
    axis.axhline(0, color="#777777", linewidth=1, zorder=0)
    axis.axvline(0, color="#777777", linewidth=1, zorder=0)
    axis.set(
        xlabel=r"$w_1$",
        ylabel=r"$w_2$",
        title="Траектории на координатной плоскости",
        xlim=(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin),
        ylim=(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    return save_figure(figure, path)
