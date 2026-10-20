import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e6e5e0"
COLORS = {"bounded": "#2a78d6", "unbounded": "#eb6834", "unbounded_prefetch": "#1baf7a"}
SHORT = {
    "bounded": "лимит 100",
    "unbounded": "без лимита",
    "unbounded_prefetch": "без лимита и prefetch",
}


@dataclass
class Run:
    name: str
    title: str
    monitor: pd.DataFrame
    tasks: pd.DataFrame
    events: dict[str, float]
    result: dict[str, Any]

    @property
    def launch(self) -> float | None:
        return self.events.get("запуск второго воркера")

    @property
    def joined(self) -> float | None:
        return self.events.get("второй воркер готов")


def style() -> None:
    plt.switch_backend("agg")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def load(out: Path, scenarios: dict[str, dict[str, Any]]) -> list[Run]:
    runs = []
    for name, scenario in scenarios.items():
        folder = out / name
        if not (folder / "monitor.csv").exists():
            continue
        events = pd.read_csv(folder / "events.csv")
        monitor = pd.read_csv(folder / "monitor.csv")
        monitor["broker_growth_mb"] = (
            monitor["broker_memory_mb"] - monitor["broker_memory_mb"].iloc[0]
        )
        runs.append(
            Run(
                name=name,
                title=scenario["title"],
                monitor=monitor,
                tasks=pd.read_csv(folder / "tasks.csv"),
                events=dict(zip(events["event"], events["elapsed"], strict=True)),
                result=json.loads((folder / "result.json").read_text(encoding="utf-8")),
            )
        )
    return runs


def per_second(frame: pd.DataFrame, column: str, window: int = 5) -> pd.Series:
    rate = frame[column].diff() / frame["elapsed"].diff()
    return rate.clip(lower=0).rolling(window, min_periods=1).mean()


def readable(value: float, _: object = None) -> str:
    if abs(value) >= 10:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:g}".replace(".", ",")


def plain_numbers(ax: Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(readable))


def worker_marker(ax: Axes, runs: list[Run], low: bool = False) -> None:
    launch = next((run.launch for run in runs if run.launch is not None), None)
    if launch is None:
        return
    ax.axvline(launch, color=SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(
        "подключили второй воркер",
        xy=(launch, 0.03 if low else 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0 if low else -4),
        textcoords="offset points",
        fontsize=8.5,
        color=SECONDARY,
        va="bottom" if low else "top",
    )


def end_labels(ax: Axes, items: list[tuple[float, float, str]]) -> None:
    figure = ax.get_figure()
    if figure is None or not items:
        return
    figure.canvas.draw()
    to_pixels = ax.transData.transform
    gap = 13 * figure.dpi / 72
    placed: list[float] = []
    for x, y, text in sorted(items, key=lambda item: to_pixels((item[0], item[1]))[1]):
        actual = to_pixels((x, y))[1]
        target = max(actual, placed[-1] + gap) if placed else actual
        placed.append(target)
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(6, (target - actual) * 72 / figure.dpi),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            va="center",
            annotation_clip=False,
        )


def frame_axes(ax: Axes, runs: list[Run], ylabel: str, room: float = 1.3) -> None:
    longest = max(run.monitor["elapsed"].max() for run in runs)
    ax.set_xlim(0, longest * room)
    ax.set_xlabel("секунды от старта источника")
    ax.set_ylabel(ylabel)


def save(figure: Any, path: Path) -> None:
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"нарисовал {path}")


def line_chart(
    runs: list[Run],
    path: Path,
    title: str,
    ylabel: str,
    series: dict[str, tuple[pd.Series, pd.Series]],
    log: bool = False,
    bottom: float = 1,
    low_marker: bool = True,
    direct_labels: bool = True,
) -> None:
    figure, ax = plt.subplots(figsize=(9, 4.6))
    labels = []
    for run in runs:
        x, y = series[run.name]
        ax.plot(x, y, color=COLORS[run.name], label=run.title)
        valid = y.dropna()
        if not valid.empty:
            labels.append((float(x[valid.index[-1]]), float(valid.iloc[-1]), SHORT[run.name]))
    if log:
        ax.set_yscale("log")
    plain_numbers(ax)
    frame_axes(ax, runs, ylabel, room=1.3 if direct_labels else 1.02)
    ax.set_ylim(bottom=bottom if log else 0)
    ax.set_title(title, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, handlelength=1.6)
    worker_marker(ax, runs, low=low_marker)
    if direct_labels:
        end_labels(ax, labels)
    save(figure, path)


def depth_chart(runs: list[Run], out: Path) -> None:
    line_chart(
        runs,
        out / "depth.png",
        "сколько задач держит брокер: в очереди и у воркеров",
        "задач, логарифмическая шкала",
        {
            run.name: (
                run.monitor["elapsed"],
                (run.monitor["ready"] + run.monitor["unacked"]).clip(lower=1),
            )
            for run in runs
        },
        log=True,
        low_marker=False,
    )


def where_chart(runs: list[Run], out: Path) -> None:
    figure, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.8), squeeze=False)
    for ax, run in zip(axes[0], runs, strict=True):
        x = run.monitor["elapsed"]
        color = COLORS[run.name]
        ax.plot(x, run.monitor["ready"], color=color, label="в очереди")
        ax.plot(x, run.monitor["unacked"], color=color, linestyle=(0, (1.5, 2)), label="у воркеров")
        ax.set_title(run.title, fontsize=10.5, pad=8)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("секунды")
        plain_numbers(ax)
        worker_marker(ax, [run])
        ax.legend(loc="center right")
    axes[0][0].set_ylabel("задач")
    figure.suptitle("где ждут задачи", x=0.01, ha="left", fontsize=12, fontweight="bold")
    figure.tight_layout()
    save(figure, out / "where_tasks_wait.png")


def latency_series(run: Run) -> tuple[pd.Series, pd.Series]:
    if run.tasks.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    buckets = (run.tasks["finished"] // 5) * 5 + 2.5
    median = run.tasks.groupby(buckets)["latency"].median()
    return pd.Series(median.index.to_numpy()), pd.Series(median.to_numpy())


def memory_chart(runs: list[Run], out: Path) -> None:
    panels = [
        ("broker_growth_mb", "прирост памяти rabbitmq"),
        ("source_rss_mb", "процесс источника"),
        ("workers_rss_mb", "все воркеры вместе"),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(13, 3.9))
    for ax, (column, title) in zip(axes, panels, strict=True):
        for run in runs:
            ax.plot(
                run.monitor["elapsed"],
                run.monitor[column].where(run.monitor[column] > 0)
                if column != "broker_growth_mb"
                else run.monitor[column],
                color=COLORS[run.name],
                label=run.title,
            )
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_xlabel("секунды")
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.2)
        worker_marker(ax, runs)
    axes[0].set_ylabel("мегабайт")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncols=3)
    figure.suptitle("память", x=0.01, y=1.07, ha="left", fontsize=12, fontweight="bold")
    figure.tight_layout()
    save(figure, out / "memory.png")


def window_mean(series: pd.Series, elapsed: pd.Series, start: float, end: float) -> float:
    picked = series[(elapsed >= start) & (elapsed < end)]
    return float(picked.mean()) if not picked.empty else float("nan")


def quantile(tasks: pd.DataFrame, start: float, end: float, level: float) -> float:
    picked = tasks[(tasks["finished"] >= start) & (tasks["finished"] < end)]["latency"]
    return float(picked.quantile(level)) if not picked.empty else float("nan")


def summary(runs: list[Run], out: Path) -> None:
    def number(value: float, digits: int = 1) -> str:
        if value != value:
            return "—"
        return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")

    header = ["показатель", *(run.title for run in runs)]
    rows: list[list[str]] = []
    metrics: dict[str, list[str]] = {}
    for run in runs:
        monitor = run.monitor
        end = float(monitor["elapsed"].max())
        launch = run.launch or end
        joined = (run.joined or launch) + 5
        source_rate = per_second(monitor, "published", window=1)
        throughput = per_second(monitor, "processed", window=1)
        alive = monitor.loc[monitor["source_rss_mb"] > 0, "elapsed"]
        source_end = float(alive.max()) if not alive.empty else end
        result = run.result
        values = {
            "опубликовано задач": number(result["published"] or 0, 0),
            "обработано задач": number(result["done"], 0),
            "осталось в очереди после остановки": number(result["left_in_queue"], 0),
            "потеряно": number(result["lost"] if result["lost"] is not None else float("nan"), 0),
            "отказов брокера источнику": number(float(monitor["rejected"].max()), 0),
            "максимум задач в брокере": number(
                float((monitor["ready"] + monitor["unacked"]).max()), 0
            ),
            "источник работал, с": number(source_end, 0),
            "скорость источника, задач/с, один воркер": number(
                window_mean(source_rate, monitor["elapsed"], 10, min(launch, source_end))
            ),
            "скорость источника, задач/с, два воркера": number(
                window_mean(source_rate, monitor["elapsed"], joined, source_end)
            ),
            "обработка, задач/с, один воркер": number(
                window_mean(throughput, monitor["elapsed"], 10, launch)
            ),
            "обработка, задач/с, два воркера": number(
                window_mean(throughput, monitor["elapsed"], joined, end)
            ),
            "медиана времени задачи, с, один воркер": number(
                quantile(run.tasks, 10, launch, 0.5), 2
            ),
            "медиана времени задачи, с, два воркера": number(
                quantile(run.tasks, joined, end, 0.5), 2
            ),
            "95-й перцентиль времени задачи, с, за весь прогон": number(
                quantile(run.tasks, 0, float("inf"), 0.95), 2
            ),
            "прирост памяти rabbitmq за прогон, мб": number(
                float(monitor["broker_growth_mb"].max())
            ),
            "пик памяти самой очереди, мб": number(float(monitor["queue_memory_mb"].max())),
            "пик памяти источника, мб": number(float(monitor["source_rss_mb"].max())),
            "пик памяти воркеров, мб": number(float(monitor["workers_rss_mb"].max())),
            "пик задач в памяти воркеров": number(float(monitor["in_flight"].max()), 0),
        }
        for key, value in values.items():
            metrics.setdefault(key, []).append(value)
    rows = [[key, *values] for key, values in metrics.items()]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def draw_all(out: Path, scenarios: dict[str, dict[str, Any]]) -> None:
    runs = load(out, scenarios)
    if not runs:
        raise SystemExit(f"в {out} нет ни одного прогона, сначала запустите experiment.py run")
    style()
    depth_chart(runs, out)
    where_chart(runs, out)
    line_chart(
        runs,
        out / "source_rate.png",
        "скорость источника",
        "задач в секунду, логарифмическая шкала",
        {
            run.name: (
                run.monitor["elapsed"],
                per_second(run.monitor, "published")
                .clip(lower=1)
                .where(run.monitor["source_rss_mb"] > 0),
            )
            for run in runs
        },
        log=True,
    )
    line_chart(
        runs,
        out / "throughput.png",
        "сколько задач воркеры обрабатывают в секунду",
        "задач в секунду",
        {run.name: (run.monitor["elapsed"], per_second(run.monitor, "processed")) for run in runs},
        direct_labels=False,
    )
    line_chart(
        runs,
        out / "latency.png",
        "время задачи от публикации до результата, медиана по 5 секундам",
        "секунд, логарифмическая шкала",
        {run.name: latency_series(run) for run in runs},
        log=True,
        bottom=0.1,
    )
    memory_chart(runs, out)
    summary(runs, out)
