from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from qlib.contrib.evaluate import risk_analysis


LayoutType = Literal["single_run", "summary_only", "batch_runs", "unknown"]


def detect_experiment_layout(experiment_dir: Path) -> LayoutType:
    if (experiment_dir / "report.csv").exists():
        return "single_run"
    if (experiment_dir / "summary.csv").exists():
        for child in experiment_dir.iterdir():
            if child.is_dir() and (child / "report.csv").exists():
                return "batch_runs"
        return "summary_only"
    return "unknown"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_line_chart(series: pd.Series, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_bar_chart(series: pd.Series, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _df_to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="table")


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _extract_experiment_summary(experiment_dir: Path) -> str:
    summary_json = experiment_dir / "summary.json"
    summary_csv = experiment_dir / "summary.csv"

    annualized_return = None
    sharpe = None
    max_drawdown = None

    if summary_json.exists():
        data = json.loads(summary_json.read_text())
        annualized_return = data.get("annualized_return")
        sharpe = data.get("sharpe")
        max_drawdown = data.get("max_drawdown")
    elif summary_csv.exists():
        df = pd.read_csv(summary_csv)
        if not df.empty:
            if "sharpe" in df.columns:
                best = df.sort_values("sharpe", ascending=False).iloc[0]
            else:
                best = df.iloc[0]
            annualized_return = best.get("annualized_return")
            sharpe = best.get("sharpe")
            max_drawdown = best.get("max_drawdown")

    if annualized_return is None and sharpe is None and max_drawdown is None:
        return "无可提取摘要"
    sharpe_text = "N/A" if sharpe is None or pd.isna(sharpe) else f"{float(sharpe):.2f}"
    return f"年化 {_format_pct(annualized_return)} | Sharpe {sharpe_text} | 最大回撤 {_format_pct(max_drawdown)}"


def build_index_html(generated: list[dict], missing: list[str]) -> str:
    lines = [
        "<html><head><meta charset='utf-8'><title>实验报告索引</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:32px;} li{margin:10px 0;} .summary{color:#555;margin-top:4px;}</style>",
        "</head><body>",
        "<h1>实验报告索引</h1>",
        "<ul>",
    ]
    for item in generated:
        lines.append(
            f"<li><a href='{escape(item['href'])}'>{escape(item['name'])}</a><div class='summary'>{escape(item['summary'])}</div></li>"
        )
    lines.append("</ul>")
    if missing:
        lines.append("<h2>未找到的实验</h2><ul>")
        for name in missing:
            lines.append(f"<li>{escape(name)}</li>")
        lines.append("</ul>")
    lines.append("</body></html>")
    return "\n".join(lines)


def _single_run_html(experiment_dir: Path, output_dir: Path) -> Path:
    _ensure_dir(output_dir)
    report = pd.read_csv(experiment_dir / "report.csv", parse_dates=["datetime"]).set_index("datetime")
    summary = json.loads((experiment_dir / "summary.json").read_text()) if (experiment_dir / "summary.json").exists() else {}
    monthly = (
        pd.read_csv(experiment_dir / "monthly_returns.csv")
        if (experiment_dir / "monthly_returns.csv").exists()
        else pd.DataFrame()
    )

    net_curve = (1 + (report["return"] - report["cost"])).cumprod()
    drawdown_curve = net_curve / net_curve.cummax() - 1
    risk_df = risk_analysis(report["return"] - report["cost"], freq="60min", mode="product").reset_index()
    risk_df.columns = ["metric", "value"]

    equity_png = output_dir / "equity_curve.png"
    drawdown_png = output_dir / "drawdown_curve.png"
    monthly_png = output_dir / "monthly_returns.png"
    _save_line_chart(net_curve, "收益曲线", equity_png)
    _save_line_chart(drawdown_curve, "回撤曲线", drawdown_png)
    if not monthly.empty:
        monthly_series = monthly.set_index(monthly.columns[0])[monthly.columns[1]]
        _save_bar_chart(monthly_series, "月度收益", monthly_png)

    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>{escape(experiment_dir.name)}</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
        img {{ max-width: 100%; border: 1px solid #ddd; }}
      </style>
    </head>
    <body>
      <h1>{escape(experiment_dir.name)}</h1>
      <h2>参数与摘要</h2>
      {_df_to_html_table(pd.DataFrame([summary])) if summary else '<p>无 summary.json</p>'}
      <h2>风险分析</h2>
      {_df_to_html_table(risk_df)}
      <div class="grid">
        <div><h2>收益曲线</h2><img src="{equity_png.name}" alt="收益曲线"></div>
        <div><h2>回撤曲线</h2><img src="{drawdown_png.name}" alt="回撤曲线"></div>
      </div>
      <h2>月度收益</h2>
      {f'<img src="{monthly_png.name}" alt="月度收益图">' if monthly_png.exists() else '<p>无月度收益图</p>'}
      {(_df_to_html_table(monthly) if not monthly.empty else '<p>无月度收益表</p>')}
    </body>
    </html>
    """
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _summary_only_html(experiment_dir: Path, output_dir: Path) -> Path:
    _ensure_dir(output_dir)
    summary = pd.read_csv(experiment_dir / "summary.csv")
    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>{escape(experiment_dir.name)}</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
      </style>
    </head>
    <body>
      <h1>{escape(experiment_dir.name)}</h1>
      <h2>实验汇总</h2>
      {_df_to_html_table(summary)}
    </body>
    </html>
    """
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def _batch_runs_html(experiment_dir: Path, output_dir: Path) -> Path:
    _ensure_dir(output_dir)
    summary = pd.read_csv(experiment_dir / "summary.csv")
    links = []
    for child in sorted(experiment_dir.iterdir()):
        if child.is_dir() and (child / "report.csv").exists():
            child_out = output_dir / child.name
            _single_run_html(child, child_out)
            links.append({"run": child.name, "report": f"{child.name}/index.html"})

    links_df = pd.DataFrame(links)
    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>{escape(experiment_dir.name)}</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
      </style>
    </head>
    <body>
      <h1>{escape(experiment_dir.name)}</h1>
      <h2>实验汇总</h2>
      {_df_to_html_table(summary)}
      <h2>子实验报告</h2>
      {_df_to_html_table(links_df) if not links_df.empty else '<p>无子实验</p>'}
    </body>
    </html>
    """
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def generate_experiment_report(experiment_dir: Path, output_dir: Path) -> Path:
    layout = detect_experiment_layout(experiment_dir)
    if layout == "single_run":
        return _single_run_html(experiment_dir, output_dir)
    if layout == "summary_only":
        return _summary_only_html(experiment_dir, output_dir)
    if layout == "batch_runs":
        return _batch_runs_html(experiment_dir, output_dir)
    raise ValueError(f"Unsupported experiment layout for {experiment_dir}")
