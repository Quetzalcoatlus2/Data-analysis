from __future__ import annotations

import io
import json
import html as htmllib
from typing import Any
from collections.abc import Callable

import numpy as np
import pandas as pd
from flask import Request, request


def _should_show_index_as_column(df: pd.DataFrame) -> bool:
    """Return True when the index carries meaningful dataset information."""
    try:
        idx = df.index
        if isinstance(idx, pd.RangeIndex):
            if idx.name in (None, "") and int(idx.start) == 0 and int(idx.step) == 1:
                return False
        return True
    except Exception:
        return False


def _display_df_with_index(df: pd.DataFrame) -> pd.DataFrame:
    """Materialize the index as the first display column when it is meaningful."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    try:
        if not _should_show_index_as_column(df):
            return df.copy()

        idx_name = str(df.index.name).strip() if df.index.name not in (None, "") else "Index"
        display_df = df.reset_index(drop=False)
        first_col = display_df.columns[0]

        if str(first_col) != idx_name:
            safe_name = idx_name
            if safe_name in display_df.columns[1:]:
                suffix = 1
                while f"{idx_name}_{suffix}" in display_df.columns[1:]:
                    suffix += 1
                safe_name = f"{idx_name}_{suffix}"
            display_df = display_df.rename(columns={first_col: safe_name})
        return display_df
    except Exception:
        return df.copy()


def build_ai_context(
    df: pd.DataFrame,
    anomalies_found: dict,
    corr_payload: dict | None,
    used_cols: list,
    is_timeseries: bool,
    forecast_horizon: int,
    contamination: float,
    *,
    is_reliable_timeseries_index: Callable[[pd.Index], bool] | None = None,
) -> str:
    """Assemble structured stats the AI can leverage for deeper analysis."""
    try:
        lines: list[str] = []
        lines.append(f"Shape: {getattr(df, 'shape', None)}")

        try:
            dtypes = {c: str(t) for c, t in df.dtypes.items()}
            lines.append("Dtypes: " + json.dumps(dtypes, ensure_ascii=False))
        except Exception:
            pass

        try:
            mv = df.isna().mean().sort_values(ascending=False)
            top_mv = mv[mv > 0].head(20)
            if not top_mv.empty:
                lines.append(
                    "Top missingness (fraction): "
                    + json.dumps({k: float(v) for k, v in top_mv.items()})
                )
        except Exception:
            pass

        try:
            nums = df.select_dtypes(include="number")
            if not nums.empty:
                desc = nums.describe().to_dict()
                compact: dict[str, dict[str, float]] = {}
                for col in nums.columns:
                    stats: dict[str, float] = {}
                    for key in ("mean", "50%", "std", "min", "max"):
                        if key in desc and col in desc[key]:
                            stats[key] = float(desc[key][col])
                    compact[str(col)] = stats
                lines.append(
                    "Numeric summary (mean, median, std, min, max): "
                    + json.dumps(compact)
                )
        except Exception:
            pass

        try:
            trend_info: dict[str, dict[str, float | int]] = {}
            for col in used_cols[:20]:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) >= 5:
                    w = min(len(s), max(20, len(s) // 5))
                    y = s.iloc[-w:]
                    x = np.arange(len(y), dtype=float)
                    y_arr = np.asarray(y.to_numpy(dtype=float), dtype=float)
                    slope, _intercept = np.polyfit(x, y_arr, 1)
                    change = float(y.iloc[-1] - y.iloc[0])
                    pct = float((change / (abs(y.iloc[0]) + 1e-12)) * 100.0)
                    trend_info[str(col)] = {
                        "window": int(w),
                        "slope_per_step": float(slope),
                        "recent_change": float(change),
                        "pct_change": pct,
                        "last": float(y.iloc[-1]),
                    }
            if trend_info:
                lines.append("Recent trends: " + json.dumps(trend_info))
        except Exception:
            pass

        try:
            if anomalies_found:
                lines.append("Anomalies summary: " + json.dumps(anomalies_found))
        except Exception:
            pass

        try:
            if corr_payload and corr_payload.get("z"):
                x = corr_payload["x"]
                y = corr_payload["y"]
                z = corr_payload["z"]
                pairs: list[tuple[float, float, Any, Any]] = []
                for i, row in enumerate(z):
                    for j, val in enumerate(row):
                        if i >= j:
                            continue
                        if val is None or isinstance(val, str):
                            continue
                        pairs.append((abs(float(val)), float(val), y[i], x[j]))
                pairs.sort(reverse=True)
                top = [{"pair": [a, b], "rho": v, "abs": av} for av, v, a, b in pairs[:15]]
                lines.append("Top correlations (Spearman): " + json.dumps(top))
        except Exception:
            pass

        try:
            if is_timeseries:
                idx_obj = df.index.dropna()
                if isinstance(idx_obj, pd.DatetimeIndex) and len(idx_obj):
                    reliable = True
                    if is_reliable_timeseries_index is not None:
                        try:
                            reliable = bool(is_reliable_timeseries_index(idx_obj))
                        except Exception:
                            reliable = True
                    if reliable:
                        idx_freq = idx_obj.freq
                        if idx_freq is not None:
                            freq = str(idx_freq)
                        else:
                            try:
                                inferred = pd.infer_freq(idx_obj)
                            except Exception:
                                inferred = None
                            freq = str(inferred) if inferred else "unknown"
                        lines.append(
                            f"Time series detected. Start: {str(idx_obj[0])}, End: {str(idx_obj[-1])}, Freq: {freq}"
                        )
        except Exception:
            pass

        lines.append(
            "User settings: "
            f"forecast_horizon={int(forecast_horizon)}, anomaly_contamination={float(contamination)}"
        )
        return "\n".join(lines)
    except Exception:
        return ""


def safe_df_head_html(df: pd.DataFrame, *, logger: Any | None = None) -> str:
    try:
        if df is None or df.shape[1] == 0:
            return "<p>No columns detected in the uploaded file.</p>"
        display_df = _display_df_with_index(df)
        with pd.option_context("display.max_columns", None, "display.width", None, "display.max_colwidth", None):
            html = display_df.head(10).to_html(
                classes=["dataframe"],
                max_cols=None,
                col_space=50,
                justify="center",
                index=False,
            )
            return f'<div style="overflow-x: auto; max-width: 100%;">{html}</div>'
    except Exception as e:
        if logger is not None:
            logger.warning("safe_df_head_html failed: %s", e)
        return "<p>Could not render head()</p>"


def safe_df_description_html(df: pd.DataFrame, *, logger: Any | None = None) -> str:
    try:
        if df is None or df.shape[1] == 0:
            return (
                df.dtypes.to_frame("dtype").to_html(classes=["dataframe"])
                if df is not None and isinstance(df, pd.DataFrame)
                else "<p>No data.</p>"
            )
        display_df = _display_df_with_index(df)
        try:
            desc = display_df.describe(include="all")
        except Exception:
            desc = display_df.select_dtypes(include="number").describe()
        with pd.option_context("display.max_columns", None, "display.width", None, "display.max_colwidth", None):
            html = desc.to_html(classes=["dataframe"], max_cols=None, col_space=50, justify="center")
            return f'<div style="overflow-x: auto; max-width: 100%;">{html}</div>'
    except Exception as e:
        if logger is not None:
            logger.warning("safe_df_description_html failed: %s", e)
        try:
            return df.dtypes.to_frame("dtype").to_html(classes=["dataframe"])
        except Exception:
            return "<p>Could not build description.</p>"


def get_dataset_overview_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build dataset overview as tabular data for web and PDF output."""
    total_rows = int(len(df.index))
    show_index_col = _should_show_index_as_column(df)
    total_cols = int(len(df.columns) + (1 if show_index_col else 0))
    display_df = _display_df_with_index(df)

    summary_rows = [
        ("Rows", total_rows),
        ("Columns (displayed)", total_cols),
        ("Index Name", str(df.index.name or "(none)")),
        ("Index Type", type(df.index).__name__),
        ("Memory Usage", f"{float(df.memory_usage(deep=True).sum()) / 1024:.1f} KB"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    non_null_counts = display_df.notna().sum()
    dtypes = display_df.dtypes.astype(str)
    columns_df = pd.DataFrame(
        {
            "#": list(range(len(display_df.columns))),
            "Column": [str(col) for col in display_df.columns],
            "Non-Null Count": [f"{int(non_null_counts[col])} non-null" for col in display_df.columns],
            "Dtype": [dtypes[col] for col in display_df.columns],
        }
    )
    return summary_df, columns_df


def safe_dataset_overview_html(df: pd.DataFrame, *, logger: Any | None = None) -> str:
    """Render dataset overview in table form for the Overview page."""
    try:
        if df is None or df.shape[1] == 0:
            return "<p>No dataset overview available.</p>"

        summary_df, columns_df = get_dataset_overview_tables(df)
        with pd.option_context("display.max_columns", None, "display.width", None, "display.max_colwidth", None):
            summary_html = summary_df.to_html(index=False, classes=["dataframe"], justify="left")
            columns_html = columns_df.to_html(
                index=False,
                classes=["dataframe"],
                max_cols=None,
                col_space=50,
                justify="left",
            )

        return (
            f'<div style="overflow-x: auto; max-width: 100%;">{summary_html}</div>'
            f'<div style="margin-top: 10px; overflow-x: auto; max-width: 100%;">{columns_html}</div>'
        )
    except Exception as e:
        if logger is not None:
            logger.warning("safe_dataset_overview_html failed: %s", e)
        return "<p>Could not render dataset overview table.</p>"


def get_cached_df_info(
    filename: str,
    df: pd.DataFrame,
    *,
    cache: Any | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    """Get cached head/description/info bundle for a DataFrame."""
    if cache is not None:
        cached = cache.get(filename)
        if cached is not None:
            return cached

    result: dict[str, Any] = {
        "head": safe_df_head_html(df, logger=logger),
        "description": safe_df_description_html(df, logger=logger),
        "overview_table_html": safe_dataset_overview_html(df, logger=logger),
        "info": None,
        "missing_values": None,
    }

    try:
        buf = io.StringIO()
        df.info(buf=buf)
        result["info"] = buf.getvalue()
    except Exception:
        result["info"] = "Unable to render DataFrame info()."

    try:
        mv = df.isnull().sum()
        mvf = mv[mv > 0]
        if not mvf.empty:
            html = mvf.to_frame("missing_count").to_html()
            result["missing_values"] = f'<div style="overflow-x: auto; max-width: 100%;">{html}</div>'
        else:
            result["missing_values"] = None
    except Exception:
        result["missing_values"] = None

    if cache is not None:
        cache.set(filename, result)
    return result


def describe_for_ai(
    df: pd.DataFrame,
    filename: str | None = None,
    *,
    describe_cache: Any | None = None,
) -> str:
    """Build a structured context block for AI prompts."""
    if filename and describe_cache is not None:
        cached = describe_cache.get(filename)
        if cached is not None:
            return cached
    try:
        if df is None or df.shape[1] == 0:
            return f"Empty or headerless table. Shape: {getattr(df, 'shape', None)}"

        parts: list[str] = []
        parts.append(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
        parts.append(f"Columns: {', '.join(str(c) for c in df.columns[:30])}")

        if df.index.name or not isinstance(df.index, pd.RangeIndex):
            idx_sample = df.index[:20].tolist()
            parts.append(f"Index ({df.index.name or 'unnamed'}): {idx_sample}")

        try:
            head_str = df.head(10).to_string()
            parts.append(f"\nFirst 10 rows:\n{head_str}")
        except Exception:
            pass

        try:
            desc = df.describe(include="all")
            parts.append(f"\nStatistics:\n{desc.to_string()}")
        except Exception:
            pass

        try:
            num_cols = df.select_dtypes(include="number").columns[:10]
            for col in num_cols:
                try:
                    sorted_df = df[[col]].dropna().sort_values(col, ascending=False)
                    top5 = sorted_df.head(5)
                    bottom5 = sorted_df.tail(5)
                    parts.append(f"\nTop 5 {col}:\n{top5.to_string()}")
                    parts.append(f"Bottom 5 {col}:\n{bottom5.to_string()}")
                except Exception:
                    pass
        except Exception:
            pass

        result = "\n".join(parts)
        if filename and describe_cache is not None:
            describe_cache.set(filename, result)
        return result
    except Exception:
        try:
            return "Columns and dtypes:\n" + str(df.dtypes)
        except Exception:
            return "No structured information available."


def offline_answer(
    df: pd.DataFrame,
    question: str = "summary",
    error=None,
    filename: str | None = None,
    *,
    sanitize_error_message: Callable[[str | None], str] | None = None,
    ai_status: dict[str, Any] | None = None,
    get_cached_numeric_df_fn: Callable[[str, pd.DataFrame], pd.DataFrame] | None = None,
    coerce_numeric_df_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> str:
    """
    Deterministic HTML answer when AI is unavailable.
    """
    try:
        q = (str(question or "")).strip().lower()
        parts: list[str] = []
        parts.append("<h3>Offline analysis</h3>")
        if error:
            try:
                reason_raw = getattr(error, "message", None) or str(error)
            except Exception:
                reason_raw = None
            try:
                if sanitize_error_message is not None:
                    reason = sanitize_error_message(reason_raw) or (
                        (ai_status or {}).get("message") or ""
                    )
                else:
                    reason = str(reason_raw or "")
            except Exception:
                reason = ""
            detail = f" Reason: {htmllib.escape(str(reason))}" if reason else ""
            parts.append(
                "<p><em>AI response unavailable. Showing a quick offline analysis instead.</em></p>"
                f"<p class=\"muted\"><small>{detail}</small></p>"
            )

        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            parts.append("<p>No data available.</p>")
            return "".join(parts)

        try:
            parts.append(f"<p><strong>Shape:</strong> {tuple(df.shape)}</p>")
            dtypes = ", ".join(
                [f"{htmllib.escape(str(c))}: {htmllib.escape(str(t))}" for c, t in df.dtypes.items()]
            )
            parts.append(f"<p><strong>Dtypes:</strong> {dtypes}</p>")
        except Exception:
            pass

        try:
            mv = df.isna().mean().sort_values(ascending=False)
            mv = mv[mv > 0].head(10)
            if not mv.empty:
                parts.append("<h4>Top missingness</h4><ul>")
                for col, frac in mv.items():
                    parts.append(f"<li><strong>{htmllib.escape(str(col))}</strong>: {round(float(frac) * 100, 2)}%</li>")
                parts.append("</ul>")
        except Exception:
            pass

        try:
            if filename and get_cached_numeric_df_fn is not None:
                df_num = get_cached_numeric_df_fn(filename, df)
            elif coerce_numeric_df_fn is not None:
                df_num = coerce_numeric_df_fn(df)
            else:
                df_num = df.select_dtypes(include="number")
            sel = df_num.select_dtypes(include="number")
            if not sel.empty:
                parts.append("<h4>Recent trends (last window)</h4><ul>")
                shown = 0
                for col in sel.columns:
                    s = pd.to_numeric(sel[col], errors="coerce").dropna()
                    if len(s) < 5:
                        continue
                    w = min(len(s), max(20, len(s) // 5))
                    y = s.iloc[-w:]
                    x = np.arange(len(y), dtype=float)
                    y_arr = np.asarray(y.to_numpy(dtype=float), dtype=float)
                    slope = np.polyfit(x, y_arr, 1)[0]
                    change = float(y.iloc[-1] - y.iloc[0])
                    pct = (change / (abs(y.iloc[0]) + 1e-12)) * 100.0
                    parts.append(
                        f"<li><strong>{htmllib.escape(str(col))}</strong>: "
                        f"slope {slope:.4g}, change {change:.4g} ({pct:.2f}%)</li>"
                    )
                    shown += 1
                    if shown >= 8:
                        break
                parts.append("</ul>")
        except Exception:
            pass

        if q and q != "summary":
            try:
                mentioned = []
                q_low = q.lower()
                for col in df.columns:
                    name = str(col)
                    if name.lower() in q_low:
                        mentioned.append(col)
                if not mentioned:
                    if filename and get_cached_numeric_df_fn is not None:
                        df_num = get_cached_numeric_df_fn(filename, df).select_dtypes(include="number")
                    elif coerce_numeric_df_fn is not None:
                        df_num = coerce_numeric_df_fn(df).select_dtypes(include="number")
                    else:
                        df_num = df.select_dtypes(include="number")
                    mentioned = list(df_num.columns[:3]) if df_num is not None else []
                if mentioned:
                    parts.append("<h4>Quick stats for relevant columns</h4><ul>")
                    for col in mentioned[:6]:
                        try:
                            s = pd.to_numeric(df[col], errors="coerce").dropna()
                            if s.empty:
                                parts.append(f"<li><strong>{htmllib.escape(str(col))}</strong>: no numeric data</li>")
                                continue
                            parts.append(
                                f"<li><strong>{htmllib.escape(str(col))}</strong>: "
                                f"mean={float(s.mean()):.4g}, median={float(s.median()):.4g}, "
                                f"std={float(s.std(ddof=1)):.4g}, min={float(s.min()):.4g}, max={float(s.max()):.4g}</li>"
                            )
                        except Exception:
                            parts.append(
                                f"<li><strong>{htmllib.escape(str(col))}</strong>: unable to compute stats</li>"
                            )
                    parts.append("</ul>")
            except Exception:
                pass

        return "".join(parts)
    except Exception:
        return "<p>Offline analysis is unavailable due to an internal error.</p>"


def _ensure_plot_dicts(items):
    """
    Normalize plot items into [{'title': str, 'img': base64}, ...]
    Accepts dicts, tuples, or raw base64 strings; drops invalid entries.
    """
    out = []
    if not items:
        return out
    for p in items:
        try:
            if isinstance(p, dict) and "img" in p:
                title = p.get("title", "")
                if not isinstance(title, str):
                    title = "" if title is None else str(title)
                out.append({"title": title, "img": p["img"]})
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                title, img = p[0], p[1]
                out.append({"title": "" if title is None else str(title), "img": img})
            elif isinstance(p, str):
                out.append({"title": "", "img": p})
        except Exception:
            continue
    return out


def _get_arg_float(name, default, req: Request | None = None):
    try:
        req = req or request
        v = (req.form.get(name) if req.method == "POST" else None)
        if v in (None, ""):
            v = req.args.get(name)
        return float(str(v)) if v not in (None, "") else default
    except Exception:
        return default


def _get_arg_int(name, default, req: Request | None = None):
    try:
        req = req or request
        v = (req.form.get(name) if req.method == "POST" else None)
        if v in (None, ""):
            v = req.args.get(name)
        return int(float(str(v))) if v not in (None, "") else default
    except Exception:
        return default


__all__ = [
    "build_ai_context",
    "safe_df_head_html",
    "safe_df_description_html",
    "get_dataset_overview_tables",
    "safe_dataset_overview_html",
    "get_cached_df_info",
    "describe_for_ai",
    "offline_answer",
    "_ensure_plot_dicts",
    "_get_arg_float",
    "_get_arg_int",
]
