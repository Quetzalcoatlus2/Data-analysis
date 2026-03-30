from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]


def _load_name_map(name_map_path: str, logger: Any | None = None) -> dict[str, Any] | None:
    try:
        if os.path.exists(name_map_path):
            with open(name_map_path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        if logger is not None:
            logger.warning("Name map load warning: %s", exc)
    return None


def _save_name_map(
    name_map_path: str,
    original_name_map: dict[str, Any],
    logger: Any | None = None,
) -> None:
    try:
        with open(name_map_path, "w", encoding="utf-8") as handle:
            json.dump(original_name_map, handle, ensure_ascii=False, indent=2)
    except Exception as exc:
        if logger is not None:
            logger.warning("Name map save warning: %s", exc)


def _safe_delete(path: str, retries: int = 3, delay: float = 0.2, logger: Any | None = None):
    """Delete a file with retries to tolerate transient locks."""
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return True, None
        except PermissionError as exc:
            if logger is not None:
                logger.warning(
                    "Delete failed (permission denied %s), attempt %d/%d: %s",
                    path,
                    i + 1,
                    retries,
                    exc,
                )
            if i < retries - 1:
                time.sleep(delay)
            else:
                return False, "Permission denied (file may be locked by OneDrive or antivirus)"
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Delete failed (%s), attempt %d/%d: %s",
                    path,
                    i + 1,
                    retries,
                    exc,
                )
            if i < retries - 1:
                time.sleep(delay)
            else:
                return False, str(exc)
    return False, "File deletion failed after retries"


def allowed_file(filename: str | None, allowed_extensions: set[str]) -> bool:
    if not filename:
        return False
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in allowed_extensions


def read_csv_fallback(
    path: str,
    *,
    supported_encodings: Iterable[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    last_err: Exception | None = None
    encodings = list(supported_encodings or SUPPORTED_ENCODINGS)
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
        except Exception:
            raise
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)
    except TypeError:
        pass
    if last_err:
        raise last_err
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode with common encodings")


def read_json_fallback(path: str, *, supported_encodings: Iterable[str] | None = None) -> pd.DataFrame:
    last_err: Exception | None = None
    encodings = list(supported_encodings or SUPPORTED_ENCODINGS)
    for enc in encodings:
        try:
            with open(path, encoding=enc, errors="strict") as handle:
                return pd.read_json(handle, orient="records")
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
        except ValueError:
            try:
                with open(path, encoding=enc, errors="strict") as handle:
                    return pd.read_json(handle, lines=True)
            except Exception:
                continue
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return pd.read_json(handle, orient="records")
    except ValueError:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                return pd.read_json(handle, lines=True)
        except Exception:
            pass
    if last_err:
        raise last_err
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode JSON with common encodings")


def read_excel_smart(path: str) -> pd.DataFrame:
    """Read first non-empty sheet and infer a time-like index where possible."""
    try:
        with pd.ExcelFile(path) as xls:
            for sheet in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet, header=0)
                    df = df.dropna(how="all").dropna(axis=1, how="all")
                    if df is not None and df.shape[1] > 0:
                        for cand in ["timestamp", "date", "time"]:
                            if cand in df.columns:
                                with pd.option_context("mode.chained_assignment", None):
                                    try:
                                        dt = pd.to_datetime(df[cand], errors="coerce")
                                        if dt.notna().any():
                                            df[cand] = dt
                                    except Exception:
                                        pass
                                with contextlib.suppress(Exception):
                                    df = df.set_index(cand)
                                break
                        else:
                            first_col = df.columns[0]
                            try:
                                maybe_dt = pd.to_datetime(df[first_col], errors="coerce")
                                if maybe_dt.notna().sum() >= max(3, int(len(df) * 0.3)):
                                    df = df.set_index(first_col)
                            except Exception:
                                pass
                        return df

                    df2 = pd.read_excel(xls, sheet_name=sheet, header=None)
                    df2 = df2.dropna(how="all").dropna(axis=1, how="all")
                    if df2 is not None and df2.shape[1] > 0:
                        header_row = df2.index[df2.notna().any(axis=1)][0] if not df2.empty else 0
                        df2.columns = df2.iloc[df2.index[df2.notna().any(axis=1)][0]]
                        df2 = df2.drop(df2.index[: header_row + 1])
                        df2 = df2.loc[:, df2.columns.notna()]
                        for cand in ["timestamp", "date", "time"]:
                            if cand in df2.columns:
                                try:
                                    dt2 = pd.to_datetime(df2[cand], errors="coerce")
                                    if dt2.notna().any():
                                        df2[cand] = dt2
                                    df2 = df2.set_index(cand)
                                except Exception:
                                    pass
                                break
                        return df2
                except Exception:
                    continue
        return pd.DataFrame()
    except Exception as exc:
        try:
            return pd.read_excel(path)
        except Exception:
            raise exc


def _cleanup_uploads_if_configured(
    *,
    upload_retention_days: int | None,
    uploads_dir: str,
    hashed_upload_re: Any,
    logger: Any | None = None,
) -> None:
    if not upload_retention_days:
        return
    cutoff = datetime.now() - timedelta(days=upload_retention_days)
    try:
        if not os.path.isdir(uploads_dir):
            return
        for name in os.listdir(uploads_dir):
            path = os.path.join(uploads_dir, name)
            if not os.path.isfile(path):
                continue
            if not hashed_upload_re.match(name):
                continue
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                try:
                    os.remove(path)
                except Exception as exc:
                    if logger is not None:
                        logger.warning("Cleanup warning: %s", exc)
    except Exception as exc:
        if logger is not None:
            logger.warning("Cleanup scan failed: %s", exc)


def _try_parse_numeric_series(s: pd.Series) -> pd.Series:
    """Best-effort conversion for object-like numeric strings."""
    if not isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")

    out = pd.to_numeric(s, errors="coerce")
    na_ratio = out.isna().mean()
    if na_ratio <= 0.25:
        return out

    ss = s.astype(str).str.strip()
    has_pct = ss.str.contains(r"%", regex=True, na=False)
    cleaned = ss.str.replace(r"[^0-9,.\-+eE]", " ", regex=True).str.replace(r"\s+", "", regex=True)

    comma_cnt = cleaned.str.count(",").sum()
    dot_cnt = cleaned.str.count(r"\.").sum()
    if comma_cnt > dot_cnt:
        attempt = cleaned.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
    else:
        attempt = cleaned.str.replace(",", "", regex=False)

    out2 = pd.to_numeric(attempt, errors="coerce")
    if has_pct.any():
        out2 = out2.where(~has_pct, out2 / 100.0)

    if out2.notna().sum() >= out.notna().sum():
        return out2
    return out


def _looks_temporal_series(s: pd.Series) -> bool:
    """Return True when a non-numeric series is primarily datetime-like text."""
    if not isinstance(s, pd.Series):
        return False
    if s.empty:
        return False
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False

    sample = s.head(200).dropna()
    if sample.empty:
        return False

    sample = sample.astype(str).str.strip()
    if sample.empty:
        return False

    name_hint = str(s.name or "").lower()
    has_name_hint = any(token in name_hint for token in ("date", "time", "timestamp", "datetime"))
    # Numeric-looking string columns (e.g. "1", "2") can parse as epoch-like
    # datetimes; require at least ~10% non-numeric evidence before temporal parse.
    numeric_only_ratio = float(sample.str.fullmatch(r"[+-]?\d+", na=False).mean())
    if numeric_only_ratio >= 0.9:
        return False

    temporal_pattern_ratio = float(
        sample.str.contains(r"[-/:]|[Tt ]\d{1,2}:\d{2}|[AaPp][Mm]|Z$", regex=True, na=False).mean()
    )
    if not has_name_hint and temporal_pattern_ratio < 0.6:
        return False

    try:
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    except Exception:
        return False
    return bool(parsed.notna().mean() >= 0.8)


def _is_active_temporal_axis_column(
    df: pd.DataFrame,
    column: object,
    *,
    is_reliable_timeseries_index: Callable[[pd.Index | Any], bool] | None = None,
) -> bool:
    """Return True when ``column`` is the temporal field already driving the x-axis.

    Categories built from the same datetime-like field that was promoted to the
    dataset index are not useful: they count each timestamp against itself. This
    helper keeps the filter narrow by only matching the actual active time axis.
    """
    if df is None or df.empty:
        return False

    if column not in df.columns:
        return False

    index_name = getattr(df.index, "name", None)
    if index_name in (None, "") or str(index_name) != str(column):
        return False

    try:
        index_is_temporal = (
            bool(is_reliable_timeseries_index(df.index))
            if callable(is_reliable_timeseries_index)
            else isinstance(df.index, pd.DatetimeIndex)
        )
    except Exception:
        index_is_temporal = isinstance(df.index, pd.DatetimeIndex)

    if not index_is_temporal:
        return False

    series = df[column]
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return True
    return _looks_temporal_series(series)


def coerce_numeric_df(
    df: pd.DataFrame,
    *,
    parse_numeric_series_fn: Callable[[pd.Series], pd.Series] = _try_parse_numeric_series,
) -> pd.DataFrame:
    """Apply robust numeric parsing to object-like columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    res: dict[Any, pd.Series] = {}
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_datetime64_any_dtype(ser) or pd.api.types.is_timedelta64_dtype(ser):
            continue
        if pd.api.types.is_numeric_dtype(ser):
            res[col] = ser.astype(float)
        else:
            if _looks_temporal_series(ser):
                continue
            coerced = parse_numeric_series_fn(ser)
            if coerced.notna().any():
                res[col] = coerced
    return pd.DataFrame(res, index=df.index)


def get_cached_numeric_df(
    filename: str,
    df: pd.DataFrame,
    *,
    numeric_df_cache: Any,
    coerce_numeric_df_fn: Callable[[pd.DataFrame], pd.DataFrame] = coerce_numeric_df,
) -> pd.DataFrame:
    cached = numeric_df_cache.get(filename)
    if cached is not None:
        return cached
    result = coerce_numeric_df_fn(df)
    numeric_df_cache.set(filename, result)
    return result


def get_dataframe_for(
    filename: str,
    *,
    dataframe_cache: Any,
    uploads_dir: str,
    upload_folder: str,
    logger: Any | None,
    read_csv_fallback_fn: Callable[..., pd.DataFrame] = read_csv_fallback,
    read_json_fallback_fn: Callable[[str], pd.DataFrame] = read_json_fallback,
    read_excel_smart_fn: Callable[[str], pd.DataFrame] = read_excel_smart,
    is_reliable_timeseries_index: Callable[[pd.Index | Any], bool],
) -> pd.DataFrame | None:
    """Best-effort loader for an uploaded dataset by hashed filename."""
    try:
        df = dataframe_cache.get(filename)
        if df is not None:
            return df

        target_uploads_dir = uploads_dir or upload_folder
        path = os.path.join(target_uploads_dir, filename)

        # Security: reject path-traversal attempts (e.g. "../outside.csv")
        try:
            real_path = os.path.realpath(path)
            real_uploads = os.path.realpath(target_uploads_dir)
            if not real_path.startswith(real_uploads + os.sep) and real_path != real_uploads:
                if logger is not None:
                    logger.warning("get_dataframe_for: path traversal blocked: %s", filename)
                return None
        except Exception:
            return None

        if not os.path.exists(path):
            if logger is not None:
                logger.info("get_dataframe_for: file not found on disk: %s", path)
            return None

        _, ext = os.path.splitext(filename)
        ext = (ext or "").lower()

        if ext == ".csv":
            df = read_csv_fallback_fn(path, parse_dates=False)
        elif ext == ".xlsx":
            df = read_excel_smart_fn(path)
        elif ext == ".json":
            df = read_json_fallback_fn(path)
            for col in ["timestamp", "date", "time"]:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                    except Exception:
                        pass
                    break
        elif ext == ".txt":
            df = read_csv_fallback_fn(path, sep=",", parse_dates=False)
        else:
            if logger is not None:
                logger.warning("get_dataframe_for: unsupported extension %s", ext)
            return None

        if not isinstance(df, pd.DataFrame):
            if logger is not None:
                logger.info("get_dataframe_for: reader returned non-DataFrame for %s", filename)
            return None

        with contextlib.suppress(Exception):
            df = df.dropna(axis=1, how="all")

        try:
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) >= 20:
                idx = pd.DatetimeIndex(df.index)
                diffs = pd.Series(idx).diff().dropna()
                non_zero = diffs[diffs > pd.Timedelta(0)]
                min_step = non_zero.min() if not non_zero.empty else pd.Timedelta(0)
                year_1970_ratio = float((idx.year == 1970).mean()) if len(idx) else 0.0
                if year_1970_ratio >= 0.95 and min_step < pd.Timedelta(microseconds=1):
                    raw_idx = pd.Index(idx.view("int64"), name=idx.name)
                    raw_num = pd.to_numeric(pd.Series(raw_idx), errors="coerce")
                    year_like = (
                        raw_num.notna().sum() >= max(5, int(0.8 * len(raw_num)))
                        and raw_num.dropna().between(1000, 3000).mean() >= 0.9
                    )
                    if year_like:
                        ts_year = pd.to_datetime(
                            raw_num.round().astype("Int64").astype(str),
                            format="%Y",
                            errors="coerce",
                            utc=False,
                        )
                        if ts_year.notna().sum() >= max(5, int(0.8 * len(ts_year))):
                            df.index = pd.DatetimeIndex(ts_year, name=idx.name)
                        else:
                            df.index = raw_idx
                    else:
                        df.index = raw_idx
        except Exception as exc:
            if logger is not None:
                logger.debug("get_dataframe_for: datetime index recovery skipped: %s", exc)

        try:
            if not is_reliable_timeseries_index(df.index) and df.shape[1] >= 1:
                # Fallback: set the first column as the index so it acts as the X-axis for graphs
                # We assign it but do NOT drop it from the DataFrame columns so it stays in tables.
                df.index = pd.Index(df.iloc[:, 0].values, name=df.columns[0])

                def _to_datetime_candidate(series_like: pd.Series) -> pd.Series:
                    ser = pd.Series(series_like)
                    num = pd.to_numeric(ser, errors="coerce")
                    year_like = (
                        num.notna().sum() >= max(5, int(0.8 * len(num)))
                        and num.dropna().between(1000, 3000).mean() >= 0.9
                    )
                    if year_like:
                        return pd.to_datetime(
                            num.round().astype("Int64").astype(str),
                            format="%Y",
                            errors="coerce",
                            utc=False,
                        )
                    if pd.api.types.is_numeric_dtype(ser):
                        return pd.Series([pd.NaT] * len(ser), index=ser.index)
                    return pd.to_datetime(ser, errors="coerce", utc=False)

                candidate_cols: list[Any] = []
                for col in df.columns:
                    lc = str(col).strip().lower()
                    if any(tok in lc for tok in ("date", "time", "timestamp", "datetime")):
                        candidate_cols.append(col)

                if not candidate_cols:
                    first_col = df.columns[0]
                    first_ser = df[first_col]
                    num_first = pd.to_numeric(first_ser, errors="coerce")
                    year_like_first = (
                        num_first.notna().sum() >= max(5, int(0.8 * len(num_first)))
                        and num_first.dropna().between(1000, 3000).mean() >= 0.9
                    )
                    if (not pd.api.types.is_numeric_dtype(first_ser)) or year_like_first:
                        candidate_cols = [first_col]

                picked = None
                for col in candidate_cols:
                    ts = _to_datetime_candidate(df[col])
                    if ts.notna().sum() >= max(5, int(0.6 * len(ts))):
                        picked = col
                        break

                if picked is not None:
                    ts = _to_datetime_candidate(df[picked])
                    # Do not drop the column so it remains available in tables
                    df.index = ts
                    df.index.name = picked
                    with contextlib.suppress(Exception):
                        df = df[ts.notna()].sort_index()
        except Exception as exc:
            if logger is not None:
                logger.debug("get_dataframe_for: datetime inference skipped: %s", exc)

        dataframe_cache.set(filename, df)
        return df
    except Exception:
        if logger is not None:
            logger.exception("get_dataframe_for failed for %s", filename)
        return None


__all__ = [
    "allowed_file",
    "_load_name_map",
    "_save_name_map",
    "_safe_delete",
    "read_csv_fallback",
    "read_json_fallback",
    "read_excel_smart",
    "_cleanup_uploads_if_configured",
    "_try_parse_numeric_series",
    "_looks_temporal_series",
    "_is_active_temporal_axis_column",
    "coerce_numeric_df",
    "get_cached_numeric_df",
    "get_dataframe_for",
]
