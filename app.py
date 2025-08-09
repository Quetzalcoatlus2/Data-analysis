import os
import io
import base64
from flask import Flask, request, render_template, redirect, url_for, flash

from datetime import datetime, timedelta
import pandas as pd
from werkzeug.utils import secure_filename

# --- Add these two lines ---
from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# AI & ML Libraries
import google.generativeai as genai
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
import hashlib
import uuid
import json  # <— add
import numpy as np  # add
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # add

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- AI Configuration with Debugging ---
print("--- 1. Attempting to configure AI... ---")
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('models/gemini-2.5-pro') 
    AI_ENABLED = True
    print("--- 2. AI configured successfully. ---")
except Exception as e:
    print(f"--- [CRITICAL] AI configuration FAILED. AI is DISABLED. Error: {e} ---")
    model = None
    AI_ENABLED = False

# Configuration
UPLOAD_FOLDER = 'datasets'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'json'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'  # Change this in a real application
app.config['DELETE_UPLOADED_AFTER_PROCESSING'] = True  # <— immediate delete after reading
app.config['UPLOAD_RETENTION_DAYS'] = None  # <— set to an int (e.g., 14) to auto-delete old files

# Add this right after the configs:
DATAFRAME_CACHE = {}              # key: hashed filename -> DataFrame
NAME_MAP_PATH = os.path.join(UPLOAD_FOLDER, "_name_map.json")  # <— add
app.config['AI_FULL_UPLOAD_MAX_MB'] = 5  # only upload full file if <= 5 MB
AI_FILE_MAP = {}  # key: hashed filename -> genai uploaded file handle

def _load_name_map():
    global ORIGINAL_NAME_MAP
    try:
        if os.path.exists(NAME_MAP_PATH):
            with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
                ORIGINAL_NAME_MAP = json.load(f)
    except Exception as e:
        print(f"Name map load warning: {e}")

def _save_name_map():
    try:
        with open(NAME_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(ORIGINAL_NAME_MAP, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Name map save warning: {e}")

# Ensure folders and load map at startup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
_load_name_map()

SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ai_summary(dataframe_description):
    print("--- 3. Calling get_ai_summary function... ---")
    if not AI_ENABLED or model is None:
        print("--- 4. AI is disabled, returning early from get_ai_summary. Check step 2. ---")
        return "AI analysis is disabled. Please check your Google API key and terminal for configuration errors."
    
    try:
        print("--- 5. Preparing to call Google API... ---")
        prompt = f"""
        You are a data analyst. Based on the following statistical description of a dataset, 
        provide a brief summary and highlight potential insights, trends, or anomalies.
        The user is interested in conclusions, prognostics, and anomalies.
        Keep your analysis concise and easy to understand.

        Strict formatting instructions:
        - Output strictly as an HTML snippet (no <html> or <body> tags).
        - Use <h3>, <p>, <ul><li>, and <strong> for structure.
        - Do NOT use Markdown symbols like #, *, or ```; no code fences.

        Data Description:
        {dataframe_description}
        """
        
        response = model.generate_content(prompt)
        print("--- 6. Google API call successful. ---")

        if not response.parts:
            block_reason = response.prompt_feedback.block_reason.name
            error_message = f"AI analysis was blocked by the content filter. Reason: {block_reason}"
            print(f"--- 7a. {error_message} ---")
            return error_message

        print("--- 7b. Successfully got AI summary. ---")
        return response.text
        
    except Exception as e:
        print(f"--- [CRITICAL] An exception occurred during the API call: {e} ---")
        return f"An error occurred during AI analysis. Check the terminal for more details. Error: {e}"

def get_ai_answer(dataframe, question):
    """Generates a specific answer to a user's question about the dataframe."""
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."
    
    # Give the AI context with the dataframe's structure and a summary
    df_head = dataframe.head().to_string()
    df_description = dataframe.describe().to_string()

    try:
        prompt = f"""
        You are a helpful data analyst. A user has a specific question about a dataset.
        Use the following information to answer the question.

        Data Summary:
        {df_description}

        First 5 rows of the dataset:
        {df_head}

        Answer the user's question strictly as an HTML snippet (no <html> or <body> tags).
        Use <h4>, <p>, <ul><li>, and <strong>. Do NOT use Markdown symbols.

        User's Question: "{question}"
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"<p>Error while answering the question: {e}</p>"

def get_ai_summary_with_file(df, file_asset=None):
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."
    # concise stats context (kept) + prefer attaching the full file if available
    df_description = df.describe().to_string()
    prompt = f"""
    You are a data analyst. Provide a concise HTML snippet with insights, trends, and anomalies.
    Use <h3>, <p>, <ul><li>, <strong>. No Markdown.

    Data Description:
    {df_description}
    """
    try:
        if file_asset:
            # Send the actual CSV file + instructions
            resp = model.generate_content([file_asset, prompt])
        else:
            resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"<p>Error during AI summary: {e}</p>"

def get_ai_answer_with_file(df, question, file_asset=None):
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled."
    df_head = df.head().to_string()
    df_description = df.describe().to_string()
    prompt = f"""
    You are a helpful data analyst. Answer as an HTML snippet (no <html>/<body>).
    Use <h4>, <p>, <ul><li>, <strong>. No Markdown.

    Context:
    Data Summary:
    {df_description}

    First 5 rows:
    {df_head}

    Question: "{question}"
    """
    try:
        if file_asset:
            resp = model.generate_content([file_asset, prompt])
        else:
            resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"<p>Error while answering the question: {e}</p>"

def generate_plot(data, title, xlabel, ylabel, forecast_data=None):
    """Helper function to generate and encode a plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    data.plot(ax=ax, label='Historical Data')
    if forecast_data is not None:
        forecast_data.plot(ax=ax, label='Forecast', linestyle='--')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

# ADD: helpers to build a proper future index and a clearer forecast plot
def _infer_future_index(idx, steps):
    # Datetime: use explicit freq or infer; fallback to median delta
    if isinstance(idx, pd.DatetimeIndex):
        freq = idx.freq or pd.infer_freq(idx)
        if freq is not None:
            offset = pd.tseries.frequencies.to_offset(freq)
        else:
            diffs = pd.Series(idx).diff().dropna()
            step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)
            offset = pd.tseries.frequencies.to_offset(step)
        start = idx[-1] + offset
        return pd.date_range(start=start, periods=steps, freq=offset)
    # Numeric/other: extend by median step or 1
    try:
        ser_idx = pd.Series(idx.astype('int64') if hasattr(idx, 'astype') else list(idx))
    except Exception:
        ser_idx = pd.Series(range(len(idx)))
    diffs = ser_idx.diff().dropna()
    step = int(diffs.median()) if not diffs.empty else 1
    last = int(ser_idx.iloc[-1])
    return pd.Index([last + step * (i + 1) for i in range(steps)])

# add: simple seasonality inference for DatetimeIndex
def _infer_seasonal_period(idx, min_seasons=2):
    if not isinstance(idx, pd.DatetimeIndex):
        return None
    freq = (idx.freqstr or pd.infer_freq(idx)) or ""
    f = freq.upper()
    # heuristics for common granularities
    if f.startswith("H"):  # hourly
        period = 24
    elif f.startswith("T") or f.startswith("MIN"):  # minutely
        period = 60
    elif f.startswith("S"):  # secondly
        period = 60
    elif f.startswith("D"):  # daily
        period = 7
    elif f.startswith("W"):  # weekly
        period = 52
    elif f.startswith("M"):  # monthly
        period = 12
    elif f.startswith("Q"):  # quarterly
        period = 4
    else:
        period = None
    # ensure enough data to estimate seasonality
    try:
        n = len(idx)
        if period is None or n < period * min_seasons:
            return None
        return period
    except Exception:
        return None

# add/replace: recent-slope forecaster to avoid flat futures
def _recent_slope_forecast(series, steps, window=None, damping=None):
    """
    Forecast using robust recent slope.
    - slope blends linear-regression slope with median step
    - optional damping (None or in (0,1)); default None for clearer trend
    """
    y = series.dropna()
    n = len(y)
    future_idx = _infer_future_index(series.index, steps)

    if n < 3:
        fc_mean = pd.Series([y.iloc[-1]] * steps, index=future_idx)
        ci = pd.concat([fc_mean, fc_mean], axis=1)
        ci.columns = ['lower', 'upper']
        return fc_mean, ci

    w = window or min(max(20, n // 5), n)
    y_win = y.iloc[-w:]

    # Linear trend + robust median step
    x = np.arange(len(y_win), dtype=float)
    slope_lr, intercept = np.polyfit(x, y_win.values, 1)
    diffs = np.diff(y_win.values)
    med_diff = float(np.median(diffs)) if len(diffs) else 0.0

    # Combine and enforce a minimum magnitude relative to recent steps
    slope = 0.5 * float(slope_lr) + 0.5 * med_diff
    baseline = max(abs(med_diff), 1e-12)
    min_mag = 0.25 * baseline
    if abs(slope) < min_mag:
        slope = np.sign(med_diff) * min_mag

    k = np.arange(1, steps + 1, dtype=float)
    if damping is not None and 0 < damping < 1:
        phi = float(damping)
        incr = (1 - np.power(phi, k)) / (1 - phi) * slope
        fc_vals = y.iloc[-1] + incr
    else:
        fc_vals = y.iloc[-1] + slope * k

    fc_mean = pd.Series(fc_vals, index=future_idx)

    # CI from residuals in the window (robust-ish)
    resid = y_win.values - (slope_lr * x + intercept)
    resid_std = float(np.nanstd(resid, ddof=1)) if len(resid) > 2 else float(np.nanstd(y_win.values, ddof=1))
    lower = fc_mean - 1.96 * resid_std
    upper = fc_mean + 1.96 * resid_std
    ci = pd.concat([lower, upper], axis=1)
    ci.columns = ['lower', 'upper']
    return fc_mean, ci

def generate_forecast_plot(history, forecast_series, title, xlabel, ylabel, conf_int=None, history_tail=200):
    fig, ax = plt.subplots(figsize=(10, 4))

    history_tail_series = history.tail(history_tail)
    history_tail_series.plot(ax=ax, label='History', color='tab:blue', linewidth=1.8)

    forecast_series.plot(
        ax=ax,
        label='Forecast',
        linestyle='--',
        color='orangered',
        linewidth=3,
        marker='o',
        markersize=3,
        zorder=3
    )

    # Confidence interval shading
    if conf_int is not None:
        try:
            lower = conf_int.iloc[:, 0]
            upper = conf_int.iloc[:, 1]
            lower.index = forecast_series.index
            upper.index = forecast_series.index
            ax.fill_between(
                forecast_series.index, lower, upper,
                color='orangered', alpha=0.22, label='95% CI', zorder=2
            )
        except Exception:
            pass

    # Clear forecast region separation
    try:
        split_x = history.index[-1]
        ax.axvline(split_x, color='gray', linestyle=':', linewidth=1.5, label='Forecast start', zorder=1)
        ax.axvspan(split_x, forecast_series.index[-1], color='orange', alpha=0.08, zorder=0)
    except Exception:
        pass

    # Focus y-limits on history tail + forecast (ignore very wide CI that can flatten visuals)
    try:
        y_stack = pd.concat([history_tail_series, forecast_series]).astype(float)
        y_min = float(np.nanmin(y_stack.values))
        y_max = float(np.nanmax(y_stack.values))
        if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
            pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
            ax.set_ylim(y_min - pad, y_max + pad)
    except Exception:
        pass

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

def read_csv_fallback(path, **kwargs):
    last_err = None
    for enc in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # If it's not a decode error, re-raise
            raise
    # Final lenient attempt for pandas>=2: replace undecodable bytes
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)
    except TypeError:
        pass
    if last_err:
        raise last_err
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode with common encodings")

def read_json_fallback(path):
    last_err = None
    for enc in SUPPORTED_ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return pd.read_json(f, orient="records")
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except ValueError:
            # Try JSON Lines if normal JSON fails
            try:
                with open(path, "r", encoding=enc, errors="strict") as f:
                    return pd.read_json(f, lines=True)
            except Exception:
                continue
    # Final lenient attempt
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return pd.read_json(f, orient="records")
    except ValueError:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return pd.read_json(f, lines=True)
    if last_err:
        raise last_err
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode JSON with common encodings")

def _cleanup_uploads_if_configured():
    days = app.config.get('UPLOAD_RETENTION_DAYS')
    if not days:
        return
    cutoff = datetime.now() - timedelta(days=days)
    try:
        for name in os.listdir(UPLOAD_FOLDER):
            path = os.path.join(UPLOAD_FOLDER, name)
            if not os.path.isfile(path):
                continue
            # only manage known data files
            if not any(name.lower().endswith(f".{ext}") for ext in ALLOWED_EXTENSIONS):
                continue
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Cleanup warning: {e}")
    except Exception as e:
        print(f"Cleanup scan failed: {e}")

# Detect if a forecast is too linear (nearly a straight line)
def _is_too_linear(series_like):
    try:
        y = pd.Series(series_like).astype(float).values
        x = np.arange(len(y), dtype=float)
        if len(y) < 3:
            return False
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return r2 > 0.985  # threshold for "too straight"
    except Exception:
        return False

# Bootstrap a "natural-looking" forecast path from recent increments
def _bootstrap_natural_path(series, steps, window=None, base_slope=None, n_samples=200, q_low=0.1, q_high=0.9):
    y = series.dropna()
    n = len(y)
    if n < 5:
        return _recent_slope_forecast(series, steps, window=None, damping=None)

    w = window or min(max(30, n // 4), n)
    y_win = y.iloc[-w:].astype(float)
    diffs = np.diff(y_win.values)
    if len(diffs) < 3 or np.allclose(diffs, 0):
        return _recent_slope_forecast(series, steps, window=None, damping=None)

    # robust center/scale for winsorization
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med))) + 1e-12
    lo_clip = med - 3.0 * mad
    hi_clip = med + 3.0 * mad

    # mild trend bias so the direction follows recent movement
    bias = base_slope if base_slope is not None else med
    bias_weight = 0.3  # small bias to avoid runaway

    future_idx = _infer_future_index(series.index, steps)
    paths = np.empty((n_samples, steps), dtype=float)

    rng = np.random.default_rng()
    for i in range(n_samples):
        incs = rng.choice(diffs, size=steps, replace=True).astype(float)
        # winsorize outliers
        incs = np.clip(incs, lo_clip, hi_clip)
        # add slight bias
        incs = incs + bias_weight * bias
        # cumulative path from last observed value
        path = y.iloc[-1] + np.cumsum(incs)
        paths[i, :] = path

    # aggregate to median and quantiles
    median_path = np.median(paths, axis=0)
    lower_path = np.quantile(paths, q_low, axis=0)
    upper_path = np.quantile(paths, q_high, axis=0)

    median_series = pd.Series(median_path, index=future_idx)
    lower_series = pd.Series(lower_path, index=future_idx)
    upper_series = pd.Series(upper_path, index=future_idx)
    conf_df = pd.concat([lower_series, upper_series], axis=1)
    conf_df.columns = ['lower', 'upper']
    return median_series, conf_df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            orig_name = secure_filename(file.filename)
            _, ext = os.path.splitext(orig_name)
            ext = ext.lower()

            # 1) Save to a temp file once (avoid re-reading the stream)
            temp_name = f"tmp_{uuid.uuid4().hex}{ext}"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
            file.save(temp_path)

            try:
                # 2) Hash the saved temp file
                hasher = hashlib.sha1()
                with open(temp_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
                        hasher.update(chunk)
                digest = hasher.hexdigest()

                # 3) Dedup by content hash
                storage_name = f"{digest}{ext}"
                final_path = os.path.join(app.config['UPLOAD_FOLDER'], storage_name)

                if os.path.exists(final_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"Warning: could not remove temp file {temp_path}: {e}")
                else:
                    os.replace(temp_path, final_path)

                # 4) Optional: upload full CSV to AI if within size cap
                try:
                    size_bytes = os.path.getsize(final_path)
                    if size_bytes <= app.config['AI_FULL_UPLOAD_MAX_MB'] * 1024 * 1024:
                        uploaded = genai.upload_file(path=final_path, mime_type="text/csv", display_name=orig_name)
                        AI_FILE_MAP[storage_name] = uploaded
                except Exception as e:
                    print(f"AI file upload skipped: {e}")

                # 5) Redirect with the original filename for display only
                return redirect(url_for('analyze_file', filename=storage_name, display=orig_name))
            except Exception as e:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception:
                    pass
                flash(f"Upload failed: {e}")
                return redirect(request.url)
    return render_template('index.html')

@app.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    display_name = request.args.get('display') or request.form.get('display') or filename

    try:
        df = DATAFRAME_CACHE.get(filename)

        if df is None:
            # Improved file reading to handle timestamps and non-UTF8 encodings
            if filename.endswith('.csv'):
                df = read_csv_fallback(filepath, index_col=0, parse_dates=True)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(filepath, index_col=0, parse_dates=True)
            elif filename.endswith('.json'):
                df = read_json_fallback(filepath)
                for col in ['timestamp', 'date', 'time']:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                            df.set_index(col, inplace=True)
                        except Exception:
                            pass
                        break
            elif filename.endswith('.txt'):
                df = read_csv_fallback(filepath, sep=',', index_col=0, parse_dates=True)
            else:
                flash('Unsupported file type')
                return redirect(url_for('upload_file'))

            # Cache for follow-up questions
            DATAFRAME_CACHE[filename] = df

            # Delete the stored hashed file immediately after successful parse (if enabled)
            if app.config.get('DELETE_UPLOADED_AFTER_PROCESSING', False):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Warning: could not delete uploaded file {filepath}: {e}")

        # DEFINE file_asset BEFORE any use (fixes possible NameError)
        file_asset = AI_FILE_MAP.get(filename)

        # --- Handle follow-up questions ---
        user_question = None
        ai_answer = None
        if request.method == 'POST':
            user_question = request.form.get('question')
            if user_question:
                ai_answer = get_ai_answer_with_file(df, user_question, file_asset)

        # --- Data Analysis & Plotting ---
        plots = []
        forecast_plots = []
        anomalies_found = {}
        numeric_cols = df.select_dtypes(include='number').columns
        is_timeseries = isinstance(df.index, pd.DatetimeIndex)

        for column in numeric_cols:
            series = df[column].dropna()
            if series.empty:
                continue

            plots.append(generate_plot(series, f'Trend for {column}', 'Timestamp' if is_timeseries else 'Index', column))

            if is_timeseries and len(series) >= 10:
                try:
                    steps = max(20, min(60, len(series) // 5))
                    conf_df = None
                    fc_mean = None

                    # 1) Holt (damped trend, no seasonality)
                    try:
                        hw = ExponentialSmoothing(
                            series, trend='add', damped_trend=True, seasonal=None,
                            initialization_method='estimated'
                        ).fit(optimized=True)

                        fc_vals = hw.forecast(steps)
                        future_idx = _infer_future_index(series.index, steps)
                        fc_mean = pd.Series(fc_vals.values, index=future_idx)

                        resid_std = float(np.nanstd(getattr(hw, 'resid', series - hw.fittedvalues), ddof=1))
                        lower = fc_mean - 1.96 * resid_std
                        upper = fc_mean + 1.96 * resid_std
                        conf_df = pd.concat([lower, upper], axis=1)
                        conf_df.columns = ['lower', 'upper']
                    except Exception as e_hw:
                        print(f"Holt-Winters (damped) failed for {column}: {e_hw}")

                    # 2) If HW nearly flat, use robust recent-slope forecast
                    need_slope = False
                    if fc_mean is not None:
                        recent = series.tail(min(len(series), 300)).values
                        diffs = np.diff(recent)
                        recent_step = float(np.median(np.abs(diffs))) if len(diffs) else 0.0
                        slope_fc = float((fc_mean.iloc[-1] - fc_mean.iloc[0]) / max(1, len(fc_mean) - 1))
                        flat_by_range = np.allclose(fc_mean.values, fc_mean.values[0], rtol=1e-3, atol=1e-6)
                        flat_by_slope = (recent_step > 0 and abs(slope_fc) < 0.25 * recent_step)
                        need_slope = flat_by_range or flat_by_slope

                    if fc_mean is None or need_slope:
                        fc_mean, conf_df = _recent_slope_forecast(series, steps, window=min(len(series), 200), damping=None)

                    # 3) Naturalize if the forecast is still too straight
                    try:
                        base_slope_est = float((fc_mean.iloc[-1] - fc_mean.iloc[0]) / max(1, len(fc_mean) - 1))
                        if _is_too_linear(fc_mean):
                            fc_mean, conf_df = _bootstrap_natural_path(
                                series, steps, window=min(len(series), 200), base_slope=base_slope_est,
                                n_samples=200, q_low=0.1, q_high=0.9
                            )
                    except Exception as e_nat:
                        print(f"Naturalization failed for {column}: {e_nat}")

                    # 4) Final fallback: ARIMA with time trend; still naturalize if straight
                    if fc_mean is None:
                        model_arima = ARIMA(series, order=(0, 1, 1), trend='t',
                                            enforce_stationarity=False, enforce_invertibility=False).fit()
                        fc = model_arima.get_forecast(steps=steps)
                        future_idx = _infer_future_index(series.index, steps)
                        fc_mean = pd.Series(fc.predicted_mean.values, index=future_idx)
                        conf_df = fc.conf_int()
                        try:
                            conf_df.index = future_idx
                        except Exception:
                            pass
                        # naturalize if too linear
                        try:
                            if _is_too_linear(fc_mean):
                                base_slope_est = float((fc_mean.iloc[-1] - fc_mean.iloc[0]) / max(1, len(fc_mean) - 1))
                                fc_mean, conf_df = _bootstrap_natural_path(
                                    series, steps, window=min(len(series), 200), base_slope=base_slope_est,
                                    n_samples=200, q_low=0.1, q_high=0.9
                                )
                        except Exception as e_nat2:
                            print(f"ARIMA naturalization failed for {column}: {e_nat2}")

                    forecast_plots.append(
                        generate_forecast_plot(
                            series, fc_mean, f'Forecast for {column}',
                            'Timestamp', column, conf_int=conf_df, history_tail=300
                        )
                    )
                except Exception as e:
                    print(f"Could not generate forecast for {column}: {e}")
        # Capture DataFrame info and missing values
        buf = io.StringIO()
        df.info(buf=buf)
        info_string = buf.getvalue()

        missing_values_data = df.isnull().sum()
        missing_values_filtered = missing_values_data[missing_values_data > 0]
        missing_values_html = None
        if not missing_values_filtered.empty:
            missing_values_html = missing_values_filtered.to_frame('missing_count').to_html()

        # AI Summary (prefer full-file asset if present)
        ai_summary = get_ai_summary_with_file(df, file_asset)

        analysis = {
            'head': df.head().to_html(),
            'description': df.describe().to_html(),
            'info': info_string,
            'missing_values': missing_values_html,
            'plots': plots,
            'forecast_plots': forecast_plots,
            'anomalies': anomalies_found,
            'ai_summary': ai_summary,
            'user_question': user_question,
            'ai_answer': ai_answer
        }

        return render_template('analysis.html', analysis=analysis, filename=filename, display_name=display_name)

    except Exception as e:
        flash(f"An error occurred while analyzing the file: {e}")
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)