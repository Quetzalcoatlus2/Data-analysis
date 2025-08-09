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

        User's Question: "{question}"

        Your Answer:
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while answering the question: {e}"


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

                # 4) Optional retention cleanup
                _cleanup_uploads_if_configured()

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
    # Prefer original display name; survive GET/POST and no-query refreshes
    display_name = request.args.get('display') or request.form.get('display') or filename

    try:
        # 1) Get df from cache if available to avoid re-reading deleted files
        df = DATAFRAME_CACHE.get(filename)

        # 2) If not cached, read from disk, then cache and optionally delete file immediately
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

        # --- Handle follow-up questions ---
        user_question = None
        ai_answer = None
        if request.method == 'POST':
            user_question = request.form.get('question')
            if user_question:
                ai_answer = get_ai_answer(df, user_question)

        # --- Data Analysis & Plotting ---
        plots = []
        forecast_plots = []
        anomalies_found = {}
        numeric_cols = df.select_dtypes(include='number').columns
        is_timeseries = isinstance(df.index, pd.DatetimeIndex)

        for column in numeric_cols:
            plots.append(generate_plot(df[column], f'Trend for {column}', 'Timestamp' if is_timeseries else 'Index', column))

            # Anomaly Detection
            model_iso = IsolationForest(contamination='auto', random_state=42)
            df['anomaly'] = model_iso.fit_predict(df[[column]])
            anomalies = df[df['anomaly'] == -1]

            if not anomalies.empty:
                anomalies_found[column] = {
                    'count': len(anomalies),
                    'min_value': anomalies[column].min(),
                    'max_value': anomalies[column].max(),
                    'mean_value': anomalies[column].mean()
                }
            df.drop('anomaly', axis=1, inplace=True)

            # Forecasting
            if is_timeseries and len(df[column]) >= 10:
                try:
                    model_arima = ARIMA(df[column], order=(5,1,0)).fit()
                    forecast = model_arima.get_forecast(steps=5)
                    forecast_index = pd.date_range(start=df.index[-1], periods=6, freq=df.index.freq)[1:]
                    forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
                    forecast_plots.append(generate_plot(df[column], f'Forecast for {column}', 'Timestamp', column, forecast_data=forecast_series))
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

        # AI Summary
        description_for_ai = df.describe().to_string()
        ai_summary = get_ai_summary(description_for_ai)

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