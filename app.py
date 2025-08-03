import os
import io
import base64
from flask import Flask, request, render_template, redirect, url_for, flash
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


print(f"--- Key found: {os.getenv('GOOGLE_API_KEY')} ---")

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- AI Configuration ---
# The key is read from the environment variable GOOGLE_API_KEY
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # --- Change this line again to the latest model ---
    model = genai.GenerativeModel('gemini-1.5-pro-latest') 
    AI_ENABLED = True
except Exception as e:
    print(f"Warning: Google Gemini client could not be initialized. AI features will be disabled. Error: {e}")
    model = None
    AI_ENABLED = False

# Configuration
UPLOAD_FOLDER = 'datasets'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'json'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey' # Change this in a real application

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ai_summary(dataframe_description):
    if not AI_ENABLED or model is None:
        return "AI analysis is disabled. Please check your Google API key."
    
    try:
        prompt = f"""
        You are a data analyst. Based on the following statistical description of a dataset, 
        provide a brief summary and highlight potential insights, trends, or anomalies.
        The user is interested in conclusions, prognostics, and anomalies.
        Keep your analysis concise and easy to understand.

        Data Description:
        {dataframe_description}
        """
        
        response = model.generate_content(prompt)

        # --- New: Check for safety blocks ---
        if not response.parts:
            # This happens if the content is blocked.
            block_reason = response.prompt_feedback.block_reason.name
            error_message = f"AI analysis was blocked by the content filter. Reason: {block_reason}"
            print(f"--- {error_message} ---") # Print to terminal
            return error_message # Show on webpage

        return response.text
        
    except Exception as e:
        # --- New: Print the full error to the terminal ---
        print(f"--- An exception occurred during the API call: {e} ---")
        return f"An error occurred during AI analysis. Check the terminal for more details. Error: {e}"

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
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('analyze_file', filename=filename))
    return render_template('index.html')

@app.route('/analyze/<filename>')
def analyze_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Improved file reading to handle timestamps
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filepath, index_col=0, parse_dates=True)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath, orient='records')
            for col in ['timestamp', 'date', 'time']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                    except Exception:
                        pass
                    break
        elif filename.endswith('.txt'):
            df = pd.read_csv(filepath, sep=',', index_col=0, parse_dates=True) 
        else:
            flash('Unsupported file type')
            return redirect(url_for('upload_file'))

        # --- Data Analysis & Plotting ---
        plots = []
        forecast_plots = []
        anomalies_found = {}
        numeric_cols = df.select_dtypes(include='number').columns
        is_timeseries = isinstance(df.index, pd.DatetimeIndex)
        
        for column in numeric_cols:
            plots.append(generate_plot(df[column], f'Trend for {column}', 'Timestamp' if is_timeseries else 'Index', column))

            # --- Anomaly Detection ---
            model_iso = IsolationForest(contamination='auto', random_state=42)
            df['anomaly'] = model_iso.fit_predict(df[[column]])
            anomalies = df[df['anomaly'] == -1]
            if not anomalies.empty:
                anomalies_found[column] = anomalies[[column]].to_html()
            df.drop('anomaly', axis=1, inplace=True)

            # --- Forecasting (Prognostics) ---
            if is_timeseries and len(df[column]) >= 10:
                try:
                    model_arima = ARIMA(df[column], order=(5,1,0)).fit()
                    forecast = model_arima.get_forecast(steps=5)
                    forecast_index = pd.date_range(start=df.index[-1], periods=6, freq=df.index.freq)[1:]
                    forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
                    forecast_plots.append(generate_plot(df[column], f'Forecast for {column}', 'Timestamp', column, forecast_data=forecast_series))
                except Exception as e:
                    print(f"Could not generate forecast for {column}: {e}")

        # --- Correctly capture DataFrame info ---
        buf = io.StringIO()
        df.info(buf=buf)
        info_string = buf.getvalue()

        # --- Correctly calculate and format missing values ---
        missing_values_data = df.isnull().sum()
        missing_values_filtered = missing_values_data[missing_values_data > 0]
        missing_values_html = None
        if not missing_values_filtered.empty:
            missing_values_html = missing_values_filtered.to_frame('missing_count').to_html()

        # --- Get AI Summary ---
        description_for_ai = df.describe().to_string()
        ai_summary = get_ai_summary(description_for_ai)

        # --- Final Analysis Package ---
        analysis = {
            'head': df.head().to_html(),
            'description': df.describe().to_html(),
            'info': info_string,
            'missing_values': missing_values_html,
            'plots': plots,
            'forecast_plots': forecast_plots,
            'anomalies': anomalies_found,
            'ai_summary': ai_summary
        }
        
        return render_template('analysis.html', analysis=analysis, filename=filename)

    except Exception as e:
        flash(f"An error occurred while analyzing the file: {e}")
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)