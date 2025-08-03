import os
import io
import base64
from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for the server
import matplotlib.pyplot as plt

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx', 'json'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey' # Change this in a real application

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
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
            # For JSON, we assume a 'records' orientation by default.
            # This works for JSON files structured like: [{"col1": "val1"}, {"col2": "val2"}]
            df = pd.read_json(filepath, orient='records')
            # Attempt to convert a common timestamp column to datetime objects and set as index
            for col in ['timestamp', 'date', 'time']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                    except Exception:
                        # Ignore if conversion fails
                        pass
                    break
        elif filename.endswith('.txt'):
            # Updated to handle comma-separated txt files like the sensor data example
            df = pd.read_csv(filepath, sep=',', index_col=0, parse_dates=True) 
        else:
            flash('Unsupported file type')
            return redirect(url_for('upload_file'))

        # --- Generate Plots ---
        plots = []
        # Select only numeric columns for plotting
        numeric_cols = df.select_dtypes(include='number').columns
        
        for column in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            df[column].plot(ax=ax)
            ax.set_title(f'Trend for {column}')
            ax.set_xlabel('Timestamp' if isinstance(df.index, pd.DatetimeIndex) else 'Index')
            ax.set_ylabel(column)
            ax.grid(True)
            
            # Save plot to a memory buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Encode image to base64 to embed in HTML
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plots.append(image_base64)
            plt.close(fig) # Close the figure to free up memory

        # --- Correctly capture DataFrame info ---
        buf = io.StringIO()
        df.info(buf=buf)
        info_string = buf.getvalue()

        # --- Correctly calculate and format missing values ---
        missing_values_data = df.isnull().sum()
        # Only include columns that have one or more missing values
        missing_values_filtered = missing_values_data[missing_values_data > 0]
        missing_values_html = None
        if not missing_values_filtered.empty:
            missing_values_html = missing_values_filtered.to_frame('missing_count').to_html()


        # Basic data analysis
        analysis = {
            'head': df.head().to_html(),
            'description': df.describe().to_html(),
            'info': info_string,
            'missing_values': missing_values_html, # This will be None if no missing values
            'plots': plots
        }
        
        # You can add more complex analysis here:
        # - Trend analysis
        # - Anomaly detection
        # - Prognostics

        return render_template('analysis.html', analysis=analysis, filename=filename)

    except Exception as e:
        flash(f"An error occurred while analyzing the file: {e}")
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)