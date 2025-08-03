import os
from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'xlsx'}

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
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filename.endswith('.txt'):
            # Assuming txt is csv-like, adjust separator if needed
            df = pd.read_csv(filepath, sep=r'\s+') 
        else:
            flash('Unsupported file type')
            return redirect(url_for('upload_file'))

        # Basic data analysis
        analysis = {
            'head': df.head().to_html(),
            'description': df.describe().to_html(),
            'info': df.info.__repr__(), # A bit of a hack to get info string
            'missing_values': df.isnull().sum().to_frame('missing_values').to_html()
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