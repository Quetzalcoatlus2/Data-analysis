# ruff: noqa: F401,F403,F405
from __future__ import annotations

from data_analysis.runtime_app import *

_LOCAL_SYMBOLS = {
    "_LOCAL_SYMBOLS",
    "_bind_runtime_globals",
    "handle_upload_file",
    "upload_file",
    "__all__",
}



def _bind_runtime_globals():
    import data_analysis.runtime_app as rt

    sync = getattr(rt, "_sync_ai_engine_state", None)
    if callable(sync):
        sync()

    g = globals()
    for key, value in rt.__dict__.items():
        if key.startswith("__") or key in _LOCAL_SYMBOLS:
            continue
        g[key] = value
    return rt


def handle_upload_file():
    _bind_runtime_globals()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            orig_name = secure_filename(file.filename or "")
            _, ext = os.path.splitext(orig_name)
            ext = ext.lower()

            
            temp_name = f"tmp_{uuid.uuid4().hex}{ext}"
            temp_path = os.path.join(app.config['UPLOADS_DIR'], temp_name)  
            file.save(temp_path)

            try:
                
                hasher = hashlib.sha256()
                with open(temp_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):  
                        hasher.update(chunk)
                digest = hasher.hexdigest()

                
                storage_name = f"{digest}{ext}"
                final_path = os.path.join(app.config['UPLOADS_DIR'], storage_name)  

                if os.path.exists(final_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        app.logger.warning("Could not remove temp file %s: %s", temp_path, e)
                else:
                    os.replace(temp_path, final_path)

                
                try:
                    size_bytes = os.path.getsize(final_path)
                    if size_bytes <= app.config['AI_FULL_UPLOAD_MAX_MB'] * 1024 * 1024:
                        uploaded = _get_genai().upload_file(
                            path=final_path,
                            mime_type="text/csv",
                            display_name=orig_name,
                        )
                        AI_FILE_MAP[storage_name] = uploaded
                except Exception as e:
                    app.logger.info("AI file upload skipped: %s", e)

                
                
                fh = request.form.get('forecast_horizon')  # legacy hidden
                fpct = request.form.get('forecast_pct')
                cont = request.form.get('contamination')
                start_view = request.form.get('view') or 'overview'
                return redirect(url_for(
                    'pages.analyze_file',
                    filename=storage_name,
                    display=orig_name,
                    forecast_horizon=fh,
                    forecast_pct=fpct,
                    contamination=cont,
                    view=start_view
                ))
            except Exception as e:
                app.logger.exception("Upload failed")
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as cleanup_err:
                    app.logger.debug("Temp cleanup failed after upload error for %s: %s", temp_path, cleanup_err)
                flash(f"Upload failed: {e}")
                return redirect(request.url)
    return render_template('index.html')

upload_file = handle_upload_file

__all__ = ["handle_upload_file", "upload_file"]
