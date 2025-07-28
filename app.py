import os
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'csv', 'tsv', 'xlsx', 'xls', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(filepath, ext):
    if ext == 'csv':
        return pd.read_csv(filepath)
    elif ext == 'tsv':
        return pd.read_csv(filepath, sep='\t')
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(filepath)
    elif ext == 'pdf':
        import tabula
        dfs = tabula.read_pdf(filepath, pages='all', multiple_tables=False)
        return dfs[0] if dfs else pd.DataFrame()
    else:
        return pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    duplicate_rows = None
    duplicate_columns = None
    row_count = 0
    col_count = 0

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = filename.rsplit('.', 1)[1].lower()
            try:
                df = read_file(filepath, ext)

                duplicate_rows = df[df.duplicated()]
                row_count = duplicate_rows.shape[0]

                duplicate_columns = df.T[df.T.duplicated()].T
                col_count = len(df.columns) - len(duplicate_columns.columns)

                return render_template('index.html', 
                    row_count=row_count, 
                    col_count=col_count, 
                    duplicate_rows=duplicate_rows.head(10).to_html(classes='table table-bordered', index=False),
                    duplicate_columns=duplicate_columns.iloc[:, :10].to_html(classes='table table-bordered', index=False)
                )

            except Exception as e:
                return render_template('index.html', error=f'Error processing file: {str(e)}')
        else:
            return render_template('index.html', error='Invalid file type')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
