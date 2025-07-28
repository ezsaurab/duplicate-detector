import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import pdfplumber
from flask import Flask, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('duplicate_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

# Configuration
class Config:
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'tsv', 'pdf', 'json', 'parquet'}
    CLEANUP_INTERVAL_HOURS = 1  # Clean up uploaded files after 1 hour
    MAX_ROWS_DISPLAY = 1000  # Maximum rows to display in results
    MAX_COLUMNS_DISPLAY = 50   # Maximum columns to display

app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_file_hash(file_path: str) -> str:
    """Generate MD5 hash of file for caching purposes."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def cleanup_old_files():
    """Remove files older than CLEANUP_INTERVAL_HOURS."""
    try:
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        cutoff_time = datetime.now() - timedelta(hours=app.config.get('CLEANUP_INTERVAL_HOURS', 1))
        
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def read_table(file_path: str, filename: str) -> Optional[pd.DataFrame]:
    """
    Read various file formats into a pandas DataFrame.
    
    Args:
        file_path: Path to the uploaded file
        filename: Original filename with extension
        
    Returns:
        DataFrame or None if file cannot be read
    """
    try:
        ext = filename.rsplit('.', 1)[-1].lower()
        df = None
        
        if ext == 'csv':
            # Try different encodings and separators
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            separators = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                        if df.shape[1] > 1:  # Successfully parsed with multiple columns
                            break
                    except:
                        continue
                if df is not None and df.shape[1] > 1:
                    break
                    
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(file_path, engine='openpyxl' if ext == 'xlsx' else None)
            
        elif ext == 'tsv':
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            
        elif ext == 'json':
            df = pd.read_json(file_path)
            
        elif ext == 'parquet':
            df = pd.read_parquet(file_path)
            
        elif ext == 'pdf':
            tables = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        table = page.extract_table()
                        if table and len(table) > 1:  # Ensure we have headers and data
                            # Clean the table data
                            cleaned_table = []
                            for row in table:
                                cleaned_row = [cell.strip() if cell else '' for cell in row]
                                cleaned_table.append(cleaned_row)
                            
                            if cleaned_table:
                                table_df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                                tables.append(table_df)
                    except Exception as e:
                        logger.warning(f"Error extracting table from page {page_num}: {str(e)}")
                        continue
                        
            if tables:
                df = pd.concat(tables, ignore_index=True)
            else:
                logger.warning("No tables found in PDF")
                return None
        
        if df is not None and not df.empty:
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            logger.info(f"Successfully loaded file: {filename}, Shape: {df.shape}")
            return df
        else:
            logger.warning(f"Empty or invalid DataFrame from file: {filename}")
            return None
            
    except Exception as e:
        logger.error(f"Error reading file {filename}: {str(e)}")
        return None

def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive duplicate analysis of the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing duplicate analysis results
    """
    try:
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_rows': {},
            'duplicate_columns': {},
            'summary': {}
        }
        
        # Analyze duplicate rows
        duplicate_row_mask = df.duplicated(keep=False)
        duplicate_rows = df[duplicate_row_mask]
        
        if not duplicate_rows.empty:
            # Limit displayed rows for performance
            display_limit = app.config.get('MAX_ROWS_DISPLAY', 1000)
            limited_rows = duplicate_rows.head(display_limit)
            
            analysis['duplicate_rows'] = {
                'count': len(duplicate_rows),
                'percentage': (len(duplicate_rows) / len(df)) * 100,
                'data': limited_rows.to_html(
                    classes='data-table table-striped', 
                    index=True,
                    escape=False,
                    max_rows=display_limit
                ),
                'truncated': len(duplicate_rows) > display_limit
            }
        else:
            analysis['duplicate_rows'] = {
                'count': 0,
                'percentage': 0,
                'data': '<div class="alert alert-success"><i class="fas fa-check-circle"></i> No duplicate rows found!</div>',
                'truncated': False
            }
        
        # Analyze duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated(keep=False)].tolist()
        
        if duplicate_cols:
            # Group duplicate column names
            duplicate_groups = {}
            for col in duplicate_cols:
                if col not in duplicate_groups:
                    duplicate_groups[col] = df.columns.tolist().count(col)
            
            analysis['duplicate_columns'] = {
                'count': len(duplicate_cols),
                'unique_names': len(duplicate_groups),
                'groups': duplicate_groups,
                'data': f'<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> Found {len(duplicate_groups)} duplicate column name(s): {", ".join(duplicate_groups.keys())}</div>'
            }
        else:
            analysis['duplicate_columns'] = {
                'count': 0,
                'unique_names': 0,
                'groups': {},
                'data': '<div class="alert alert-success"><i class="fas fa-check-circle"></i> No duplicate columns found!</div>'
            }
        
        # Generate summary statistics
        analysis['summary'] = {
            'file_health_score': calculate_health_score(analysis),
            'recommendations': generate_recommendations(analysis),
            'processing_time': datetime.now().isoformat()
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing duplicates: {str(e)}")
        raise

def calculate_health_score(analysis: Dict[str, Any]) -> int:
    """Calculate a data quality health score (0-100)."""
    score = 100
    
    # Deduct points for duplicate rows
    duplicate_row_percentage = analysis['duplicate_rows'].get('percentage', 0)
    score -= min(duplicate_row_percentage * 2, 50)  # Max 50 points deduction
    
    # Deduct points for duplicate columns
    duplicate_col_count = analysis['duplicate_columns'].get('count', 0)
    score -= min(duplicate_col_count * 10, 30)  # Max 30 points deduction
    
    return max(int(score), 0)

def generate_recommendations(analysis: Dict[str, Any]) -> list:
    """Generate data quality recommendations."""
    recommendations = []
    
    if analysis['duplicate_rows']['count'] > 0:
        recommendations.append({
            'type': 'warning',
            'message': f"Remove {analysis['duplicate_rows']['count']} duplicate rows to improve data quality"
        })
    
    if analysis['duplicate_columns']['count'] > 0:
        recommendations.append({
            'type': 'error', 
            'message': "Rename duplicate column headers to ensure unique identifiers"
        })
    
    if not recommendations:
        recommendations.append({
            'type': 'success',
            'message': "Great! Your data appears to be clean with no duplicates found"
        })
    
    return recommendations

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file size limit exceeded."""
    return render_template('index.html', 
                         error=f"File too large. Maximum size allowed: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB"), 413

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('index.html', 
                         error="An internal error occurred. Please try again."), 500

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route handling file upload and duplicate detection."""
    
    # Clean up old files periodically
    cleanup_old_files()
    
    if request.method == "GET":
        return render_template("index.html")
    
    # Handle POST request
    try:
        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        
        if not allowed_file(file.filename):
            allowed_ext = ', '.join(app.config['ALLOWED_EXTENSIONS'])
            return render_template('index.html', 
                                 error=f"File type not supported. Allowed types: {allowed_ext}")
        
        # Secure the filename and create temporary file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(filepath)
        logger.info(f"File uploaded: {filename}, Size: {os.path.getsize(filepath)} bytes")
        
        # Read and analyze the file
        df = read_table(filepath, filename)
        
        if df is None:
            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            return render_template('index.html', 
                                 error="Unable to read the file. Please check the file format and try again.")
        
        # Perform duplicate analysis
        analysis = analyze_duplicates(df)
        
        # Clean up the uploaded file after processing
        try:
            os.remove(filepath)
        except Exception as e:
            logger.warning(f"Could not remove temporary file {filepath}: {str(e)}")
        
        # Prepare template variables for backward compatibility
        template_vars = {
            'row_count': analysis['duplicate_rows']['count'],
            'duplicate_rows': analysis['duplicate_rows']['data'],
            'col_count': analysis['duplicate_columns']['count'], 
            'duplicate_columns': analysis['duplicate_columns']['data'],
            'analysis': analysis,  # Full analysis for enhanced features
            'filename': filename
        }
        
        return render_template('index.html', **template_vars)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return render_template('index.html', 
                             error="An error occurred while processing your file. Please try again.")

@app.route("/api/health")
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024*1024)
    })

@app.route("/api/supported-formats")
def supported_formats():
    """API endpoint to get supported file formats."""
    return jsonify({
        'supported_formats': list(app.config['ALLOWED_EXTENSIONS']),
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024*1024)
    })

if __name__ == "__main__":
    # Ensure proper startup
    logger.info("Starting Duplicate Detector application...")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    logger.info(f"Supported formats: {app.config['ALLOWED_EXTENSIONS']}")
    
    app.run(debug=True, host='0.0.0.0', port=5050)