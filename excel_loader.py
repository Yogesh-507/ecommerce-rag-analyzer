import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import streamlit as st
from pathlib import Path
import io

class ExcelLoader:
    """
    Handles loading and processing of Excel (.xlsx) and CSV (.csv) files for RAG applications.
    Converts tabular data into structured documents that prevent LLM hallucinations.
    """
    
    def __init__(self):
        self.supported_formats = ['.xlsx', '.csv', '.xls']
    
    def load_file(self, uploaded_file) -> Tuple[List[str], Dict[str, Any]]:
        """
        Main entry point for loading Excel/CSV files.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple of (documents_list, file_summary)
        """
        try:
            # Determine file type and read accordingly
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Read file into pandas DataFrame
            df = self._read_file(uploaded_file, file_extension)
            
            # Validate the DataFrame
            self._validate_dataframe(df)
            
            # Generate file summary
            file_summary = self._generate_summary(df, uploaded_file.name)
            
            # Convert DataFrame to documents
            documents = self._dataframe_to_documents(df)
            
            return documents, file_summary
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            raise e
    
    def _read_file(self, uploaded_file, file_extension: str) -> pd.DataFrame:
        """Read file based on its extension."""
        try:
            if file_extension in ['.xlsx', '.xls']:
                # Read Excel file
                df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == '.xlsx' else 'xlrd')
            elif file_extension == '.csv':
                # Read CSV file with various encodings and separators
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Could not read file. Please ensure it's a valid {file_extension} file. Error: {str(e)}")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate the loaded DataFrame."""
        if df.empty:
            raise ValueError("The uploaded file is empty or contains no readable data.")
        
        if len(df.columns) == 0:
            raise ValueError("The uploaded file has no columns.")
        
        # Check for reasonable size limits (adjust based on your needs)
        max_rows = 50000  # Increased limit for production use
        if len(df) > max_rows:
            raise ValueError(f"File too large. Maximum {max_rows:,} rows allowed. Your file has {len(df):,} rows.")
    
    def _generate_summary(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Generate a comprehensive summary of the dataset."""
        summary = {
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'column_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'has_missing_values': df.isnull().any().any(),
            'missing_value_counts': df.isnull().sum().to_dict(),
        }
        
        # Add data type analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        summary.update({
            'numeric_columns': numeric_columns,
            'text_columns': text_columns,
            'date_columns': date_columns,
        })
        
        # Add basic statistics for numeric columns
        if numeric_columns:
            summary['numeric_stats'] = df[numeric_columns].describe().to_dict()
        
        return summary
    
    def _dataframe_to_documents(self, df: pd.DataFrame) -> List[str]:
        """
        Convert DataFrame to list of structured document strings.
        Each row becomes a document with full context.
        """
        documents = []
        
        # Clean column names (remove special characters, spaces)
        clean_columns = [str(col).strip().replace('\n', ' ').replace('\r', '') for col in df.columns]
        
        # Process each row
        for idx, row in df.iterrows():
            # Create structured document string
            row_parts = []
            
            for col_idx, (original_col, clean_col) in enumerate(zip(df.columns, clean_columns)):
                value = row.iloc[col_idx]
                
                # Handle different data types
                if pd.isna(value):
                    formatted_value = "NULL"
                elif isinstance(value, (int, float)):
                    if pd.isna(value):
                        formatted_value = "NULL"
                    else:
                        formatted_value = str(value)
                elif isinstance(value, str):
                    # Clean string values
                    formatted_value = str(value).strip().replace('\n', ' ').replace('\r', '')
                    # Escape single quotes in the value
                    formatted_value = formatted_value.replace("'", "\\'")
                else:
                    formatted_value = str(value)
                
                row_parts.append(f"{clean_col}='{formatted_value}'")
            
            # Create the structured document
            document = f"Row {idx + 1}: {'; '.join(row_parts)}"
            documents.append(document)
        
        return documents
    
    def display_summary(self, summary: Dict[str, Any]) -> None:
        """Display file summary in Streamlit UI."""
        st.subheader("📊 Data Summary")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{summary['total_rows']:,}")
        with col2:
            st.metric("Total Columns", summary['total_columns'])
        with col3:
            st.metric("File Size", f"{summary['memory_usage_mb']:.2f} MB")
        
        # Column information
        st.subheader("📋 Column Information")
        
        # Create a DataFrame for column info
        column_info = []
        for col in summary['column_names']:
            col_type = str(summary['column_types'][col])
            missing_count = summary['missing_value_counts'][col]
            missing_pct = (missing_count / summary['total_rows']) * 100
            
            column_info.append({
                'Column Name': col,
                'Data Type': col_type,
                'Missing Values': f"{missing_count} ({missing_pct:.1f}%)"
            })
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)
        
        # Data quality warnings
        if summary['has_missing_values']:
            st.warning("⚠️ This dataset contains missing values. The analysis will handle them appropriately.")
        
        # Show data types breakdown
        if summary['numeric_columns'] or summary['text_columns']:
            st.subheader("🔍 Data Types Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                if summary['numeric_columns']:
                    st.write("**Numeric Columns:**")
                    for col in summary['numeric_columns']:
                        st.write(f"• {col}")
            
            with col2:
                if summary['text_columns']:
                    st.write("**Text Columns:**")
                    for col in summary['text_columns']:
                        st.write(f"• {col}")
        
        # Show sample data
        st.subheader("👀 Sample Data Preview")
        st.info("This is how your data will be processed for analysis:")
        
        # Create a small sample to show the document format
        if summary['total_rows'] > 0:
            sample_text = "Row 1: " + "; ".join([f"{col}='sample_value'" for col in summary['column_names'][:5]])
            if len(summary['column_names']) > 5:
                sample_text += "; ..."
            st.code(sample_text, language="text")


def validate_uploaded_file(uploaded_file) -> bool:
    """
    Validate uploaded file before processing.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        bool: True if valid, False otherwise
    """
    if uploaded_file is None:
        return False
    
    # Check file extension
    allowed_extensions = ['.pdf', '.xlsx', '.csv', '.xls']
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension not in allowed_extensions:
        st.error(f"Unsupported file type: {file_extension}. Please upload a PDF, Excel (.xlsx), or CSV file.")
        return False
    
    # Check file size (50MB limit)
    max_size_mb = 50
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        st.error(f"File too large: {file_size_mb:.1f}MB. Maximum size allowed: {max_size_mb}MB")
        return False
    
    return True
