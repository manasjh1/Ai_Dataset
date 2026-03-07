import subprocess
import tempfile
import os
import logging
from pathlib import Path
from typing import Optional
import shutil

logger = logging.getLogger(__name__)

class DocumentConverter:
    """
    A scalable document converter using LibreOffice headless mode
    """
    
    def __init__(self, libreoffice_path: Optional[str] = None):
        """
        Initialize the DocumentConverter
        
        Args:
            libreoffice_path: Custom path to LibreOffice executable. 
                            If None, will try to find it automatically.
        """
        self.libreoffice_path = libreoffice_path or self._find_libreoffice()
        if not self.libreoffice_path:
            raise RuntimeError("LibreOffice not found. Please install LibreOffice or provide the path.")
    
    def _find_libreoffice(self) -> Optional[str]:
        """
        Try to find LibreOffice executable in common locations
        """
        common_paths = [
            # Windows paths
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            # Linux paths
            "/usr/bin/libreoffice",
            "/usr/bin/soffice",
            "/snap/bin/libreoffice",
            # macOS paths
            "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        ]
        
        # First try 'soffice' command (should work if LibreOffice is in PATH)
        try:
            result = subprocess.run(['soffice', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return 'soffice'
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try common installation paths
        for path in common_paths:
            if os.path.isfile(path):
                return path
        
        return None
    
    def convert_to_pdf(self, input_file_path: str, output_dir: Optional[str] = None) -> str:
        """
        Convert DOC/DOCX file to PDF using LibreOffice
        
        Args:
            input_file_path: Path to the input DOC/DOCX file
            output_dir: Directory where PDF will be saved. If None, uses temp directory.
            
        Returns:
            str: Path to the generated PDF file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If conversion fails
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file not found: {input_file_path}")
        
        # Use provided output directory or create a temporary one
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            cleanup_output_dir = True
        else:
            cleanup_output_dir = False
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Build LibreOffice command
            cmd = [
                self.libreoffice_path,
                '--headless',
                '--convert-to', 'pdf',
                '--outdir', output_dir,
                input_file_path
            ]
            
            logger.info(f"Converting {input_file_path} to PDF using command: {' '.join(cmd)}")
            
            # Execute conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode != 0:
                error_msg = f"LibreOffice conversion failed: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Find the generated PDF file
            input_name = Path(input_file_path).stem
            pdf_path = os.path.join(output_dir, f"{input_name}.pdf")
            
            if not os.path.exists(pdf_path):
                # Sometimes LibreOffice might generate with different name
                pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
                if pdf_files:
                    pdf_path = os.path.join(output_dir, pdf_files[0])
                else:
                    raise RuntimeError("PDF file was not generated")
            
            logger.info(f"Successfully converted to PDF: {pdf_path}")
            return pdf_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("LibreOffice conversion timed out")
        except Exception as e:
            if cleanup_output_dir and os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
            raise RuntimeError(f"Conversion failed: {str(e)}")
    
    def is_supported_format(self, filename: str) -> bool:
        """
        Check if the file format is supported for conversion
        """
        supported_extensions = {'.doc', '.docx', '.odt', '.rtf', '.txt'}
        file_ext = Path(filename).suffix.lower()
        return file_ext in supported_extensions
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        """
        Get the file type based on extension
        """
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pdf':
            return 'pdf'
        elif file_ext in ['.doc', '.docx']:
            return 'doc'
        else:
            return 'other'

# Global converter instance
_converter = None

def get_converter() -> DocumentConverter:
    """
    Get a global DocumentConverter instance
    """
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter

def convert_doc_to_pdf(input_file_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convenience function to convert DOC/DOCX to PDF
    """
    converter = get_converter()
    return converter.convert_to_pdf(input_file_path, output_dir)