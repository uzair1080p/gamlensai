"""
Utility functions for uploading CSV files to n8n webhook
"""

import requests
import os
from typing import Optional

def upload_csv_to_n8n(file_path: str, filename: Optional[str] = None) -> dict:
    """
    Upload a CSV file to the n8n webhook for processing
    
    Args:
        file_path: Path to the CSV file to upload
        filename: Optional custom filename (defaults to file basename)
    
    Returns:
        dict: Response from the webhook
        
    Raises:
        requests.RequestException: If the upload fails
        ValueError: If the file doesn't exist or webhook URL is not configured
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    webhook_url = os.getenv("N8N_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("N8N_WEBHOOK_URL not found in environment variables")
    
    if not filename:
        filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            data = {'filename': filename}
            
            response = requests.post(webhook_url, files=files, data=data, timeout=30)
            response.raise_for_status()
            
            return {
                'success': True,
                'status_code': response.status_code,
                'response': response.text,
                'filename': filename
            }
    
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'filename': filename
        }

def test_n8n_connection() -> dict:
    """
    Test connection to the n8n webhook
    
    Returns:
        dict: Connection test result
    """
    webhook_url = os.getenv("N8N_WEBHOOK_URL")
    if not webhook_url:
        return {
            'success': False,
            'error': 'N8N_WEBHOOK_URL not found in environment variables'
        }
    
    try:
        # Send a simple GET request to test connectivity
        response = requests.get(webhook_url, timeout=10)
        return {
            'success': True,
            'status_code': response.status_code,
            'url': webhook_url
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'url': webhook_url
        }
