import os
import requests
from dotenv import load_dotenv

load_dotenv()

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

def upload_csv_to_n8n(file_path: str, filename: str):
    if not N8N_WEBHOOK_URL:
        return {"success": False, "error": "N8N_WEBHOOK_URL environment variable not set."}

    try:
        print(f"[WEBHOOK DEBUG] Uploading file: {filename} to {N8N_WEBHOOK_URL}")
        
        with open(file_path, 'rb') as f:
            # Prepare files for multipart/form-data upload
            files = {
                'file': (filename, f, 'text/csv')
            }
            
            # Prepare data fields (filename as separate field)
            # Try multiple field names that n8n workflows commonly expect
            data = {
                'filename': filename,
                'source_file': filename,  # Alternative field name
                'file_name': filename,    # Another common variation
                'name': filename          # Simple field name
            }
            
            print(f"[WEBHOOK DEBUG] Sending files: {list(files.keys())}")
            print(f"[WEBHOOK DEBUG] Sending data: {data}")
            
            response = requests.post(N8N_WEBHOOK_URL, files=files, data=data)
        
        print(f"[WEBHOOK DEBUG] Response status: {response.status_code}")
        print(f"[WEBHOOK DEBUG] Response headers: {dict(response.headers)}")
        print(f"[WEBHOOK DEBUG] Response content: {response.text[:500]}...")
        
        response.raise_for_status()
        
        # Handle different response types
        try:
            response_json = response.json()
            return {"success": True, "response": response_json}
        except ValueError:
            # If response is not JSON, return the text content
            return {"success": True, "response": response.text}
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Network or HTTP error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"An unexpected error occurred: {e}"}

def test_n8n_connection():
    if not N8N_WEBHOOK_URL:
        return {"success": False, "error": "N8N_WEBHOOK_URL environment variable not set."}
    
    try:
        response = requests.get(N8N_WEBHOOK_URL)
        response.raise_for_status()
        return {"success": True, "message": "Successfully connected to n8n webhook."}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Failed to connect to n8n webhook: {e}"}

def test_filename_detection():
    """Test what fields the n8n webhook can detect"""
    if not N8N_WEBHOOK_URL:
        return {"success": False, "error": "N8N_WEBHOOK_URL environment variable not set."}
    
    try:
        # Create a small test CSV content
        test_csv_content = "test,header\n1,2\n3,4"
        test_filename = "test_filename_detection.csv"
        
        # Send test request with filename in multiple formats
        files = {
            'file': (test_filename, test_csv_content, 'text/csv')
        }
        
        data = {
            'filename': test_filename,
            'source_file': test_filename,
            'file_name': test_filename,
            'name': test_filename,
            'test': 'true'  # Flag to indicate this is a test
        }
        
        print(f"[TEST] Sending test filename detection request...")
        print(f"[TEST] Files: {list(files.keys())}")
        print(f"[TEST] Data: {data}")
        
        response = requests.post(N8N_WEBHOOK_URL, files=files, data=data)
        
        print(f"[TEST] Response status: {response.status_code}")
        print(f"[TEST] Response: {response.text}")
        
        return {"success": True, "response": response.text}
        
    except Exception as e:
        return {"success": False, "error": f"Test failed: {e}"}
