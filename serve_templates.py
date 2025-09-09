#!/usr/bin/env python3
"""
Simple static file server for serving GameLens AI templates
Run this alongside the Streamlit app to serve template files directly
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Configuration
PORT = 8505  # Different port from Streamlit (8504)
TEMPLATE_FILES = [
    "Data_Template_GameLens_AI.csv",
    "DATA_TEMPLATE_GUIDE.md"
]

class TemplateHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve template files with proper headers"""
    
    def end_headers(self):
        # Add CORS headers to allow cross-origin requests
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        """Handle GET requests for template files"""
        # Extract filename from path
        filename = self.path.lstrip('/')
        
        # Check if it's one of our template files
        if filename in TEMPLATE_FILES:
            if os.path.exists(filename):
                # Set appropriate content type
                if filename.endswith('.csv'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/csv')
                    self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                    self.end_headers()
                    
                    # Read and send file content
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.wfile.write(content.encode('utf-8'))
                    return
                elif filename.endswith('.md'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/markdown')
                    self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
                    self.end_headers()
                    
                    # Read and send file content
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.wfile.write(content.encode('utf-8'))
                    return
        
        # For other files, use default behavior
        super().do_GET()

def main():
    """Start the template server"""
    # Check if template files exist
    missing_files = [f for f in TEMPLATE_FILES if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing template files: {missing_files}")
        print("Please ensure the template files exist in the current directory.")
        sys.exit(1)
    
    # Start server
    try:
        with socketserver.TCPServer(("", PORT), TemplateHandler) as httpd:
            print(f"🚀 Template server started on port {PORT}")
            print(f"📁 Serving template files:")
            for file in TEMPLATE_FILES:
                print(f"   - http://localhost:{PORT}/{file}")
            print(f"\n💡 Use these URLs in your Streamlit app or access directly")
            print(f"🛑 Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n🛑 Template server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Please stop other services or change the port.")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
