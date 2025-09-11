#!/bin/bash

# GameLens AI v2.0 - Quick Deployment Script
# Usage: ./deploy.sh [production|development]

set -e  # Exit on any error

DEPLOYMENT_MODE=${1:-development}
APP_NAME="gamelens-ai"
APP_DIR="/opt/gamelens-ai"
SERVICE_USER="gamelens"
SERVICE_PORT="8501"

echo "🚀 GameLens AI v2.0 Deployment Script"
echo "======================================"
echo "Mode: $DEPLOYMENT_MODE"
echo ""

# Check if running as root for system setup
if [[ $EUID -eq 0 ]]; then
    echo "⚠️  Running as root. Setting up system dependencies..."
    
    # Update system
    apt update && apt upgrade -y
    
    # Install system dependencies
    apt install -y python3 python3-venv python3-pip build-essential libpq-dev
    
    # Install PostgreSQL for production
    if [ "$DEPLOYMENT_MODE" = "production" ]; then
        echo "📊 Installing PostgreSQL..."
        apt install -y postgresql postgresql-contrib
        systemctl start postgresql
        systemctl enable postgresql
    fi
    
    # Create service user
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /bin/bash -d $APP_DIR $SERVICE_USER
        echo "👤 Created service user: $SERVICE_USER"
    fi
    
    # Create application directory
    mkdir -p $APP_DIR
    chown $SERVICE_USER:$SERVICE_USER $APP_DIR
    
    echo "✅ System setup complete. Please run as non-root user to continue."
    exit 0
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "glai" ]; then
    echo "❌ Error: Please run this script from the GameLens AI project directory"
    exit 1
fi

echo "📁 Setting up application..."

# Create virtual environment
if [ ! -d "gamlens_env" ]; then
    python3 -m venv gamlens_env
    echo "✅ Created virtual environment"
fi

# Activate virtual environment
source gamlens_env/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment configuration
if [ ! -f ".env" ]; then
    cp env.example .env
    echo "📝 Created .env file. Please edit it with your configuration:"
    echo "   - DATABASE_URL (for PostgreSQL or SQLite)"
    echo "   - OPENAI_API_KEY (for GPT naming)"
    echo ""
    echo "Example for PostgreSQL:"
    echo "DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/gamelens"
    echo ""
    echo "Example for SQLite:"
    echo "DATABASE_URL=sqlite:///./gamelens.db"
    echo ""
    read -p "Press Enter after you've configured .env file..."
fi

# Initialize database
echo "🗄️  Initializing database..."
make db-upgrade

# Setup demo data (optional)
read -p "Do you want to setup demo data? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Setting up demo data..."
    python demo_setup.py
fi

# Create systemd service for production
if [ "$DEPLOYMENT_MODE" = "production" ]; then
    echo "🔧 Creating systemd service..."
    
    CURRENT_DIR=$(pwd)
    SERVICE_FILE="/etc/systemd/system/${APP_NAME}.service"
    
    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=GameLens AI Streamlit App
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$CURRENT_DIR
Environment=PATH=$CURRENT_DIR/gamlens_env/bin
ExecStart=$CURRENT_DIR/gamlens_env/bin/streamlit run pages/2_🚀_Train_Predict_Validate_FAQ.py --server.port $SERVICE_PORT --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable $APP_NAME
    sudo systemctl start $APP_NAME
    
    echo "✅ Service created and started"
    echo "📊 Check status: sudo systemctl status $APP_NAME"
    echo "📋 View logs: sudo journalctl -u $APP_NAME -f"
    
else
    echo "🚀 Starting development server..."
    echo "Access the application at: http://localhost:$SERVICE_PORT"
    echo "Press Ctrl+C to stop"
    
    streamlit run pages/2_🚀_Train_Predict_Validate_FAQ.py --server.port $SERVICE_PORT --server.address 0.0.0.0
fi

echo ""
echo "🎉 GameLens AI v2.0 deployment complete!"
echo ""
echo "📋 Useful commands:"
echo "  - Check status: sudo systemctl status $APP_NAME"
echo "  - View logs: sudo journalctl -u $APP_NAME -f"
echo "  - Restart: sudo systemctl restart $APP_NAME"
echo "  - Stop: sudo systemctl stop $APP_NAME"
echo ""
echo "🌐 Access your application at: http://your-server-ip:$SERVICE_PORT"
