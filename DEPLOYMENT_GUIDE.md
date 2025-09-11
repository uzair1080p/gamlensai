# GameLens AI v2.0 - Server Deployment Guide

## 🚀 **Quick Deployment Steps**

### **1. Server Setup**
```bash
# Connect to your server
ssh your-username@your-server-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3+ and pip
sudo apt install python3 python3-venv python3-pip -y

# Install PostgreSQL (recommended for production)
sudo apt install postgresql postgresql-contrib -y

# Install system dependencies
sudo apt install build-essential libpq-dev -y
```

### **2. Clone and Setup Repository**
```bash
# Clone your repository
git clone https://github.com/uzair1080p/gamlensai.git
cd gamlensai

# Create virtual environment
python3 -m venv gamlens_env
source gamlens_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Database Configuration**

#### **Option A: PostgreSQL (Recommended for Production)**
```bash
# Create database and user
sudo -u postgres psql
CREATE DATABASE gamelens;
CREATE USER gamelens_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE gamelens TO gamelens_user;
\q

# Update environment configuration
cp env.example .env
nano .env
```

**Update `.env` file:**
```env
DATABASE_URL=postgresql+psycopg2://gamelens_user:your_secure_password@localhost:5432/gamelens
OPENAI_API_KEY=your_openai_api_key_here
```

#### **Option B: SQLite (Simple Setup)**
```bash
# Just copy the example and update API key
cp env.example .env
nano .env
```

**Update `.env` file:**
```env
DATABASE_URL=sqlite:///./gamelens.db
OPENAI_API_KEY=your_openai_api_key_here
```

### **4. Initialize Database**
```bash
# Run database migrations
make db-upgrade

# Or manually:
alembic upgrade head
```

### **5. Setup Demo Data (Optional)**
```bash
# Create demo data and train initial model
python demo_setup.py
```

### **6. Run the Application**

#### **Development Mode**
```bash
# Start Streamlit app
streamlit run pages/2_🚀_Train_Predict_Validate_FAQ.py --server.port 8501
```

#### **Production Mode with Systemd Service**
```bash
# Create systemd service file
sudo nano /etc/systemd/system/gamelens-ai.service
```

**Service file content:**
```ini
[Unit]
Description=GameLens AI Streamlit App
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/gamlensai
Environment=PATH=/path/to/gamlensai/gamlens_env/bin
ExecStart=/path/to/gamlensai/gamlens_env/bin/streamlit run pages/2_🚀_Train_Predict_Validate_FAQ.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable gamelens-ai
sudo systemctl start gamelens-ai

# Check status
sudo systemctl status gamelens-ai
```

### **7. Nginx Configuration (Optional but Recommended)**

```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/gamelens-ai
```

**Nginx configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

```bash
# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/gamelens-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### **8. SSL Certificate (Optional)**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

## 🔧 **Environment Variables**

Create a `.env` file with these variables:

```env
# Database Configuration
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/gamelens

# OpenAI API Key (for GPT naming)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom model parameters
LIGHTGBM_LEARNING_RATE=0.05
LIGHTGBM_MAX_DEPTH=6
LIGHTGBM_N_ESTIMATORS=100

# Optional: Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 📁 **Directory Structure on Server**

```
/home/your-username/gamlensai/
├── glai/                          # Core application
├── pages/                         # Streamlit pages
├── alembic/                       # Database migrations
├── artifacts/                     # Model storage
├── data/                          # Data storage
├── tests/                         # Test suite
├── .env                          # Environment configuration
├── requirements.txt              # Dependencies
├── Makefile                      # Development commands
└── gamelens.db                   # SQLite database (if using SQLite)
```

## 🛠️ **Useful Commands**

```bash
# Check application status
sudo systemctl status gamelens-ai

# View logs
sudo journalctl -u gamelens-ai -f

# Restart application
sudo systemctl restart gamelens-ai

# Update application
cd /path/to/gamlensai
git pull origin main
source gamlens_env/bin/activate
pip install -r requirements.txt
make db-upgrade
sudo systemctl restart gamelens-ai

# Database management
make db-upgrade    # Run migrations
make db-reset      # Reset database (WARNING: deletes data)
make db-status     # Check migration status
```

## 🔒 **Security Considerations**

1. **Firewall Configuration:**
```bash
# Allow only necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

2. **Database Security:**
- Use strong passwords
- Limit database user permissions
- Enable SSL for database connections

3. **Environment Variables:**
- Never commit `.env` file to git
- Use strong API keys
- Rotate keys regularly

## 📊 **Monitoring and Maintenance**

### **Log Monitoring**
```bash
# Application logs
sudo journalctl -u gamelens-ai -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### **Performance Monitoring**
```bash
# Check system resources
htop
df -h
free -h

# Check application status
curl http://localhost:8501
```

### **Backup Strategy**
```bash
# Database backup (PostgreSQL)
pg_dump -h localhost -U gamelens_user gamelens > backup_$(date +%Y%m%d).sql

# Application backup
tar -czf gamelens_backup_$(date +%Y%m%d).tar.gz /path/to/gamlensai
```

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Port already in use:**
```bash
sudo lsof -i :8501
sudo kill -9 <PID>
```

2. **Database connection issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U gamelens_user -d gamelens
```

3. **Permission issues:**
```bash
# Fix file permissions
sudo chown -R your-username:your-username /path/to/gamlensai
chmod +x /path/to/gamlensai/gamlens_env/bin/*
```

4. **Memory issues:**
```bash
# Check memory usage
free -h
# Consider upgrading server or optimizing models
```

## 🎯 **Quick Start Commands**

```bash
# Complete setup in one go
git clone https://github.com/uzair1080p/gamlensai.git
cd gamlensai
python3.9 -m venv gamlens_env
source gamlens_env/bin/activate
pip install -r requirements.txt
cp env.example .env
# Edit .env with your settings
make db-upgrade
python demo_setup.py
streamlit run pages/2_🚀_Train_Predict_Validate_FAQ.py --server.port 8501 --server.address 0.0.0.0
```

## 📞 **Support**

If you encounter issues:
1. Check the logs: `sudo journalctl -u gamelens-ai -f`
2. Verify environment variables in `.env`
3. Ensure all dependencies are installed
4. Check database connectivity
5. Verify firewall and port configurations

Your GameLens AI v2.0 system should now be running on your server! 🚀
