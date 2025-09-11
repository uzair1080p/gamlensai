# 🚀 Quick Server Deployment

## **One-Command Deployment**

```bash
# Clone and deploy in one command
git clone https://github.com/uzair1080p/gamlensai.git && cd gamlensai && chmod +x deploy.sh && ./deploy.sh development
```

## **Step-by-Step Deployment**

### **1. Connect to your server**
```bash
ssh your-username@your-server-ip
```

### **2. Clone the repository**
```bash
git clone https://github.com/uzair1080p/gamlensai.git
cd gamlensai
```

### **3. Run the deployment script**
```bash
# Make script executable
chmod +x deploy.sh

# For development (simple setup)
./deploy.sh development

# For production (with systemd service)
./deploy.sh production
```

### **4. Configure environment (when prompted)**
Edit the `.env` file with your settings:
```bash
nano .env
```

**For SQLite (simple):**
```env
DATABASE_URL=sqlite:///./gamelens.db
OPENAI_API_KEY=your_openai_api_key_here
```

**For PostgreSQL (production):**
```env
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/gamelens
OPENAI_API_KEY=your_openai_api_key_here
```

### **5. Access your application**
- **Development**: http://your-server-ip:8501
- **Production**: http://your-server-ip:8501 (or configure domain)

## **Manual Deployment (if script fails)**

```bash
# 1. Install dependencies
sudo apt update
sudo apt install python3 python3-venv python3-pip -y

# 2. Setup application
python3 -m venv gamlens_env
source gamlens_env/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp env.example .env
nano .env  # Edit with your settings

# 4. Initialize database
make db-upgrade

# 5. Setup demo data (optional)
python demo_setup.py

# 6. Run application
streamlit run pages/2_🚀_Train_Predict_Validate_FAQ.py --server.port 8501 --server.address 0.0.0.0
```

## **Troubleshooting**

### **Port already in use**
```bash
sudo lsof -i :8501
sudo kill -9 <PID>
```

### **Permission issues**
```bash
sudo chown -R $USER:$USER .
chmod +x deploy.sh
```

### **Database issues**
```bash
# Check if database file exists
ls -la gamelens.db

# Reset database if needed
rm gamelens.db
make db-upgrade
```

### **Check application status**
```bash
# If using systemd service
sudo systemctl status gamelens-ai
sudo journalctl -u gamelens-ai -f

# If running manually
ps aux | grep streamlit
```

## **Production Setup with Nginx**

```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx config
sudo nano /etc/nginx/sites-available/gamelens-ai
```

**Nginx configuration:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/gamelens-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## **SSL Certificate (Optional)**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

## **Firewall Configuration**

```bash
# Allow necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw allow 8501  # Streamlit (if not using Nginx)
sudo ufw enable
```

Your GameLens AI v2.0 system should now be running! 🎉
