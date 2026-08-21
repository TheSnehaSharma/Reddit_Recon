# Streamlit App Deployment Guide

## 🚀 Deploy to Streamlit Cloud

### Prerequisites
1. GitHub account with Reddit_Recon repository
2. Databricks workspace with SQL Warehouse
3. Unity Catalog with `workspace.redditrecon.posts_gold` table

### Step-by-Step Deployment

#### 1. Go to Streamlit Cloud
Visit: https://share.streamlit.io/

#### 2. Connect GitHub Repository
- Click "New app"
- Select repository: `TheSnehaSharma/Reddit_Recon`
- Branch: `main`
- Main file path: `streamlit_app/app.py`

#### 3. Configure Secrets
Click "Advanced settings" → "Secrets" and add:

```toml
DATABRICKS_SERVER_HOSTNAME = "your-workspace.cloud.databricks.com"
DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/your-warehouse-id"
DATABRICKS_TOKEN = "your-databricks-token"
```

**How to get these values:**

**a) DATABRICKS_SERVER_HOSTNAME**
- Go to your Databricks workspace
- URL format: `https://your-workspace.cloud.databricks.com`
- Copy just: `your-workspace.cloud.databricks.com`

**b) DATABRICKS_HTTP_PATH**
- Go to SQL Warehouses in Databricks
- Click your warehouse
- Go to "Connection details" tab
- Copy the "Server hostname" → use as `DATABRICKS_SERVER_HOSTNAME`
- Copy the "HTTP path" → use as `DATABRICKS_HTTP_PATH`

**c) DATABRICKS_TOKEN**
- Go to Settings → Developer → Access tokens
- Click "Generate new token"
- Copy the token (save it securely!)
- Use this as `DATABRICKS_TOKEN`

#### 4. Deploy
- Click "Deploy!"
- Wait for installation and deployment (2-3 minutes)
- Your app will be live at: `https://share.streamlit.io/your-username/reddit-recon`

---

## 🐛 Troubleshooting

### Error: "installer returned a non-zero exit code"

**Solution:**
```bash
# Update requirements.txt with flexible versions
streamlit>=1.29.0
pandas>=2.0.0
plotly>=5.18.0
databricks-sql-connector>=3.0.0
pyarrow>=10.0.0
```

### Error: "Database connection failed"

**Check:**
1. SQL Warehouse is running in Databricks
2. Secrets are correctly configured
3. Token has not expired
4. Network access is allowed

### Error: "Table not found"

**Ensure:**
1. Gold Layer pipeline has been run
2. Table exists: `workspace.redditrecon.posts_gold`
3. Token has SELECT permission on the table

---

## 🏃 Local Development

### Run Locally

```bash
# Clone repository
git clone https://github.com/TheSnehaSharma/Reddit_Recon.git
cd Reddit_Recon/streamlit_app

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your Databricks credentials

# Run app
streamlit run app.py
```

App will be available at: http://localhost:8501

### Test Without Databricks

If you want to test the UI without connecting to Databricks:

1. Comment out database connection code in `app.py`
2. Use mock data instead of real queries
3. Test UI/UX and styling

---

## 📊 Performance Tips

### Optimize Query Performance
- Use date range filters to limit data
- Enable result caching (24-hour TTL)
- Use Serverless SQL Warehouse for auto-scaling

### Reduce Load Times
- Keep `ttl=86400` in `@st.cache_data` decorators
- Limit number of displayed records
- Use lazy loading for large datasets

---

## 🔒 Security Best Practices

1. **Never commit secrets to Git**
   - `.streamlit/secrets.toml` is in `.gitignore`
   - Use Streamlit Cloud secrets management

2. **Use read-only tokens**
   - Token should have SELECT permission only
   - No CREATE/DELETE/UPDATE permissions needed

3. **Rotate tokens regularly**
   - Generate new tokens every 90 days
   - Update Streamlit secrets after rotation

---

## 📝 Environment Variables

The app requires these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABRICKS_SERVER_HOSTNAME` | Databricks workspace URL | `adb-123456789.azuredatabricks.net` |
| `DATABRICKS_HTTP_PATH` | SQL Warehouse HTTP path | `/sql/1.0/warehouses/abc123def456` |
| `DATABRICKS_TOKEN` | Personal access token | `dapi...` |

---

## 🎨 Customization

### Change Theme Colors
Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4500"      # Reddit orange
backgroundColor = "#F6F7F8"   # Light gray
secondaryBackgroundColor = "#FFFFFF"  # White
textColor = "#1A1A1B"         # Dark gray
```

### Modify Cache Duration
In `app.py`, change:

```python
@st.cache_data(ttl=86400)  # 24 hours
```

To:

```python
@st.cache_data(ttl=3600)   # 1 hour
```

---

## 📞 Support

**Issues:** https://github.com/TheSnehaSharma/Reddit_Recon/issues  
**Email:** devsnehasharma@gmail.com  
**Streamlit Docs:** https://docs.streamlit.io/

---

Built with ❤️ using Streamlit and Databricks
