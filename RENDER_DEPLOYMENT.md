# 🚀 Render Deployment Guide - ML Trading System

## **Environment Variables Setup (24/7 Persistent Storage)**

Your ML Trading System now uses **environment variables** for persistent API key storage, making it perfect for 24/7 Render deployment!

### **✅ What's Changed:**

1. **Persistent Storage**: API keys are saved to environment variables
2. **Auto-Loading**: Server automatically loads credentials on startup
3. **24/7 Ready**: Survives server restarts and deployments
4. **Secure**: Environment variables are encrypted in Render

---

## **🔧 Render Deployment Steps**

### **Step 1: Prepare Your Repository**

Make sure your repository has these files:
```
├── trading_api.py          # Python API server
├── requirements.txt        # Python dependencies
├── user_data/
│   └── models/            # Your trained ML models
│       ├── scaler.pkl
│       ├── xgboost_performance.pkl
│       ├── xgboost_aggressive.pkl
│       ├── xgboost_balanced.pkl
│       └── ensemble_weights.json
└── paper-pilot-live-main/  # React webapp
    ├── package.json
    ├── src/
    └── ...
```

### **Step 2: Deploy Backend API to Render**

1. **Go to [Render Dashboard](https://dashboard.render.com/)**
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**
4. **Configure the service:**

   **Basic Settings:**
   - **Name**: `ml-trading-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python trading_api.py`
   - **Instance Type**: `Starter` (free) or `Standard` (recommended)

   **Environment Variables:**
   ```
   BINANCE_API_KEY=your_testnet_api_key_here
   BINANCE_API_SECRET=your_testnet_secret_key_here
   ```

5. **Click "Create Web Service"**

### **Step 3: Deploy Frontend to Render**

1. **Click "New +" → "Static Site"**
2. **Connect your GitHub repository**
3. **Configure:**

   **Basic Settings:**
   - **Name**: `ml-trading-webapp`
   - **Build Command**: `cd paper-pilot-live-main && npm install && npm run build`
   - **Publish Directory**: `paper-pilot-live-main/dist`
   - **Environment**: `Node`

4. **Click "Create Static Site"**

### **Step 4: Update Frontend API URL**

After deployment, update the API URL in your frontend:

```typescript
// In paper-pilot-live-main/src/hooks/useTradingAPI.ts
const API_BASE_URL = 'https://your-api-service-name.onrender.com'; // Your Render API URL
```

### **Step 5: Configure Binance Testnet**

1. **Get Binance Testnet API Keys:**
   - Go to [Binance Testnet](https://testnet.binance.vision/)
   - Create account and generate API keys

2. **Set Environment Variables in Render:**
   - Go to your API service dashboard
   - Click "Environment" tab
   - Add:
     ```
     BINANCE_API_KEY=your_testnet_api_key
     BINANCE_API_SECRET=your_testnet_secret_key
     ```
   - Click "Save Changes"

3. **Redeploy** (automatic after saving environment variables)

---

## **🎯 How It Works Now**

### **Automatic Credential Loading:**
```python
# Server automatically loads credentials on startup
if has_credentials_in_env():
    api_key, api_secret = load_credentials_from_env()
    binance_client = Client(api_key, api_secret, testnet=True)
    logger.info("✅ Binance API auto-configured from environment variables")
```

### **Persistent Storage:**
```python
# When you configure API keys via webapp
save_credentials_to_env(config.api_key, config.api_secret)
# Credentials are saved to environment variables
```

### **Health Check:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "binance_configured": true,
  "credentials_from_env": true
}
```

---

## **🔒 Security Benefits**

✅ **Encrypted Storage**: Environment variables are encrypted in Render
✅ **No File Storage**: Credentials never stored in files
✅ **Automatic Loading**: No manual configuration needed
✅ **24/7 Persistence**: Survives all server restarts
✅ **Testnet Only**: Safe for testing (no real money)

---

## **📊 Monitoring Your Deployment**

### **API Health Check:**
```bash
curl https://your-api-service-name.onrender.com/api/health
```

### **Expected Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "binance_configured": true,
  "credentials_from_env": true
}
```

### **Trading Signal Test:**
```bash
curl https://your-api-service-name.onrender.com/api/trading-signal
```

---

## **🚨 Important Notes**

1. **Free Tier Limitations**: Render free tier sleeps after 15 minutes of inactivity
2. **Upgrade Recommended**: For 24/7 trading, use Render's paid plans
3. **Testnet Only**: Always use Binance Testnet API keys
4. **Model Files**: Make sure your trained models are in the repository
5. **Environment Variables**: Set them in Render dashboard, not in code

---

## **🎉 You're All Set!**

Your ML Trading System is now:
- ✅ **24/7 Ready** with persistent credential storage
- ✅ **Auto-Configuring** on server restarts
- ✅ **Secure** with encrypted environment variables
- ✅ **Deployed** on Render cloud infrastructure

**Your algorithmic trading system will now run continuously and automatically reconnect to Binance Testnet after any server restart! 🚀**
