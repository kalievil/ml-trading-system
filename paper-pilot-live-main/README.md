# Trading Dashboard - Standalone Web App

A modern, responsive trading dashboard built with React, TypeScript, and Tailwind CSS. This is a **standalone web application** that can be easily deployed to any static hosting service like Render, Vercel, or Netlify.

## Features

- 📊 **Clean Trading Dashboard** - Start with empty state, build your trading history
- 💰 **Portfolio Management** - Track balance, P&L, win rate, and active positions
- 📈 **Performance Charts** - Visualize trading performance over time (when data exists)
- ⚙️ **API Settings** - Configure Binance Testnet credentials (demo mode)
- 📱 **Responsive Design** - Works perfectly on desktop, tablet, and mobile
- 🔄 **Real-time Updates** - Live position updates when active trades exist

## Tech Stack

- **Frontend**: React 18 + TypeScript
- **Styling**: Tailwind CSS + shadcn/ui components
- **Charts**: Recharts
- **State Management**: React Hooks + Local Storage
- **Build Tool**: Vite

## Local Development

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Open browser**: Navigate to `http://localhost:5173`

## Deployment on Render

This app is ready for deployment on Render as a **Static Site**:

1. **Connect your repository** to Render
2. **Choose "Static Site"** as the service type
3. **Configure build settings**:
   - **Build Command**: `npm run build`
   - **Publish Directory**: `dist`
   - **Node Version**: `18` (or latest)

4. **Deploy** - Render will automatically build and deploy your app

## Features Overview

### Dashboard
- Real-time account balance and P&L tracking
- Win rate and trade statistics
- Active positions monitoring
- Performance charts and metrics

### Clean State Management
- Starts with empty trading history
- $10,000 starting balance
- Ready for real trading data integration
- Local storage persistence for user data

### API Settings
- Binance Testnet API credential management
- Local storage for credential persistence
- Connection testing (demo mode)
- Secure credential masking

### Responsive Design
- Mobile-first approach
- Tablet and desktop optimized
- Modern UI with dark/light theme support
- Smooth animations and transitions

## Data Persistence

The app uses **localStorage** to persist:
- API credentials
- Trading data and positions
- User preferences

**Note**: Data is stored locally in the browser and will persist between sessions.

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## License

MIT License - Feel free to use this project for your own trading dashboard needs.

---

**Ready for deployment!** 🚀 This standalone web app can be deployed to any static hosting service without any backend dependencies.