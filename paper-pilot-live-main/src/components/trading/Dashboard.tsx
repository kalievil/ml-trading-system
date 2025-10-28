import { Card } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Activity, DollarSign, RefreshCw } from "lucide-react";
import MetricCard from "./MetricCard";
import PerformanceChart from "./PerformanceChart";
import RecentTrades from "./RecentTrades";
import { useRealTradingData } from "@/hooks/useRealTradingData";
import { Button } from "@/components/ui/button";

const Dashboard = () => {
  const { accountInfo, positions, tradeHistory, tradingPerformance, isLoading, error, refreshData, isConnected } = useRealTradingData();

  if (!isConnected) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Activity className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Connect to Binance API to view trading data</p>
          <p className="text-sm text-muted-foreground mt-2">Go to Settings tab to configure your API keys</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading trading data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Activity className="w-12 h-12 mx-auto mb-4 text-destructive" />
          <p className="text-destructive mb-4">Error loading trading data</p>
          <p className="text-sm text-muted-foreground mb-4">{error}</p>
          <Button onClick={refreshData} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Calculate metrics from real data
  const totalTrades = tradingPerformance?.total_trades || 0;
  const winRate = tradingPerformance?.win_rate_percent || 0;
  const totalBalance = tradingPerformance?.current_portfolio_value || 0;
  const btcBalance = tradingPerformance?.btc_balance || 0;
  const usdtBalance = tradingPerformance?.usdt_balance || 0;
  const realizedPnL = tradingPerformance?.realized_pnl || 0;
  const unrealizedPnL = tradingPerformance?.unrealized_pnl || 0;
  const totalPnL = tradingPerformance?.total_pnl || 0;
  const totalReturnPercent = tradingPerformance?.total_return_percent || 0;
  const dailyReturnPercent = tradingPerformance?.daily_return_percent || 0;
  const avgWin = tradingPerformance?.average_win || 0;
  const avgLoss = tradingPerformance?.average_loss || 0;

  return (
    <div className="space-y-6">
      {/* Header with refresh button */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Trading Dashboard</h2>
        <Button onClick={refreshData} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Performance Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total P&L"
          value={`${totalPnL >= 0 ? '+' : ''}$${totalPnL.toFixed(2)}`}
          icon={totalPnL >= 0 ? TrendingUp : TrendingDown}
          trend={totalPnL >= 0 ? "up" : "down"}
          trendValue={`${totalReturnPercent >= 0 ? '+' : ''}${totalReturnPercent.toFixed(1)}%`}
          className={totalPnL >= 0 ? "profit-text" : "loss-text"}
        />
        
        <MetricCard
          title="Unrealized P&L"
          value={`${unrealizedPnL >= 0 ? '+' : ''}$${unrealizedPnL.toFixed(2)}`}
          icon={unrealizedPnL >= 0 ? TrendingUp : TrendingDown}
          trend={unrealizedPnL >= 0 ? "up" : "down"}
          trendValue={`${dailyReturnPercent >= 0 ? '+' : ''}${dailyReturnPercent.toFixed(1)}% daily`}
          className={unrealizedPnL >= 0 ? "profit-text" : "loss-text"}
        />
        
        <MetricCard
          title="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          icon={Activity}
          trend="up"
          trendValue={`${tradingPerformance?.completed_trades || 0}/${tradingPerformance?.total_trades || 0} trades`}
        />
        
        <MetricCard
          title="Avg Win/Loss"
          value={`$${avgWin.toFixed(0)} / $${Math.abs(avgLoss).toFixed(0)}`}
          icon={DollarSign}
          trend="up"
          trendValue={`${tradingPerformance?.profitable_trades || 0}W / ${tradingPerformance?.losing_trades || 0}L`}
        />
      </div>

      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Portfolio Value"
          value={`$${totalBalance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`}
          icon={DollarSign}
          trend="up"
          trendValue={`${accountInfo?.account_type || 'SPOT'}`}
        />
        
        <MetricCard
          title="BTC Balance"
          value={`${btcBalance.toFixed(6)} BTC`}
          icon={TrendingUp}
          trend="up"
          trendValue={`$${(btcBalance * (tradingPerformance?.current_price || 0)).toFixed(2)}`}
        />
        
        <MetricCard
          title="USDT Balance"
          value={`${usdtBalance.toFixed(2)} USDT`}
          icon={DollarSign}
          trend="up"
          trendValue="Available"
        />
        
        <MetricCard
          title="Total Return"
          value={`${totalReturnPercent >= 0 ? '+' : ''}${totalReturnPercent.toFixed(1)}%`}
          icon={totalReturnPercent >= 0 ? TrendingUp : TrendingDown}
          trend={totalReturnPercent >= 0 ? "up" : "down"}
          trendValue="All Time"
          className={totalReturnPercent >= 0 ? "profit-text" : "loss-text"}
        />
      </div>

      {/* Charts and Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 p-6 bg-card border-border">
          <h3 className="text-lg font-semibold mb-4">Performance</h3>
          <PerformanceChart />
        </Card>

        <Card className="p-6 bg-card border-border">
          <h3 className="text-lg font-semibold mb-4">Recent Trades</h3>
          <RecentTrades trades={tradeHistory.slice(0, 5)} />
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
