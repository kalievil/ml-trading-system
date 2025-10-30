import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Activity, RefreshCw, Zap, ShieldCheck, AlertTriangle } from "lucide-react";
import { useTradingAPI, type TradingSignal } from "@/hooks/useTradingAPI";

const Signals = () => {
  const { isConnected, getMLSignal } = useTradingAPI();
  const [signal, setSignal] = useState<TradingSignal | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSignal = async () => {
    if (!isConnected) {
      setSignal(null);
      setIsLoading(false);
      return;
    }
    try {
      setIsLoading(true);
      setError(null);
      const s = await getMLSignal();
      setSignal(s);
    } catch (e: any) {
      setError(e?.message || "Failed to load signal");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSignal();
    const id = setInterval(fetchSignal, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected]);

  if (!isConnected) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Activity className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Connect to Binance API to view signals</p>
          <p className="text-sm text-muted-foreground mt-2">Go to Settings tab to configure your API keys</p>
        </div>
      </div>
    );
  }

  if (isLoading && !signal) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading signal...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Activity className="w-12 h-12 mx-auto mb-4 text-destructive" />
          <p className="text-destructive mb-4">Error loading signal</p>
          <p className="text-sm text-muted-foreground mb-4">{error}</p>
          <Button onClick={fetchSignal} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  const signalText = signal?.signal || "HOLD";
  const confidencePct = (signal?.confidence ?? 0) * 100;
  const leverage = signal?.leverage ?? 1;
  const positionSizePct = (signal?.position_size ?? 0) * 100;
  const currentPrice = signal?.current_price ?? 0;
  const stopLoss = signal?.stop_loss ?? 0;
  const takeProfit = signal?.take_profit ?? 0;
  const halted = !!signal?.trading_halted;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Current ML Signal</h2>
        <Button onClick={fetchSignal} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="p-6 bg-card border-border">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              <h3 className="font-semibold">Signal</h3>
            </div>
            <Badge variant={signalText === 'BUY' ? 'default' : signalText === 'SELL' ? 'destructive' : 'secondary'}>
              {signalText}
            </Badge>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="space-y-1">
              <p className="text-muted-foreground">Confidence</p>
              <p className="font-semibold mono-nums">{confidencePct.toFixed(1)}%</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Leverage</p>
              <p className="font-semibold mono-nums">{leverage.toFixed(2)}x</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Position Size</p>
              <p className="font-semibold mono-nums">{positionSizePct.toFixed(1)}%</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Current Price</p>
              <p className="font-semibold mono-nums">${currentPrice.toLocaleString()}</p>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-card border-border">
          <div className="flex items-center gap-2 mb-3">
            <ShieldCheck className="w-5 h-5" />
            <h3 className="font-semibold">Risk Controls</h3>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="space-y-1">
              <p className="text-muted-foreground">Stop Loss</p>
              <p className="font-semibold mono-nums">${stopLoss.toLocaleString()}</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Take Profit</p>
              <p className="font-semibold mono-nums">${takeProfit.toLocaleString()}</p>
            </div>
            <div className="space-y-1 col-span-2">
              <p className="text-muted-foreground">Reason</p>
              <p className="font-semibold text-sm leading-snug">{signal?.reason || '—'}</p>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-card border-border">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-5 h-5" />
            <h3 className="font-semibold">Algorithm</h3>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="space-y-1">
              <p className="text-muted-foreground">Name</p>
              <p className="font-semibold">{signal?.algorithm || 'ML System'}</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Models Loaded</p>
              <p className="font-semibold">{signal?.models_loaded ? 'Yes' : 'No'}</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Trading Halted</p>
              <Badge variant={halted ? 'destructive' : 'secondary'}>{halted ? 'Yes' : 'No'}</Badge>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Timestamp</p>
              <p className="font-semibold mono-nums">{signal?.timestamp || '—'}</p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Signals;


