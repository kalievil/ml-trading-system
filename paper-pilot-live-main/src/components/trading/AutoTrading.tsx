import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Activity, Power, RefreshCw } from "lucide-react";
import { useTradingAPI } from "@/hooks/useTradingAPI";

const AutoTrading = () => {
  const { isConnected, getAutoTradingStatus, enableAutoTrading, disableAutoTrading } = useTradingAPI();
  const [enabled, setEnabled] = useState<boolean>(false);
  const [mlReady, setMlReady] = useState<boolean>(false);
  const [binanceConnected, setBinanceConnected] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    try {
      setError(null);
      const st = await getAutoTradingStatus();
      setEnabled(!!st?.enabled);
      setMlReady(!!st?.ml_system_ready);
      setBinanceConnected(!!st?.binance_connected);
    } catch (e: any) {
      setError(e?.message || 'Failed to get status');
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected]);

  const onToggle = async (checked: boolean) => {
    try {
      setLoading(true);
      setError(null);
      if (checked) {
        await enableAutoTrading();
      } else {
        await disableAutoTrading();
      }
      await refresh();
    } catch (e: any) {
      setError(e?.message || 'Action failed');
    } finally {
      setLoading(false);
    }
  };

  if (!isConnected) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <Activity className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Connect to Binance API to control auto-trading</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Auto-Trading</h2>
        <div className="flex items-center gap-2">
          <Button onClick={refresh} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <Card className="p-6 bg-card border-border">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">Status</p>
            <div className="flex items-center gap-2">
              <Badge variant={enabled ? 'default' : 'secondary'} className={enabled ? 'bg-[hsl(var(--profit))]' : ''}>
                {enabled ? 'Enabled' : 'Disabled'}
              </Badge>
              <Badge variant={mlReady ? 'secondary' : 'destructive'}>
                ML {mlReady ? 'Ready' : 'Not Ready'}
              </Badge>
              <Badge variant={binanceConnected ? 'secondary' : 'destructive'}>
                Binance {binanceConnected ? 'OK' : 'Not Connected'}
              </Badge>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Power className={`w-5 h-5 ${enabled ? 'text-[hsl(var(--profit))]' : 'text-muted-foreground'}`} />
            <Switch checked={enabled} onCheckedChange={onToggle} disabled={loading} />
          </div>
        </div>
        {error && <p className="text-sm text-destructive mt-4">{error}</p>}
      </Card>
    </div>
  );
};

export default AutoTrading;


