import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RefreshCw, TrendingUp, TrendingDown, DollarSign } from "lucide-react";
import { cn } from "@/lib/utils";
import { useRealTradingData } from "@/hooks/useRealTradingData";

const TradeHistory = () => {
  const { tradeHistory, isLoading, error, refreshData, isConnected } = useRealTradingData();

  if (!isConnected) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-muted-foreground">Connect to Binance API to view trade history</p>
          <p className="text-sm text-muted-foreground mt-2">Go to Settings tab to configure your API keys</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading trade history...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-destructive mb-4">Error loading trade history</p>
          <p className="text-sm text-muted-foreground mb-4">{error}</p>
          <Button onClick={refreshData} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  // Helper function to format time
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-card border-border">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold">Trade History</h2>
            <p className="text-sm text-muted-foreground">All executed trades from Binance</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="mono-nums">
              {tradeHistory.length} Trades
            </Badge>
            <Button onClick={refreshData} variant="outline" size="sm">
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>

        {tradeHistory.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-lg mb-2">No trade history</p>
            <p className="text-sm">Execute trades to see them here</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent border-border">
                      <TableHead>Time</TableHead>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead className="text-right">Quantity</TableHead>
                      <TableHead className="text-right">Entry Price</TableHead>
                      <TableHead className="text-right">Exit Price</TableHead>
                      <TableHead className="text-right">P&L</TableHead>
                      <TableHead className="text-right">P&L %</TableHead>
                      <TableHead className="text-right">Total Value</TableHead>
                      <TableHead className="text-right">Commission</TableHead>
                    </TableRow>
                  </TableHeader>
              <TableBody>
                {tradeHistory.map((trade) => (
                  <TableRow key={trade.id} className="hover:bg-muted/50">
                    <TableCell className="text-sm text-muted-foreground">
                      {formatTime(trade.time)}
                    </TableCell>
                    <TableCell className="font-medium">{trade.symbol}</TableCell>
                    <TableCell>
                      <Badge 
                        variant={trade.side === 'BUY' ? 'default' : 'destructive'}
                        className={cn(
                          trade.side === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        )}
                      >
                        {trade.side}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right mono-nums">
                      {trade.quantity.toFixed(6)} BTC
                    </TableCell>
                    <TableCell className="text-right mono-nums">
                      {trade.entry_price ? `$${trade.entry_price.toLocaleString()}` : '-'}
                    </TableCell>
                    <TableCell className="text-right mono-nums">
                      {trade.exit_price ? `$${trade.exit_price.toLocaleString()}` : '-'}
                    </TableCell>
                    <TableCell className={cn(
                      "text-right mono-nums font-semibold",
                      trade.pnl !== null ? (trade.pnl >= 0 ? "profit-text" : "loss-text") : ""
                    )}>
                      {trade.pnl !== null ? (
                        <>
                          {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                        </>
                      ) : '-'}
                    </TableCell>
                    <TableCell className={cn(
                      "text-right mono-nums font-semibold",
                      trade.pnl_percent !== null ? (trade.pnl_percent >= 0 ? "profit-text" : "loss-text") : ""
                    )}>
                      {trade.pnl_percent !== null ? (
                        <>
                          {trade.pnl_percent >= 0 ? '+' : ''}{trade.pnl_percent.toFixed(2)}%
                        </>
                      ) : '-'}
                    </TableCell>
                    <TableCell className="text-right mono-nums">
                      ${trade.total_value.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right mono-nums">
                      {trade.commission.toFixed(6)} {trade.commission_asset}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </Card>
    </div>
  );
};

export default TradeHistory;