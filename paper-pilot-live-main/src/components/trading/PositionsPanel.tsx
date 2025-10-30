import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Eye, RefreshCw, X } from "lucide-react";
import { cn } from "../../lib/utils.ts";
import { useRealTradingData } from "@/hooks/useRealTradingData";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

const PositionsPanel = () => {
  const { positions, isLoading, error, refreshData, closePositionAndRefresh, isConnected } = useRealTradingData();
  const { toast } = useToast();
  const [closingPosition, setClosingPosition] = useState<string | null>(null);

  const handleClosePosition = async (symbol: string) => {
    try {
      setClosingPosition(symbol);
      
      // Show confirmation dialog
      const confirmed = window.confirm(
        `Are you sure you want to close your ${symbol} position?\n\nThis will sell all your BTC holdings at market price.`
      );
      
      if (!confirmed) {
        setClosingPosition(null);
        return;
      }

      const result = await closePositionAndRefresh(symbol);
      
      toast({
        title: "Position Closed",
        description: `Successfully closed ${symbol} position. Sold ${result.btc_sold} BTC for ~$${result.estimated_usdt_received.toFixed(2)}`,
      });
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Error Closing Position",
        description: error.message || "Failed to close position",
      });
    } finally {
      setClosingPosition(null);
    }
  };

  if (!isConnected) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-muted-foreground">Connect to Binance API to view positions</p>
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
          <p className="text-muted-foreground">Loading positions...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <p className="text-destructive mb-4">Error loading positions</p>
          <p className="text-sm text-muted-foreground mb-4">{error}</p>
          <Button onClick={refreshData} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-card border-border">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold">Active Positions</h2>
            <p className="text-sm text-muted-foreground">Your current holdings from Binance</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="mono-nums">
              {positions.length} Open
            </Badge>
            <Button onClick={refreshData} variant="outline" size="sm">
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>

        {positions.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-lg mb-2">No active positions</p>
            <p className="text-sm">Execute trades to see positions here</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent border-border">
                  <TableHead>Symbol</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead className="text-right">Entry Price</TableHead>
                  <TableHead className="text-right">Current Price</TableHead>
                  <TableHead className="text-right">Amount</TableHead>
                  <TableHead className="text-right">Unrealized P&L</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((position) => (
                  <TableRow key={position.symbol} className="hover:bg-muted/50">
                    <TableCell className="font-medium">{position.symbol}</TableCell>
                    <TableCell>
                      <Badge 
                        variant={position.side === 'LONG' ? 'default' : 'destructive'}
                        className={cn(
                          position.side === 'LONG' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        )}
                      >
                        {position.side}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right mono-nums">
                      ${position.entry_price.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right mono-nums">
                      ${position.current_price.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-right mono-nums">
                      {position.amount.toFixed(6)} BTC
                    </TableCell>
                    <TableCell className={cn(
                      "text-right mono-nums font-semibold",
                      position.unrealized_pnl >= 0 ? "profit-text" : "loss-text"
                    )}>
                      {position.unrealized_pnl >= 0 ? '+' : ''}${position.unrealized_pnl.toFixed(2)}
                      <div className="text-xs text-muted-foreground">
                        ({position.unrealized_pnl_percent >= 0 ? '+' : ''}{position.unrealized_pnl_percent.toFixed(2)}%)
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center gap-2">
                        <Button variant="ghost" size="sm">
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button 
                          variant="destructive" 
                          size="sm"
                          onClick={() => handleClosePosition(position.symbol)}
                          disabled={closingPosition === position.symbol}
                        >
                          {closingPosition === position.symbol ? (
                            <RefreshCw className="w-4 h-4 animate-spin" />
                          ) : (
                            <X className="w-4 h-4" />
                          )}
                        </Button>
                      </div>
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

export default PositionsPanel;