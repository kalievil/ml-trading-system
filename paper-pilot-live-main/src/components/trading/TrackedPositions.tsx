import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTradingAPI } from "@/hooks/useTradingAPI";

interface TrackedPosition {
  position_id: string;
  side: 'BUY' | 'SELL';
  entry_price: number;
  current_price: number;
  stop_loss: number;
  take_profit: number;
  unrealized_pnl_percent?: number;
  entry_time: string;
  order_id?: string;
}

const TrackedPositions = () => {
  const { isConnected, getTrackedPositions } = useTradingAPI();
  const [positions, setPositions] = useState<TrackedPosition[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const refresh = async () => {
    if (!isConnected) return;
    try {
      setLoading(true);
      const res = await getTrackedPositions();
      setPositions(res || []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Tracked Positions (SL/TP)</h2>
        <Button onClick={refresh} variant="outline" size="sm" disabled={loading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      <Card className="p-6 bg-card border-border">
        {positions.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-sm">No tracked positions</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent border-border">
                  <TableHead>ID</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead className="text-right">Entry</TableHead>
                  <TableHead className="text-right">Current</TableHead>
                  <TableHead className="text-right">Stop Loss</TableHead>
                  <TableHead className="text-right">Take Profit</TableHead>
                  <TableHead className="text-right">Unrealized %</TableHead>
                  <TableHead>Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((p) => (
                  <TableRow key={p.position_id} className="hover:bg-muted/50">
                    <TableCell className="font-medium mono-nums">{p.position_id}</TableCell>
                    <TableCell>
                      <Badge variant={p.side === 'BUY' ? 'default' : 'destructive'}>
                        {p.side}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right mono-nums">${p.entry_price.toLocaleString()}</TableCell>
                    <TableCell className="text-right mono-nums">${p.current_price.toLocaleString()}</TableCell>
                    <TableCell className="text-right mono-nums">${p.stop_loss.toLocaleString()}</TableCell>
                    <TableCell className="text-right mono-nums">${p.take_profit.toLocaleString()}</TableCell>
                    <TableCell className={cn('text-right mono-nums', (p.unrealized_pnl_percent ?? 0) >= 0 ? 'profit-text' : 'loss-text')}>
                      {(p.unrealized_pnl_percent ?? 0).toFixed(2)}%
                    </TableCell>
                    <TableCell className="mono-nums text-sm">
                      {p.entry_time}
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

export default TrackedPositions;


