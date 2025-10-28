import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface RealTrade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  commission: number;
  commission_asset: string;
  time: number;
  is_buyer: boolean;
  is_maker: boolean;
}

interface RecentTradesProps {
  trades?: RealTrade[];
}

const RecentTrades = ({ trades = [] }: RecentTradesProps) => {
  if (trades.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <p>No recent trades</p>
        <p className="text-sm">Execute trades to see them here</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {trades.map((trade) => {
        const timeAgo = getTimeAgo(trade.time);
        const isBuy = trade.side === 'BUY';
        
        return (
          <div key={trade.id} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
            <div className="flex items-center gap-3">
              <div className={cn(
                "p-2 rounded-full",
                isBuy ? "bg-green-100 text-green-600" : "bg-red-100 text-red-600"
              )}>
                {isBuy ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
              </div>
              <div>
                <p className="font-medium">{trade.symbol}</p>
                <p className="text-sm text-muted-foreground">{timeAgo}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="font-medium">
                {isBuy ? '+' : '-'}{trade.quantity.toFixed(6)} BTC
              </p>
              <p className="text-sm text-muted-foreground">
                ${trade.price.toLocaleString()}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
};

// Helper function to format time ago
function getTimeAgo(timestamp: number): string {
  const now = Date.now();
  const diff = now - timestamp;
  
  const minutes = Math.floor(diff / (1000 * 60));
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  
  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'Just now';
}

export default RecentTrades;