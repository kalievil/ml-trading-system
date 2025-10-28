import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Trade } from "@/hooks/useTradingData";
import { CheckCircle2, TrendingUp, Shield, Activity } from "lucide-react";

interface TradeDetailsDialogProps {
  trade: Trade | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const TradeDetailsDialog = ({ trade, open, onOpenChange }: TradeDetailsDialogProps) => {
  if (!trade) return null;

  const confirmations = trade.confirmation_details;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <span>Trade Details - {trade.symbol}</span>
            <Badge 
              variant="outline"
              className={cn(
                "mono-nums",
                trade.side === 'LONG' ? 'long-bg profit-text' : 'short-bg loss-text'
              )}
            >
              {trade.side}
            </Badge>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Trade Summary */}
          <Card className="p-4 bg-card border-border">
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Trade Summary
            </h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-muted-foreground">Entry Price</p>
                <p className="font-mono font-semibold">
                  ${trade.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Size</p>
                <p className="font-mono font-semibold">{trade.size.toFixed(6)} BTC</p>
              </div>
              {confirmations?.leverage && (
                <div>
                  <p className="text-muted-foreground">Leverage</p>
                  <p className="font-mono font-semibold text-primary">{confirmations.leverage.toFixed(1)}x</p>
                </div>
              )}
              {confirmations?.positionSizePct && (
                <div>
                  <p className="text-muted-foreground">Position Size</p>
                  <p className="font-mono font-semibold">{confirmations.positionSizePct.toFixed(1)}%</p>
                </div>
              )}
              <div>
                <p className="text-muted-foreground">Stop Loss</p>
                <p className="font-mono font-semibold text-loss">
                  ${(trade.stop_loss || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Take Profit</p>
                <p className="font-mono font-semibold text-profit">
                  ${(trade.take_profit || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Confidence</p>
                <p className="font-mono font-semibold">{((trade.confidence || 0) * 100).toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-muted-foreground">Entry Time</p>
                <p className="text-xs">{new Date(trade.entry_time).toLocaleString()}</p>
              </div>
            </div>
          </Card>

          {/* Technical Indicators */}
          {confirmations && (
            <>
              <Card className="p-4 bg-card border-border">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-profit" />
                  Technical Indicators
                </h3>
                <ul className="space-y-2">
                  {confirmations.indicators.map((indicator, idx) => (
                    <li key={idx} className="flex items-center gap-2 text-sm">
                      <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                      {indicator}
                    </li>
                  ))}
                </ul>
              </Card>

              <Card className="p-4 bg-card border-border">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-primary" />
                  Technical Analysis
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {confirmations.technicalAnalysis}
                </p>
              </Card>

              <Card className="p-4 bg-card border-border">
                <h3 className="font-semibold mb-3 flex items-center gap-2">
                  <Shield className="w-4 h-4 text-warning" />
                  Risk Assessment
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {confirmations.riskAssessment}
                </p>
              </Card>

              <Card className="p-4 bg-card border-border">
                <h3 className="font-semibold mb-3">Market Condition</h3>
                <p className="text-sm text-muted-foreground">
                  {confirmations.marketCondition}
                </p>
              </Card>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TradeDetailsDialog;
