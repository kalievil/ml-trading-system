import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useTradingAPI } from "@/hooks/useTradingAPI";
import { useToast } from "@/hooks/use-toast";

const ManualTrade = () => {
  const { isConnected, manualTrade } = useTradingAPI();
  const { toast } = useToast();
  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [amountUsdt, setAmountUsdt] = useState<string>("");
  const [amountBtc, setAmountBtc] = useState<string>("");
  const [percentage, setPercentage] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const onSubmit = async () => {
    try {
      setLoading(true);
      const payload: any = { side };
      if (side === 'BUY') {
        if (percentage) payload.percentage = Math.max(0, Math.min(1, Number(percentage)));
        else if (amountUsdt) payload.amount_usdt = Number(amountUsdt);
      } else {
        if (percentage) payload.percentage = Math.max(0, Math.min(1, Number(percentage)));
        else if (amountBtc) payload.amount_btc = Number(amountBtc);
      }
      const res = await manualTrade(payload);
      toast({ title: `Manual ${side} executed`, description: JSON.stringify(res) });
      setAmountUsdt("");
      setAmountBtc("");
      setPercentage("");
    } catch (e: any) {
      toast({ variant: "destructive", title: "Manual trade failed", description: e?.message || 'Error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Manual Trade</h2>
      <Card className="p-6 bg-card border-border space-y-6">
        <div>
          <Label className="mb-2 block">Side</Label>
          <RadioGroup value={side} onValueChange={(v) => setSide(v as any)} className="flex gap-4">
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="BUY" id="side-buy" />
              <Label htmlFor="side-buy">BUY</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="SELL" id="side-sell" />
              <Label htmlFor="side-sell">SELL</Label>
            </div>
          </RadioGroup>
        </div>

        {side === 'BUY' ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="amount-usdt">Amount (USDT)</Label>
              <Input id="amount-usdt" placeholder="e.g. 100" value={amountUsdt} onChange={(e) => setAmountUsdt(e.target.value)} />
            </div>
            <div>
              <Label htmlFor="percentage">Percentage of USDT (0-1)</Label>
              <Input id="percentage" placeholder="e.g. 0.25" value={percentage} onChange={(e) => setPercentage(e.target.value)} />
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label htmlFor="amount-btc">Amount (BTC)</Label>
              <Input id="amount-btc" placeholder="e.g. 0.01" value={amountBtc} onChange={(e) => setAmountBtc(e.target.value)} />
            </div>
            <div>
              <Label htmlFor="percentage">Percentage of BTC (0-1)</Label>
              <Input id="percentage" placeholder="e.g. 0.5" value={percentage} onChange={(e) => setPercentage(e.target.value)} />
            </div>
          </div>
        )}

        <div className="flex gap-3">
          <Button onClick={onSubmit} disabled={loading || !isConnected}>{loading ? 'Submitting...' : 'Submit'}</Button>
        </div>
      </Card>
    </div>
  );
};

export default ManualTrade;


