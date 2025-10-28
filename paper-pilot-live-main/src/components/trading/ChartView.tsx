import { Card } from "@/components/ui/card";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useEffect, useState } from 'react';

const ChartView = () => {
  const [priceData, setPriceData] = useState<Array<{ time: string; price: number }>>([]);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [stats24h, setStats24h] = useState({ high: 0, low: 0, volume: 0 });

  useEffect(() => {
    const fetchBinanceData = async () => {
      try {
        // Fetch klines data
        const klinesResponse = await fetch('https://testnet.binancefuture.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=50');
        const klinesData = await klinesResponse.json();
        
        const formattedData = klinesData.map((candle: any) => ({
          time: new Date(candle[0]).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          price: parseFloat(candle[4]) // Close price
        }));
        
        setPriceData(formattedData);
        
        if (formattedData.length > 0) {
          const latest = formattedData[formattedData.length - 1].price;
          const previous = formattedData[0].price;
          setCurrentPrice(latest);
          setPriceChange(((latest - previous) / previous) * 100);
        }

        // Fetch 24h ticker stats
        const tickerResponse = await fetch('https://testnet.binancefuture.com/fapi/v1/ticker/24hr?symbol=BTCUSDT');
        const tickerData = await tickerResponse.json();
        
        setStats24h({
          high: parseFloat(tickerData.highPrice),
          low: parseFloat(tickerData.lowPrice),
          volume: parseFloat(tickerData.quoteVolume)
        });
      } catch (error) {
        console.error('Error fetching Binance data:', error);
      }
    };

    fetchBinanceData();
    const interval = setInterval(fetchBinanceData, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-card border-border">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold">BTC/USDT</h2>
            <p className="text-sm text-muted-foreground">Binance Testnet</p>
          </div>
          <div className="text-right">
            <p className="text-3xl font-bold mono-nums">${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
            <p className={`text-sm mono-nums ${priceChange >= 0 ? 'profit-text' : 'loss-text'}`}>
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
            </p>
          </div>
        </div>

        <div className="h-[500px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={priceData}>
              <defs>
                <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis 
                dataKey="time" 
                stroke="hsl(var(--muted-foreground))"
                style={{ fontSize: '12px' }}
              />
              <YAxis 
                stroke="hsl(var(--muted-foreground))"
                style={{ fontSize: '12px' }}
                domain={['dataMin - 200', 'dataMax + 200']}
                tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
              />
              <Area 
                type="monotone" 
                dataKey="price" 
                stroke="hsl(var(--primary))" 
                strokeWidth={2}
                fill="url(#priceGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-1">24h High</p>
          <p className="text-xl font-bold mono-nums">
            ${stats24h.high > 0 ? stats24h.high.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '-'}
          </p>
        </Card>
        <Card className="p-4 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-1">24h Low</p>
          <p className="text-xl font-bold mono-nums">
            ${stats24h.low > 0 ? stats24h.low.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '-'}
          </p>
        </Card>
        <Card className="p-4 bg-card border-border">
          <p className="text-sm text-muted-foreground mb-1">24h Volume</p>
          <p className="text-xl font-bold mono-nums">
            ${stats24h.volume > 0 ? (stats24h.volume / 1000000).toFixed(1) + 'M' : '-'}
          </p>
        </Card>
      </div>
    </div>
  );
};

export default ChartView;
