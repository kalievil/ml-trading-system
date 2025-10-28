import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useEffect, useState } from 'react';
import { useTradingData } from '@/hooks/useTradingData';

const PerformanceChart = () => {
  const [data, setData] = useState<Array<{ date: string; balance: number }>>([]);
  const { algorithmState } = useTradingData();

  useEffect(() => {
    // Only show data if there are actual trades or positions
    if (algorithmState && algorithmState.total_trades > 0) {
      // Generate minimal chart data based on current balance
      const now = new Date();
      const chartData = [];
      
      // Show last 7 days with current balance
      for (let i = 6; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        chartData.push({
          date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          balance: algorithmState.current_balance
        });
      }
      
      setData(chartData);
    } else {
      // No data - show empty state
      setData([]);
    }
  }, [algorithmState]);

  // Show empty state if no data
  if (data.length === 0) {
    return (
      <div className="h-[300px] flex items-center justify-center">
        <div className="text-center text-muted-foreground">
          <p className="text-sm">No trading data available</p>
          <p className="text-xs mt-1">Start trading to see performance charts</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis 
            dataKey="date" 
            stroke="hsl(var(--muted-foreground))"
            style={{ fontSize: '12px' }}
          />
          <YAxis 
            stroke="hsl(var(--muted-foreground))"
            style={{ fontSize: '12px' }}
            tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
              fontSize: '12px',
            }}
            formatter={(value: number) => [`$${value.toLocaleString()}`, 'Balance']}
          />
          <Line 
            type="monotone" 
            dataKey="balance" 
            stroke="hsl(var(--primary))" 
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PerformanceChart;