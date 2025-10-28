import { Card } from "@/components/ui/card";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string;
  icon: LucideIcon;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  className?: string;
}

const MetricCard = ({ title, value, icon: Icon, trend, trendValue, className }: MetricCardProps) => {
  return (
    <Card className="p-6 bg-card border-border hover:border-primary/50 transition-colors">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm text-muted-foreground mb-1">{title}</p>
          <h3 className={cn("text-2xl font-bold mono-nums", className)}>{value}</h3>
          {trendValue && (
            <p className={cn(
              "text-xs mt-2 mono-nums",
              trend === "up" && "profit-text",
              trend === "down" && "loss-text",
              trend === "neutral" && "text-muted-foreground"
            )}>
              {trendValue}
            </p>
          )}
        </div>
        <div className={cn(
          "w-10 h-10 rounded-lg flex items-center justify-center",
          trend === "up" && "bg-[hsl(var(--profit)/0.1)]",
          trend === "down" && "bg-[hsl(var(--loss)/0.1)]",
          trend === "neutral" && "bg-muted"
        )}>
          <Icon className={cn(
            "w-5 h-5",
            trend === "up" && "profit-text",
            trend === "down" && "loss-text",
            trend === "neutral" && "text-muted-foreground"
          )} />
        </div>
      </div>
    </Card>
  );
};

export default MetricCard;
