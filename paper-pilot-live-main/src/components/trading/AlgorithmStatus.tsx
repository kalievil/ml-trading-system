import { Badge } from "@/components/ui/badge";
import { Activity } from "lucide-react";
import { useTradingData } from "@/hooks/useTradingData";

const AlgorithmStatus = () => {
  const { algorithmState } = useTradingData();

  if (!algorithmState) {
    return null;
  }

  const isActive = algorithmState.is_active;
  const confidence = algorithmState.confidence || 0;
  const hasTrades = algorithmState.total_trades > 0;

  // Show different status based on trading activity
  const getStatusText = () => {
    if (!hasTrades) {
      return "Ready";
    }
    return isActive ? "Active" : "Inactive";
  };

  const getConfidenceText = () => {
    if (!hasTrades) {
      return "Ready to Trade";
    }
    return `Confidence: ${(confidence * 100).toFixed(0)}%`;
  };

  return (
    <div className="flex items-center gap-4">
      <div className="text-right hidden md:block">
        <p className="text-xs text-muted-foreground">Algorithm Status</p>
        <p className="text-sm font-semibold mono-nums">{getConfidenceText()}</p>
      </div>
      <Badge 
        variant={isActive ? "default" : "secondary"}
        className={isActive && hasTrades ? "bg-[hsl(var(--profit))] hover:bg-[hsl(var(--profit))] animate-pulse-glow" : ""}
      >
        <Activity className="w-3 h-3 mr-1" />
        {getStatusText()}
      </Badge>
    </div>
  );
};

export default AlgorithmStatus;
