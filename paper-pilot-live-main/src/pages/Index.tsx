import { useState } from "react";
import { Button } from "@/components/ui/button";
import Dashboard from "@/components/trading/Dashboard";
import ChartView from "@/components/trading/ChartView";
import PositionsPanel from "@/components/trading/PositionsPanel";
import TradeHistory from "@/components/trading/TradeHistory";
import Signals from "@/components/trading/Signals";
import AutoTrading from "@/components/trading/AutoTrading";
import TrackedPositions from "@/components/trading/TrackedPositions";
import ManualTrade from "@/components/trading/ManualTrade";
import AlgorithmStatus from "@/components/trading/AlgorithmStatus";
import ApiSettings from "@/components/trading/ApiSettings";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshCw } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [activeTab, setActiveTab] = useState("dashboard");
  const { toast } = useToast();

  const handleRefresh = () => {
    // Refresh the page to reload mock data
    window.location.reload();
    toast({
      title: "Refreshed",
      description: "Trading data has been refreshed.",
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center">
                <span className="text-xl font-bold">K</span>
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">KDE Trading Platform</h1>
                <p className="text-xs text-muted-foreground">Algorithmic Paper Trading</p>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <AlgorithmStatus />
              <Button
                variant="ghost"
                size="sm"
                onClick={handleRefresh}
                className="gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-card border border-border">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="signals">Signals</TabsTrigger>
            <TabsTrigger value="chart">Chart</TabsTrigger>
            <TabsTrigger value="positions">Positions</TabsTrigger>
            <TabsTrigger value="tracked">Tracked Positions</TabsTrigger>
            <TabsTrigger value="autotrading">Auto-Trading</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
            <TabsTrigger value="manual">Manual Trade</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="space-y-6 animate-slide-up">
            <Dashboard />
          </TabsContent>

          <TabsContent value="signals" className="space-y-6 animate-slide-up">
            <Signals />
          </TabsContent>

          <TabsContent value="chart" className="space-y-6 animate-slide-up">
            <ChartView />
          </TabsContent>

          <TabsContent value="positions" className="space-y-6 animate-slide-up">
            <PositionsPanel />
          </TabsContent>

          <TabsContent value="tracked" className="space-y-6 animate-slide-up">
            <TrackedPositions />
          </TabsContent>

          <TabsContent value="autotrading" className="space-y-6 animate-slide-up">
            <AutoTrading />
          </TabsContent>

          <TabsContent value="history" className="space-y-6 animate-slide-up">
            <TradeHistory />
          </TabsContent>

          <TabsContent value="manual" className="space-y-6 animate-slide-up">
            <ManualTrade />
          </TabsContent>

          <TabsContent value="settings" className="space-y-6 animate-slide-up">
            <ApiSettings />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Index;
