import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Shield, Eye, EyeOff, Key, CheckCircle2, AlertTriangle, Wifi, WifiOff, FileText } from "lucide-react";
import { z } from "zod";
import { useTradingAPI } from "@/hooks/useTradingAPI";

const apiCredentialsSchema = z.object({
  binance_api_key: z.string()
    .trim()
    .min(20, "API Key must be at least 20 characters")
    .max(100, "API Key must be less than 100 characters"),
  binance_secret_key: z.string()
    .trim()
    .min(20, "Secret Key must be at least 20 characters")
    .max(100, "Secret Key must be less than 100 characters"),
});

const ApiSettings = () => {
  const [apiKey, setApiKey] = useState("");
  const [secretKey, setSecretKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [showSecretKey, setShowSecretKey] = useState(false);
  const [loading, setLoading] = useState(false);
  const [hasCredentials, setHasCredentials] = useState(false);
  const [autoTradingEnabled, setAutoTradingEnabled] = useState(false);
  const [autoTradingStatus, setAutoTradingStatus] = useState<any>(null);
  const { toast } = useToast();
  const { isConnected, configureBinance, healthCheck, getMLSignal, getTestSignal, executeTestTrade, getAutoTradingStatus, enableAutoTrading, disableAutoTrading } = useTradingAPI();

  useEffect(() => {
    checkExistingCredentials();
  }, []);

  const checkExistingCredentials = async () => {
    try {
      // Check localStorage first
      const stored = localStorage.getItem('api-credentials');
      if (stored) {
        const credentials = JSON.parse(stored);
        setHasCredentials(true);
        setApiKey('••••••••••••••••' + credentials.binance_api_key.slice(-4));
      }
      
      // Check API health to see if credentials are loaded from file
      try {
        const health = await healthCheck();
        if (health && health.credentials_from_file) {
          setCredentialsFromFile(true);
          setHasCredentials(true);
          setApiKey('••••••••••••••••' + 'FILE');
        }
      } catch (error) {
        console.error('Error checking API health:', error);
      }
    } catch (error) {
      console.error('Error checking credentials:', error);
    }
  };

  const handleSave = async () => {
    try {
      setLoading(true);
      
      // Validate input
      const validation = apiCredentialsSchema.safeParse({
        binance_api_key: apiKey,
        binance_secret_key: secretKey,
      });

      if (!validation.success) {
        const firstError = validation.error.errors[0];
        toast({
          variant: "destructive",
          title: "Validation Error",
          description: firstError.message,
        });
        return;
      }

      // Configure Binance API with backend
      await configureBinance(apiKey, secretKey);

      // Save to localStorage
      const credentials = {
        binance_api_key: apiKey,
        binance_secret_key: secretKey,
        saved_at: new Date().toISOString(),
      };
      
      localStorage.setItem('api-credentials', JSON.stringify(credentials));

      toast({
        title: "Success",
        description: "API credentials saved and configured successfully!",
      });

      setHasCredentials(true);
      setApiKey('••••••••••••••••' + apiKey.slice(-4));
      setSecretKey('');
      setShowApiKey(false);
      setShowSecretKey(false);
    } catch (error: any) {
      console.error('Error saving credentials:', error);
      toast({
        variant: "destructive",
        title: "Error",
        description: error.message || "Failed to save credentials",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTest = async () => {
    try {
      setLoading(true);
      
      // Test API connection
      const health = await healthCheck();
      
      if (health && health.models_loaded && health.binance_configured) {
        toast({
          title: "Connection Test",
          description: "Successfully connected to ML Trading API and Binance Testnet!",
          variant: "default",
        });
      } else {
        toast({
          title: "Connection Test",
          description: "API server not ready. Please check if Python backend is running.",
          variant: "destructive",
        });
      }
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Test Failed",
        description: error.message || "Failed to test connection",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleClearData = () => {
    localStorage.clear();
    window.location.reload();
    toast({
      title: "Data Cleared",
      description: "All trading data has been reset to initial state.",
    });
  };

  const handleMLSignal = async () => {
    try {
      setLoading(true);
      const signal = await getMLSignal();
      if (signal) {
        toast({
          title: "ML Signal Generated",
          description: `${signal.signal || 'HOLD'} signal at $${signal.current_price?.toFixed(2) || signal.entry_price?.toFixed(2) || 'N/A'} - Confidence: ${(signal.confidence * 100).toFixed(0)}%`,
        });
      }
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "ML Signal Error",
        description: error.message || "Failed to generate ML signal",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTestSignal = async () => {
    try {
      setLoading(true);
      const signal = await getTestSignal();
      if (signal) {
        toast({
          title: "Test Signal Generated",
          description: `${signal.side} signal at $${signal.entry_price.toFixed(2)} - Confidence: ${(signal.confidence * 100).toFixed(0)}%`,
        });
      }
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Test Failed",
        description: error.message,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleTestTrade = async () => {
    try {
      setLoading(true);
      const result = await executeTestTrade();
      toast({
        title: "Test Trade Executed",
        description: `Successfully executed ${result.side} order for ${result.quantity} ${result.symbol}`,
      });
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Test Trade Failed",
        description: error.message,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleEnableAutoTrading = async () => {
    try {
      setLoading(true);
      const result = await enableAutoTrading();
      setAutoTradingEnabled(true);
      toast({
        title: "Auto Trading Enabled",
        description: "ML algorithm will now automatically execute trades when signals meet criteria",
      });
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Auto Trading Error",
        description: error.message || "Failed to enable auto trading",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDisableAutoTrading = async () => {
    try {
      setLoading(true);
      const result = await disableAutoTrading();
      setAutoTradingEnabled(false);
      toast({
        title: "Auto Trading Disabled",
        description: "ML algorithm will no longer automatically execute trades",
      });
    } catch (error: any) {
      toast({
        variant: "destructive",
        title: "Auto Trading Error",
        description: error.message || "Failed to disable auto trading",
      });
    } finally {
      setLoading(false);
    }
  };

  const checkAutoTradingStatus = async () => {
    try {
      const status = await getAutoTradingStatus();
      setAutoTradingStatus(status);
      setAutoTradingEnabled(status.enabled);
    } catch (error) {
      console.error('Failed to get auto trading status:', error);
    }
  };

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Shield className="h-6 w-6 text-primary" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold">API Settings</h3>
            <p className="text-sm text-muted-foreground">
              Configure your Binance Testnet API credentials for ML-powered trading
            </p>
          </div>
          <div className="flex items-center gap-2">
            {isConnected ? (
              <div className="flex items-center gap-1 text-green-600">
                <Wifi className="h-4 w-4" />
                <span className="text-sm font-medium">
                  {credentialsFromFile ? "Connected (File)" : "Connected"}
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-1 text-red-600">
                <WifiOff className="h-4 w-4" />
                <span className="text-sm font-medium">Disconnected</span>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="api-key">Binance API Key</Label>
            <div className="flex gap-2">
              <Input
                id="api-key"
                type={showApiKey ? "text" : "password"}
                placeholder="Enter your Binance API key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                disabled={hasCredentials && !showApiKey}
                className="flex-1"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => setShowApiKey(!showApiKey)}
                disabled={loading}
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="secret-key">Binance Secret Key</Label>
            <div className="flex gap-2">
              <Input
                id="secret-key"
                type={showSecretKey ? "text" : "password"}
                placeholder="Enter your Binance secret key"
                value={secretKey}
                onChange={(e) => setSecretKey(e.target.value)}
                disabled={hasCredentials && !showSecretKey}
                className="flex-1"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => setShowSecretKey(!showSecretKey)}
                disabled={loading}
              >
                {showSecretKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        </div>

        <div className="flex gap-3 pt-2">
          <Button
            onClick={handleSave}
            disabled={loading || !apiKey || !secretKey}
            className="flex-1"
          >
            {loading ? "Saving..." : hasCredentials ? "Update Credentials" : "Save Credentials"}
          </Button>
          {hasCredentials && (
            <Button
              onClick={handleTest}
              disabled={loading}
              variant="outline"
            >
              Test Connection
            </Button>
          )}
        </div>

            <div className="pt-4 border-t border-border">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-medium">🤖 Automatic ML Trading</h4>
                  <p className="text-xs text-muted-foreground">
                    Enable ML algorithm to automatically execute trades when signals meet criteria
                  </p>
                  {autoTradingStatus && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Status: {autoTradingStatus.enabled ? "Enabled" : "Disabled"} | 
                      Min Interval: {Math.floor(autoTradingStatus.min_trade_interval / 60)}min | 
                      Engine Ready: {autoTradingStatus.trading_engine_ready ? "Yes" : "No"}
                    </div>
                  )}
                </div>
                <div className="flex gap-2">
                  {autoTradingEnabled ? (
                    <Button
                      onClick={handleDisableAutoTrading}
                      disabled={loading}
                      variant="destructive"
                      size="sm"
                    >
                      {loading ? "Disabling..." : "Disable Auto Trading"}
                    </Button>
                  ) : (
                    <Button
                      onClick={handleEnableAutoTrading}
                      disabled={loading || !isConnected}
                      variant="default"
                      size="sm"
                    >
                      {loading ? "Enabling..." : "Enable Auto Trading"}
                    </Button>
                  )}
                  <Button
                    onClick={checkAutoTradingStatus}
                    disabled={loading}
                    variant="outline"
                    size="sm"
                  >
                    Refresh Status
                  </Button>
                </div>
              </div>
            </div>
            
            <div className="pt-4 border-t border-border">
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium mb-2">🧪 Binance API Testing</h4>
              <p className="text-xs text-muted-foreground mb-3">
                Test the Binance API connection with simple trades (no ML conditions)
              </p>
              <div className="flex gap-2">
                <Button
                  onClick={handleMLSignal}
                  disabled={loading || !isConnected}
                  variant="outline"
                  size="sm"
                >
                  {loading ? "Generating..." : "Generate ML Signal"}
                </Button>
                <Button
                  onClick={handleTestSignal}
                  disabled={loading || !isConnected}
                  variant="outline"
                  size="sm"
                >
                  {loading ? "Testing..." : "Generate Test Signal"}
                </Button>
                <Button
                  onClick={handleTestTrade}
                  disabled={loading || !isConnected}
                  variant="outline"
                  size="sm"
                >
                  {loading ? "Executing..." : "Execute Test Trade"}
                </Button>
              </div>
            </div>
            
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-4">
              {isConnected ? (
                <>
                  <Wifi className="h-4 w-4 text-green-500" />
                  <span>Backend API: Connected</span>
                  {credentialsFromFile && (
                    <>
                      <FileText className="h-4 w-4 text-blue-500 ml-2" />
                      <span>Credentials loaded from file</span>
                    </>
                  )}
                </>
              ) : (
                <>
                  <WifiOff className="h-4 w-4 text-red-500" />
                  <span>Backend API: Disconnected</span>
                </>
              )}
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-medium">Reset Data</h4>
                <p className="text-xs text-muted-foreground">Clear all trading data and start fresh</p>
              </div>
              <Button
                onClick={handleClearData}
                variant="destructive"
                size="sm"
              >
                Clear All Data
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default ApiSettings;