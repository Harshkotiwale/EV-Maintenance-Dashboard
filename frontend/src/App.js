import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Bell, Battery, Thermometer, Zap, Activity, AlertTriangle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert';
import './dashboard.css'; // New custom styles

const Dashboard = () => {
  const [sensorData, setSensorData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selectedVehicle, setSelectedVehicle] = useState('EV001');
  const [maintenanceScore, setMaintenanceScore] = useState(85);
  const [batteryHealth, setBatteryHealth] = useState(92);

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      const newReading = {
        timestamp: new Date().toISOString(),
        battery_temp: 25 + Math.random() * 10,
        battery_voltage: 375 + Math.random() * 25,
        motor_temp: 50 + Math.random() * 15,
        battery_health: batteryHealth - Math.random(),
        motor_vibration: 2 + Math.random() * 2,
        power_output: 75 + Math.random() * 25,
      };

      setSensorData(prev => [...prev.slice(-20), newReading]);
      setBatteryHealth(prev => Math.max(prev - 0.1, 70));
      setMaintenanceScore(prev => Math.max(prev - 0.2, 0));

      // Generate random alerts
      if (Math.random() < 0.1) {
        setAlerts(prev => [
          ...prev,
          {
            id: Date.now(),
            type: Math.random() > 0.5 ? 'warning' : 'error',
            message: Math.random() > 0.5 ? 'High motor temperature!' : 'Battery degradation detected!',
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [batteryHealth]);

  const MetricCard = ({ icon: Icon, title, value, unit, trend }) => (
    <div className="metric-card">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className="text-blue-500" size={20} />
          <h3 className="text-sm font-medium">{title}</h3>
        </div>
        {trend && (
          <span className={`trend ${trend > 0 ? 'positive' : 'negative'}`}>
            {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}%
          </span>
        )}
      </div>
      <div className="flex items-baseline">
        <span className="text-2xl font-bold">{value.toFixed(1)}</span>
        <span className="ml-1 text-sm">{unit}</span>
      </div>
    </div>
  );

  return (
    <div className="dashboard-container">
      {/* Background Animation */}
      <div className="background-animation"></div>

      <div className="content">
        <header className="header">
          <div>
            <h1 className="title">EV Maintenance Dashboard</h1>
            <p className="subtitle">Real-time monitoring and predictive maintenance</p>
          </div>
          <select
            value={selectedVehicle}
            onChange={(e) => setSelectedVehicle(e.target.value)}
            className="vehicle-selector"
          >
            <option value="EV001">Vehicle EV001</option>
            <option value="EV002">Vehicle EV002</option>
            <option value="EV003">Vehicle EV003</option>
          </select>
        </header>

        {/* Main Metrics */}
        <div className="metrics-grid">
          <MetricCard icon={Activity} title="Maintenance Score" value={maintenanceScore} unit="%" trend={-2.5} />
          <MetricCard icon={Battery} title="Battery Health" value={batteryHealth} unit="%" trend={-1.2} />
          <MetricCard
            icon={Thermometer}
            title="Motor Temperature"
            value={sensorData.length ? sensorData[sensorData.length - 1].motor_temp : 0}
            unit="°C"
          />
          <MetricCard
            icon={Zap}
            title="Power Output"
            value={sensorData.length ? sensorData[sensorData.length - 1].power_output : 0}
            unit="kW"
          />
        </div>

        {/* Charts */}
        <div className="charts-grid">
          <div className="chart-card">
            <h3 className="chart-title">Temperature Trends</h3>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sensorData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" tick={false} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="battery_temp" stroke="#8884d8" name="Battery Temp" />
                  <Line type="monotone" dataKey="motor_temp" stroke="#82ca9d" name="Motor Temp" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="chart-card">
            <h3 className="chart-title">Power & Vibration</h3>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sensorData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" tick={false} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="power_output" stroke="#ff7300" name="Power Output" />
                  <Line type="monotone" dataKey="motor_vibration" stroke="#0088fe" name="Vibration" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Alerts Section */}
        <div className="alerts-section">
          <h3 className="alerts-title">
            Recent Alerts <Bell className="icon" />
          </h3>
          <div className="alerts-list">
            {alerts.slice(-5).reverse().map((alert) => (
              <Alert key={alert.id} variant={alert.type === 'error' ? 'destructive' : 'default'}>
                <AlertTriangle className="icon" />
                <AlertTitle>{alert.type === 'error' ? 'Critical Alert' : 'Warning'}</AlertTitle>
                <AlertDescription>
                  {alert.message} <span className="timestamp">{new Date(alert.timestamp).toLocaleTimeString()}</span>
                </AlertDescription>
              </Alert>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
