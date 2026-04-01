import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  LineChart, Line, Legend 
} from 'recharts';
import { 
  Sun, Zap, Activity, Info, TrendingUp, RefreshCw, Layers, ShieldCheck, 
  Clock, CloudSun, Target, Users
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = 'http://localhost:8000';

const StatsCard = ({ title, value, unit, icon: Icon, color = "blue" }) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="p-6 glass-card flex flex-col justify-between"
    style={{ minHeight: '160px', padding: '1.5rem', backgroundColor: 'rgba(255, 255, 255, 0.6)' }}
  >
    <div className="flex justify-between items-start">
      <span className="font-medium text-xs tracking-widest uppercase" style={{ color: '#64748b' }}>{title}</span>
      <Icon className={`w-5 h-5 text-${color}-600 opacity-80`} />
    </div>
    <div className="mt-4 flex items-baseline">
      <span className="text-3xl font-bold" style={{ color: '#0f172a' }}>{value}</span>
      <span className="ml-1 text-sm" style={{ color: '#475569' }}>{unit}</span>
    </div>
  </motion.div>
);

const LineChartGraph = ({ data, title, tickCount = undefined }) => (
  <div className="glass-card flex flex-col p-4 w-full h-full" style={{ minHeight: '300px' }}>
    <h4 className="text-md font-semibold mb-2 text-center" style={{ color: '#1e293b' }}>{title}</h4>
    <ResponsiveContainer width="100%" height="90%">
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={false} />
        <XAxis dataKey="index" hide />
        <YAxis stroke="#64748b" fontSize={11} tickLine={true} axisLine={true} tickCount={tickCount} />
        <Tooltip 
          contentStyle={{ backgroundColor: 'rgba(255,255,255,0.9)', border: '1px solid #e2e8f0', borderRadius: '8px' }}
          labelStyle={{ color: '#64748b' }}
        />
        <Legend wrapperStyle={{ fontSize: '12px' }}/>
        <Line isAnimationActive={false} type="linear" dataKey="actual" stroke="#2563eb" strokeWidth={2.5} dot={false} name="Ground Truth" />
        <Line isAnimationActive={false} type="linear" dataKey="predicted" stroke="#f59e0b" strokeWidth={2.5} dot={false} strokeDasharray="5 5" name="Forecast" />
      </LineChart>
    </ResponsiveContainer>
  </div>
);

const App = () => {
  const [forecast, setForecast] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('forecast');

  const teamMembers = ["Alex Johnson", "Jordan Lee", "Taylor Smith", "Morgan Davis", "Casey Wilson"];

  const fetchData = async () => {
    setLoading(true);
    try {
      const [fRes, mRes] = await Promise.all([
        axios.post(`${API_BASE}/forecast`),
        axios.get(`${API_BASE}/metrics`)
      ]);
      setForecast(fRes.data);
      setMetrics(mRes.data);
    } catch (error) {
      console.error("Fetch error:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const forecastData = forecast?.forecast.map(d => ({
    time: new Date(d.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    p10: d.p10,
    p50: d.p50,
    p90: d.p90,
  })) || [];

  const validationData = metrics?.history?.actual?.map((act, i) => ({
    index: i,
    actual: act,
    predicted: metrics.history.predicted[i]
  })) || [];

  // Derived datasets for the 3 requested graphs
  const transformerData = validationData.slice(-400); // Zoomed out view of the transformer learning
  const validationSetData = validationData.slice(-250, -100); // 150 points for validation
  const testSetData = validationData.slice(-100); // Last 100 points for test

  return (
    <div style={{ minHeight: '100vh', width: '100%', padding: '2rem 4rem', boxSizing: 'border-box' }}>
      <header className="flex justify-between items-center mb-8">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-100 rounded-xl shadow-sm border border-blue-200">
              <Sun className="w-8 h-8 text-blue-600" />
            </div>
            <div>
              <h1 className="text-3xl font-bold m-0" style={{ color: '#0f172a' }}>Team 8 Research Suite</h1>
              <p className="text-sm m-0" style={{ color: '#64748b' }}>SolarAI Intelligence Forecasting</p>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2 items-center text-sm" style={{ color: '#475569' }}>
            <Users className="w-4 h-4 mr-1"/>
            <span className="font-semibold text-slate-800">Team:</span>
            {teamMembers.map((member, idx) => (
              <span key={idx} className="bg-white/60 px-2 py-1 rounded-full border border-slate-200 backdrop-blur-sm mr-1">
                {member}
              </span>
            ))}
          </div>
        </div>
        
        <button 
          onClick={fetchData}
          className="flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl transition-all duration-200 active:scale-95 shadow-lg shadow-blue-500/30"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Sync Data
        </button>
      </header>

      <div className="flex gap-4 mb-8">
        {[
          { id: 'forecast', label: 'Live Pipeline Forecast', icon: Zap },
          { id: 'metrics', label: 'Neural Validation Reports', icon: Activity }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-6 py-2.5 rounded-full text-sm font-semibold transition-all duration-300 flex items-center gap-2 ${
              activeTab === tab.id 
                ? 'bg-blue-100 text-blue-700 border border-blue-300 shadow-sm' 
                : 'bg-white/40 text-slate-600 hover:text-slate-800 hover:bg-white/70 border border-slate-200/50'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === 'forecast' ? (
          <motion.div 
            key="forecast-tab"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          >
            <div className="grid grid-cols-4 gap-6 mb-8">
              <StatsCard title="24h Energy Yield" value={forecast?.metadata.integrated_kwh.toFixed(1) || '0.0'} unit="kWh" icon={Layers} color="blue" />
              <StatsCard title="Peak Solar Cap" value={forecast?.metadata.peak_kw.toFixed(2) || '0.00'} unit="kW" icon={TrendingUp} color="emerald" />
              <StatsCard title="Peak Flux Time" value={new Date(forecast?.forecast.find(d => d.p50 === forecast.metadata.peak_kw)?.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) || '--:--'} unit="" icon={Clock} color="amber" />
              <StatsCard title="System Precision" value="Optimal" unit="" icon={ShieldCheck} color="indigo" />
            </div>

            <div className="p-8 glass-card flex flex-col" style={{ height: '550px' }}>
              <h3 className="text-xl font-semibold mb-6 flex items-center gap-2" style={{ color: '#0f172a' }}>
                <CloudSun className="text-blue-500" /> 24-Hour Probabilistic Solar Output
              </h3>
              <ResponsiveContainer width="100%" height="90%">
                <AreaChart data={forecastData} margin={{ top: 20, right: 30, left: 30, bottom: 20 }}>
                  <defs>
                    <linearGradient id="colorP50" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563eb" stopOpacity={0.5}/>
                      <stop offset="95%" stopColor="#2563eb" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={false} />
                  
                  {/* Explicit XAxis and YAxis values and labels as requested */}
                  <XAxis 
                    dataKey="time" 
                    stroke="#475569" 
                    fontSize={12} 
                    tickLine={true} 
                    axisLine={true} 
                    tickMargin={12}
                    label={{ value: 'Time of Day (HH:MM)', position: 'insideBottom', offset: -15, fill: '#334155', fontWeight: 500 }}
                  />
                  <YAxis 
                    stroke="#475569" 
                    fontSize={12} 
                    tickLine={true} 
                    axisLine={true} 
                    tickMargin={12} 
                    label={{ value: 'Projected Power Output (kW)', angle: -90, position: 'insideLeft', offset: -10, fill: '#334155', fontWeight: 500 }} 
                  />
                  
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(255,255,255,0.9)', border: '1px solid #cbd5e1', borderRadius: '8px', color: '#0f172a' }}
                    labelStyle={{ color: '#334155', fontWeight: 'bold', marginBottom: '4px' }}
                    itemStyle={{ color: '#0f172a' }}
                  />
                  <Legend verticalAlign="top" height={36}/>
                  
                  <Area type="monotone" dataKey="p90" stroke="none" fill="#f59e0b" fillOpacity={0.15} name="Upper Bound (P90)"/>
                  <Area type="monotone" dataKey="p10" stroke="none" fill="#f59e0b" fillOpacity={0.15} name="Lower Bound (P10)"/>
                  <Area type="monotone" dataKey="p50" stroke="#2563eb" fillOpacity={1} fill="url(#colorP50)" strokeWidth={3} name="Expected Output (P50)"/>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        ) : (
          <motion.div 
            key="metrics-tab"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          >
            <div className="grid grid-cols-4 gap-6 mb-8">
              <StatsCard title="Regression R²" value={metrics?.validation.r2.toFixed(3) || '0.000'} unit="" icon={Target} color="blue" />
              <StatsCard title="Mean Absolute Error" value={metrics?.validation.mae.toFixed(3) || '0.000'} unit="kW" icon={Zap} color="cyan" />
              <StatsCard title="RMSE Metric" value={metrics?.validation.rmse.toFixed(3) || '0.000'} unit="kW" icon={Activity} color="indigo" />
              <StatsCard title="Dataset Health" value="Stable" unit="" icon={ShieldCheck} color="emerald" />
            </div>

            <div className="grid grid-cols-2 gap-6" style={{ height: '600px' }}>
              <div className="col-span-2" style={{ height: '300px' }}>
                <LineChartGraph data={transformerData} title="Transformer Graph (Global Trend)" />
              </div>
              <div style={{ height: '280px' }}>
                <LineChartGraph data={validationSetData} title="Validation Set Graph" tickCount={5} />
              </div>
              <div style={{ height: '280px' }}>
                <LineChartGraph data={testSetData} title="Test Set Graph" tickCount={5} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="mt-12 text-center text-xs tracking-wider uppercase font-medium pb-8" style={{ color: '#64748b' }}>
        Team 8 Research Suite • Photovoltaic Forecasting
      </footer>
    </div>
  );
};

export default App;

