import React, { useState } from 'react';
import axios from 'axios';
import { Toaster, toast } from 'sonner';
import { Pickaxe, Activity } from 'lucide-react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';

function App() {
  const [params, setParams] = useState({
    k_sites: 5,
    uncertainty_penalty: 0.30,
    n_mc: 50,
    depth_slice: 10,
    threshold: 0.25,
    mineral_price: 50000.0,
    drill_cost: 15000.0
  });

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const runAnalysis = async () => {
    setLoading(true);
    const toastId = toast.loading('Running probabilistic inversion model...', {
      description: `Simulating ${params.n_mc} MC dropout passes. Please wait.`,
    });

    try {
      const response = await axios.post('/api/analyze', params);
      setData(response.data);
      toast.success('Inference Complete', {
        id: toastId,
        description: `Identified ${params.k_sites} optimal drill sites with positive ROI.`,
      });
    } catch (error) {
      console.error("Error fetching analysis data:", error);
      toast.error('Analysis Failed', {
        id: toastId,
        description: 'Check if the FastAPI backend is running.',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex bg-transparent min-h-screen text-white">
      {/* Sonner Toast Container centered at bottom */}
      <Toaster position="bottom-right" theme="dark" richColors toastOptions={{
        style: { background: 'rgba(22, 26, 35, 0.95)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.08)' }
      }} />

      <Sidebar
        params={params}
        setParams={setParams}
        onRunAnalyze={runAnalysis}
        loading={loading}
      />

      <div className="flex-1 ml-80 p-8 flex flex-col min-h-screen relative z-10">
        <header className="mb-8 border-b border-white/5 pb-6">
          <div className="flex justify-between items-end">
            <div>
              <h1 className="text-4xl font-extrabold tracking-tight flex items-center gap-3 mb-2">
                <Pickaxe size={40} className="text-blue-400" />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
                  GeoVision AI
                </span>
              </h1>
              <p className="text-gray-400 flex items-center gap-2">
                <Activity size={18} className="text-gray-500" />
                Probabilistic Subsurface Mineral Intelligence
              </p>
            </div>
            {data && data.kpis && data.kpis.proj_total_profit > 0 && (
              <div className="text-right metric-card !p-4 !rounded-xl !bg-emerald-900/10 !border-emerald-500/20">
                <p className="text-sm font-medium text-emerald-400/70 mb-1 uppercase tracking-wider">Total Projected ROI</p>
                <div className="text-4xl font-bold text-emerald-400">${data.kpis.proj_total_profit.toLocaleString()}</div>
              </div>
            )}
          </div>
        </header>

        <Dashboard data={data} params={params} loading={loading} />
      </div>
    </div>
  );
}

export default App;
