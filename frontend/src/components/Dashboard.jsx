import React, { useState } from 'react';
import ReactPlotly from 'react-plotly.js';
import { Download, Layers, MapPin, Target, Activity } from 'lucide-react';
const Plot = typeof ReactPlotly === 'object' && ReactPlotly.default ? ReactPlotly.default : ReactPlotly;

const Dashboard = ({ data, params, loading }) => {
    const [activeTab, setActiveTab] = useState('maps');
    const [hoverData, setHoverData] = useState(null);

    if (loading && !data) {
        return (
            <div className="flex-1 space-y-6 animate-pulse">
                <div className="grid grid-cols-4 gap-4 mb-8">
                    {[1, 2, 3, 4].map(i => <div key={i} className="h-28 rounded-xl skeleton"></div>)}
                </div>
                <div className="h-14 w-1/3 rounded-lg skeleton mb-6"></div>
                <div className="h-[600px] w-full rounded-2xl skeleton"></div>
            </div>
        );
    }

    if (!data) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-20 text-gray-400 glass-panel border border-white/5 rounded-3xl mx-10 mt-10 shadow-2xl relative overflow-hidden group">
                <div className="absolute inset-0 bg-blue-500/5 mix-blend-overlay group-hover:bg-blue-500/10 transition duration-1000"></div>
                <div className="text-7xl mb-6 text-blue-500/80 drop-shadow-lg transform transition group-hover:scale-110 duration-700">🌍</div>
                <h2 className="text-3xl font-extrabold mb-3 text-white tracking-tight">Ready to infer</h2>
                <p className="text-lg">Adjust parameters in the sidebar and trigger the analysis model.</p>
                <p className="mt-8 text-sm px-4 py-2 rounded-full glass-panel border border-white/10 text-emerald-400 shadow-md">
                    💡 Tip: Request caching is enabled for real-time slider iteration.
                </p>
            </div>
        );
    }

    const { kpis, heatmap, scatter3d, drill_targets } = data;

    const downloadCSV = () => {
        if (!drill_targets || drill_targets.length === 0) return;
        const headers = Object.keys(drill_targets[0]).join(',');
        const rows = drill_targets.map(row => Object.values(row).join(',')).join('\n');
        const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows;

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "drill_targets.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className={`flex-1 transition-opacity duration-300 ${loading ? 'opacity-40 pointer-events-none' : 'opacity-100'}`}>
            {/* KPIs */}
            <div className="grid grid-cols-4 gap-5 mb-8">
                <div className="metric-card group relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition"><Target size={40} /></div>
                    <div className="text-xs font-semibold tracking-wider text-gray-400 mb-2 uppercase">Top Site Score</div>
                    <div className="text-4xl font-black text-white">{kpis.top_score.toFixed(3)}</div>
                </div>
                <div className="metric-card group relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition text-blue-400"><Layers size={40} /></div>
                    <div className="text-xs font-semibold tracking-wider text-gray-400 mb-2 uppercase">Mean Confidence</div>
                    <div className="text-4xl font-black text-blue-400">{kpis.mean_confidence.toFixed(1)}%</div>
                </div>
                <div className="metric-card group relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition text-orange-400"><MapPin size={40} /></div>
                    <div className="text-xs font-semibold tracking-wider text-gray-400 mb-2 uppercase">High Value Voxels</div>
                    <div className="text-4xl font-black text-orange-400">{kpis.high_value_voxels}</div>
                </div>
                <div className="metric-card group relative overflow-hidden border-emerald-500/30 shadow-emerald-900/20">
                    <div className="absolute top-0 right-0 p-4 opacity-20 transition text-emerald-400"><Activity size={40} /></div>
                    <div className="text-xs font-semibold tracking-wider text-gray-400 mb-2 uppercase">Total ROI</div>
                    <div className="text-3xl font-black text-emerald-400">${(kpis.proj_total_profit).toLocaleString()}</div>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-white/5 mb-6 gap-2">
                <button
                    className={`px-6 py-3 font-semibold text-sm rounded-t-lg transition-all duration-200 ${activeTab === 'maps' ? 'bg-white/5 text-blue-400 border-b-2 border-blue-500' : 'text-gray-400 hover:text-white hover:bg-white/5 border-b-2 border-transparent'}`}
                    onClick={() => setActiveTab('maps')}
                >
                    Probability Maps
                </button>
                <button
                    className={`px-6 py-3 font-semibold text-sm rounded-t-lg transition-all duration-200 ${activeTab === '3d' ? 'bg-white/5 text-blue-400 border-b-2 border-blue-500' : 'text-gray-400 hover:text-white hover:bg-white/5 border-b-2 border-transparent'}`}
                    onClick={() => setActiveTab('3d')}
                >
                    3D Ore Body Model
                </button>
                <button
                    className={`px-6 py-3 font-semibold text-sm rounded-t-lg transition-all duration-200 ${activeTab === 'targets' ? 'bg-white/5 text-emerald-400 border-b-2 border-emerald-500' : 'text-gray-400 hover:text-white hover:bg-white/5 border-b-2 border-transparent'}`}
                    onClick={() => setActiveTab('targets')}
                >
                    Targets & Financials
                </button>
            </div>

            {/* Tab Content */}
            <div className="glass-panel rounded-2xl p-6 shadow-2xl relative">

                {hoverData && (activeTab === 'maps' || activeTab === '3d') && (
                    <div className="absolute top-4 right-4 z-50 glass-panel border-gray-700 text-sm p-4 rounded-xl shadow-2xl min-w-48 pointer-events-none transform transition-all duration-200">
                        <div className="text-gray-400 font-medium mb-1">Coordinates: ({hoverData.x}, {hoverData.y})</div>
                        <div className="font-extrabold text-xl text-orange-400">{(hoverData.z * 100).toFixed(1)}% Yield</div>
                    </div>
                )}

                {loading && data && (
                    <div className="absolute inset-0 z-40 glass-panel rounded-2xl flex items-center justify-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                    </div>
                )}

                {activeTab === 'maps' && (
                    <div className="grid grid-cols-2 gap-6 relative z-10">
                        <div className="w-full bg-[#0a0c10]/80 rounded-xl p-2 border border-white/5">
                            <Plot
                                data={[{ z: heatmap.mean, type: 'heatmap', colorscale: 'Hot', hoverinfo: 'x+y+z' }]}
                                onHover={(e) => setHoverData({ x: e.points[0].x, y: e.points[0].y, z: e.points[0].z })}
                                onUnhover={() => setHoverData(null)}
                                layout={{ title: `Mineral Probability (Depth: ${params.depth_slice})`, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#ffffff', family: 'Inter' }, margin: { t: 40, l: 30, r: 30, b: 30 }, autosize: true }}
                                useResizeHandler={true} className="w-full h-[500px]"
                            />
                        </div>
                        <div className="w-full bg-[#0a0c10]/80 rounded-xl p-2 border border-white/5">
                            <Plot
                                data={[{ z: heatmap.uncert, type: 'heatmap', colorscale: 'Blues', hoverinfo: 'x+y+z' }]}
                                onHover={(e) => setHoverData({ x: e.points[0].x, y: e.points[0].y, z: e.points[0].z })}
                                onUnhover={() => setHoverData(null)}
                                layout={{ title: `Prediction Uncertainty (Depth: ${params.depth_slice})`, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#ffffff', family: 'Inter' }, margin: { t: 40, l: 30, r: 30, b: 30 }, autosize: true }}
                                useResizeHandler={true} className="w-full h-[500px]"
                            />
                        </div>
                    </div>
                )}

                {activeTab === '3d' && (
                    <div className="w-full bg-[#0a0c10]/80 rounded-xl border border-white/5 overflow-hidden">
                        <Plot
                            data={[{
                                type: 'scatter3d', mode: 'markers', x: scatter3d.x, y: scatter3d.y, z: scatter3d.z, hoverinfo: 'x+y+z+text',
                                text: scatter3d.prob.map(p => `Prob: ${(p * 100).toFixed(1)}%`),
                                marker: { color: scatter3d.prob, colorscale: 'Hot', size: 4, opacity: 0.85, colorbar: { title: "Probability", x: 1.1 } }
                            }]}
                            layout={{
                                title: "3D Predicted Ore Body Simulation", height: 700, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#ffffff', family: 'Inter' },
                                scene: {
                                    xaxis: { title: "X Grid", backgroundcolor: '#11151f', showbackground: true, gridcolor: '#2a3142' },
                                    yaxis: { title: "Y Grid", backgroundcolor: '#11151f', showbackground: true, gridcolor: '#2a3142' },
                                    zaxis: { title: "Depth", backgroundcolor: '#11151f', showbackground: true, gridcolor: '#2a3142' },
                                    camera: { eye: { x: 1.4, y: 1.4, z: 1.1 } }
                                }, margin: { t: 50, l: 0, r: 0, b: 0 }
                            }}
                            useResizeHandler={true} className="w-full"
                        />
                    </div>
                )}

                {activeTab === 'targets' && (
                    <div className="relative z-10">
                        <div className="flex justify-between items-center mb-6">
                            <h3 className="text-xl font-bold">Financial Modeling & Drill Targets</h3>
                            <button onClick={downloadCSV} className="bg-white/10 hover:bg-white/20 text-white px-5 py-2.5 rounded-lg text-sm font-semibold flex items-center gap-2 transition backdrop-blur">
                                <Download size={18} /> Export CSV
                            </button>
                        </div>

                        <div className="overflow-hidden rounded-xl border border-white/10 bg-black/20">
                            <table className="w-full text-left border-collapse">
                                <thead className="bg-black/40">
                                    <tr>
                                        <th className="py-4 px-6 font-semibold text-xs tracking-wider uppercase text-gray-400">Rank</th>
                                        <th className="py-4 px-6 font-semibold text-xs tracking-wider uppercase text-gray-400">Target (X, Y)</th>
                                        <th className="py-4 px-6 font-semibold text-xs tracking-wider uppercase text-gray-400">Probability</th>
                                        <th className="py-4 px-6 font-semibold text-xs tracking-wider uppercase text-gray-400">Risk Profile</th>
                                        <th className="py-4 px-6 font-semibold text-xs tracking-wider uppercase text-blue-400">Proj. Revenue</th>
                                        <th className="py-4 px-6 font-semibold text-xs tracking-wider uppercase text-emerald-400">Net Profit</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-white/5">
                                    {drill_targets.map((row, i) => (
                                        <tr key={i} className="hover:bg-white/5 transition-colors group">
                                            <td className="py-4 px-6 font-bold text-gray-300">#{row.Rank}</td>
                                            <td className="py-4 px-6 font-medium">({row['Grid X']}, {row['Grid Y']})</td>
                                            <td className="py-4 px-6">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-20 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                                                        <div className="h-full bg-gradient-to-r from-orange-500 to-yellow-400" style={{ width: `${row['Avg Probability'] * 100}%` }}></div>
                                                    </div>
                                                    <span className="text-orange-400 font-semibold text-sm">{(row['Avg Probability'] * 100).toFixed(1)}%</span>
                                                </div>
                                            </td>
                                            <td className="py-4 px-6 text-gray-500 text-sm">{(row.Uncertainty * 100).toFixed(1)}% ±</td>
                                            <td className="py-4 px-6 font-semibold text-blue-200">${row['Proj. Revenue ($)'].toLocaleString()}</td>
                                            <td className={`py-4 px-6 font-black ${row['Net Profit ($)'] > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                                ${row['Net Profit ($)'].toLocaleString()}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;
