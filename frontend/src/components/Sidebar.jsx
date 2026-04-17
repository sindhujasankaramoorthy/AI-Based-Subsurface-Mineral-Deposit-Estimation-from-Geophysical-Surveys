import React from 'react';
import { Settings, Loader2, Play, DollarSign, Target, SlidersHorizontal } from 'lucide-react';

const Sidebar = ({ params, setParams, onRunAnalyze, loading }) => {
    const handleChange = (e) => {
        const { name, value } = e.target;
        setParams(prev => ({ ...prev, [name]: parseFloat(value) }));
    };

    return (
        <div className="w-[340px] glass-panel min-h-screen p-6 border-r border-white/5 flex flex-col fixed left-0 top-0 overflow-y-auto shadow-2xl z-20">
            <h2 className="text-2xl font-black mb-1 flex items-center gap-3 text-white tracking-tight">
                <Settings size={28} className="text-blue-400" /> API Controls
            </h2>
            <p className="text-sm text-gray-400 mb-8 font-medium">Fine-tune geospatial inference parameters</p>

            <div className="space-y-8 flex-1 text-gray-200">

                {/* Financial Section */}
                <div className="pb-6 border-b border-white/5">
                    <h3 className="text-xs font-bold text-blue-400 mb-4 uppercase tracking-widest flex items-center gap-2">
                        <DollarSign size={16} /> Financial Model
                    </h3>
                    <div className="space-y-4">
                        <div className="group">
                            <label className="block text-[13px] font-medium text-gray-400 mb-1.5 group-hover:text-blue-300 transition">Est. Mineral Price ($)</label>
                            <input
                                type="number" name="mineral_price" value={params.mineral_price} onChange={handleChange}
                                className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition shadow-inner"
                            />
                        </div>
                        <div className="group">
                            <label className="block text-[13px] font-medium text-gray-400 mb-1.5 group-hover:text-blue-300 transition">Cost Per Drill Site ($)</label>
                            <input
                                type="number" name="drill_cost" value={params.drill_cost} onChange={handleChange}
                                className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2.5 text-white text-sm focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition shadow-inner"
                            />
                        </div>
                    </div>
                </div>

                {/* Exploration Section */}
                <div className="pb-6 border-b border-white/5">
                    <h3 className="text-xs font-bold text-orange-400 mb-4 uppercase tracking-widest flex items-center gap-2">
                        <Target size={16} /> Targeting Substation
                    </h3>
                    <div className="space-y-6">
                        <div className="group">
                            <label className="text-[13px] font-medium text-gray-400 mb-2 flex justify-between group-hover:text-orange-300 transition">
                                <span>Top-K Drill Sites</span>
                                <span className="text-orange-400 font-bold bg-orange-500/10 px-2 py-0.5 rounded">{params.k_sites}</span>
                            </label>
                            <input type="range" min="1" max="10" step="1" name="k_sites" value={params.k_sites} onChange={handleChange} className="w-full accent-orange-500 cursor-pointer" />
                        </div>

                        <div className="group">
                            <label className="text-[13px] font-medium text-gray-400 mb-2 flex justify-between group-hover:text-orange-300 transition">
                                <span>Uncertainty Penalty (Risk)</span>
                                <span className="text-orange-400 font-bold bg-orange-500/10 px-2 py-0.5 rounded">{params.uncertainty_penalty.toFixed(2)}</span>
                            </label>
                            <input type="range" min="0.0" max="1.0" step="0.05" name="uncertainty_penalty" value={params.uncertainty_penalty} onChange={handleChange} className="w-full accent-orange-500 cursor-pointer" />
                        </div>
                    </div>
                </div>

                {/* Model Inference */}
                <div>
                    <h3 className="text-xs font-bold text-purple-400 mb-4 uppercase tracking-widest flex items-center gap-2">
                        <SlidersHorizontal size={16} /> Inference Config
                    </h3>
                    <div className="space-y-6">
                        <div className="group">
                            <label className="text-[13px] font-medium text-gray-400 mb-2 flex justify-between group-hover:text-purple-300 transition">
                                <span>MC Dropout Samples</span>
                                <span className="text-purple-400 font-bold bg-purple-500/10 px-2 py-0.5 rounded">{params.n_mc}</span>
                            </label>
                            <input type="range" min="10" max="100" step="1" name="n_mc" value={params.n_mc} onChange={handleChange} className="w-full accent-purple-500 cursor-pointer" />
                        </div>

                        <div className="group">
                            <label className="text-[13px] font-medium text-gray-400 mb-2 flex justify-between group-hover:text-purple-300 transition">
                                <span>Depth Slice Matrix</span>
                                <span className="text-purple-400 font-bold bg-purple-500/10 px-2 py-0.5 rounded">Z={params.depth_slice}</span>
                            </label>
                            <input type="range" min="0" max="19" step="1" name="depth_slice" value={params.depth_slice} onChange={handleChange} className="w-full accent-purple-500 cursor-pointer" />
                        </div>

                        <div className="group">
                            <label className="text-[13px] font-medium text-gray-400 mb-2 flex justify-between group-hover:text-purple-300 transition">
                                <span>3D Ore Threshold</span>
                                <span className="text-purple-400 font-bold bg-purple-500/10 px-2 py-0.5 rounded">{params.threshold.toFixed(2)}</span>
                            </label>
                            <input type="range" min="0.10" max="0.90" step="0.05" name="threshold" value={params.threshold} onChange={handleChange} className="w-full accent-purple-500 cursor-pointer" />
                        </div>
                    </div>
                </div>
            </div>

            <button
                onClick={onRunAnalyze}
                disabled={loading}
                className="w-full mt-8 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-3.5 px-4 rounded-xl transition-all duration-300 disabled:opacity-50 flex justify-center items-center gap-2 shadow-[0_0_20px_rgba(79,70,229,0.3)] hover:shadow-[0_0_25px_rgba(79,70,229,0.5)] border border-white/10 shrink-0 transform hover:-translate-y-0.5"
            >
                {loading ? (
                    <>
                        <Loader2 className="animate-spin" size={20} />
                        Inferring Network...
                    </>
                ) : (
                    <>
                        <Play size={20} className="fill-white" />
                        Engage AI Analysis
                    </>
                )}
            </button>
        </div>
    );
};

export default Sidebar;
