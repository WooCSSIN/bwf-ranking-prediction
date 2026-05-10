import { useState } from 'react'

const API_URL = "https://bwf-ranking-prediction.onrender.com/api/v1/rankings";

function App() {
  const [draw, setDraw] = useState("MS");
  const [region, setRegion] = useState("Global");
  const [topN, setTopN] = useState(10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const fetchPredictions = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/${draw}/${region}?top_n=${topN}`);
      if (!response.ok) {
        throw new Error(`Model chưa được huấn luyện cho thể loại ${draw} - ${region}`);
      }
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-200 p-4 md:p-12 flex flex-col items-center selection:bg-blue-500/30">
      <div className="max-w-4xl w-full">
        <header className="text-center mb-16 mt-8">
          <div className="inline-block mb-4 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-sm font-medium tracking-wide">
            Powered by LightGBM & MLOps
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-emerald-400 mb-6 tracking-tight">
            BWF AI Predictor
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto leading-relaxed">
            Dự đoán Bảng xếp hạng cầu lông thế giới trong tương lai bằng Trí tuệ nhân tạo.
          </p>
        </header>

        {/* Controls Container - Glassmorphism */}
        <div className="bg-slate-900/40 backdrop-blur-2xl border border-slate-800/60 rounded-3xl p-8 mb-12 shadow-2xl shadow-black/50 flex flex-col md:flex-row gap-6 items-end relative overflow-hidden">
          {/* Subtle gradient glow inside the card */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-1/2 bg-blue-500/5 blur-[100px] pointer-events-none"></div>

          <div className="flex-1 w-full relative z-10">
            <label className="block text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Hạng mục</label>
            <select 
              value={draw} 
              onChange={(e) => setDraw(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3.5 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all hover:border-slate-700"
            >
              <option value="MS">Đơn Nam (MS)</option>
              <option value="WS">Đơn Nữ (WS)</option>
              <option value="MD">Đôi Nam (MD)</option>
              <option value="WD">Đôi Nữ (WD)</option>
              <option value="XD">Đôi Nam Nữ (XD)</option>
            </select>
          </div>

          <div className="flex-1 w-full relative z-10">
            <label className="block text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Khu vực</label>
            <select 
              value={region} 
              onChange={(e) => setRegion(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3.5 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all hover:border-slate-700"
            >
              <option value="Global">Toàn cầu (Global)</option>
              <option value="Asia">Châu Á (Asia)</option>
              <option value="Europe">Châu Âu (Europe)</option>
            </select>
          </div>

          <div className="flex-1 w-full relative z-10">
            <label className="block text-sm font-semibold text-slate-400 mb-2 uppercase tracking-wider">Top VĐV</label>
            <input 
              type="number" 
              value={topN} 
              onChange={(e) => setTopN(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-3.5 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all hover:border-slate-700"
              min="1" max="50"
            />
          </div>

          <button 
            onClick={fetchPredictions}
            disabled={loading}
            className="w-full md:w-auto px-10 py-3.5 relative z-10 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold rounded-xl shadow-[0_0_20px_rgba(37,99,235,0.3)] hover:shadow-[0_0_30px_rgba(37,99,235,0.5)] transform transition-all duration-200 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            {loading ? "Đang dự đoán..." : "Phân tích AI"}
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 p-5 rounded-2xl mb-8 text-center backdrop-blur-sm">
            <span className="font-semibold mr-2">⚠️ Lỗi:</span>{error}
          </div>
        )}

        {/* Results Container */}
        {results && (
          <div className="animate-in fade-in slide-in-from-bottom-8 duration-1000">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
              <div>
                <h2 className="text-3xl font-bold text-slate-100 tracking-tight">
                  Bảng xếp hạng Tương lai
                </h2>
                <p className="text-slate-400 mt-1">Phiên bản Mô hình: {results.model_version}</p>
              </div>
              <div className="text-left md:text-right bg-slate-900/50 border border-slate-800 px-4 py-2 rounded-xl">
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Thời gian dự kiến</p>
                <p className="text-emerald-400 font-mono font-bold text-lg">{results.prediction_date}</p>
              </div>
            </div>

            <div className="space-y-4">
              {results.rankings.map((player, idx) => (
                <div 
                  key={idx} 
                  className="bg-slate-900/40 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 flex items-center gap-6 hover:bg-slate-800/60 hover:border-slate-700 transition-all duration-300 group"
                  style={{ animationDelay: `${idx * 50}ms` }}
                >
                  <div className="w-14 h-14 shrink-0 flex items-center justify-center rounded-2xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 text-2xl font-black text-slate-300 group-hover:scale-110 group-hover:text-white group-hover:border-blue-500/50 shadow-inner transition-all duration-300">
                    {player.rank}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <h3 className="text-2xl font-bold text-slate-100 truncate group-hover:text-blue-400 transition-colors">
                      {player.player_name}
                    </h3>
                    <div className="flex items-center gap-3 mt-2">
                      <span className="text-xs px-3 py-1 rounded-full bg-slate-800/80 text-slate-300 font-semibold tracking-widest border border-slate-700/50 shadow-sm">
                        {player.country_code}
                      </span>
                    </div>
                  </div>

                  <div className="text-right shrink-0">
                    <p className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-500 font-mono tracking-tight">
                      {Math.round(player.predicted_points).toLocaleString()}
                    </p>
                    <p className="text-xs text-slate-500 uppercase tracking-widest font-bold mt-1">
                      Điểm BWF
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

export default App
