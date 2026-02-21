import { useSimulationStore } from '../store/useSimulationStore';
import './TopBar.css';

export function TopBar() {
  const mode = useSimulationStore((state) => state.mode);
  const setMode = useSimulationStore((state) => state.setMode);

  return (
    <div className="top-bar">
      <div className="title">HPC Simulation Viewer</div>
      <div className="mode-selector">
        <button
          className={`mode-btn ${mode === 'nbody' ? 'active' : ''}`}
          onClick={() => setMode('nbody')}
        >
          N-Body
        </button>
        <button
          className={`mode-btn ${mode === 'montecarlo' ? 'active' : ''}`}
          onClick={() => setMode('montecarlo')}
        >
          Monte Carlo
        </button>
      </div>
    </div>
  );
}
