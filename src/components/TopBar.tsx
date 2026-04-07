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
          N-Body (Precomputed)
        </button>
        <button
          className={`mode-btn ${mode === 'nbody-realtime' ? 'active' : ''}`}
          onClick={() => setMode('nbody-realtime')}
        >
          N-Body (Real-Time)
        </button>
      </div>
    </div>
  );
}
