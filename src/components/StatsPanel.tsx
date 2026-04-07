import { useSimulationStore } from '../store/useSimulationStore';
import './StatsPanel.css';

import { getParticleCount } from '../simulation/particleData';

export function StatsPanel() {
  const nbodyCurrentFrame = useSimulationStore((state) => state.nbody.currentFrame);
  const nbodyNumSnapshots = useSimulationStore((state) => state.nbody.snapshots.length);
  const nbodyInitialSnapshot = useSimulationStore((state) => state.nbody.snapshots[0]);

  const numParticles = nbodyInitialSnapshot ? getParticleCount(nbodyInitialSnapshot) : 0;

  return (
    <div className="stats-panel">
      <div className="stat-row">
        <span className="stat-label">Particles:</span>
        <span className="stat-value">{numParticles.toLocaleString()}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Snapshots:</span>
        <span className="stat-value">{nbodyNumSnapshots.toLocaleString()}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Frame:</span>
        <span className="stat-value">{nbodyCurrentFrame}</span>
      </div>
    </div>
  );
}
