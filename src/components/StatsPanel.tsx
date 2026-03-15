import { useMemo } from 'react';
import { useSimulationStore } from '../store/useSimulationStore';
import './StatsPanel.css';

import { getParticleCount } from '../simulation/particleData';

export function StatsPanel() {
  const mode = useSimulationStore((state) => state.mode);

  // Use granular selectors to avoid large object diffs on every frame
  const nbodyCurrentFrame = useSimulationStore((state) => state.nbody.currentFrame);
  const nbodyNumSnapshots = useSimulationStore((state) => state.nbody.snapshots.length);
  const nbodyInitialSnapshot = useSimulationStore((state) => state.nbody.snapshots[0]);

  const mcPhotons = useSimulationStore((state) => state.montecarlo.photons);
  const mcAbsorbed = useSimulationStore((state) => state.montecarlo.absorbed);
  const mcEscaped = useSimulationStore((state) => state.montecarlo.escaped);

  // Optimization: Prevent unnecessary array allocations (.filter) on every render
  const activePhotonsCount = useMemo(() => {
    let count = 0;
    for (let i = 0; i < mcPhotons.length; i++) {
      if (mcPhotons[i].alive) {
        count++;
      }
    }
    return count;
  }, [mcPhotons]);

  if (mode === 'nbody') {
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

  return (
    <div className="stats-panel">
      <div className="stat-row">
        <span className="stat-label">Active Photons:</span>
        <span className="stat-value">
          {activePhotonsCount}
        </span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Absorbed:</span>
        <span className="stat-value">{mcAbsorbed}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Escaped:</span>
        <span className="stat-value">{mcEscaped}</span>
      </div>
    </div>
  );
}
