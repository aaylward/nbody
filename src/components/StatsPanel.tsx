import { useSimulationStore } from '../store/useSimulationStore';
import './StatsPanel.css';

export function StatsPanel() {
  const mode = useSimulationStore((state) => state.mode);
  const nbody = useSimulationStore((state) => state.nbody);
  const montecarlo = useSimulationStore((state) => state.montecarlo);

  if (mode === 'nbody') {
    const numParticles = nbody.snapshots[0]?.length || 0;
    const numSnapshots = nbody.snapshots.length;

    return (
      <div className="stats-panel">
        <div className="stat-row">
          <span className="stat-label">Particles:</span>
          <span className="stat-value">{numParticles.toLocaleString()}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Snapshots:</span>
          <span className="stat-value">{numSnapshots.toLocaleString()}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Frame:</span>
          <span className="stat-value">{nbody.currentFrame}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="stats-panel">
      <div className="stat-row">
        <span className="stat-label">Active Photons:</span>
        <span className="stat-value">
          {montecarlo.photons.filter((p) => p.alive).length}
        </span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Absorbed:</span>
        <span className="stat-value">{montecarlo.absorbed}</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Escaped:</span>
        <span className="stat-value">{montecarlo.escaped}</span>
      </div>
    </div>
  );
}
