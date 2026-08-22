import { useEffect, useState } from 'react';
import { useRealtimeStore } from '../store/useRealtimeStore';
import './StatsPanel.css'; // Reuse existing stats panel styles

export function PerformanceStats() {
  const simulation = useRealtimeStore((state) => state.simulation);
  const physicsFrameCount = useRealtimeStore((state) => state.physicsFrameCount);
  const updateStats = useRealtimeStore((state) => state.updateStats);

  // Use granular selectors to avoid whole-state subscription
  const renderFPS = useRealtimeStore((state) => state.stats.renderFPS);
  const physicsFPS = useRealtimeStore((state) => state.stats.physicsFPS);
  const physicsAvg = useRealtimeStore((state) => state.stats.physicsAvg);
  const physicsP95 = useRealtimeStore((state) => state.stats.physicsP95);
  const renderAvg = useRealtimeStore((state) => state.stats.renderAvg);
  const renderP95 = useRealtimeStore((state) => state.stats.renderP95);

  const [visible, setVisible] = useState(false);

  // Update stats periodically
  useEffect(() => {
    if (!simulation) return;

    const interval = setInterval(() => {
      updateStats();
    }, 500); // Update every 500ms

    return () => clearInterval(interval);
  }, [simulation, updateStats]);

  if (!simulation) return null;

  const getFPSColor = (fps: number, target: number) => {
    if (fps >= target * 0.9) return '#00ff00';
    if (fps >= target * 0.7) return '#ffff00';
    return '#ff4444';
  };

  return (
    <div className={`stats-panel ${!visible ? 'hidden' : ''}`}>
      <div className="stats-header">
        <h4>Performance Stats</h4>
        <button
          className="stats-toggle"
          onClick={() => setVisible(!visible)}
          aria-label={visible ? 'Hide stats' : 'Show stats'}
        >
          {visible ? '−' : '+'}
        </button>
      </div>

      {visible && (
        <div className="stats-content">
          <div className="stat-row">
            <span className="stat-label">Render FPS:</span>
            <span className="stat-value" style={{ color: getFPSColor(renderFPS, 60) }}>
              {renderFPS.toFixed(1)}
            </span>
            <span className="stat-target">/ 60</span>
          </div>

          <div className="stat-row">
            <span className="stat-label">Physics FPS:</span>
            <span className="stat-value" style={{ color: getFPSColor(physicsFPS, 20) }}>
              {physicsFPS.toFixed(1)}
            </span>
            <span className="stat-target">/ 20</span>
          </div>

          <div className="stat-row">
            <span className="stat-label">Physics Time:</span>
            <span className="stat-value">{physicsAvg.toFixed(1)}ms</span>
            <span className="stat-detail">(P95: {physicsP95.toFixed(1)}ms)</span>
          </div>

          <div className="stat-row">
            <span className="stat-label">Render Time:</span>
            <span className="stat-value">{renderAvg.toFixed(1)}ms</span>
            <span className="stat-detail">(P95: {renderP95.toFixed(1)}ms)</span>
          </div>

          <div className="stat-row">
            <span className="stat-label">Physics Frames:</span>
            <span className="stat-value">{physicsFrameCount.toLocaleString()}</span>
          </div>

          <div className="stat-row">
            <span className="stat-label">Particles:</span>
            <span className="stat-value">{simulation.getParticleCount().toLocaleString()}</span>
          </div>

          <div className="stats-info">
            <small>
              ℹ️ Render @ 60 FPS via interpolation
              <br />
              Physics @ ~{simulation.targetPhysicsFPS} FPS target
            </small>
          </div>
        </div>
      )}
    </div>
  );
}
