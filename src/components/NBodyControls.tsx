import { useState } from 'react';
import { useSimulationStore } from '../store/useSimulationStore';
import { generateNBodyDemo } from '../simulation/nbody';
import './NBodyControls.css';

export function NBodyControls() {
  const {
    nbody,
    setNBodySnapshots,
    setNBodyFrame,
    setNBodyPlaying,
    setNBodyAnimationSpeed,
    setNBodyNumParticles,
    setNBodyNumSnapshots,
    setNBodyDeltaT,
  } = useSimulationStore();

  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    setProgress(0);
    setProgressMessage('Starting...');

    try {
      const snapshots = await generateNBodyDemo({
        numParticles: nbody.numParticles,
        numSnapshots: nbody.numSnapshots,
        deltaT: nbody.deltaT,
        onProgress: (p, msg) => {
          setProgress(p);
          setProgressMessage(msg);
        },
      });

      setNBodySnapshots(snapshots);
      setProgressMessage('Complete!');
    } catch (error) {
      console.error('Generation failed:', error);
      setProgressMessage('Error: ' + (error as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className={`control-panel ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="control-header">
        <h3>N-Body Simulation</h3>
        <button
          className="toggle-button"
          onClick={() => setIsCollapsed(!isCollapsed)}
          aria-label={isCollapsed ? 'Expand controls' : 'Collapse controls'}
        >
          {isCollapsed ? '▶' : '◀'}
        </button>
      </div>

      {!isCollapsed && (
        <>
      {generating ? (
        <div className="demo-notice">
          <div
            style={{
              background: '#333',
              borderRadius: '4px',
              overflow: 'hidden',
              marginTop: '8px',
            }}
          >
            <div
              style={{
                width: `${progress}%`,
                height: '20px',
                background: 'linear-gradient(90deg, #00ff00, #00cc00)',
                transition: 'width 0.3s',
              }}
            />
          </div>
          <div style={{ marginTop: '5px', fontSize: '11px' }}>{progressMessage}</div>
        </div>
      ) : (
        <div className="demo-notice">
          Click "Generate Demo" to see particles in motion!
        </div>
      )}

      <div className="control-group">
        <label>
          Number of Particles: <span className="value-display">{nbody.numParticles.toLocaleString()}</span>
        </label>
        <input
          type="range"
          min="100"
          max="50000"
          step="100"
          value={nbody.numParticles}
          onChange={(e) => setNBodyNumParticles(parseInt(e.target.value))}
          disabled={generating}
        />
      </div>

      <div className="control-group">
        <label>
          Number of Timesteps: <span className="value-display">{nbody.numSnapshots.toLocaleString()}</span>
        </label>
        <input
          type="range"
          min="50"
          max="10000"
          step="50"
          value={nbody.numSnapshots}
          onChange={(e) => setNBodyNumSnapshots(parseInt(e.target.value))}
          disabled={generating}
        />
      </div>

      <div className="control-group">
        <label>
          Delta T (Timestep Size): <span className="value-display">{nbody.deltaT.toFixed(3)}</span>
        </label>
        <input
          type="range"
          min="0.001"
          max="0.2"
          step="0.001"
          value={nbody.deltaT}
          onChange={(e) => setNBodyDeltaT(parseFloat(e.target.value))}
          disabled={generating}
        />
      </div>

      <div className="control-group">
        <button className="primary" onClick={handleGenerate} disabled={generating}>
          Generate Demo Data
        </button>
      </div>

      <div className="control-group">
        <label>
          Time Step: <span className="value-display">{nbody.currentFrame}</span>
        </label>
        <input
          type="range"
          min="0"
          max={Math.max(0, nbody.snapshots.length - 1)}
          value={nbody.currentFrame}
          onChange={(e) => setNBodyFrame(parseInt(e.target.value))}
          disabled={nbody.snapshots.length === 0}
        />
      </div>

      <div className="control-group">
        <label>
          Animation Speed:{' '}
          <span className="value-display">{nbody.animationSpeed.toFixed(1)}x</span>
        </label>
        <input
          type="range"
          min="0.1"
          max="5"
          step="0.1"
          value={nbody.animationSpeed}
          onChange={(e) => setNBodyAnimationSpeed(parseFloat(e.target.value))}
        />
      </div>

      <button
        className="primary"
        onClick={() => setNBodyPlaying(!nbody.playing)}
        disabled={nbody.snapshots.length === 0}
      >
        {nbody.playing ? 'Pause' : 'Play'} Animation
      </button>

      <button className="primary" onClick={() => window.location.reload()}>
        Reset View
      </button>
        </>
      )}
    </div>
  );
}
