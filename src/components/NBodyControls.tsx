import { useState } from 'react';
import { useSimulationStore } from '../store/useSimulationStore';
import { generateNBodyDemo, initRealTimeSimulation, activeSimulation } from '../simulation/nbody';
import './NBodyControls.css';

function PlaybackControls() {
  const isRealTime = useSimulationStore((state) => state.nbody.isRealTime);
  const playing = useSimulationStore((state) => state.nbody.playing);
  const currentFrame = useSimulationStore((state) => state.nbody.currentFrame);
  const snapshotsLength = useSimulationStore((state) => state.nbody.snapshots.length);
  const animationSpeed = useSimulationStore((state) => state.nbody.animationSpeed);

  const setNBodyFrame = useSimulationStore((state) => state.setNBodyFrame);
  const setNBodyAnimationSpeed = useSimulationStore((state) => state.setNBodyAnimationSpeed);
  const setNBodyPlaying = useSimulationStore((state) => state.setNBodyPlaying);

  return (
    <>
      {!isRealTime && (
        <>
          <div className="control-group">
            <label>
              Time Step: <span className="value-display">{currentFrame}</span>
            </label>
            <input
              type="range"
              min="0"
              max={Math.max(0, snapshotsLength - 1)}
              value={currentFrame}
              onChange={(e) => setNBodyFrame(parseInt(e.target.value))}
              disabled={snapshotsLength === 0}
            />
          </div>

          <div className="control-group">
            <label>
              Animation Speed:{' '}
              <span className="value-display">{animationSpeed.toFixed(1)}x</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="5"
              step="0.1"
              value={animationSpeed}
              onChange={(e) => setNBodyAnimationSpeed(parseFloat(e.target.value))}
            />
          </div>
        </>
      )}

      <button
        className="primary"
        onClick={() => setNBodyPlaying(!playing)}
        disabled={snapshotsLength === 0}
      >
        {playing ? 'Pause' : 'Play'} {isRealTime ? 'Simulation' : 'Animation'}
      </button>
    </>
  );
}

export function NBodyControls() {
  // Use granular selectors to avoid re-rendering on every frame (currentFrame change)
  const isRealTime = useSimulationStore((state) => state.nbody.isRealTime);
  const numParticles = useSimulationStore((state) => state.nbody.numParticles);
  const numSnapshots = useSimulationStore((state) => state.nbody.numSnapshots);
  const deltaT = useSimulationStore((state) => state.nbody.deltaT);

  const setNBodySnapshots = useSimulationStore((state) => state.setNBodySnapshots);
  const setNBodyPlaying = useSimulationStore((state) => state.setNBodyPlaying);
  const setNBodyNumParticles = useSimulationStore((state) => state.setNBodyNumParticles);
  const setNBodyNumSnapshots = useSimulationStore((state) => state.setNBodyNumSnapshots);
  const setNBodyDeltaT = useSimulationStore((state) => state.setNBodyDeltaT);
  const setNBodyRealTime = useSimulationStore((state) => state.setNBodyRealTime);
  const setNBodySimulationTimestamp = useSimulationStore((state) => state.setNBodySimulationTimestamp);

  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [statusMessage, setStatusMessage] = useState('');
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleGenerate = async () => {
    // Access current state directly to avoid subscribing
    const nbodyState = useSimulationStore.getState().nbody;

    setGenerating(true);
    setProgress(0);
    setProgressMessage('Starting...');
    setStatusMessage('');

    try {
      if (nbodyState.isRealTime) {
        // Real-time mode
        const success = await initRealTimeSimulation({
            numParticles: nbodyState.numParticles,
            numSnapshots: 0, // Not used
            deltaT: nbodyState.deltaT,
            onProgress: (p, msg) => {
                setProgress(p);
                setProgressMessage(msg);
            }
        });

        if (success) {
            setNBodySnapshots([new Float32Array(0)]); // Dummy
            setNBodyPlaying(true);
            setNBodySimulationTimestamp(Date.now());
            setProgressMessage('Simulation Running');
        } else {
             setProgressMessage('Failed to init GPU');
        }

      } else {
        // Offline mode
        const snapshots = await generateNBodyDemo({
            numParticles: nbodyState.numParticles,
            numSnapshots: nbodyState.numSnapshots,
            deltaT: nbodyState.deltaT,
            onProgress: (p, msg) => {
            setProgress(p);
            setProgressMessage(msg);
            },
        });

        setNBodySnapshots(snapshots);
        setProgressMessage('Complete!');
      }
    } catch (error) {
      console.error('Generation failed:', error);
      setProgressMessage('Error: ' + (error as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  const handleBenchmark = async () => {
    // Access current state directly
    const nbodyState = useSimulationStore.getState().nbody;

    if (!nbodyState.isRealTime) {
        alert("Please enable 'Real-time GPU Mode' first.");
        return;
    }

    setGenerating(true);
    setProgress(0);
    setProgressMessage('Initializing Benchmark...');

    try {
        const numParticles = 50000;
        const numFrames = 60;

        const success = await initRealTimeSimulation({
            numParticles,
            numSnapshots: 0,
            deltaT: 0.016,
            onProgress: (p, msg) => {
                setProgress(p);
                setProgressMessage(msg);
            }
        });

        if (!success || !activeSimulation) {
            throw new Error("Failed to init simulation for benchmark");
        }

        setProgressMessage(`Benchmarking ${numParticles.toLocaleString()} particles...`);

        // Run benchmark
        const start = performance.now();
        for (let i = 0; i < numFrames; i++) {
            activeSimulation.step(0.016);
            // We must wait for GPU to finish to measure time accurately?
            // step() submits work. getParticleData() waits for it.
            // So we read back every frame to simulate rendering load + sync.
            await activeSimulation.getParticleData();

            setProgress(((i + 1) / numFrames) * 100);
        }
        const end = performance.now();

        const totalTime = end - start;
        const avgFrameTime = totalTime / numFrames;
        const fps = 1000 / avgFrameTime;

        const msg = `Benchmark Result: ${fps.toFixed(1)} FPS (${avgFrameTime.toFixed(1)}ms/frame)`;
        console.log(msg);
        setStatusMessage(msg);

        // Clean up or leave it running?
        // Let's leave it in a "done" state but maybe not running loop
        setNBodyPlaying(false);

    } catch (e) {
        console.error("Benchmark failed", e);
        setStatusMessage("Benchmark Failed: " + (e as Error).message);
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
        <div className="control-group">
            <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                    type="checkbox"
                    checked={isRealTime}
                    onChange={(e) => setNBodyRealTime(e.target.checked)}
                    style={{ marginRight: '8px' }}
                />
                Real-time GPU Mode
            </label>
        </div>

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
          <div style={{ marginTop: '5px', fontSize: '11px', whiteSpace: 'pre-wrap' }} id="progress-message">{progressMessage}</div>
        </div>
      ) : (
        <div className="demo-notice">
            {statusMessage ? (
                <span id="status-message" style={{ fontWeight: 'bold' }}>{statusMessage}</span>
            ) : (
                isRealTime ? 'Click "Start Simulation" to run in real-time!' : 'Click "Generate Demo" to see particles in motion!'
            )}
        </div>
      )}

      <div className="control-group">
        <label>
          Number of Particles: <span className="value-display">{numParticles.toLocaleString()}</span>
        </label>
        <input
          type="range"
          min="100"
          max={isRealTime ? "1000000" : "50000"}
          step="100"
          value={numParticles}
          onChange={(e) => setNBodyNumParticles(parseInt(e.target.value))}
          disabled={generating}
        />
      </div>

      {!isRealTime && (
        <div className="control-group">
            <label>
            Number of Timesteps: <span className="value-display">{numSnapshots.toLocaleString()}</span>
            </label>
            <input
            type="range"
            min="50"
            max="10000"
            step="50"
            value={numSnapshots}
            onChange={(e) => setNBodyNumSnapshots(parseInt(e.target.value))}
            disabled={generating}
            />
        </div>
      )}

      <div className="control-group">
        <label>
          Delta T (Timestep Size): <span className="value-display">{deltaT.toFixed(3)}</span>
        </label>
        <input
          type="range"
          min="0.001"
          max="0.2"
          step="0.001"
          value={deltaT}
          onChange={(e) => setNBodyDeltaT(parseFloat(e.target.value))}
          disabled={generating}
        />
      </div>

      <div className="control-group" style={{ display: 'flex', gap: '8px' }}>
        <button className="primary" onClick={handleGenerate} disabled={generating} style={{ flex: 2 }}>
          {isRealTime ? 'Start Simulation' : 'Generate Demo Data'}
        </button>
        {isRealTime && (
            <button className="secondary" onClick={handleBenchmark} disabled={generating} style={{ flex: 1, fontSize: '11px', padding: '4px' }}>
                Benchmark
            </button>
        )}
      </div>

      <PlaybackControls />

      <button className="primary" onClick={() => window.location.reload()}>
        Reset View
      </button>
        </>
      )}
    </div>
  );
}
