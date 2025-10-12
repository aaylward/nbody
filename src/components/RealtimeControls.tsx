/**
 * Controls for real-time N-body simulation
 */

import { useState, useEffect } from 'react';
import { useRealtimeStore } from '../store/useRealtimeStore';
import { initGPU } from '../simulation/nbody';
import './NBodyControls.css'; // Reuse existing styles

export function RealtimeControls() {
  const { simulation, isRunning, error, backend, startSimulation, stopSimulation, resetSimulation, setTargetFPS, setBackend, theta, setTheta } = useRealtimeStore();
  const [numParticles, setNumParticles] = useState(50000);
  const [targetPhysicsFPS, setTargetPhysicsFPS] = useState(20);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [gpuDevice, setGPUDevice] = useState<GPUDevice | null>(null);

  // Initialize GPU on mount
  useEffect(() => {
    async function init() {
      const hasGPU = await initGPU();
      if (hasGPU && navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const device = await adapter.requestDevice();
          setGPUDevice(device);
        }
      }
    }
    init();
  }, []);

  const handleStart = async () => {
    if ((backend === 'gpu' || backend === 'gpu-barnes-hut') && !gpuDevice) {
      console.error('GPU not available');
      return;
    }
    await startSimulation(numParticles, gpuDevice || undefined);
  };

  const handleStop = () => {
    stopSimulation();
  };

  const handleReset = () => {
    resetSimulation();
  };

  return (
    <div className={`control-panel ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="control-header">
        <h3>Real-Time Simulation</h3>
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
          {error && (
            <div className="demo-notice" style={{ color: '#ff4444' }}>
              Error: {error}
            </div>
          )}

          {(backend === 'gpu' || backend === 'gpu-barnes-hut') && !gpuDevice && (
            <div className="demo-notice" style={{ color: '#ff8800' }}>
              WebGPU not available. Switch to CPU Barnes-Hut backend.
            </div>
          )}

          {!isRunning && (
            <div className="demo-notice">
              Configure parameters and click "Start" to begin real-time physics!
            </div>
          )}

          {isRunning && backend === 'gpu' && (
            <div className="demo-notice" style={{ color: '#00ff00' }}>
              ✓ Simulation running - GPU Brute-Force O(N²)
            </div>
          )}

          {isRunning && backend === 'gpu-barnes-hut' && (
            <div className="demo-notice" style={{ color: '#00ff00' }}>
              ✓ Simulation running - GPU Barnes-Hut O(N log N) ⚡
            </div>
          )}

          {isRunning && backend === 'cpu-barnes-hut' && (
            <div className="demo-notice" style={{ color: '#00ff00' }}>
              ✓ Simulation running - CPU Barnes-Hut O(N log N)
            </div>
          )}

          <div className="control-group">
            <label>
              Backend:{' '}
              <span className="value-display">
                {backend === 'gpu' ? 'GPU Brute-Force' :
                 backend === 'gpu-barnes-hut' ? 'GPU Barnes-Hut' :
                 'CPU Barnes-Hut'}
              </span>
            </label>
            <select
              value={backend}
              onChange={(e) => setBackend(e.target.value as typeof backend)}
              disabled={isRunning}
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#1a1a1a',
                color: 'white',
                border: '1px solid #333',
                borderRadius: '4px',
                fontSize: '14px',
              }}
            >
              <option value="gpu-barnes-hut">GPU Barnes-Hut O(N log N) - ⚡ Fastest (Phase 3)</option>
              <option value="cpu-barnes-hut">CPU Barnes-Hut O(N log N) - Fast</option>
              <option value="gpu">GPU Brute-Force O(N²) - Slow</option>
            </select>
            <div style={{ fontSize: '11px', color: '#aaa', marginTop: '4px' }}>
              GPU Barnes-Hut: 100-1000x faster for large N!
            </div>
          </div>

          {(backend === 'cpu-barnes-hut' || backend === 'gpu-barnes-hut') && (
            <div className="control-group">
              <label>
                Barnes-Hut Theta (θ):{' '}
                <span className="value-display">{theta.toFixed(2)}</span>
              </label>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.1"
                value={theta}
                onChange={(e) => setTheta(parseFloat(e.target.value))}
              />
              <div style={{ fontSize: '11px', color: '#aaa', marginTop: '4px' }}>
                Lower = more accurate, slower | Higher = faster, less accurate
              </div>
            </div>
          )}

          <div className="control-group">
            <label>
              Number of Particles:{' '}
              <span className="value-display">{numParticles.toLocaleString()}</span>
            </label>
            <input
              type="range"
              min="1000"
              max="100000"
              step="1000"
              value={numParticles}
              onChange={(e) => setNumParticles(parseInt(e.target.value))}
              disabled={isRunning}
            />
            <div style={{ fontSize: '11px', color: '#aaa', marginTop: '4px' }}>
              {backend === 'gpu' ? 'Recommended: <50k (GPU brute-force)' :
               backend === 'gpu-barnes-hut' ? 'Can handle 100k-1M particles!' :
               'CPU: up to 90k particles'}
            </div>
          </div>

          <div className="control-group">
            <label>
              Target Physics FPS:{' '}
              <span className="value-display">{targetPhysicsFPS}</span>
            </label>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={targetPhysicsFPS}
              onChange={(e) => {
                const fps = parseInt(e.target.value);
                setTargetPhysicsFPS(fps);
                if (simulation) {
                  setTargetFPS(fps);
                }
              }}
            />
            <div style={{ fontSize: '11px', color: '#aaa', marginTop: '4px' }}>
              Rendering always at 60 FPS via interpolation
            </div>
          </div>

          <div className="control-group">
            {!isRunning ? (
              <button className="primary" onClick={handleStart} disabled={(backend === 'gpu' || backend === 'gpu-barnes-hut') && !gpuDevice}>
                Start Simulation
              </button>
            ) : (
              <button className="primary" onClick={handleStop}>
                Stop Simulation
              </button>
            )}
          </div>

          <div className="control-group">
            <button className="primary" onClick={handleReset}>
              Reset
            </button>
          </div>
        </>
      )}
    </div>
  );
}
