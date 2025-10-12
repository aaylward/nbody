import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import { useSimulationStore } from './store/useSimulationStore';
import { NBodyVisualization } from './components/NBodyVisualization';
import { RealtimeVisualization } from './components/RealtimeVisualization';
import { MonteCarloVisualization } from './components/MonteCarloVisualization';
import { TopBar } from './components/TopBar';
import { NBodyControls } from './components/NBodyControls';
import { RealtimeControls } from './components/RealtimeControls';
import { PerformanceStats } from './components/PerformanceStats';
import { MonteCarloControls } from './components/MonteCarloControls';
import { StatsPanel } from './components/StatsPanel';
import './App.css';

function App() {
  const mode = useSimulationStore((state) => state.mode);

  const isNBody = mode === 'nbody' || mode === 'nbody-realtime';

  return (
    <div className="viewer-container">
      <TopBar />

      <div className="canvas-area">
        <Canvas
          camera={{ position: isNBody ? [0, 0, 200] : [150, 150, 150], fov: 75 }}
        >
          <color attach="background" args={['#000510']} />
          <ambientLight intensity={2} />
          <directionalLight position={[1, 1, 1]} intensity={0.5} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          <Stats />

          {mode === 'nbody' && <NBodyVisualization />}
          {mode === 'nbody-realtime' && <RealtimeVisualization />}
          {mode === 'montecarlo' && <MonteCarloVisualization />}
        </Canvas>

        {mode === 'nbody' && <NBodyControls />}
        {mode === 'nbody-realtime' && (
          <>
            <RealtimeControls />
            <PerformanceStats />
          </>
        )}
        {mode === 'montecarlo' && <MonteCarloControls />}

        {mode !== 'nbody-realtime' && <StatsPanel />}
      </div>
    </div>
  );
}

export default App;
