import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stats } from '@react-three/drei';
import { useSimulationStore } from './store/useSimulationStore';
import { NBodyVisualization } from './components/NBodyVisualization';
import { RealtimeVisualization } from './components/RealtimeVisualization';
import { TopBar } from './components/TopBar';
import { NBodyControls } from './components/NBodyControls';
import { RealtimeControls } from './components/RealtimeControls';
import { PerformanceStats } from './components/PerformanceStats';
import { StatsPanel } from './components/StatsPanel';
import './App.css';

function App() {
  const mode = useSimulationStore((state) => state.mode);

  return (
    <div className="viewer-container">
      <TopBar />

      <div className="canvas-area">
        <Canvas camera={{ position: [0, 0, 200], fov: 75 }}>
          <color attach="background" args={['#000510']} />
          <ambientLight intensity={2} />
          <directionalLight position={[1, 1, 1]} intensity={0.5} />
          <OrbitControls enableDamping dampingFactor={0.05} />
          <Stats />

          {mode === 'nbody' && <NBodyVisualization />}
          {mode === 'nbody-realtime' && <RealtimeVisualization />}
        </Canvas>

        {mode === 'nbody' && <NBodyControls />}
        {mode === 'nbody-realtime' && (
          <>
            <RealtimeControls />
            <PerformanceStats />
          </>
        )}

        {mode !== 'nbody-realtime' && <StatsPanel />}
      </div>
    </div>
  );
}

export default App;
