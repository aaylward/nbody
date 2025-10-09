import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useSimulationStore } from './store/useSimulationStore';
import { NBodyVisualization } from './components/NBodyVisualization';
import { MonteCarloVisualization } from './components/MonteCarloVisualization';
import { TopBar } from './components/TopBar';
import { NBodyControls } from './components/NBodyControls';
import { MonteCarloControls } from './components/MonteCarloControls';
import { StatsPanel } from './components/StatsPanel';
import './App.css';

function App() {
  const mode = useSimulationStore((state) => state.mode);

  return (
    <div className="viewer-container">
      <TopBar />

      <div className="canvas-area">
        <Canvas
          camera={{ position: mode === 'nbody' ? [0, 0, 200] : [150, 150, 150], fov: 75 }}
        >
          <color attach="background" args={['#000510']} />
          <ambientLight intensity={2} />
          <directionalLight position={[1, 1, 1]} intensity={0.5} />
          <OrbitControls enableDamping dampingFactor={0.05} />

          {mode === 'nbody' ? <NBodyVisualization /> : <MonteCarloVisualization />}
        </Canvas>

        {mode === 'nbody' ? <NBodyControls /> : <MonteCarloControls />}
        <StatsPanel />
      </div>
    </div>
  );
}

export default App;
