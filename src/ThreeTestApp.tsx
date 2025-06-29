import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere } from '@react-three/drei';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useTektraStore } from './store';
import './App.css';

// Create QueryClient instance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Simple Three.js scene
const SimpleScene: React.FC = () => {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <Sphere args={[1, 32, 32]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#6366F1" />
      </Sphere>
      <OrbitControls />
    </>
  );
};

const ThreeTestContent: React.FC = () => {
  const { modelStatus } = useTektraStore();

  return (
    <div className="min-h-screen bg-primary-bg text-text-primary overflow-hidden">
      {/* Header Bar */}
      <div className="fixed top-0 left-0 right-0 z-50 h-16 bg-secondary-bg border-b border-border-primary flex items-center justify-between px-6">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
            <span className="text-white font-bold text-sm">T</span>
          </div>
          <h1 className="text-lg font-semibold text-text-primary">Tektra</h1>
        </div>
        <div className="text-text-secondary">Testing Three.js</div>
      </div>

      {/* Main Content */}
      <div className="pt-20 pb-12 px-4">
        <div className="text-center mb-4">
          <h2 className="text-xl font-bold text-text-primary">
            Three.js Integration Test
          </h2>
          <p className="text-text-secondary text-sm">
            Testing React Three Fiber components
          </p>
        </div>

        {/* Three.js Canvas */}
        <div className="h-80 bg-surface rounded-card border border-border-primary overflow-hidden">
          <Canvas
            camera={{ position: [0, 0, 5], fov: 50 }}
            gl={{ antialias: true }}
          >
            <Suspense fallback={null}>
              <SimpleScene />
            </Suspense>
          </Canvas>
        </div>

        <div className="mt-4 text-center">
          <p className="text-sm text-text-secondary">
            If you see a rotating blue sphere above, Three.js is working!
          </p>
        </div>
      </div>

      {/* Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-center px-4 text-xs">
        <span className="text-text-tertiary">Three.js Test - React Three Fiber</span>
      </div>
    </div>
  );
};

const ThreeTestApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThreeTestContent />
    </QueryClientProvider>
  );
};

export default ThreeTestApp;