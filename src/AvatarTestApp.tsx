import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sphere, MeshDistortMaterial } from '@react-three/drei';
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

// Simple Avatar Head (without the complex animations)
const SimpleAvatarHead: React.FC = () => {
  return (
    <Sphere args={[1, 64, 64]} position={[0, 0, 0]}>
      <MeshDistortMaterial
        color="#6366F1"
        attach="material"
        distort={0.2}
        speed={1}
        roughness={0.4}
        metalness={0.1}
      />
    </Sphere>
  );
};

// Simple Avatar Scene
const SimpleAvatarScene: React.FC = () => {
  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#6366F1" />
      
      <SimpleAvatarHead />
      
      <OrbitControls
        enableZoom={true}
        enablePan={false}
        enableRotate={true}
        minDistance={2}
        maxDistance={8}
      />
    </>
  );
};

const AvatarTestContent: React.FC = () => {
  const { avatarState } = useTektraStore();

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
        <div className="text-text-secondary">Testing Avatar</div>
      </div>

      {/* Main Content */}
      <div className="pt-20 pb-12 px-4">
        <div className="text-center mb-4">
          <h2 className="text-xl font-bold text-text-primary">
            Avatar Component Test
          </h2>
          <p className="text-text-secondary text-sm">
            Testing simplified Avatar3D component
          </p>
        </div>

        {/* Avatar Canvas */}
        <div className="h-80 bg-gradient-to-br from-secondary-bg to-surface rounded-card border border-border-primary overflow-hidden">
          <Canvas
            camera={{ position: [0, 0, 4], fov: 50 }}
            gl={{ 
              antialias: true, 
              alpha: true,
              powerPreference: "high-performance"
            }}
            onCreated={(state) => {
              state.gl.setClearColor('#13141A');
            }}
          >
            <Suspense fallback={null}>
              <SimpleAvatarScene />
            </Suspense>
          </Canvas>
        </div>

        <div className="mt-4 text-center">
          <p className="text-sm text-text-secondary">
            Avatar Expression: <span className="text-accent capitalize">{avatarState.expression}</span>
          </p>
          <p className="text-xs text-text-tertiary mt-2">
            If you see a distorted blue sphere above, the Avatar component is working!
          </p>
        </div>
      </div>

      {/* Status Bar */}
      <div className="fixed bottom-0 left-0 right-0 z-50 h-8 bg-primary-bg border-t border-border-primary flex items-center justify-center px-4 text-xs">
        <span className="text-text-tertiary">Avatar Test - Simplified 3D Avatar</span>
      </div>
    </div>
  );
};

const AvatarTestApp: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <AvatarTestContent />
    </QueryClientProvider>
  );
};

export default AvatarTestApp;