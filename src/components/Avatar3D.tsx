import React, { useRef, useEffect, useState, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Environment, 
  PerspectiveCamera,
  Html,
  useProgress,
  Sphere,
  MeshDistortMaterial,
} from '@react-three/drei';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import {
  Minimize2,
  Maximize2,
  Volume2,
  VolumeX,
  RotateCcw,
  Settings,
  Camera,
  Monitor,
  Eye,
} from 'lucide-react';
import { useTektraStore } from '../store';

// Avatar Head Component with enhanced animations
const AvatarHead: React.FC<{ 
  expression: string;
  isSpeaking: boolean;
  isListening: boolean;
  eyeTracking: boolean;
  audioLevel?: number;
}> = ({ expression, isSpeaking, isListening, eyeTracking, audioLevel = 0 }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera, mouse } = useThree();
  const [baseDistortion, setBaseDistortion] = useState(0.1);
  
  // Animate the avatar based on state
  useFrame((state) => {
    if (!meshRef.current) return;
    
    // Breathing animation - more subtle and natural
    const breathingScale = 1 + Math.sin(state.clock.elapsedTime * 1.2) * 0.015;
    meshRef.current.scale.setScalar(breathingScale);
    
    // Eye tracking - subtle head movement following mouse
    if (eyeTracking) {
      const targetRotationY = mouse.x * 0.08;
      const targetRotationX = -mouse.y * 0.04;
      
      meshRef.current.rotation.y = THREE.MathUtils.lerp(
        meshRef.current.rotation.y,
        targetRotationY,
        0.03
      );
      meshRef.current.rotation.x = THREE.MathUtils.lerp(
        meshRef.current.rotation.x,
        targetRotationX,
        0.03
      );
    }
    
    // Enhanced lip-sync animation based on audio level
    if (isSpeaking) {
      // Use actual audio level if available, otherwise simulate
      const audioIntensity = audioLevel > 0 ? audioLevel : Math.random() * 0.8 + 0.2;
      const lipSyncDistortion = baseDistortion + audioIntensity * 0.4;
      
      // Add phoneme-like variations for more realistic lip movement
      const phonemeVariation = Math.sin(state.clock.elapsedTime * 12) * 0.1;
      const finalDistortion = lipSyncDistortion + phonemeVariation;
      
      if (meshRef.current.material?.uniforms?.distort) {
        meshRef.current.material.uniforms.distort.value = THREE.MathUtils.lerp(
          meshRef.current.material.uniforms.distort.value,
          finalDistortion,
          0.2
        );
      }
    } else {
      // Return to neutral expression gradually
      if (meshRef.current.material?.uniforms?.distort) {
        meshRef.current.material.uniforms.distort.value = THREE.MathUtils.lerp(
          meshRef.current.material.uniforms.distort.value,
          baseDistortion,
          0.1
        );
      }
    }
    
    // Listening animation with gentle pulsing
    if (isListening) {
      const glowIntensity = 1.5 + Math.sin(state.clock.elapsedTime * 3) * 0.3;
      if (meshRef.current.material?.uniforms?.speed) {
        meshRef.current.material.uniforms.speed.value = glowIntensity;
      }
    } else {
      if (meshRef.current.material?.uniforms?.speed) {
        meshRef.current.material.uniforms.speed.value = THREE.MathUtils.lerp(
          meshRef.current.material.uniforms.speed.value,
          1.0,
          0.05
        );
      }
    }
    
    // Expression-based morphing
    const targetDistortion = getExpressionDistortion(expression);
    setBaseDistortion(prev => THREE.MathUtils.lerp(prev, targetDistortion, 0.02));
  });

  // Get distortion value based on expression
  const getExpressionDistortion = (expr: string) => {
    switch (expr) {
      case 'happy': return 0.15;
      case 'excited': return 0.25;
      case 'surprised': return 0.3;
      case 'thinking': return 0.05;
      case 'concerned': return 0.08;
      case 'friendly': return 0.12;
      default: return 0.1;
    }
  };

  // Expression-based colors
  const getExpressionColor = () => {
    switch (expression) {
      case 'happy': return '#10B981'; // Green
      case 'thinking': return '#F59E0B'; // Orange  
      case 'surprised': return '#EF4444'; // Red
      case 'concerned': return '#8B5CF6'; // Purple
      case 'excited': return '#F59E0B'; // Orange
      case 'friendly': return '#06B6D4'; // Cyan
      default: return '#6366F1'; // Default accent
    }
  };

  return (
    <Sphere ref={meshRef} args={[1, 64, 64]} position={[0, 0, 0]}>
      <MeshDistortMaterial
        color={getExpressionColor()}
        attach="material"
        distort={isSpeaking ? 0.3 : 0.1}
        speed={isListening ? 2 : 1}
        roughness={0.4}
        metalness={0.1}
      />
    </Sphere>
  );
};

// Loading component
const Loader: React.FC = () => {
  const { progress } = useProgress();
  
  return (
    <Html center>
      <div className="flex flex-col items-center space-y-4">
        <div className="w-12 h-12 border-4 border-accent border-t-transparent rounded-full animate-spin" />
        <p className="text-text-secondary text-sm">Loading Avatar... {Math.round(progress)}%</p>
      </div>
    </Html>
  );
};

// Avatar Scene Component with enhanced lighting and effects
const AvatarScene: React.FC = () => {
  const { avatarState } = useTektraStore();
  const controlsRef = useRef();
  const [audioLevel, setAudioLevel] = useState(0);

  // Simulate audio level changes when speaking
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (avatarState.isSpeaking) {
      interval = setInterval(() => {
        // Simulate realistic audio level fluctuations
        const baseLevel = 0.3 + Math.random() * 0.7;
        const variation = Math.sin(Date.now() * 0.01) * 0.2;
        setAudioLevel(Math.max(0, Math.min(1, baseLevel + variation)));
      }, 50); // Update 20 times per second for smooth animation
    } else {
      setAudioLevel(0);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [avatarState.isSpeaking]);

  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 0, 4]} fov={50} />
      <OrbitControls
        ref={controlsRef}
        enableZoom={true}
        enablePan={false}
        enableRotate={true}
        minDistance={2}
        maxDistance={8}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 1.5}
        autoRotate={avatarState.animation.autoRotate}
        autoRotateSpeed={0.3}
      />
      
      {/* Enhanced Lighting Setup */}
      <ambientLight intensity={0.3} />
      <pointLight 
        position={[5, 5, 5]} 
        intensity={1.2} 
        color="#ffffff" 
        castShadow
      />
      <pointLight 
        position={[-5, -5, -5]} 
        intensity={0.6} 
        color="#6366F1"
      />
      
      {/* Rim lighting for depth */}
      <pointLight 
        position={[0, 0, -8]} 
        intensity={0.8} 
        color="#818CF8"
      />
      
      {/* Environment */}
      <Environment preset="studio" />
      
      {/* Avatar with enhanced props */}
      <AvatarHead
        expression={avatarState.expression}
        isSpeaking={avatarState.isSpeaking}
        isListening={avatarState.isListening}
        eyeTracking={avatarState.animation.eyeTracking}
        audioLevel={audioLevel}
      />
      
      {/* Background gradient sphere with dynamic opacity */}
      <Sphere args={[20, 32, 32]} position={[0, 0, -10]}>
        <meshBasicMaterial 
          color="#13141A" 
          side={THREE.BackSide}
          transparent
          opacity={avatarState.isListening ? 0.9 : 0.7}
        />
      </Sphere>
      
      {/* Particle effects when speaking */}
      {avatarState.isSpeaking && (
        <group>
          {Array.from({ length: 20 }).map((_, i) => (
            <Sphere
              key={i}
              args={[0.02, 8, 8]}
              position={[
                (Math.random() - 0.5) * 6,
                (Math.random() - 0.5) * 6,
                (Math.random() - 0.5) * 6
              ]}
            >
              <meshBasicMaterial 
                color="#6366F1" 
                transparent 
                opacity={0.1 + Math.random() * 0.3} 
              />
            </Sphere>
          ))}
        </group>
      )}
    </>
  );
};

// Avatar Controls Overlay
const AvatarControls: React.FC<{
  isMinimized: boolean;
  onToggleMinimize: () => void;
  onReset: () => void;
  muteAnimations: boolean;
  onToggleMute: () => void;
}> = ({ isMinimized, onToggleMinimize, onReset, muteAnimations, onToggleMute }) => {
  const [showControls, setShowControls] = useState(false);

  return (
    <motion.div
      className="absolute top-4 right-4 z-10"
      initial={{ opacity: 0 }}
      animate={{ opacity: showControls ? 1 : 0 }}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      <div className="flex space-x-2">
        {/* Reset View */}
        <button
          onClick={onReset}
          className="p-2 glass-dark rounded-button hover:bg-white/20 transition-colors"
          title="Reset view"
        >
          <RotateCcw className="w-4 h-4 text-white" />
        </button>

        {/* Mute Animations */}
        <button
          onClick={onToggleMute}
          className="p-2 glass-dark rounded-button hover:bg-white/20 transition-colors"
          title={muteAnimations ? "Enable animations" : "Disable animations"}
        >
          {muteAnimations ? (
            <VolumeX className="w-4 h-4 text-white" />
          ) : (
            <Volume2 className="w-4 h-4 text-white" />
          )}
        </button>

        {/* Minimize/Maximize */}
        <button
          onClick={onToggleMinimize}
          className="p-2 glass-dark rounded-button hover:bg-white/20 transition-colors"
          title={isMinimized ? "Expand avatar" : "Minimize avatar"}
        >
          {isMinimized ? (
            <Maximize2 className="w-4 h-4 text-white" />
          ) : (
            <Minimize2 className="w-4 h-4 text-white" />
          )}
        </button>
      </div>
    </motion.div>
  );
};

// Enhanced Status Indicators
const StatusIndicators: React.FC<{ audioLevel?: number }> = ({ audioLevel = 0 }) => {
  const { avatarState } = useTektraStore();

  return (
    <div className="absolute bottom-4 left-4 flex flex-col space-y-2">
      <div className="flex space-x-2">
        {/* Speaking Indicator with Audio Level */}
        {avatarState.isSpeaking && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            className="flex items-center space-x-2 px-3 py-1.5 glass-dark rounded-full"
          >
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
              {/* Audio level bars */}
              <div className="flex space-x-0.5">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-0.5 h-2 rounded-full transition-all duration-100 ${
                      audioLevel > (i + 1) * 0.2 
                        ? 'bg-success' 
                        : 'bg-success/20'
                    }`}
                    style={{
                      height: audioLevel > (i + 1) * 0.2 
                        ? `${8 + (audioLevel * 8)}px` 
                        : '4px'
                    }}
                  />
                ))}
              </div>
            </div>
            <span className="text-xs text-white">Speaking</span>
          </motion.div>
        )}

        {/* Listening Indicator */}
        {avatarState.isListening && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            className="flex items-center space-x-2 px-3 py-1.5 glass-dark rounded-full"
          >
            <div className="w-2 h-2 bg-accent rounded-full animate-ping" />
            <span className="text-xs text-white">Listening</span>
          </motion.div>
        )}

        {/* Expression Indicator */}
        <motion.div 
          className="flex items-center space-x-2 px-3 py-1.5 glass-dark rounded-full"
          animate={{ 
            scale: avatarState.isSpeaking || avatarState.isListening ? 1.05 : 1 
          }}
          transition={{ duration: 0.2 }}
        >
          <div className={`w-2 h-2 rounded-full ${getExpressionColor(avatarState.expression)}`} />
          <span className="text-xs text-white capitalize">{avatarState.expression}</span>
        </motion.div>
      </div>

      {/* Eye Tracking Indicator */}
      {avatarState.animation.eyeTracking && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center space-x-2 px-2 py-1 glass-dark rounded-full"
        >
          <Eye className="w-3 h-3 text-accent" />
          <span className="text-xs text-white/70">Eye Tracking</span>
        </motion.div>
      )}
    </div>
  );

  function getExpressionColor(expression: string) {
    switch (expression) {
      case 'happy': return 'bg-green-400';
      case 'thinking': return 'bg-yellow-400';
      case 'surprised': return 'bg-red-400';
      case 'concerned': return 'bg-purple-400';
      case 'excited': return 'bg-orange-400';
      case 'friendly': return 'bg-cyan-400';
      default: return 'bg-accent';
    }
  }
};

// Main Avatar Container
interface Avatar3DProps {
  className?: string;
}

const Avatar3D: React.FC<Avatar3DProps> = ({ className = '' }) => {
  const { avatarState, setAvatarState } = useTektraStore();
  const [muteAnimations, setMuteAnimations] = useState(false);
  const [currentAudioLevel, setCurrentAudioLevel] = useState(0);
  
  const containerHeight = avatarState.isMinimized ? 'h-20' : 'h-80';

  // Update audio level for parent component access
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (avatarState.isSpeaking) {
      interval = setInterval(() => {
        const baseLevel = 0.3 + Math.random() * 0.7;
        const variation = Math.sin(Date.now() * 0.01) * 0.2;
        setCurrentAudioLevel(Math.max(0, Math.min(1, baseLevel + variation)));
      }, 50);
    } else {
      setCurrentAudioLevel(0);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [avatarState.isSpeaking]);

  const handleToggleMinimize = () => {
    setAvatarState({ isMinimized: !avatarState.isMinimized });
  };

  const handleReset = () => {
    // Reset camera position - would need to implement with controls ref
    console.log('Reset avatar view');
  };

  const handleToggleMute = () => {
    setMuteAnimations(!muteAnimations);
  };

  return (
    <motion.div
      animate={{ height: avatarState.isMinimized ? 80 : 320 }}
      transition={{ duration: 0.3 }}
      className={`
        relative bg-gradient-to-br from-secondary-bg to-surface
        border border-border-primary rounded-card overflow-hidden
        shadow-avatar
        ${className}
      `}
    >
      {/* Avatar Canvas */}
      {avatarState.isVisible && !avatarState.isMinimized && (
        <Canvas
          className="w-full h-full"
          gl={{ 
            antialias: true, 
            alpha: true,
            powerPreference: "high-performance"
          }}
          dpr={Math.min(window.devicePixelRatio, 2)}
          onCreated={(state) => {
            state.gl.setClearColor('#13141A');
          }}
        >
          <Suspense fallback={<Loader />}>
            <AvatarScene />
          </Suspense>
        </Canvas>
      )}

      {/* Minimized Avatar */}
      {avatarState.isMinimized && (
        <div className="flex items-center justify-between h-full px-4">
          <div className="flex items-center space-x-3">
            <div className={`
              w-12 h-12 rounded-full bg-gradient-to-br from-accent to-accent-light
              flex items-center justify-center
              ${avatarState.isSpeaking ? 'animate-pulse' : ''}
            `}>
              <span className="text-lg">🤖</span>
            </div>
            <div>
              <p className="text-sm font-medium text-text-primary capitalize">
                {avatarState.expression}
              </p>
              <p className="text-xs text-text-tertiary">
                {avatarState.isSpeaking ? 'Speaking...' : 
                 avatarState.isListening ? 'Listening...' : 'Ready'}
              </p>
            </div>
          </div>
          
          <button
            onClick={handleToggleMinimize}
            className="p-2 hover:bg-surface-hover rounded-button transition-colors"
          >
            <Maximize2 className="w-4 h-4 text-text-secondary" />
          </button>
        </div>
      )}

      {/* Controls Overlay */}
      {!avatarState.isMinimized && (
        <div 
          className="absolute inset-0"
          onMouseEnter={() => {}}
          onMouseLeave={() => {}}
        >
          <AvatarControls
            isMinimized={avatarState.isMinimized}
            onToggleMinimize={handleToggleMinimize}
            onReset={handleReset}
            muteAnimations={muteAnimations}
            onToggleMute={handleToggleMute}
          />
          
          <StatusIndicators audioLevel={currentAudioLevel} />
        </div>
      )}

      {/* Performance Info */}
      <div className="absolute top-2 left-2 text-xs text-text-tertiary font-mono opacity-50">
        60fps
      </div>
    </motion.div>
  );
};

export default Avatar3D;