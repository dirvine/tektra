"use client"

import React, { Suspense, useRef, useEffect, useState, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Environment, Html, Text3D, Sphere, Box } from '@react-three/drei'
import * as THREE from 'three'

interface AvatarRendererProps {
  expression?: string
  speaking?: boolean
  audioData?: Float32Array
  gesture?: string
  className?: string
  size?: { width: number; height: number }
  style?: string
  gender?: string
}

interface VisemeData {
  viseme: string
  time: number
  intensity: number
}

interface AvatarMeshProps {
  expression: string
  speaking: boolean
  audioData?: Float32Array
  gesture: string
  style: string
  gender: string
}

// Expression morph targets mapping
const EXPRESSION_MORPHS = {
  neutral: { mouth: 0, eyebrows: 0, eyes: 0, cheeks: 0 },
  happy: { mouth: 0.8, eyebrows: 0.3, eyes: 0.2, cheeks: 0.6 },
  sad: { mouth: -0.5, eyebrows: -0.4, eyes: -0.3, cheeks: -0.2 },
  angry: { mouth: -0.3, eyebrows: -0.8, eyes: 0.4, cheeks: 0 },
  surprised: { mouth: 0.6, eyebrows: 0.9, eyes: 0.8, cheeks: 0.2 },
  confused: { mouth: -0.2, eyebrows: -0.2, eyes: 0.1, cheeks: 0 },
  thinking: { mouth: 0.1, eyebrows: 0.2, eyes: 0.3, cheeks: 0 },
  excited: { mouth: 0.9, eyebrows: 0.5, eyes: 0.6, cheeks: 0.8 },
  calm: { mouth: 0.1, eyebrows: 0.1, eyes: 0, cheeks: 0.1 },
  focused: { mouth: 0, eyebrows: 0.4, eyes: 0.2, cheeks: 0 },
  worried: { mouth: -0.3, eyebrows: -0.5, eyes: -0.1, cheeks: -0.1 },
  relaxed: { mouth: 0.2, eyebrows: -0.1, eyes: -0.1, cheeks: 0.2 },
  determined: { mouth: 0.1, eyebrows: 0.6, eyes: 0.3, cheeks: 0.1 },
  friendly: { mouth: 0.5, eyebrows: 0.2, eyes: 0.1, cheeks: 0.4 }
}

// Gesture animations
const GESTURE_ANIMATIONS = {
  idle: { armRotation: [0, 0, 0], headRotation: [0, 0, 0] },
  wave: { armRotation: [0, 0, 0.5], headRotation: [0, 0.2, 0] },
  nod: { armRotation: [0, 0, 0], headRotation: [0.3, 0, 0] },
  shake_head: { armRotation: [0, 0, 0], headRotation: [0, 0.3, 0] },
  point: { armRotation: [0, 0, 0.8], headRotation: [0, 0.1, 0] },
  thumbs_up: { armRotation: [0, 0, 0.2], headRotation: [0, 0, 0] },
  shrug: { armRotation: [0.2, 0, 0.3], headRotation: [0, 0, 0.1] },
  thinking_pose: { armRotation: [0.1, 0, 0], headRotation: [0.2, 0.1, 0] }
}

// Lip-sync viseme mapping
const VISEME_MORPHS = {
  sil: { mouth: 0 },     // Silence
  aa: { mouth: 0.8 },    // "ah" 
  ae: { mouth: 0.6 },    // "cat"
  ah: { mouth: 0.7 },    // "father"
  ao: { mouth: 0.5 },    // "off" 
  aw: { mouth: 0.4 },    // "cow"
  ay: { mouth: 0.3 },    // "hide"
  b: { mouth: 0.1 },     // "bee"
  ch: { mouth: 0.2 },    // "cheese"
  d: { mouth: 0.2 },     // "dee"
  dh: { mouth: 0.3 },    // "thee"
  eh: { mouth: 0.4 },    // "red"
  er: { mouth: 0.3 },    // "bird"
  ey: { mouth: 0.2 },    // "ate"
  f: { mouth: 0.2 },     // "fee"
  g: { mouth: 0.3 },     // "green"
  hh: { mouth: 0.1 },    // "hee"
  ih: { mouth: 0.2 },    // "bit"
  iy: { mouth: 0.1 },    // "beat"
  jh: { mouth: 0.3 },    // "gee"
  k: { mouth: 0.2 },     // "key"
  l: { mouth: 0.3 },     // "lee"
  m: { mouth: 0.0 },     // "me"
  n: { mouth: 0.2 },     // "knee"
  ng: { mouth: 0.2 },    // "ping"
  ow: { mouth: 0.6 },    // "boat"
  oy: { mouth: 0.5 },    // "toy"
  p: { mouth: 0.0 },     // "pee"
  r: { mouth: 0.4 },     // "read"
  s: { mouth: 0.1 },     // "sea"
  sh: { mouth: 0.2 },    // "she"
  t: { mouth: 0.2 },     // "tea"
  th: { mouth: 0.3 },    // "think"
  uh: { mouth: 0.3 },    // "book"
  uw: { mouth: 0.6 },    // "cool"
  v: { mouth: 0.2 },     // "vee"
  w: { mouth: 0.6 },     // "we"
  y: { mouth: 0.1 },     // "yes"
  z: { mouth: 0.1 },     // "zee"
  zh: { mouth: 0.2 }     // "leisure"
}

// Simple procedural avatar mesh
function AvatarMesh({ expression, speaking, audioData, gesture, style, gender }: AvatarMeshProps) {
  const meshRef = useRef<THREE.Group>(null)
  const headRef = useRef<THREE.Mesh>(null)
  const mouthRef = useRef<THREE.Mesh>(null)
  const eyesRef = useRef<THREE.Group>(null)
  const [currentViseme, setCurrentViseme] = useState<string>('sil')
  const [gestureAnimation, setGestureAnimation] = useState(0)

  // Get expression morph values
  const expressionMorphs = EXPRESSION_MORPHS[expression as keyof typeof EXPRESSION_MORPHS] || EXPRESSION_MORPHS.neutral
  const gestureMorphs = GESTURE_ANIMATIONS[gesture as keyof typeof GESTURE_ANIMATIONS] || GESTURE_ANIMATIONS.idle

  // Animate gestures and expressions
  useFrame((state, delta) => {
    if (!meshRef.current || !headRef.current) return

    // Apply gesture animations
    if (gesture !== 'idle') {
      setGestureAnimation(prev => Math.min(prev + delta * 2, 1))
    } else {
      setGestureAnimation(prev => Math.max(prev - delta * 2, 0))
    }

    // Animate head rotation for gestures
    const targetHeadRotation = gestureMorphs.headRotation
    if (headRef.current) {
      headRef.current.rotation.x = THREE.MathUtils.lerp(
        headRef.current.rotation.x, 
        targetHeadRotation[0] * gestureAnimation, 
        delta * 3
      )
      headRef.current.rotation.y = THREE.MathUtils.lerp(
        headRef.current.rotation.y, 
        targetHeadRotation[1] * gestureAnimation, 
        delta * 3
      )
      headRef.current.rotation.z = THREE.MathUtils.lerp(
        headRef.current.rotation.z, 
        targetHeadRotation[2] * gestureAnimation, 
        delta * 3
      )
    }

    // Simulate lip-sync from audio data
    if (speaking && audioData) {
      const volume = audioData.reduce((sum, val) => sum + Math.abs(val), 0) / audioData.length
      const normalizedVolume = Math.min(volume * 10, 1)
      
      // Simple volume-based mouth animation
      if (mouthRef.current) {
        const mouthScale = 1 + normalizedVolume * 0.3
        mouthRef.current.scale.setScalar(mouthScale)
      }
    } else if (mouthRef.current) {
      // Return to neutral mouth position
      mouthRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), delta * 5)
    }

    // Idle breathing animation
    const breathe = Math.sin(state.clock.elapsedTime * 2) * 0.02
    meshRef.current.position.y = breathe
  })

  // Color scheme based on gender and style
  const avatarColors = useMemo(() => {
    const baseColors = {
      masculine: {
        skin: '#E8B899',
        hair: '#8B4513',
        eyes: '#4169E1'
      },
      feminine: {
        skin: '#F5DEB3',
        hair: '#8B4513',
        eyes: '#228B22'
      },
      neutral: {
        skin: '#DEB887',
        hair: '#696969',
        eyes: '#708090'
      }
    }

    return baseColors[gender as keyof typeof baseColors] || baseColors.neutral
  }, [gender])

  return (
    <group ref={meshRef} position={[0, -0.5, 0]}>
      {/* Head */}
      <mesh ref={headRef} position={[0, 0.8, 0]}>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color={avatarColors.skin} />
      </mesh>

      {/* Eyes */}
      <group ref={eyesRef} position={[0, 0.9, 0.3]}>
        <mesh position={[-0.15, 0, 0]}>
          <sphereGeometry args={[0.08, 16, 16]} />
          <meshStandardMaterial color={avatarColors.eyes} />
        </mesh>
        <mesh position={[0.15, 0, 0]}>
          <sphereGeometry args={[0.08, 16, 16]} />
          <meshStandardMaterial color={avatarColors.eyes} />
        </mesh>
        
        {/* Eyebrows */}
        <mesh position={[-0.15, 0.1 + expressionMorphs.eyebrows * 0.1, 0]}>
          <boxGeometry args={[0.2, 0.05, 0.05]} />
          <meshStandardMaterial color={avatarColors.hair} />
        </mesh>
        <mesh position={[0.15, 0.1 + expressionMorphs.eyebrows * 0.1, 0]}>
          <boxGeometry args={[0.2, 0.05, 0.05]} />
          <meshStandardMaterial color={avatarColors.hair} />
        </mesh>
      </group>

      {/* Mouth */}
      <mesh ref={mouthRef} position={[0, 0.65, 0.35]}>
        <sphereGeometry args={[0.1, 16, 8]} />
        <meshStandardMaterial color="#FF6B6B" />
      </mesh>

      {/* Hair */}
      <mesh position={[0, 1.2, 0]}>
        <sphereGeometry args={[0.55, 16, 16]} />
        <meshStandardMaterial color={avatarColors.hair} />
      </mesh>

      {/* Body */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.3, 0.4, 0.8, 16]} />
        <meshStandardMaterial color="#4A90E2" />
      </mesh>

      {/* Arms */}
      <group position={[0, 0.2, 0]} rotation={gestureMorphs.armRotation.map(r => r * gestureAnimation) as [number, number, number]}>
        <mesh position={[-0.6, 0, 0]}>
          <cylinderGeometry args={[0.08, 0.08, 0.6, 8]} />
          <meshStandardMaterial color={avatarColors.skin} />
        </mesh>
        <mesh position={[0.6, 0, 0]}>
          <cylinderGeometry args={[0.08, 0.08, 0.6, 8]} />
          <meshStandardMaterial color={avatarColors.skin} />
        </mesh>
      </group>
    </group>
  )
}

// Loading component
function LoadingAvatar() {
  return (
    <Html center>
      <div className="flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-sm text-gray-600">Loading avatar...</span>
      </div>
    </Html>
  )
}

// Main avatar renderer component
export function AvatarRenderer({
  expression = 'neutral',
  speaking = false,
  audioData,
  gesture = 'idle',
  className = '',
  size = { width: 400, height: 400 },
  style = 'realistic',
  gender = 'neutral'
}: AvatarRendererProps) {
  return (
    <div 
      className={`${className} border rounded-lg overflow-hidden bg-gradient-to-b from-blue-50 to-blue-100`}
      style={{ width: size.width, height: size.height }}
    >
      <Canvas
        camera={{ position: [0, 0, 3], fov: 50 }}
        style={{ width: '100%', height: '100%' }}
      >
        <Suspense fallback={<LoadingAvatar />}>
          {/* Lighting */}
          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <pointLight position={[-10, -10, -5]} intensity={0.5} />
          
          {/* Environment */}
          <Environment preset="studio" />
          
          {/* Avatar */}
          <AvatarMesh
            expression={expression}
            speaking={speaking}
            audioData={audioData}
            gesture={gesture}
            style={style}
            gender={gender}
          />
          
          {/* Controls */}
          <OrbitControls
            enableZoom={false}
            enablePan={false}
            maxPolarAngle={Math.PI / 2}
            minPolarAngle={Math.PI / 3}
          />
        </Suspense>
      </Canvas>
      
      {/* Status overlay */}
      <div className="absolute top-2 left-2 text-xs bg-black/20 text-white px-2 py-1 rounded">
        {expression} {speaking && '• Speaking'} {gesture !== 'idle' && `• ${gesture}`}
      </div>
    </div>
  )
}

export default AvatarRenderer