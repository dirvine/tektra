"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Camera, 
  CameraOff, 
  Mic, 
  MicOff, 
  UserPlus, 
  LogIn, 
  Shield, 
  Eye,
  EyeOff,
  CheckCircle,
  AlertCircle,
  Square
} from 'lucide-react'

interface BiometricAuthProps {
  mode: 'register' | 'login'
  onAuthSuccess?: (userData: any) => void
  onAuthFailed?: (error: string) => void
  className?: string
}

interface AuthStep {
  id: string
  title: string
  description: string
  completed: boolean
  current: boolean
}

export default function BiometricAuth({ 
  mode, 
  onAuthSuccess, 
  onAuthFailed, 
  className 
}: BiometricAuthProps) {
  // State
  const [currentStep, setCurrentStep] = useState(0)
  const [userId, setUserId] = useState('')
  const [pin, setPin] = useState('')
  const [showPin, setShowPin] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  
  // Camera state
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [faceDetected, setFaceDetected] = useState(false)
  const [faceCapture, setFaceCapture] = useState<string | null>(null)
  
  // Voice state
  const [isRecording, setIsRecording] = useState(false)
  const [voiceCapture, setVoiceCapture] = useState<Blob | null>(null)
  const [recordingDuration, setRecordingDuration] = useState(0)
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null)
  
  // Steps definition
  const registerSteps: AuthStep[] = [
    {
      id: 'setup',
      title: 'Setup Account',
      description: 'Enter your user ID and PIN',
      completed: false,
      current: true
    },
    {
      id: 'face',
      title: 'Face Recognition',
      description: 'Capture your face for identification',
      completed: false,
      current: false
    },
    {
      id: 'voice',
      title: 'Voice Recognition',
      description: 'Record your voice for verification',
      completed: false,
      current: false
    },
    {
      id: 'complete',
      title: 'Complete',
      description: 'Finalize your secure account',
      completed: false,
      current: false
    }
  ]
  
  const loginSteps: AuthStep[] = [
    {
      id: 'capture',
      title: 'Biometric Capture',
      description: 'Capture face and voice for authentication',
      completed: false,
      current: true
    },
    {
      id: 'verify',
      title: 'Verify PIN',
      description: 'Enter your PIN to unlock vault',
      completed: false,
      current: false
    },
    {
      id: 'complete',
      title: 'Complete',
      description: 'Access your secure vault',
      completed: false,
      current: false
    }
  ]
  
  const steps = mode === 'register' ? registerSteps : loginSteps
  
  // Initialize camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      setIsCameraOn(true)
      setError('')
      
      // Start face detection
      startFaceDetection()
    } catch (error) {
      console.error('Camera access failed:', error)
      setError('Could not access camera. Please check permissions.')
    }
  }
  
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    setIsCameraOn(false)
    setFaceDetected(false)
  }
  
  // Face detection (simplified - in production would use face detection library)
  const startFaceDetection = () => {
    const detectFace = () => {
      if (videoRef.current && isCameraOn) {
        // Simplified face detection simulation
        setFaceDetected(Math.random() > 0.3) // Simulate face detection
        setTimeout(detectFace, 500)
      }
    }
    detectFace()
  }
  
  const captureFace = () => {
    if (!videoRef.current || !canvasRef.current || !faceDetected) return
    
    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    if (!context) return
    
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    context.drawImage(videoRef.current, 0, 0)
    
    const imageData = canvas.toDataURL('image/jpeg', 0.8)
    setFaceCapture(imageData)
    setSuccess('Face captured successfully!')
    
    // Update step progress
    steps[currentStep].completed = true
    if (mode === 'register') {
      setCurrentStep(2) // Move to voice step
    }
  }
  
  // Voice recording
  const startVoiceRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      
      const audioChunks: Blob[] = []
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data)
      }
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
        setVoiceCapture(audioBlob)
        setSuccess('Voice sample captured!')
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())
        
        // Clear timer
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current)
        }
        
        // Update step progress
        steps[currentStep].completed = true
        if (mode === 'register') {
          setCurrentStep(3) // Move to completion
        }
      }
      
      mediaRecorder.start()
      setIsRecording(true)
      setRecordingDuration(0)
      setError('')
      
      // Start timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration(prev => {
          if (prev >= 5) { // Auto-stop after 5 seconds
            stopVoiceRecording()
            return prev
          }
          return prev + 0.1
        })
      }, 100)
      
    } catch (error) {
      console.error('Microphone access failed:', error)
      setError('Could not access microphone. Please check permissions.')
    }
  }
  
  const stopVoiceRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      setRecordingDuration(0)
      
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current)
      }
    }
  }
  
  // Convert blob to base64
  const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        resolve(result.split(',')[1]) // Remove data URL prefix
      }
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }
  
  // Submit authentication
  const submitAuth = async () => {
    if (!faceCapture || !voiceCapture || !pin.trim()) {
      setError('Please complete all authentication steps')
      return
    }
    
    if (mode === 'register' && !userId.trim()) {
      setError('Please enter a user ID')
      return
    }
    
    try {
      setIsLoading(true)
      setProgress(0)
      setError('')
      
      // Convert captures to base64
      const faceImageData = faceCapture.split(',')[1]
      const voiceAudioData = await blobToBase64(voiceCapture)
      
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90))
      }, 200)
      
      // Prepare request
      const endpoint = mode === 'register' ? '/api/v1/security/register' : '/api/v1/security/authenticate'
      const payload = mode === 'register' 
        ? {
            user_id: userId,
            face_image: faceImageData,
            voice_audio: voiceAudioData,
            pin: pin
          }
        : {
            face_image: faceImageData,
            voice_audio: voiceAudioData,
            pin: pin
          }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })
      
      clearInterval(progressInterval)
      setProgress(100)
      
      const result = await response.json()
      
      if (result.success) {
        setSuccess(mode === 'register' ? 'Account created successfully!' : 'Authentication successful!')
        steps[steps.length - 1].completed = true
        
        if (onAuthSuccess) {
          onAuthSuccess(result)
        }
      } else {
        setError(result.error || 'Authentication failed')
        if (onAuthFailed) {
          onAuthFailed(result.error || 'Authentication failed')
        }
      }
      
    } catch (error) {
      console.error('Authentication error:', error)
      setError('Network error occurred')
      if (onAuthFailed) {
        onAuthFailed('Network error occurred')
      }
    } finally {
      setIsLoading(false)
    }
  }
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera()
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current)
      }
    }
  }, [])
  
  return (
    <div className={`max-w-2xl mx-auto ${className}`}>
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Shield className="h-6 w-6 text-blue-600" />
            <CardTitle>
              {mode === 'register' ? 'Create Secure Account' : 'Secure Login'}
            </CardTitle>
          </div>
          
          {/* Progress Steps */}
          <div className="flex items-center justify-between mt-4">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center">
                <div className={`
                  w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                  ${step.completed ? 'bg-green-600 text-white' : 
                    step.current ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'}
                `}>
                  {step.completed ? <CheckCircle className="h-4 w-4" /> : index + 1}
                </div>
                {index < steps.length - 1 && (
                  <div className={`w-12 h-0.5 mx-2 ${
                    step.completed ? 'bg-green-600' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
          
          <div className="text-center mt-2">
            <h3 className="font-medium">{steps[currentStep]?.title}</h3>
            <p className="text-sm text-gray-600">{steps[currentStep]?.description}</p>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Error/Success Messages */}
          {error && (
            <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-lg">
              <AlertCircle className="h-4 w-4 text-red-600" />
              <span className="text-sm text-red-800">{error}</span>
            </div>
          )}
          
          {success && (
            <div className="flex items-center space-x-2 p-3 bg-green-50 border border-green-200 rounded-lg">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <span className="text-sm text-green-800">{success}</span>
            </div>
          )}
          
          {/* User ID Input (Register only) */}
          {mode === 'register' && currentStep === 0 && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">User ID</label>
                <Input
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="Enter unique user ID"
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">PIN</label>
                <div className="relative">
                  <Input
                    type={showPin ? "text" : "password"}
                    value={pin}
                    onChange={(e) => setPin(e.target.value)}
                    placeholder="Enter 4-8 digit PIN"
                    className="w-full pr-10"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3"
                    onClick={() => setShowPin(!showPin)}
                  >
                    {showPin ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                </div>
              </div>
              <Button
                onClick={() => {
                  if (userId.trim() && pin.trim()) {
                    steps[0].completed = true
                    setCurrentStep(1)
                    setSuccess('Setup completed!')
                  } else {
                    setError('Please fill in all fields')
                  }
                }}
                className="w-full"
                disabled={!userId.trim() || !pin.trim()}
              >
                Continue to Face Capture
              </Button>
            </div>
          )}
          
          {/* Face Capture */}
          {((mode === 'register' && currentStep === 1) || (mode === 'login' && currentStep === 0)) && (
            <div className="space-y-4">
              <div className="text-center">
                <h4 className="font-medium mb-2">Face Recognition Setup</h4>
                <p className="text-sm text-gray-600 mb-4">
                  Position your face in the camera and ensure good lighting
                </p>
              </div>
              
              <div className="relative bg-gray-100 rounded-lg overflow-hidden aspect-video">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  style={{ display: isCameraOn ? 'block' : 'none' }}
                />
                <canvas
                  ref={canvasRef}
                  className="hidden"
                />
                {!isCameraOn && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <CameraOff className="h-12 w-12 text-gray-400" />
                  </div>
                )}
                
                {/* Face Detection Indicator */}
                {isCameraOn && (
                  <div className={`absolute top-4 left-4 px-2 py-1 rounded text-xs font-medium ${
                    faceDetected ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
                  }`}>
                    {faceDetected ? 'Face Detected' : 'No Face Detected'}
                  </div>
                )}
                
                {/* Capture Preview */}
                {faceCapture && (
                  <div className="absolute bottom-4 right-4 w-16 h-16 border-2 border-white rounded overflow-hidden">
                    <img src={faceCapture} alt="Face capture" className="w-full h-full object-cover" />
                  </div>
                )}
              </div>
              
              <div className="flex space-x-2">
                <Button
                  onClick={isCameraOn ? stopCamera : startCamera}
                  variant={isCameraOn ? "destructive" : "default"}
                  className="flex-1"
                >
                  {isCameraOn ? (
                    <>
                      <CameraOff className="h-4 w-4 mr-2" />
                      Stop Camera
                    </>
                  ) : (
                    <>
                      <Camera className="h-4 w-4 mr-2" />
                      Start Camera
                    </>
                  )}
                </Button>
                
                <Button
                  onClick={captureFace}
                  disabled={!isCameraOn || !faceDetected}
                  variant="outline"
                  className="flex-1"
                >
                  <Camera className="h-4 w-4 mr-2" />
                  Capture Face
                </Button>
              </div>
            </div>
          )}
          
          {/* Voice Capture */}
          {((mode === 'register' && currentStep === 2) || (mode === 'login' && currentStep === 0 && faceCapture)) && (
            <div className="space-y-4">
              <div className="text-center">
                <h4 className="font-medium mb-2">Voice Recognition Setup</h4>
                <p className="text-sm text-gray-600 mb-4">
                  Record a voice sample by speaking clearly for 3-5 seconds
                </p>
              </div>
              
              <div className="bg-gray-100 rounded-lg p-8 text-center">
                {isRecording ? (
                  <div className="space-y-4">
                    <div className="w-16 h-16 bg-red-600 rounded-full mx-auto flex items-center justify-center animate-pulse">
                      <Mic className="h-8 w-8 text-white" />
                    </div>
                    <p className="text-lg font-medium">Recording...</p>
                    <p className="text-sm text-gray-600">
                      {(recordingDuration).toFixed(1)}s / 5.0s
                    </p>
                    <Progress value={(recordingDuration / 5) * 100} className="w-48 mx-auto" />
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="w-16 h-16 bg-gray-300 rounded-full mx-auto flex items-center justify-center">
                      <MicOff className="h-8 w-8 text-gray-600" />
                    </div>
                    <p className="text-lg font-medium">Ready to Record</p>
                    <p className="text-sm text-gray-600">
                      Click start and speak naturally
                    </p>
                  </div>
                )}
                
                {voiceCapture && (
                  <Badge variant="secondary" className="mt-2">
                    Voice Sample Captured
                  </Badge>
                )}
              </div>
              
              <Button
                onClick={isRecording ? stopVoiceRecording : startVoiceRecording}
                variant={isRecording ? "destructive" : "default"}
                className="w-full"
              >
                {isRecording ? (
                  <>
                    <Square className="h-4 w-4 mr-2" />
                    Stop Recording
                  </>
                ) : (
                  <>
                    <Mic className="h-4 w-4 mr-2" />
                    Start Recording
                  </>
                )}
              </Button>
            </div>
          )}
          
          {/* PIN Entry for Login */}
          {mode === 'login' && faceCapture && voiceCapture && currentStep === 1 && (
            <div className="space-y-4">
              <div className="text-center">
                <h4 className="font-medium mb-2">Enter PIN</h4>
                <p className="text-sm text-gray-600 mb-4">
                  Enter your PIN to unlock your secure vault
                </p>
              </div>
              
              <div className="relative">
                <Input
                  type={showPin ? "text" : "password"}
                  value={pin}
                  onChange={(e) => setPin(e.target.value)}
                  placeholder="Enter your PIN"
                  className="w-full pr-10 text-center text-lg"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3"
                  onClick={() => setShowPin(!showPin)}
                >
                  {showPin ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          )}
          
          {/* Progress Bar */}
          {isLoading && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Processing...</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          )}
          
          {/* Submit Button */}
          {((mode === 'register' && currentStep === 3) || 
            (mode === 'login' && faceCapture && voiceCapture && pin.trim())) && (
            <Button
              onClick={submitAuth}
              disabled={isLoading}
              className="w-full"
              size="lg"
            >
              {isLoading ? (
                'Processing...'
              ) : mode === 'register' ? (
                <>
                  <UserPlus className="h-4 w-4 mr-2" />
                  Create Secure Account
                </>
              ) : (
                <>
                  <LogIn className="h-4 w-4 mr-2" />
                  Unlock Vault
                </>
              )}
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  )
}