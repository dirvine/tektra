"use client"

import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { 
  Mic, 
  MicOff, 
  Camera, 
  CameraOff, 
  Send, 
  Bot, 
  User, 
  Settings,
  Volume2,
  Square
} from 'lucide-react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  type: 'text' | 'audio' | 'image'
  duration?: number
}

interface UnifiedInterfaceProps {
  className?: string
}

export default function UnifiedInterface({ className }: UnifiedInterfaceProps) {
  // State
  const [messages, setMessages] = useState<Message[]>([])
  const [inputText, setInputText] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [phi4Status, setPhi4Status] = useState<'loading' | 'ready' | 'error'>('loading')

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Initialize WebSocket connection
  useEffect(() => {
    initializeWebSocket()
    checkPhi4Status()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const initializeWebSocket = () => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/ws`
      
      wsRef.current = new WebSocket(wsUrl)
      
      wsRef.current.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
      }
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          handleWebSocketMessage(data)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }
      
      wsRef.current.onclose = () => {
        setIsConnected(false)
        console.log('WebSocket disconnected')
        // Try to reconnect after 3 seconds
        setTimeout(initializeWebSocket, 3000)
      }
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setIsConnected(false)
      }
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error)
      setIsConnected(false)
    }
  }

  const handleWebSocketMessage = (data: any) => {
    if (data.type === 'chat_response') {
      const newMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.message,
        timestamp: new Date(),
        type: 'text'
      }
      setMessages(prev => [...prev, newMessage])
    } else if (data.type === 'transcription') {
      const newMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        content: data.text,
        timestamp: new Date(),
        type: 'audio'
      }
      setMessages(prev => [...prev, newMessage])
    }
    setIsLoading(false)
  }

  const checkPhi4Status = async () => {
    try {
      const response = await fetch('/api/v1/audio/api/v1/audio/phi4/info')
      if (response.ok) {
        const data = await response.json()
        setPhi4Status(data.is_loaded ? 'ready' : 'loading')
        
        // Auto-load Phi-4 if not loaded
        if (!data.is_loaded) {
          await loadPhi4()
        }
      } else {
        setPhi4Status('error')
      }
    } catch (error) {
      console.error('Error checking Phi-4 status:', error)
      setPhi4Status('error')
    }
  }

  const loadPhi4 = async () => {
    try {
      setPhi4Status('loading')
      const response = await fetch('/api/v1/audio/api/v1/audio/phi4/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (response.ok) {
        setPhi4Status('ready')
      } else {
        setPhi4Status('error')
      }
    } catch (error) {
      console.error('Error loading Phi-4:', error)
      setPhi4Status('error')
    }
  }

  const sendTextMessage = async () => {
    if (!inputText.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText,
      timestamp: new Date(),
      type: 'text'
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)
    
    // Send via WebSocket
    wsRef.current.send(JSON.stringify({
      type: 'chat_message',
      message: inputText,
      model: 'phi-4'
    }))

    setInputText('')
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendTextMessage()
    }
  }

  const toggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop()
      }
      setIsRecording(false)
    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const mediaRecorder = new MediaRecorder(stream)
        mediaRecorderRef.current = mediaRecorder

        const audioChunks: Blob[] = []
        
        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data)
        }

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
          await sendAudioMessage(audioBlob)
          
          // Stop all tracks
          stream.getTracks().forEach(track => track.stop())
        }

        mediaRecorder.start()
        setIsRecording(true)
      } catch (error) {
        console.error('Error starting recording:', error)
        alert('Could not access microphone. Please check permissions.')
      }
    }
  }

  const sendAudioMessage = async (audioBlob: Blob) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

    setIsLoading(true)
    
    // Convert to base64 and send via WebSocket
    const reader = new FileReader()
    reader.onload = () => {
      const base64Audio = (reader.result as string).split(',')[1]
      wsRef.current?.send(JSON.stringify({
        type: 'audio_message',
        audio_data: base64Audio,
        format: 'webm'
      }))
    }
    reader.readAsDataURL(audioBlob)
  }

  const toggleCamera = async () => {
    if (isCameraOn) {
      // Turn off camera
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream
        stream.getTracks().forEach(track => track.stop())
        videoRef.current.srcObject = null
      }
      setIsCameraOn(false)
    } else {
      // Turn on camera
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 } 
        })
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          videoRef.current.play()
        }
        setIsCameraOn(true)
      } catch (error) {
        console.error('Error starting camera:', error)
        alert('Could not access camera. Please check permissions.')
      }
    }
  }

  const takeSnapshot = async () => {
    if (!videoRef.current || !isCameraOn) return

    const canvas = document.createElement('canvas')
    const context = canvas.getContext('2d')
    if (!context) return

    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    context.drawImage(videoRef.current, 0, 0)

    canvas.toBlob(async (blob) => {
      if (!blob || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return

      setIsLoading(true)
      
      // Convert to base64 and send via WebSocket
      const reader = new FileReader()
      reader.onload = () => {
        const base64Image = (reader.result as string).split(',')[1]
        
        const userMessage: Message = {
          id: Date.now().toString(),
          role: 'user',
          content: '📸 Image captured',
          timestamp: new Date(),
          type: 'image'
        }
        setMessages(prev => [...prev, userMessage])

        wsRef.current?.send(JSON.stringify({
          type: 'image_message',
          image_data: base64Image,
          format: 'png'
        }))
      }
      reader.readAsDataURL(blob)
    }, 'image/png')
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const getStatusColor = () => {
    if (phi4Status === 'ready' && isConnected) return 'bg-green-500'
    if (phi4Status === 'loading') return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const getStatusText = () => {
    if (phi4Status === 'ready' && isConnected) return 'Phi-4 Ready'
    if (phi4Status === 'loading') return 'Loading Phi-4...'
    if (phi4Status === 'error') return 'Phi-4 Error'
    return 'Disconnected'
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-2">
          <Bot className="h-6 w-6 text-blue-600" />
          <h1 className="text-xl font-semibold">Tektra AI Assistant</h1>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${getStatusColor()}`} />
          <span className="text-sm text-gray-600">{getStatusText()}</span>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
                <Bot className="h-12 w-12 mb-4 text-gray-400" />
                <h3 className="text-lg font-medium mb-2">Welcome to Tektra AI</h3>
                <p>Start a conversation by typing, speaking, or taking a photo!</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[70%] rounded-lg px-4 py-2 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    <div className="flex items-center space-x-2 mb-1">
                      {message.role === 'user' ? (
                        <User className="h-4 w-4" />
                      ) : (
                        <Bot className="h-4 w-4" />
                      )}
                      <span className="text-xs opacity-75">
                        {formatTime(message.timestamp)}
                      </span>
                      {message.type !== 'text' && (
                        <Badge variant="secondary" className="text-xs">
                          {message.type}
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg px-4 py-2">
                  <div className="flex items-center space-x-2">
                    <Bot className="h-4 w-4" />
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t p-4">
            <div className="flex space-x-2">
              <div className="flex-1">
                <Textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message..."
                  className="resize-none"
                  rows={2}
                />
              </div>
              <div className="flex flex-col space-y-2">
                <Button
                  onClick={sendTextMessage}
                  disabled={!inputText.trim() || !isConnected}
                  size="sm"
                >
                  <Send className="h-4 w-4" />
                </Button>
                <Button
                  onClick={toggleRecording}
                  variant={isRecording ? "destructive" : "outline"}
                  size="sm"
                  disabled={!isConnected}
                >
                  {isRecording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Camera Panel */}
        <div className="w-80 border-l flex flex-col">
          <div className="p-4 border-b">
            <h3 className="font-medium">Camera</h3>
          </div>
          
          <div className="flex-1 p-4">
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden mb-4">
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                style={{ display: isCameraOn ? 'block' : 'none' }}
              />
              {!isCameraOn && (
                <div className="w-full h-full flex items-center justify-center">
                  <CameraOff className="h-12 w-12 text-gray-400" />
                </div>
              )}
            </div>
            
            <div className="space-y-2">
              <Button
                onClick={toggleCamera}
                variant={isCameraOn ? "destructive" : "default"}
                className="w-full"
              >
                {isCameraOn ? (
                  <>
                    <CameraOff className="h-4 w-4 mr-2" />
                    Turn Off Camera
                  </>
                ) : (
                  <>
                    <Camera className="h-4 w-4 mr-2" />
                    Turn On Camera
                  </>
                )}
              </Button>
              
              {isCameraOn && (
                <Button
                  onClick={takeSnapshot}
                  variant="outline"
                  className="w-full"
                  disabled={!isConnected}
                >
                  <Camera className="h-4 w-4 mr-2" />
                  Take Photo
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}