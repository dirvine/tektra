"use client"

import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
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
  Square,
  Shield,
  UserPlus,
  LogIn,
  Lock,
  Unlock
} from 'lucide-react'

import BiometricAuth from '@/components/security/BiometricAuth'
import SecurityStatus from '@/components/security/SecurityStatus'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  type: 'text' | 'audio' | 'image'
  duration?: number
}

interface UserSession {
  sessionToken: string
  user_id: string
  vault_stats: any
  match_score: number
}

interface SecureUnifiedInterfaceProps {
  className?: string
}

export default function SecureUnifiedInterface({ className }: SecureUnifiedInterfaceProps) {
  // Authentication state
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [showAuth, setShowAuth] = useState(false)
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login')
  const [userSession, setUserSession] = useState<UserSession | null>(null)
  
  // Chat state
  const [messages, setMessages] = useState<Message[]>([])
  const [inputText, setInputText] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [phi4Status, setPhi4Status] = useState<'loading' | 'ready' | 'error'>('loading')
  const [phi4Progress, setPhi4Progress] = useState({ progress: 0, message: '' })

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

  // Initialize WebSocket connection with authentication
  const initializeWebSocket = () => {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl = `${protocol}//${window.location.host}/ws`
      
      wsRef.current = new WebSocket(wsUrl)
      
      wsRef.current.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
        
        // Send authentication if available
        if (userSession?.sessionToken) {
          wsRef.current?.send(JSON.stringify({
            type: 'authenticate',
            data: { session_token: userSession.sessionToken }
          }))
        }
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
        // Try to reconnect after 3 seconds if authenticated
        if (isAuthenticated) {
          setTimeout(initializeWebSocket, 3000)
        }
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
    console.log('WebSocket message received:', data)
    
    if (data.type === 'connection_established') {
      console.log('WebSocket connection established:', data.data)
    } else if (data.type === 'ai_response_chunk') {
      // Handle streaming AI responses
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1]
        if (lastMessage && lastMessage.role === 'assistant' && lastMessage.id === 'streaming') {
          // Update existing streaming message
          return prev.slice(0, -1).concat({
            ...lastMessage,
            content: lastMessage.content + data.data.content
          })
        } else {
          // Create new streaming message
          return [...prev, {
            id: 'streaming',
            role: 'assistant',
            content: data.data.content,
            timestamp: new Date(),
            type: 'text'
          }]
        }
      })
    } else if (data.type === 'ai_response_complete') {
      // Finalize streaming message and save to vault
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1]
        if (lastMessage && lastMessage.id === 'streaming') {
          const finalMessage = {
            ...lastMessage,
            id: Date.now().toString()
          }
          
          // Save to vault if authenticated
          if (userSession?.sessionToken) {
            saveMessageToVault(finalMessage)
          }
          
          return prev.slice(0, -1).concat(finalMessage)
        }
        return prev
      })
      setIsLoading(false)
    } else if (data.type === 'transcription_final') {
      const newMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        content: data.data.text,
        timestamp: new Date(),
        type: 'audio'
      }
      setMessages(prev => [...prev, newMessage])
      
      // Save to vault if authenticated
      if (userSession?.sessionToken) {
        saveMessageToVault(newMessage)
      }
    } else if (data.type === 'error') {
      console.error('WebSocket error:', data.data.message)
      setIsLoading(false)
    }
  }

  // Save message to encrypted vault
  const saveMessageToVault = async (message: Message) => {
    if (!userSession?.sessionToken) return
    
    try {
      await fetch('/api/v1/security/vault/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${userSession.sessionToken}`
        },
        body: JSON.stringify({
          conversation_id: 'main_conversation',
          message: {
            id: message.id,
            role: message.role,
            content: message.content,
            type: message.type,
            timestamp: message.timestamp.toISOString()
          }
        })
      })
    } catch (error) {
      console.error('Failed to save message to vault:', error)
    }
  }

  // Authentication handlers
  const handleAuthSuccess = (userData: any) => {
    setUserSession({
      sessionToken: userData.session_token,
      user_id: userData.user_id,
      vault_stats: userData.vault_stats,
      match_score: userData.match_score
    })
    setIsAuthenticated(true)
    setShowAuth(false)
    
    // Load conversation history from vault
    loadConversationHistory(userData.session_token)
    
    // Reconnect WebSocket with authentication
    if (wsRef.current) {
      wsRef.current.close()
    }
    setTimeout(initializeWebSocket, 1000)
  }

  const handleAuthFailed = (error: string) => {
    console.error('Authentication failed:', error)
    // Show error to user
  }

  const handleLogout = async () => {
    if (userSession?.sessionToken) {
      try {
        await fetch('/api/v1/security/logout', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${userSession.sessionToken}`
          }
        })
      } catch (error) {
        console.error('Logout error:', error)
      }
    }
    
    setIsAuthenticated(false)
    setUserSession(null)
    setMessages([])
    
    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close()
    }
    setIsConnected(false)
  }

  // Load conversation history from vault
  const loadConversationHistory = async (sessionToken: string) => {
    try {
      const response = await fetch('/api/v1/security/vault/conversations', {
        headers: {
          'Authorization': `Bearer ${sessionToken}`
        }
      })
      
      if (response.ok) {
        const conversations = await response.json()
        if (conversations.length > 0) {
          // Load messages from most recent conversation
          const recentConversation = conversations[0]
          const historyMessages = recentConversation.messages.map((msg: any) => ({
            id: msg.id,
            role: msg.role,
            content: msg.content,
            timestamp: new Date(msg.timestamp),
            type: msg.type || 'text'
          }))
          setMessages(historyMessages)
        }
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error)
    }
  }

  const checkPhi4Status = async () => {
    try {
      const response = await fetch('/api/v1/audio/phi4/info')
      if (response.ok) {
        const data = await response.json()
        setPhi4Status(data.is_loaded ? 'ready' : (data.is_loading ? 'loading' : 'error'))
        
        if (data.load_progress) {
          setPhi4Progress({
            progress: data.load_progress.progress * 100,
            message: data.load_progress.message || ''
          })
        }
        
        if (!data.is_loaded && !data.is_loading) {
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
      const response = await fetch('/api/v1/audio/phi4/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      if (response.ok) {
        const pollProgress = setInterval(async () => {
          try {
            const statusResponse = await fetch('/api/v1/audio/phi4/info')
            if (statusResponse.ok) {
              const data = await statusResponse.json()
              
              if (data.load_progress) {
                setPhi4Progress({
                  progress: data.load_progress.progress * 100,
                  message: data.load_progress.message || ''
                })
              }
              
              if (data.is_loaded) {
                setPhi4Status('ready')
                clearInterval(pollProgress)
              } else if (data.load_progress?.status === 'error') {
                setPhi4Status('error')
                clearInterval(pollProgress)
              }
            }
          } catch (error) {
            console.error('Error polling Phi-4 status:', error)
          }
        }, 2000)
        
        setTimeout(() => clearInterval(pollProgress), 300000)
      } else {
        setPhi4Status('error')
      }
    } catch (error) {
      console.error('Error loading Phi-4:', error)
      setPhi4Status('error')
    }
  }

  // Anonymize query for external APIs if needed
  const anonymizeQuery = async (query: string) => {
    if (!userSession?.sessionToken) return query
    
    try {
      const response = await fetch('/api/v1/security/anonymize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${userSession.sessionToken}`
        },
        body: JSON.stringify({ query, preserve_technical: true })
      })
      
      if (response.ok) {
        const result = await response.json()
        return result.anonymized_query
      }
    } catch (error) {
      console.error('Query anonymization failed:', error)
    }
    
    return query
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
    
    // Save to vault if authenticated
    if (userSession?.sessionToken) {
      saveMessageToVault(userMessage)
    }
    
    // Anonymize query if needed for external processing
    const processedQuery = await anonymizeQuery(inputText)
    
    // Send via WebSocket
    wsRef.current.send(JSON.stringify({
      type: 'chat_message',
      data: {
        content: processedQuery,
        conversation_id: 'main_conversation',
        model: 'phi-4',
        temperature: 0.7,
        stream: true,
        session_token: userSession?.sessionToken
      }
    }))

    setInputText('')
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendTextMessage()
    }
  }

  // Initialize on mount
  useEffect(() => {
    checkPhi4Status()
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // Show authentication screen if not authenticated
  if (!isAuthenticated) {
    return (
      <div className={`flex flex-col h-full ${className}`}>
        {/* Header */}
        <div className="border-b">
          <div className="flex items-center justify-between p-4">
            <div className="flex items-center space-x-2">
              <Shield className="h-6 w-6 text-blue-600" />
              <h1 className="text-xl font-semibold">Tektra Secure AI Assistant</h1>
            </div>
            
            <div className="flex items-center space-x-2">
              <Lock className="h-4 w-4 text-red-600" />
              <span className="text-sm text-red-600">Secure Mode - Authentication Required</span>
            </div>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center p-8">
          {!showAuth ? (
            <Card className="w-full max-w-md">
              <CardHeader className="text-center">
                <CardTitle className="flex items-center justify-center space-x-2">
                  <Shield className="h-6 w-6 text-blue-600" />
                  <span>Secure Access Required</span>
                </CardTitle>
                <p className="text-sm text-gray-600 mt-2">
                  This AI assistant uses biometric authentication and encrypted vaults to protect your conversations and data.
                </p>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <Button
                    onClick={() => {
                      setAuthMode('login')
                      setShowAuth(true)
                    }}
                    className="w-full"
                  >
                    <LogIn className="h-4 w-4 mr-2" />
                    Login
                  </Button>
                  
                  <Button
                    onClick={() => {
                      setAuthMode('register')
                      setShowAuth(true)
                    }}
                    variant="outline"
                    className="w-full"
                  >
                    <UserPlus className="h-4 w-4 mr-2" />
                    Register
                  </Button>
                </div>
                
                <div className="text-center">
                  <p className="text-xs text-gray-500">
                    Biometric data is processed locally and never transmitted
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="w-full">
              <div className="mb-4 text-center">
                <Button
                  onClick={() => setShowAuth(false)}
                  variant="ghost"
                  size="sm"
                >
                  ← Back to Options
                </Button>
              </div>
              
              <BiometricAuth
                mode={authMode}
                onAuthSuccess={handleAuthSuccess}
                onAuthFailed={handleAuthFailed}
              />
            </div>
          )}
        </div>
      </div>
    )
  }

  const getStatusColor = () => {
    if (phi4Status === 'ready' && isConnected) return 'bg-green-500'
    if (phi4Status === 'loading') return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const getStatusText = () => {
    if (phi4Status === 'ready' && isConnected) return 'Phi-4 Ready & Secured'
    if (phi4Status === 'loading') {
      return `Loading Phi-4... ${phi4Progress.progress.toFixed(0)}%`
    }
    if (phi4Status === 'error') return 'Phi-4 Error'
    return 'Disconnected'
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Header */}
      <div className="border-b">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center space-x-2">
            <Unlock className="h-6 w-6 text-green-600" />
            <h1 className="text-xl font-semibold">Tektra Secure AI Assistant</h1>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${getStatusColor()}`} />
              <span className="text-sm text-gray-600">{getStatusText()}</span>
            </div>
            
            <SecurityStatus
              sessionToken={userSession?.sessionToken}
              userData={userSession}
              onLogout={handleLogout}
            />
          </div>
        </div>
        
        {/* Progress Bar */}
        {phi4Status === 'loading' && (
          <div className="px-4 pb-2">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${phi4Progress.progress}%` }}
              />
            </div>
            {phi4Progress.message && (
              <p className="text-xs text-gray-500 mt-1">{phi4Progress.message}</p>
            )}
          </div>
        )}
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
                <Shield className="h-12 w-12 mb-4 text-green-400" />
                <h3 className="text-lg font-medium mb-2">Secure Session Active</h3>
                <p>Your conversations are encrypted and stored securely in your personal vault.</p>
                <p className="text-sm mt-2">Start chatting - your privacy is protected!</p>
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
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                      {message.type !== 'text' && (
                        <Badge variant="secondary" className="text-xs">
                          {message.type}
                        </Badge>
                      )}
                      {message.role === 'user' && (
                        <Shield className="h-3 w-3 opacity-75" title="Encrypted in vault" />
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
                  placeholder="Type your secure message..."
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
                  onClick={() => {/* Voice recording functionality */}}
                  variant="outline"
                  size="sm"
                  disabled={!isConnected}
                >
                  <Mic className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
              <span>🔒 End-to-end encrypted • Locally processed</span>
              <span>User: {userSession?.user_id}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}