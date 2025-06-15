"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { 
  Mic, 
  MicOff, 
  Send, 
  Settings, 
  Bot, 
  User, 
  Copy, 
  RefreshCw,
  MoreVertical,
  Sidebar,
  Activity,
  Camera,
  ArrowLeft
} from 'lucide-react'
import { api } from '@/lib/api'
import { chatWebSocket } from '@/lib/websocket'
import ConversationSidebar, { ConversationWithMessages } from './ConversationSidebar'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  conversationId?: number
  metadata?: Record<string, unknown>
}

interface EnhancedChatInterfaceProps {
  onNavigate?: (tab: string) => void
}

export default function EnhancedChatInterface({ onNavigate }: EnhancedChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [selectedModel, setSelectedModel] = useState('phi-3-mini')
  const [isConnected, setIsConnected] = useState(false)
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null)
  const [currentConversation, setCurrentConversation] = useState<ConversationWithMessages | null>(null)
  const [showSidebar, setShowSidebar] = useState(true)
  const [userId] = useState('1') // TODO: Get from auth context
  
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Connect to WebSocket with user ID
    const connectWebSocket = async () => {
      try {
        // Update WebSocket URL to include user ID
        const wsUrl = `ws://localhost:8000/ws/chat/${userId}`
        const customChatWebSocket = {
          ...chatWebSocket,
          url: wsUrl
        }
        
        await customChatWebSocket.connect()
        setIsConnected(true)
      } catch (error) {
        console.error('WebSocket connection failed:', error)
        setIsConnected(false)
      }
    }

    connectWebSocket()

    // Listen for conversation created
    chatWebSocket.on('conversation_created', (message) => {
      const data = message.data as { conversation_id?: number; title?: string }
      if (data.conversation_id) {
        setCurrentConversationId(data.conversation_id)
        console.log(`New conversation created: ${data.conversation_id}`)
      }
    })

    // Listen for AI response start
    chatWebSocket.on('ai_response_start', (message) => {
      const data = message.data as { conversation_id?: number; model?: string }
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        conversationId: data.conversation_id
      }])
    })

    // Listen for AI response tokens (streaming)
    chatWebSocket.on('ai_response_token', (message) => {
      const data = message.data as { token?: string }
      setMessages(prev => {
        const updatedMessages = [...prev]
        const lastMessage = updatedMessages[updatedMessages.length - 1]
        if (lastMessage && lastMessage.role === 'assistant') {
          lastMessage.content += data.token || ''
        }
        return updatedMessages
      })
    })

    // Listen for AI response completion
    chatWebSocket.on('ai_response_complete', (message) => {
      setIsLoading(false)
      const data = message.data as { 
        conversation_id?: number
        full_response?: string
        tokens_used?: number
        model?: string 
      }
      console.log(`AI response completed. Tokens used: ${data.tokens_used}, Model: ${data.model}`)
    })

    // Listen for AI response errors
    chatWebSocket.on('ai_response_error', (message) => {
      setIsLoading(false)
      const data = message.data as { error?: string }
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Error: ${data.error || 'Unknown error occurred'}`,
        timestamp: new Date()
      }])
    })

    return () => {
      chatWebSocket.disconnect()
    }
  }, [userId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async (content: string) => {
    if (!content.trim()) return

    // Add user message to UI immediately
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
      conversationId: currentConversationId || undefined
    }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      if (isConnected) {
        // Send via WebSocket for real-time response
        chatWebSocket.send({
          type: 'chat',
          data: {
            message: content,
            model: selectedModel,
            conversation_id: currentConversationId
          }
        })
      } else {
        // Fallback: Use HTTP API to create/use conversation
        let conversationId = currentConversationId
        
        if (!conversationId) {
          // Create new conversation
          const convResponse = await api.createConversation(undefined, selectedModel)
          if (convResponse.data) {
            conversationId = (convResponse.data as { id: number }).id
            setCurrentConversationId(conversationId)
          }
        }

        if (conversationId) {
          // Add message to conversation
          await api.addMessageToConversation(conversationId, content, 'user')
        }

        // Send chat message
        const response = await api.sendChatMessage(content, selectedModel)
        if (response.data) {
          setMessages(prev => [...prev, {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: (response.data as { response?: string })?.response || 'No response',
            timestamp: new Date(),
            conversationId
          }])

          // Save assistant response to conversation
          if (conversationId) {
            await api.addMessageToConversation(
              conversationId, 
              (response.data as { response?: string })?.response || 'No response',
              'assistant'
            )
          }
        }
        setIsLoading(false)
      }
    } catch (error) {
      console.error('Failed to send message:', error)
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const handleNewConversation = () => {
    setMessages([])
    setCurrentConversationId(null)
    setCurrentConversation(null)
  }

  const handleConversationSelect = (conversation: ConversationWithMessages) => {
    setCurrentConversation(conversation)
    setCurrentConversationId(conversation.id)
    
    // Load messages from conversation
    const conversationMessages: Message[] = conversation.messages.map(msg => ({
      id: msg.id.toString(),
      role: msg.role as 'user' | 'assistant' | 'system',
      content: msg.content,
      timestamp: new Date(msg.created_at),
      conversationId: conversation.id,
      metadata: msg.metadata
    }))
    
    setMessages(conversationMessages)
    setSelectedModel(conversation.model_name)
  }

  const toggleRecording = async () => {
    if (isRecording) {
      setIsRecording(false)
      try {
        await api.stopRecording()
        // TODO: Get recorded audio and transcribe
      } catch (error) {
        console.error('Failed to stop recording:', error)
      }
    } else {
      setIsRecording(true)
      try {
        await api.startRecording()
      } catch (error) {
        console.error('Failed to start recording:', error)
        setIsRecording(false)
      }
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const regenerateResponse = async (messageIndex: number) => {
    if (messageIndex < 1) return
    
    const userMessage = messages[messageIndex - 1]
    if (userMessage.role !== 'user') return

    // Remove the assistant's response and regenerate
    setMessages(prev => prev.slice(0, messageIndex))
    await sendMessage(userMessage.content)
  }

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Conversation Sidebar */}
      {showSidebar && (
        <div className="w-80 border-r">
          <ConversationSidebar
            currentConversation={currentConversationId}
            onConversationSelect={handleConversationSelect}
            onNewConversation={handleNewConversation}
          />
        </div>
      )}

      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col">
        <Card className="h-full flex flex-col border-0 rounded-none">
          <CardHeader className="border-b">
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {onNavigate && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onNavigate('dashboard')}
                  >
                    <ArrowLeft className="h-4 w-4" />
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowSidebar(!showSidebar)}
                >
                  <Sidebar className="h-4 w-4" />
                </Button>
                <span>
                  {currentConversation ? currentConversation.title : 'AI Chat Assistant'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                {onNavigate && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onNavigate('avatar')}
                      title="Avatar Control"
                    >
                      <User className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onNavigate('robot')}
                      title="Robot Control"
                    >
                      <Activity className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onNavigate('camera')}
                      title="Camera Control"
                    >
                      <Camera className="h-4 w-4" />
                    </Button>
                  </>
                )}
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="text-sm text-muted-foreground">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
                <Button variant="ghost" size="sm">
                  <Settings className="h-4 w-4" />
                </Button>
              </div>
            </CardTitle>
            <div className="flex items-center gap-2">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="text-sm border rounded px-2 py-1"
              >
                <option value="phi-3-mini">Phi-3 Mini</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                <option value="llama-2-7b-chat">Llama 2 7B Chat</option>
              </select>
              {currentConversation && (
                <span className="text-xs text-muted-foreground">
                  {currentConversation.messages.length} messages
                </span>
              )}
            </div>
          </CardHeader>

          <CardContent className="flex-1 overflow-y-auto space-y-4 p-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <Bot className="h-12 w-12 text-muted-foreground/50 mb-4" />
                <h3 className="text-lg font-semibold mb-2">Start a conversation</h3>
                <p className="text-muted-foreground">
                  Ask me anything! I&apos;m here to help with your questions and tasks.
                </p>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`group max-w-[80%] rounded-lg px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                  }`}
                >
                  <div className="flex items-start gap-2 mb-2">
                    {message.role === 'user' ? (
                      <User className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    ) : (
                      <Bot className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    </div>
                    
                    {/* Message actions */}
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0"
                        onClick={() => copyToClipboard(message.content)}
                      >
                        <Copy className="h-3 w-3" />
                      </Button>
                      {message.role === 'assistant' && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-6 w-6 p-0"
                          onClick={() => regenerateResponse(index)}
                        >
                          <RefreshCw className="h-3 w-3" />
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0"
                      >
                        <MoreVertical className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <p className="text-xs opacity-70">
                      {formatTimestamp(message.timestamp)}
                    </p>
                    {message.conversationId && (
                      <p className="text-xs opacity-50">
                        Conv #{message.conversationId}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-muted rounded-lg px-4 py-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Bot className="h-4 w-4" />
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-current rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                      <span className="text-sm">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </CardContent>

          <CardFooter className="border-t p-4">
            <div className="flex gap-2 w-full">
              <Button
                variant={isRecording ? "destructive" : "outline"}
                size="icon"
                onClick={toggleRecording}
              >
                {isRecording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
              </Button>
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your message..."
                className="flex-1 min-h-0 resize-none"
                rows={1}
              />
              <Button
                onClick={() => sendMessage(input)}
                disabled={!input.trim() || isLoading}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </CardFooter>
        </Card>
      </div>
    </div>
  )
}