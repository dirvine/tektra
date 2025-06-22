"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { MessageSquare, Mic, Volume2, Loader2 } from 'lucide-react'
import AudioRecorder from './AudioRecorder'
import AudioPlayer from './AudioPlayer'
import { api } from '@/lib/api'

interface VoiceMessage {
  id: string
  type: 'user' | 'assistant'
  text: string
  audioData?: string
  timestamp: Date
  language?: string
  confidence?: number
  method?: string
}

interface VoiceConversationProps {
  className?: string
  onError?: (error: string) => void
}

export function VoiceConversation({ className = "", onError }: VoiceConversationProps) {
  const [messages, setMessages] = useState<VoiceMessage[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentStep, setCurrentStep] = useState<string>('')
  const [serviceInfo, setServiceInfo] = useState<any>(null)
  const [isServiceLoaded, setIsServiceLoaded] = useState(false)

  useEffect(() => {
    loadServiceInfo()
  }, [])

  const loadServiceInfo = async () => {
    try {
      const response = await api.getAudioServiceInfo()
      if (response.data) {
        setServiceInfo(response.data)
        setIsServiceLoaded(true)
      }
    } catch (error) {
      console.error('Failed to load audio service info:', error)
      onError?.('Failed to load audio services')
    }
  }

  const handleRecordingComplete = async (audioBlob: Blob) => {
    if (isProcessing) return

    setIsProcessing(true)
    setCurrentStep('Analyzing audio...')

    try {
      // Convert blob to file
      const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })

      // Add user message placeholder
      const userMessageId = Date.now().toString()
      setMessages(prev => [...prev, {
        id: userMessageId,
        type: 'user',
        text: 'Processing voice message...',
        timestamp: new Date()
      }])

      setCurrentStep('Transcribing speech...')

      // Send to voice conversation API
      const response = await api.voiceConversation(audioFile)

      if (response.error) {
        throw new Error(response.error)
      }

      const conversationData = response.data

      // Update user message with transcription
      setMessages(prev => prev.map(msg => 
        msg.id === userMessageId 
          ? {
              ...msg,
              text: conversationData.conversation.user_input.text,
              language: conversationData.conversation.user_input.language,
              confidence: conversationData.conversation.user_input.confidence,
              method: conversationData.conversation.user_input.method
            }
          : msg
      ))

      setCurrentStep('Generating AI response...')

      // Add AI response message
      const assistantMessage: VoiceMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        text: conversationData.conversation.ai_response.text,
        audioData: conversationData.conversation.audio_response.data,
        timestamp: new Date(),
        method: conversationData.conversation.ai_response.method
      }

      setMessages(prev => [...prev, assistantMessage])

      setCurrentStep('Complete!')

    } catch (error) {
      console.error('Voice conversation failed:', error)
      onError?.(error instanceof Error ? error.message : 'Voice conversation failed')
      
      // Remove the placeholder message on error
      setMessages(prev => prev.filter(msg => msg.id !== userMessageId))
    } finally {
      setIsProcessing(false)
      setCurrentStep('')
    }
  }

  const clearConversation = () => {
    setMessages([])
  }

  const getMethodBadgeColor = (method?: string) => {
    switch (method) {
      case 'phi4': return 'bg-blue-500'
      case 'phi4_simulated': return 'bg-purple-500'
      case 'whisper': return 'bg-green-500'
      case 'fallback': return 'bg-gray-500'
      default: return 'bg-gray-400'
    }
  }

  if (!isServiceLoaded) {
    return (
      <Card className={className}>
        <CardContent className="p-6">
          <div className="flex items-center justify-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span>Loading audio services...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Voice Conversation
          {serviceInfo && (
            <div className="flex gap-1 ml-auto">
              {serviceInfo.whisper_available && (
                <Badge variant="secondary" className="text-xs">
                  <Mic className="h-3 w-3 mr-1" />
                  STT
                </Badge>
              )}
              {serviceInfo.tts_available && (
                <Badge variant="secondary" className="text-xs">
                  <Volume2 className="h-3 w-3 mr-1" />
                  TTS
                </Badge>
              )}
            </div>
          )}
        </CardTitle>
        {messages.length > 0 && (
          <Button
            onClick={clearConversation}
            variant="outline"
            size="sm"
            className="w-fit"
          >
            Clear Conversation
          </Button>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Service Status */}
        {serviceInfo && (
          <div className="text-xs text-muted-foreground space-y-1">
            <div>Primary STT: {serviceInfo.model_info?.primary_stt || 'Unknown'}</div>
            <div>Languages: {serviceInfo.supported_languages?.slice(0, 5).join(', ')}...</div>
          </div>
        )}

        {/* Messages */}
        <div className="space-y-4 max-h-96 overflow-y-auto">
          {messages.map((message) => (
            <div key={message.id} className="space-y-2">
              <div
                className={`flex ${
                  message.type === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.type === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    {message.type === 'user' ? (
                      <Mic className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    ) : (
                      <Volume2 className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    )}
                    <div className="space-y-2 flex-1">
                      <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                      
                      {/* Audio Player for AI responses */}
                      {message.type === 'assistant' && message.audioData && (
                        <AudioPlayer
                          audioData={message.audioData}
                          autoPlay={false}
                          showControls={true}
                          className="mt-2"
                        />
                      )}
                      
                      {/* Message metadata */}
                      <div className="flex items-center gap-2 text-xs opacity-70">
                        <span>{message.timestamp.toLocaleTimeString()}</span>
                        {message.method && (
                          <Badge 
                            variant="secondary" 
                            className={`text-xs h-4 ${getMethodBadgeColor(message.method)}`}
                          >
                            {message.method}
                          </Badge>
                        )}
                        {message.confidence && (
                          <span>{Math.round(message.confidence * 100)}%</span>
                        )}
                        {message.language && (
                          <span>{message.language.toUpperCase()}</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Processing Status */}
        {isProcessing && (
          <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{currentStep}</span>
          </div>
        )}

        <Separator />

        {/* Audio Recorder */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Record Your Message</h4>
          <AudioRecorder
            onRecordingComplete={handleRecordingComplete}
            onError={onError}
            maxDuration={60}
          />
        </div>

        {/* Help Text */}
        <div className="text-xs text-muted-foreground text-center">
          Click "Start Recording" to begin a voice conversation. The AI will respond with both text and speech.
        </div>
      </CardContent>
    </Card>
  )
}

export default VoiceConversation