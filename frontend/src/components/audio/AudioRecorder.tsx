"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Mic, MicOff, Square, Play, Pause, Volume2, VolumeX } from 'lucide-react'

interface AudioRecorderProps {
  onRecordingComplete?: (audioBlob: Blob) => void
  onError?: (error: string) => void
  maxDuration?: number // seconds
  className?: string
}

export function AudioRecorder({ 
  onRecordingComplete, 
  onError, 
  maxDuration = 30,
  className = "" 
}: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [volume, setVolume] = useState(0)
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const volumeIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyzerRef = useRef<AnalyserNode | null>(null)

  useEffect(() => {
    // Check microphone permission on mount
    checkMicrophonePermission()
    
    return () => {
      cleanup()
    }
  }, [])

  const checkMicrophonePermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      setHasPermission(true)
      stream.getTracks().forEach(track => track.stop()) // Stop the test stream
    } catch (error) {
      setHasPermission(false)
      console.error('Microphone permission denied:', error)
    }
  }

  const cleanup = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (volumeIntervalRef.current) {
      clearInterval(volumeIntervalRef.current)
      volumeIntervalRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
  }

  const startRecording = async () => {
    try {
      if (!hasPermission) {
        await checkMicrophonePermission()
        if (!hasPermission) {
          onError?.('Microphone permission is required for voice recording')
          return
        }
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      })
      
      streamRef.current = stream
      chunksRef.current = []

      // Setup audio context for volume monitoring
      audioContextRef.current = new AudioContext()
      const source = audioContextRef.current.createMediaStreamSource(stream)
      analyzerRef.current = audioContextRef.current.createAnalyser()
      analyzerRef.current.fftSize = 256
      source.connect(analyzerRef.current)

      // Setup MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setAudioBlob(blob)
        setAudioUrl(URL.createObjectURL(blob))
        onRecordingComplete?.(blob)
        
        // Cleanup
        stream.getTracks().forEach(track => track.stop())
        if (audioContextRef.current) {
          audioContextRef.current.close()
          audioContextRef.current = null
        }
      }

      mediaRecorder.start(100) // Collect data every 100ms
      setIsRecording(true)
      setRecordingTime(0)

      // Start recording timer
      intervalRef.current = setInterval(() => {
        setRecordingTime(prev => {
          const newTime = prev + 1
          if (newTime >= maxDuration) {
            stopRecording()
          }
          return newTime
        })
      }, 1000)

      // Start volume monitoring
      volumeIntervalRef.current = setInterval(() => {
        if (analyzerRef.current) {
          const dataArray = new Uint8Array(analyzerRef.current.frequencyBinCount)
          analyzerRef.current.getByteFrequencyData(dataArray)
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length
          setVolume(average)
        }
      }, 100)

    } catch (error) {
      console.error('Failed to start recording:', error)
      onError?.('Failed to start recording. Please check microphone permissions.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      
      if (volumeIntervalRef.current) {
        clearInterval(volumeIntervalRef.current)
        volumeIntervalRef.current = null
      }
      
      setVolume(0)
    }
  }

  const playRecording = () => {
    if (audioUrl && audioRef.current) {
      audioRef.current.play()
      setIsPlaying(true)
    }
  }

  const pauseRecording = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      setIsPlaying(false)
      setIsPaused(true)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getVolumeColor = (vol: number) => {
    if (vol < 30) return 'bg-green-500'
    if (vol < 60) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <Card className={className}>
      <CardContent className="p-4 space-y-4">
        {/* Permission Check */}
        {hasPermission === false && (
          <div className="text-center text-red-500 text-sm">
            Microphone permission required. Please allow access and refresh.
          </div>
        )}

        {/* Recording Controls */}
        <div className="flex items-center justify-center gap-4">
          {!isRecording ? (
            <Button
              onClick={startRecording}
              disabled={hasPermission === false}
              variant="default"
              size="lg"
              className="gap-2"
            >
              <Mic className="h-5 w-5" />
              Start Recording
            </Button>
          ) : (
            <Button
              onClick={stopRecording}
              variant="destructive"
              size="lg"
              className="gap-2"
            >
              <Square className="h-4 w-4" />
              Stop ({formatTime(recordingTime)})
            </Button>
          )}
        </div>

        {/* Recording Progress */}
        {isRecording && (
          <div className="space-y-2">
            <Progress 
              value={(recordingTime / maxDuration) * 100} 
              className="h-2"
            />
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <span>{formatTime(recordingTime)} / {formatTime(maxDuration)}</span>
              <div className="flex items-center gap-2">
                {volume > 0 ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                <div className="w-16 h-2 bg-gray-200 rounded">
                  <div 
                    className={`h-full rounded transition-all duration-100 ${getVolumeColor(volume)}`}
                    style={{ width: `${Math.min(volume, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Playback Controls */}
        {audioUrl && !isRecording && (
          <div className="space-y-2">
            <audio
              ref={audioRef}
              src={audioUrl}
              onEnded={() => {
                setIsPlaying(false)
                setIsPaused(false)
              }}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
            />
            
            <div className="flex items-center justify-center gap-2">
              {!isPlaying ? (
                <Button
                  onClick={playRecording}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                >
                  <Play className="h-4 w-4" />
                  {isPaused ? 'Resume' : 'Play Recording'}
                </Button>
              ) : (
                <Button
                  onClick={pauseRecording}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                >
                  <Pause className="h-4 w-4" />
                  Pause
                </Button>
              )}
            </div>
          </div>
        )}

        {/* Recording Indicator */}
        {isRecording && (
          <div className="flex items-center justify-center gap-2 text-red-500">
            <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            <span className="text-sm font-medium">Recording...</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default AudioRecorder