"use client"

import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Slider } from '@/components/ui/slider'
import { Play, Pause, Volume2, VolumeX, RotateCcw } from 'lucide-react'

interface AudioPlayerProps {
  audioData?: string // Base64 encoded audio data
  audioUrl?: string // Direct URL to audio
  autoPlay?: boolean
  showControls?: boolean
  className?: string
  onPlayStart?: () => void
  onPlayEnd?: () => void
  onError?: (error: string) => void
}

export function AudioPlayer({ 
  audioData, 
  audioUrl, 
  autoPlay = false,
  showControls = true,
  className = "",
  onPlayStart,
  onPlayEnd,
  onError
}: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [volume, setVolume] = useState(0.8)
  const [isMuted, setIsMuted] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const progressIntervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (audioData || audioUrl) {
      loadAudio()
    }
    
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current)
      }
    }
  }, [audioData, audioUrl])

  useEffect(() => {
    if (autoPlay && audioRef.current && !error) {
      playAudio()
    }
  }, [autoPlay, error])

  const loadAudio = () => {
    setIsLoading(true)
    setError(null)

    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.src = ''
    }

    try {
      const audio = new Audio()
      audioRef.current = audio

      // Set audio source
      if (audioData) {
        // Decode base64 and create blob URL
        const binaryString = atob(audioData)
        const bytes = new Uint8Array(binaryString.length)
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i)
        }
        const blob = new Blob([bytes], { type: 'audio/mpeg' })
        audio.src = URL.createObjectURL(blob)
      } else if (audioUrl) {
        audio.src = audioUrl
      }

      audio.volume = volume
      audio.preload = 'metadata'

      // Event listeners
      audio.addEventListener('loadedmetadata', () => {
        setDuration(audio.duration)
        setIsLoading(false)
      })

      audio.addEventListener('play', () => {
        setIsPlaying(true)
        onPlayStart?.()
        startProgressTracking()
      })

      audio.addEventListener('pause', () => {
        setIsPlaying(false)
        stopProgressTracking()
      })

      audio.addEventListener('ended', () => {
        setIsPlaying(false)
        setCurrentTime(0)
        onPlayEnd?.()
        stopProgressTracking()
      })

      audio.addEventListener('error', (e) => {
        const errorMessage = 'Failed to load audio'
        setError(errorMessage)
        setIsLoading(false)
        onError?.(errorMessage)
      })

      audio.addEventListener('timeupdate', () => {
        setCurrentTime(audio.currentTime)
      })

    } catch (error) {
      const errorMessage = 'Failed to create audio player'
      setError(errorMessage)
      setIsLoading(false)
      onError?.(errorMessage)
    }
  }

  const startProgressTracking = () => {
    progressIntervalRef.current = setInterval(() => {
      if (audioRef.current) {
        setCurrentTime(audioRef.current.currentTime)
      }
    }, 100)
  }

  const stopProgressTracking = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current)
      progressIntervalRef.current = null
    }
  }

  const playAudio = () => {
    if (audioRef.current && !error) {
      audioRef.current.play().catch((error) => {
        console.error('Failed to play audio:', error)
        setError('Failed to play audio')
      })
    }
  }

  const pauseAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause()
    }
  }

  const togglePlayPause = () => {
    if (isPlaying) {
      pauseAudio()
    } else {
      playAudio()
    }
  }

  const seekTo = (time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time
      setCurrentTime(time)
    }
  }

  const handleProgressChange = (value: number[]) => {
    const newTime = (value[0] / 100) * duration
    seekTo(newTime)
  }

  const handleVolumeChange = (value: number[]) => {
    const newVolume = value[0] / 100
    setVolume(newVolume)
    if (audioRef.current) {
      audioRef.current.volume = newVolume
    }
    if (newVolume > 0 && isMuted) {
      setIsMuted(false)
    }
  }

  const toggleMute = () => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted
      setIsMuted(!isMuted)
    }
  }

  const restart = () => {
    seekTo(0)
    if (!isPlaying) {
      playAudio()
    }
  }

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  if (error) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="text-center text-red-500 text-sm">
            {error}
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isLoading) {
    return (
      <Card className={className}>
        <CardContent className="p-4">
          <div className="flex items-center justify-center gap-2">
            <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-sm">Loading audio...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!audioData && !audioUrl) {
    return null
  }

  return (
    <Card className={className}>
      <CardContent className="p-4 space-y-3">
        {/* Main Controls */}
        <div className="flex items-center gap-3">
          <Button
            onClick={togglePlayPause}
            variant="outline"
            size="sm"
            disabled={!duration}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>

          <Button
            onClick={restart}
            variant="ghost"
            size="sm"
            disabled={!duration}
          >
            <RotateCcw className="h-4 w-4" />
          </Button>

          <div className="flex-1 space-y-1">
            {/* Progress Bar */}
            <Slider
              value={[duration ? (currentTime / duration) * 100 : 0]}
              onValueChange={handleProgressChange}
              max={100}
              step={1}
              className="w-full"
              disabled={!duration}
            />
            
            {/* Time Display */}
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>
        </div>

        {/* Volume Controls */}
        {showControls && (
          <div className="flex items-center gap-2">
            <Button
              onClick={toggleMute}
              variant="ghost"
              size="sm"
            >
              {isMuted || volume === 0 ? (
                <VolumeX className="h-4 w-4" />
              ) : (
                <Volume2 className="h-4 w-4" />
              )}
            </Button>
            
            <Slider
              value={[isMuted ? 0 : volume * 100]}
              onValueChange={handleVolumeChange}
              max={100}
              step={1}
              className="w-20"
            />
          </div>
        )}

        {/* Status Indicator */}
        {isPlaying && (
          <div className="flex items-center justify-center gap-2 text-green-600">
            <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse" />
            <span className="text-xs">Playing AI Response</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default AudioPlayer