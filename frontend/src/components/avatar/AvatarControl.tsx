"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Play, RotateCcw } from 'lucide-react'
import { api } from '@/lib/api'

interface AvatarStatus {
  active: boolean
  current_expression: string
  current_gesture: string
  speaking: boolean
  position: { x: number; y: number; z: number }
}

export default function AvatarControl() {
  const [status, setStatus] = useState<AvatarStatus | null>(null)
  const [selectedExpression, setSelectedExpression] = useState('neutral')
  const [selectedGesture, setSelectedGesture] = useState('wave')
  const [speechText, setSpeechText] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const expressions = [
    'neutral', 'happy', 'sad', 'angry', 'surprised',
    'confused', 'thinking', 'excited', 'calm'
  ]

  const gestures = [
    'wave', 'nod', 'shake_head', 'point', 'thumbs_up',
    'shrug', 'clap', 'peace_sign', 'thinking_pose'
  ]

  useEffect(() => {
    loadAvatarStatus()
  }, [])

  const loadAvatarStatus = async () => {
    try {
      const response = await api.getAvatarStatus()
      if (response.data) {
        setStatus(response.data as AvatarStatus)
      }
    } catch (error) {
      console.error('Failed to load avatar status:', error)
    }
  }

  const setExpression = async () => {
    if (!selectedExpression) return
    setIsLoading(true)

    try {
      const response = await api.setAvatarExpression(selectedExpression, 1.0)
      if ((response.data as { status?: string })?.status === 'success') {
        loadAvatarStatus()
      }
    } catch (error) {
      console.error('Failed to set expression:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const triggerGesture = async () => {
    if (!selectedGesture) return
    setIsLoading(true)

    try {
      const response = await api.triggerAvatarGesture(selectedGesture, 1.0)
      if ((response.data as { status?: string })?.status === 'success') {
        loadAvatarStatus()
      }
    } catch (error) {
      console.error('Failed to trigger gesture:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const makeSpeak = async () => {
    if (!speechText.trim()) return
    setIsLoading(true)

    try {
      const response = await api.makeAvatarSpeak(speechText, selectedExpression)
      if ((response.data as { status?: string })?.status === 'success') {
        loadAvatarStatus()
        setSpeechText('')
      }
    } catch (error) {
      console.error('Failed to make avatar speak:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const resetAvatar = async () => {
    setIsLoading(true)

    try {
      // Reset avatar to default state
      await api.setAvatarExpression('neutral', 1.0)
      await loadAvatarStatus()
    } catch (error) {
      console.error('Failed to reset avatar:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Avatar Control</span>
          <div className={`w-3 h-3 rounded-full ${status?.active ? 'bg-green-500' : 'bg-gray-400'}`} />
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Status Display */}
        {status && (
          <div className="bg-muted p-4 rounded-lg">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Expression:</span> {status.current_expression}
              </div>
              <div>
                <span className="font-medium">Gesture:</span> {status.current_gesture}
              </div>
              <div>
                <span className="font-medium">Speaking:</span> {status.speaking ? 'Yes' : 'No'}
              </div>
              <div>
                <span className="font-medium">Position:</span> 
                {` X:${status.position.x} Y:${status.position.y} Z:${status.position.z}`}
              </div>
            </div>
          </div>
        )}

        {/* Expression Control */}
        <div>
          <h3 className="font-medium mb-2">Expression</h3>
          <div className="flex gap-2">
            <select
              value={selectedExpression}
              onChange={(e) => setSelectedExpression(e.target.value)}
              className="flex-1 border rounded px-3 py-2"
            >
              {expressions.map(expr => (
                <option key={expr} value={expr}>
                  {expr.charAt(0).toUpperCase() + expr.slice(1)}
                </option>
              ))}
            </select>
            <Button onClick={setExpression} disabled={isLoading}>
              <Play className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Gesture Control */}
        <div>
          <h3 className="font-medium mb-2">Gesture</h3>
          <div className="flex gap-2">
            <select
              value={selectedGesture}
              onChange={(e) => setSelectedGesture(e.target.value)}
              className="flex-1 border rounded px-3 py-2"
            >
              {gestures.map(gesture => (
                <option key={gesture} value={gesture}>
                  {gesture.replace('_', ' ').charAt(0).toUpperCase() + gesture.replace('_', ' ').slice(1)}
                </option>
              ))}
            </select>
            <Button onClick={triggerGesture} disabled={isLoading}>
              <Play className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Speech Control */}
        <div>
          <h3 className="font-medium mb-2">Speech</h3>
          <div className="flex gap-2">
            <Input
              value={speechText}
              onChange={(e) => setSpeechText(e.target.value)}
              placeholder="Enter text for avatar to speak..."
              className="flex-1"
            />
            <Button onClick={makeSpeak} disabled={isLoading || !speechText.trim()}>
              <Play className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Reset Button */}
        <div className="flex justify-center">
          <Button onClick={resetAvatar} variant="outline" disabled={isLoading}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset Avatar
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}