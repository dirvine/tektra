"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Activity, Bot, Camera, Mic, Settings, User } from 'lucide-react'
import EnhancedChatInterface from '@/components/chat/EnhancedChatInterface'
import AvatarControl from '@/components/avatar/AvatarControl'
import RobotControl from '@/components/robot/RobotControl'
import { api } from '@/lib/api'

interface SystemStatus {
  backend: boolean
  models: number
  robots: number
  avatar: boolean
  camera: boolean
  audio: boolean
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('chat')
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    backend: false,
    models: 0,
    robots: 0,
    avatar: false,
    camera: false,
    audio: false
  })

  useEffect(() => {
    checkSystemStatus()
    const interval = setInterval(checkSystemStatus, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const checkSystemStatus = async () => {
    try {
      // Check backend health
      const healthResponse = await api.healthCheck()
      const backendHealthy = healthResponse.status === 200

      // Check models
      const modelsResponse = await api.getModels()
      const modelCount = Array.isArray(modelsResponse.data) ? modelsResponse.data.length : 0

      // Check robots
      const robotsResponse = await api.getRobots()
      const robotCount = Array.isArray(robotsResponse.data) ? robotsResponse.data.length : 0

      // Check avatar
      const avatarResponse = await api.getAvatarStatus()
      const avatarActive = (avatarResponse.data as { active?: boolean })?.active || false

      // Check camera
      const cameraResponse = await api.getCameraStatus()
      const cameraConnected = (cameraResponse.data as { connected?: boolean })?.connected || false

      // Check audio (simplified)
      const audioConnected = true // Assume audio is available

      setSystemStatus({
        backend: backendHealthy,
        models: modelCount,
        robots: robotCount,
        avatar: avatarActive,
        camera: cameraConnected,
        audio: audioConnected
      })
    } catch (error) {
      console.error('Failed to check system status:', error)
      setSystemStatus(prev => ({ ...prev, backend: false }))
    }
  }

  const tabs = [
    { id: 'chat', label: 'Chat', icon: Bot },
    { id: 'avatar', label: 'Avatar', icon: User },
    { id: 'robot', label: 'Robot', icon: Activity },
    { id: 'camera', label: 'Camera', icon: Camera },
    { id: 'audio', label: 'Audio', icon: Mic },
  ]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'avatar':
        return <AvatarControl />
      case 'robot':
        return <RobotControl />
      case 'camera':
        return (
          <Card>
            <CardHeader>
              <CardTitle>Camera Control</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center text-muted-foreground">
                Camera interface coming in Week 2...
              </div>
            </CardContent>
          </Card>
        )
      case 'audio':
        return (
          <Card>
            <CardHeader>
              <CardTitle>Audio Control</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center text-muted-foreground">
                Audio interface coming in Week 2...
              </div>
            </CardContent>
          </Card>
        )
      default:
        return null
    }
  }

  // For chat tab, show full-screen interface
  if (activeTab === 'chat') {
    return <EnhancedChatInterface onNavigate={setActiveTab} />
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">Tektra AI Assistant</h1>
            <div className="flex items-center gap-4">
              {/* System Status Indicators */}
              <div className="flex items-center gap-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${systemStatus.backend ? 'bg-green-500' : 'bg-red-500'}`} />
                <span>Backend</span>
              </div>
              <div className="text-sm text-muted-foreground">
                {systemStatus.models} Models | {systemStatus.robots} Robots
              </div>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle>Navigation</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {tabs.map((tab) => {
                  const Icon = tab.icon
                  return (
                    <Button
                      key={tab.id}
                      variant={activeTab === tab.id ? "default" : "ghost"}
                      className="w-full justify-start"
                      onClick={() => setActiveTab(tab.id)}
                    >
                      <Icon className="h-4 w-4 mr-2" />
                      {tab.label}
                    </Button>
                  )
                })}
              </CardContent>
            </Card>

            {/* Quick Status */}
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>System Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Backend</span>
                  <div className={`w-3 h-3 rounded-full ${systemStatus.backend ? 'bg-green-500' : 'bg-red-500'}`} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Avatar</span>
                  <div className={`w-3 h-3 rounded-full ${systemStatus.avatar ? 'bg-green-500' : 'bg-gray-400'}`} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Camera</span>
                  <div className={`w-3 h-3 rounded-full ${systemStatus.camera ? 'bg-green-500' : 'bg-gray-400'}`} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Audio</span>
                  <div className={`w-3 h-3 rounded-full ${systemStatus.audio ? 'bg-green-500' : 'bg-gray-400'}`} />
                </div>
                <div className="text-xs text-muted-foreground pt-2 border-t">
                  {systemStatus.models} AI Models Available
                  <br />
                  {systemStatus.robots} Robots Connected
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {renderTabContent()}
          </div>
        </div>
      </div>
    </div>
  )
}