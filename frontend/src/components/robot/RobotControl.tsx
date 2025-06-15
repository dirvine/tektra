"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { AlertTriangle, Play, RefreshCw } from 'lucide-react'
import { api } from '@/lib/api'
import { robotWebSocket } from '@/lib/websocket'

interface Robot {
  id: string
  name: string
  status: string
  position: Record<string, number>
  battery: number
  last_command: string
  uptime: number
}

export default function RobotControl() {
  const [robots, setRobots] = useState<Robot[]>([])
  const [selectedRobot, setSelectedRobot] = useState<string>('')
  const [selectedCommand, setSelectedCommand] = useState('move_to')
  const [commandParams, setCommandParams] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isConnected, setIsConnected] = useState(false)

  const commands = [
    'move_to', 'pick_up', 'place', 'rotate', 'stop', 'home',
    'navigate', 'turn', 'open_gripper', 'close_gripper'
  ]

  useEffect(() => {
    loadRobots()
    
    // Connect to robot WebSocket
    robotWebSocket.connect().then(() => {
      setIsConnected(true)
    }).catch(console.error)

    // Listen for robot status updates
    robotWebSocket.on('robot_status', (message) => {
      const msgData = message.data as { robot_id?: string; status?: string; position?: Record<string, number> }
      setRobots(prev => prev.map(robot => 
        robot.id === msgData.robot_id 
          ? { ...robot, status: msgData.status || robot.status, position: msgData.position || robot.position }
          : robot
      ))
    })

    return () => {
      robotWebSocket.disconnect()
    }
  }, [])

  const loadRobots = async () => {
    try {
      const response = await api.getRobots()
      if (response.data && Array.isArray(response.data)) {
        setRobots(response.data as Robot[])
        if (response.data.length > 0) {
          setSelectedRobot((response.data[0] as Robot).id)
        }
      }
    } catch (error) {
      console.error('Failed to load robots:', error)
    }
  }

  const sendCommand = async () => {
    if (!selectedRobot || !selectedCommand) return
    setIsLoading(true)

    try {
      let parameters = {}
      if (commandParams.trim()) {
        try {
          parameters = JSON.parse(commandParams)
        } catch {
          // If not valid JSON, treat as simple key-value
          parameters = { value: commandParams }
        }
      }

      if (isConnected) {
        // Send via WebSocket for real-time updates
        robotWebSocket.send({
          type: 'command',
          data: {
            command: selectedCommand,
            parameters
          }
        })
      } else {
        // Fallback to HTTP API
        const response = await api.sendRobotCommand(selectedRobot, selectedCommand, parameters)
        if ((response.data as { status?: string })?.status === 'success') {
          loadRobots()
        }
      }
    } catch (error) {
      console.error('Failed to send command:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const emergencyStop = async (robotId: string) => {
    try {
      await api.emergencyStop(robotId)
      loadRobots()
    } catch (error) {
      console.error('Failed to emergency stop:', error)
    }
  }

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'connected':
      case 'idle':
        return 'bg-green-500'
      case 'moving':
      case 'executing':
        return 'bg-blue-500'
      case 'error':
        return 'bg-red-500'
      case 'disconnected':
        return 'bg-gray-400'
      default:
        return 'bg-yellow-500'
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Robot Control</span>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-muted-foreground">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Robot Selection */}
        <div>
          <h3 className="font-medium mb-2">Select Robot</h3>
          <select
            value={selectedRobot}
            onChange={(e) => setSelectedRobot(e.target.value)}
            className="w-full border rounded px-3 py-2"
          >
            {robots.map(robot => (
              <option key={robot.id} value={robot.id}>
                {robot.name} ({robot.id})
              </option>
            ))}
          </select>
        </div>

        {/* Robot Status */}
        {selectedRobot && robots.find(r => r.id === selectedRobot) && (
          <div className="bg-muted p-4 rounded-lg">
            {(() => {
              const robot = robots.find(r => r.id === selectedRobot)!
              return (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{robot.name}</span>
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(robot.status)}`} />
                      <span className="text-sm">{robot.status}</span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Battery:</span> {robot.battery}%
                    </div>
                    <div>
                      <span className="font-medium">Uptime:</span> {formatUptime(robot.uptime)}
                    </div>
                    <div>
                      <span className="font-medium">Last Command:</span> {robot.last_command}
                    </div>
                    <div>
                      <span className="font-medium">Position:</span> 
                      {Object.entries(robot.position).map(([key, value]) => 
                        ` ${key.toUpperCase()}:${value}`
                      ).join(',')}
                    </div>
                  </div>
                </div>
              )
            })()}
          </div>
        )}

        {/* Command Control */}
        <div>
          <h3 className="font-medium mb-2">Send Command</h3>
          <div className="space-y-2">
            <select
              value={selectedCommand}
              onChange={(e) => setSelectedCommand(e.target.value)}
              className="w-full border rounded px-3 py-2"
            >
              {commands.map(cmd => (
                <option key={cmd} value={cmd}>
                  {cmd.replace('_', ' ').charAt(0).toUpperCase() + cmd.replace('_', ' ').slice(1)}
                </option>
              ))}
            </select>
            <Input
              value={commandParams}
              onChange={(e) => setCommandParams(e.target.value)}
              placeholder="Parameters (JSON or simple value)"
            />
            <Button 
              onClick={sendCommand} 
              disabled={isLoading || !selectedRobot}
              className="w-full"
            >
              <Play className="h-4 w-4 mr-2" />
              Send Command
            </Button>
          </div>
        </div>

        {/* Emergency Stop */}
        {selectedRobot && (
          <div className="border-t pt-4">
            <Button 
              onClick={() => emergencyStop(selectedRobot)}
              variant="destructive"
              className="w-full"
            >
              <AlertTriangle className="h-4 w-4 mr-2" />
              Emergency Stop
            </Button>
          </div>
        )}

        {/* Refresh Button */}
        <div className="flex justify-center">
          <Button onClick={loadRobots} variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh Status
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}