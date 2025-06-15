"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { 
  TrendingUp,
  MessageSquare,
  Clock,
  Brain,
  Zap,
  ChevronRight,
  ArrowUp,
  ArrowDown,
  Target,
  Activity
} from 'lucide-react'
import { api } from '@/lib/api'

interface AnalyticsWidgetProps {
  userId: number
  onViewDetails?: () => void
  compact?: boolean
}

interface QuickStats {
  totalConversations: number
  totalMessages: number
  avgResponseTime: number
  efficiency: number
  trends: {
    conversations: number
    efficiency: number
  }
  topModel: string
  todayActivity: number
}

export default function AnalyticsWidget({ 
  userId, 
  onViewDetails, 
  compact = false 
}: AnalyticsWidgetProps) {
  const [stats, setStats] = useState<QuickStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadQuickStats()
  }, [userId])

  const loadQuickStats = async () => {
    setIsLoading(true)
    try {
      const response = await api.getConversationAnalytics(undefined, 7)
      if (response.data) {
        // Mock data - in real implementation this would come from the API
        const quickStats: QuickStats = {
          totalConversations: 127,
          totalMessages: 2846,
          avgResponseTime: 2.3,
          efficiency: 92,
          trends: {
            conversations: 15, // +15% this week
            efficiency: 8      // +8% this week
          },
          topModel: 'phi-3-mini',
          todayActivity: 12
        }
        setStats(quickStats)
      }
    } catch (error) {
      console.error('Failed to load quick stats:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const formatTime = (seconds: number): string => {
    if (seconds < 1) {
      return `${Math.round(seconds * 1000)}ms`
    } else if (seconds < 60) {
      return `${seconds.toFixed(1)}s`
    }
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  }

  const getTrendIcon = (value: number) => {
    return value > 0 ? (
      <ArrowUp className="h-3 w-3 text-green-500" />
    ) : value < 0 ? (
      <ArrowDown className="h-3 w-3 text-red-500" />
    ) : null
  }

  const getTrendColor = (value: number) => {
    return value > 0 ? 'text-green-600' : value < 0 ? 'text-red-600' : 'text-gray-600'
  }

  if (isLoading) {
    return (
      <Card className={compact ? "w-80" : "w-full"}>
        <CardContent className="flex items-center justify-center py-8">
          <div className="text-center">
            <Activity className="h-6 w-6 animate-pulse mx-auto mb-2 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Loading analytics...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!stats) {
    return (
      <Card className={compact ? "w-80" : "w-full"}>
        <CardContent className="flex items-center justify-center py-8">
          <div className="text-center">
            <p className="text-sm text-muted-foreground">Analytics unavailable</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (compact) {
    return (
      <Card className="w-80">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Quick Stats
            </div>
            {onViewDetails && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onViewDetails}
                className="h-6 w-6 p-0"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Key Metrics */}
          <div className="grid grid-cols-2 gap-3 text-center">
            <div>
              <p className="text-2xl font-bold text-blue-600">{stats.totalConversations}</p>
              <p className="text-xs text-muted-foreground">Conversations</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-green-600">{stats.efficiency}%</p>
              <p className="text-xs text-muted-foreground">Efficiency</p>
            </div>
          </div>

          {/* Trends */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>This week</span>
              <div className="flex items-center gap-1">
                {getTrendIcon(stats.trends.conversations)}
                <span className={getTrendColor(stats.trends.conversations)}>
                  {stats.trends.conversations > 0 ? '+' : ''}{stats.trends.conversations}%
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span>Performance</span>
              <div className="flex items-center gap-1">
                {getTrendIcon(stats.trends.efficiency)}
                <span className={getTrendColor(stats.trends.efficiency)}>
                  {stats.trends.efficiency > 0 ? '+' : ''}{stats.trends.efficiency}%
                </span>
              </div>
            </div>
          </div>

          {/* Today's Activity */}
          <div className="pt-2 border-t">
            <div className="flex items-center justify-between text-sm">
              <span>Today's activity</span>
              <Badge variant="outline">{stats.todayActivity} messages</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Analytics Overview
          </div>
          {onViewDetails && (
            <Button variant="outline" onClick={onViewDetails}>
              View Details
              <ChevronRight className="h-4 w-4 ml-2" />
            </Button>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="flex items-center justify-center gap-1 mb-2">
              <MessageSquare className="h-4 w-4 text-blue-500" />
              <span className="text-sm font-medium">Conversations</span>
            </div>
            <p className="text-2xl font-bold">{stats.totalConversations}</p>
            <div className="flex items-center justify-center gap-1 mt-1">
              {getTrendIcon(stats.trends.conversations)}
              <span className={`text-xs ${getTrendColor(stats.trends.conversations)}`}>
                {stats.trends.conversations > 0 ? '+' : ''}{stats.trends.conversations}%
              </span>
            </div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center gap-1 mb-2">
              <Clock className="h-4 w-4 text-purple-500" />
              <span className="text-sm font-medium">Avg Response</span>
            </div>
            <p className="text-2xl font-bold">{formatTime(stats.avgResponseTime)}</p>
            <p className="text-xs text-muted-foreground">response time</p>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center gap-1 mb-2">
              <Target className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium">Efficiency</span>
            </div>
            <p className="text-2xl font-bold">{stats.efficiency}%</p>
            <div className="flex items-center justify-center gap-1 mt-1">
              {getTrendIcon(stats.trends.efficiency)}
              <span className={`text-xs ${getTrendColor(stats.trends.efficiency)}`}>
                {stats.trends.efficiency > 0 ? '+' : ''}{stats.trends.efficiency}%
              </span>
            </div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center gap-1 mb-2">
              <Brain className="h-4 w-4 text-indigo-500" />
              <span className="text-sm font-medium">Top Model</span>
            </div>
            <p className="text-lg font-bold">{stats.topModel}</p>
            <p className="text-xs text-muted-foreground">most used</p>
          </div>
        </div>

        {/* Performance Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Overall Performance</span>
            <span className="text-sm text-muted-foreground">{stats.efficiency}%</span>
          </div>
          <Progress value={stats.efficiency} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Poor</span>
            <span>Good</span>
            <span>Excellent</span>
          </div>
        </div>

        {/* Quick Insights */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-3 rounded-lg bg-blue-50 border border-blue-200">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="h-4 w-4 text-blue-500" />
              <span className="text-sm font-medium text-blue-900">Peak Performance</span>
            </div>
            <p className="text-xs text-blue-700">
              Your efficiency is up {stats.trends.efficiency}% this week!
            </p>
          </div>

          <div className="p-3 rounded-lg bg-green-50 border border-green-200">
            <div className="flex items-center gap-2 mb-1">
              <Activity className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium text-green-900">Growing Usage</span>
            </div>
            <p className="text-xs text-green-700">
              {stats.todayActivity} messages sent today, keeping the momentum!
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}