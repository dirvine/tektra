"use client"

import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'

export interface AnalyticsMetrics {
  conversations: number
  messages: number
  tokens: number
  avgResponseTime: number
  activeModels: number
  efficiency: number
}

export interface UsagePattern {
  hourly: Array<{ hour: number, count: number }>
  daily: Array<{ day: string, count: number }>
  weekly: Array<{ week: string, count: number }>
  monthly: Array<{ month: string, count: number }>
}

export interface ModelAnalytics {
  name: string
  usage: number
  percentage: number
  avgResponseTime: number
  efficiency: number
  tokens: number
}

export interface TrendData {
  metric: string
  current: number
  previous: number
  change: number
  trend: 'up' | 'down' | 'stable'
}

export interface AnalyticsInsight {
  type: 'performance' | 'usage' | 'efficiency' | 'recommendation'
  title: string
  description: string
  value?: number
  trend?: 'positive' | 'negative' | 'neutral'
  actionable?: boolean
}

export interface UseAnalyticsOptions {
  userId: number
  timeRange?: number // days
  autoRefresh?: boolean
  refreshInterval?: number // milliseconds
}

export interface UseAnalyticsReturn {
  // Data
  metrics: AnalyticsMetrics | null
  patterns: UsagePattern | null
  models: ModelAnalytics[]
  trends: TrendData[]
  insights: AnalyticsInsight[]
  
  // State
  isLoading: boolean
  error: string | null
  lastUpdated: Date | null
  
  // Actions
  refresh: () => Promise<void>
  setTimeRange: (days: number) => void
  exportData: (format: 'json' | 'csv') => Promise<void>
  
  // Computed values
  getTopModel: () => ModelAnalytics | null
  getEfficiencyTrend: () => TrendData | null
  getPeakUsageHour: () => number
  getProductivityScore: () => number
}

export function useAnalytics({
  userId,
  timeRange = 30,
  autoRefresh = false,
  refreshInterval = 300000 // 5 minutes
}: UseAnalyticsOptions): UseAnalyticsReturn {
  const [metrics, setMetrics] = useState<AnalyticsMetrics | null>(null)
  const [patterns, setPatterns] = useState<UsagePattern | null>(null)
  const [models, setModels] = useState<ModelAnalytics[]>([])
  const [trends, setTrends] = useState<TrendData[]>([])
  const [insights, setInsights] = useState<AnalyticsInsight[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [currentTimeRange, setCurrentTimeRange] = useState(timeRange)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      // Load analytics data
      const response = await api.getConversationAnalytics(undefined, currentTimeRange)
      
      if (response.data) {
        // In a real implementation, this would parse the actual API response
        // For now, we'll generate realistic mock data
        
        const mockMetrics: AnalyticsMetrics = {
          conversations: 127 + Math.floor(Math.random() * 20),
          messages: 2846 + Math.floor(Math.random() * 200),
          tokens: 1250000 + Math.floor(Math.random() * 100000),
          avgResponseTime: 2.3 + (Math.random() - 0.5) * 0.4,
          activeModels: 5,
          efficiency: 92 + Math.floor(Math.random() * 8)
        }
        
        const mockPatterns: UsagePattern = {
          hourly: generateHourlyPattern(),
          daily: generateDailyPattern(currentTimeRange),
          weekly: generateWeeklyPattern(Math.ceil(currentTimeRange / 7)),
          monthly: generateMonthlyPattern(Math.ceil(currentTimeRange / 30))
        }
        
        const mockModels: ModelAnalytics[] = [
          {
            name: 'phi-3-mini',
            usage: 67,
            percentage: 45,
            avgResponseTime: 1.8,
            efficiency: 95,
            tokens: 450000
          },
          {
            name: 'llama-3-8b',
            usage: 34,
            percentage: 28,
            avgResponseTime: 3.2,
            efficiency: 88,
            tokens: 520000
          },
          {
            name: 'mistral-7b',
            usage: 18,
            percentage: 15,
            avgResponseTime: 2.7,
            efficiency: 91,
            tokens: 280000
          },
          {
            name: 'gemma-2b',
            usage: 8,
            percentage: 12,
            avgResponseTime: 1.5,
            efficiency: 93,
            tokens: 120000
          }
        ]
        
        const mockTrends: TrendData[] = [
          {
            metric: 'conversations',
            current: mockMetrics.conversations,
            previous: mockMetrics.conversations - 15,
            change: 15,
            trend: 'up'
          },
          {
            metric: 'efficiency',
            current: mockMetrics.efficiency,
            previous: mockMetrics.efficiency - 8,
            change: 8,
            trend: 'up'
          },
          {
            metric: 'responseTime',
            current: mockMetrics.avgResponseTime,
            previous: mockMetrics.avgResponseTime + 0.3,
            change: -0.3,
            trend: 'up' // Lower response time is better
          }
        ]
        
        const mockInsights: AnalyticsInsight[] = generateInsights(mockMetrics, mockModels, mockPatterns)
        
        setMetrics(mockMetrics)
        setPatterns(mockPatterns)
        setModels(mockModels)
        setTrends(mockTrends)
        setInsights(mockInsights)
        setLastUpdated(new Date())
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load analytics'
      setError(errorMessage)
      console.error('Analytics loading error:', err)
    } finally {
      setIsLoading(false)
    }
  }, [userId, currentTimeRange])

  const setTimeRange = useCallback((days: number) => {
    setCurrentTimeRange(days)
  }, [])

  const exportData = useCallback(async (format: 'json' | 'csv') => {
    try {
      const data = {
        metrics,
        patterns,
        models,
        trends,
        insights,
        exportedAt: new Date().toISOString(),
        timeRange: currentTimeRange
      }
      
      if (format === 'json') {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `tektra-analytics-${Date.now()}.json`
        link.click()
        URL.revokeObjectURL(url)
      } else if (format === 'csv') {
        // Simple CSV export for metrics
        const csvData = [
          ['Metric', 'Value'],
          ['Conversations', metrics?.conversations || 0],
          ['Messages', metrics?.messages || 0],
          ['Tokens', metrics?.tokens || 0],
          ['Avg Response Time', metrics?.avgResponseTime || 0],
          ['Efficiency', metrics?.efficiency || 0]
        ]
        
        const csvContent = csvData.map(row => row.join(',')).join('\n')
        const blob = new Blob([csvContent], { type: 'text/csv' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `tektra-analytics-${Date.now()}.csv`
        link.click()
        URL.revokeObjectURL(url)
      }
    } catch (err) {
      console.error('Export failed:', err)
      setError('Failed to export data')
    }
  }, [metrics, patterns, models, trends, insights, currentTimeRange])

  const getTopModel = useCallback((): ModelAnalytics | null => {
    if (models.length === 0) return null
    return models.reduce((top, current) => 
      current.usage > top.usage ? current : top
    )
  }, [models])

  const getEfficiencyTrend = useCallback((): TrendData | null => {
    return trends.find(t => t.metric === 'efficiency') || null
  }, [trends])

  const getPeakUsageHour = useCallback((): number => {
    if (!patterns?.hourly) return 14 // Default to 2 PM
    return patterns.hourly.reduce((peak, current) => 
      current.count > peak.count ? current : peak
    ).hour
  }, [patterns])

  const getProductivityScore = useCallback((): number => {
    if (!metrics) return 0
    
    // Calculate productivity score based on various factors
    const efficiencyScore = metrics.efficiency
    const responseScore = Math.max(0, 100 - (metrics.avgResponseTime * 20))
    const usageScore = Math.min(100, (metrics.conversations / 30) * 100) // Normalize to 30 conversations
    
    return Math.round((efficiencyScore * 0.4 + responseScore * 0.3 + usageScore * 0.3))
  }, [metrics])

  // Auto-refresh effect
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      const interval = setInterval(refresh, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refreshInterval, refresh])

  // Initial load and time range changes
  useEffect(() => {
    refresh()
  }, [refresh])

  return {
    metrics,
    patterns,
    models,
    trends,
    insights,
    isLoading,
    error,
    lastUpdated,
    refresh,
    setTimeRange,
    exportData,
    getTopModel,
    getEfficiencyTrend,
    getPeakUsageHour,
    getProductivityScore
  }
}

// Helper functions for generating mock data
function generateHourlyPattern(): Array<{ hour: number, count: number }> {
  const hours = []
  for (let i = 0; i < 24; i++) {
    let count = 0
    if (i >= 9 && i <= 17) {
      count = Math.floor(Math.random() * 15) + 5 // Work hours
    } else if (i >= 19 && i <= 22) {
      count = Math.floor(Math.random() * 8) + 2 // Evening
    } else {
      count = Math.floor(Math.random() * 3) // Other hours
    }
    hours.push({ hour: i, count })
  }
  return hours
}

function generateDailyPattern(days: number): Array<{ day: string, count: number }> {
  const pattern = []
  const now = new Date()
  
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(now)
    date.setDate(date.getDate() - i)
    
    const dayName = date.toLocaleDateString('en-US', { weekday: 'short' })
    const isWeekend = date.getDay() === 0 || date.getDay() === 6
    const count = isWeekend 
      ? Math.floor(Math.random() * 8) + 2
      : Math.floor(Math.random() * 20) + 10
    
    pattern.push({ day: dayName, count })
  }
  
  return pattern
}

function generateWeeklyPattern(weeks: number): Array<{ week: string, count: number }> {
  const pattern = []
  const now = new Date()
  
  for (let i = weeks - 1; i >= 0; i--) {
    const date = new Date(now)
    date.setDate(date.getDate() - (i * 7))
    
    const weekStart = new Date(date)
    weekStart.setDate(date.getDate() - date.getDay())
    
    const week = `Week of ${weekStart.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
    const count = Math.floor(Math.random() * 100) + 50
    
    pattern.push({ week, count })
  }
  
  return pattern
}

function generateMonthlyPattern(months: number): Array<{ month: string, count: number }> {
  const pattern = []
  const now = new Date()
  
  for (let i = months - 1; i >= 0; i--) {
    const date = new Date(now)
    date.setMonth(date.getMonth() - i)
    
    const month = date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
    const count = Math.floor(Math.random() * 500) + 200
    
    pattern.push({ month, count })
  }
  
  return pattern
}

function generateInsights(
  metrics: AnalyticsMetrics, 
  models: ModelAnalytics[], 
  patterns: UsagePattern
): AnalyticsInsight[] {
  const insights: AnalyticsInsight[] = []
  
  // Performance insights
  if (metrics.efficiency > 90) {
    insights.push({
      type: 'performance',
      title: 'Excellent Efficiency',
      description: `Your ${metrics.efficiency}% efficiency rate is outstanding! You're making great use of AI assistance.`,
      value: metrics.efficiency,
      trend: 'positive'
    })
  }
  
  // Usage insights
  const topModel = models[0]
  if (topModel) {
    insights.push({
      type: 'usage',
      title: 'Preferred Model',
      description: `You use ${topModel.name} for ${topModel.percentage}% of conversations. It's giving you ${topModel.avgResponseTime.toFixed(1)}s average response time.`,
      trend: 'neutral'
    })
  }
  
  // Efficiency recommendations
  if (metrics.avgResponseTime > 3.0) {
    insights.push({
      type: 'recommendation',
      title: 'Speed Optimization',
      description: 'Consider using lighter models like Phi-3 Mini for simple tasks to improve response times.',
      actionable: true,
      trend: 'neutral'
    })
  }
  
  // Peak usage insights
  const peakHour = patterns.hourly.reduce((peak, current) => 
    current.count > peak.count ? current : peak
  )
  
  insights.push({
    type: 'usage',
    title: 'Peak Activity',
    description: `You're most active at ${peakHour.hour}:00. This might be your optimal productivity window.`,
    value: peakHour.hour,
    trend: 'neutral'
  })
  
  return insights
}