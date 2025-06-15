"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { 
  BarChart3,
  TrendingUp,
  MessageSquare,
  Clock,
  Brain,
  Users,
  Calendar,
  Zap,
  Target,
  Award,
  Activity,
  PieChart,
  LineChart,
  RefreshCw,
  Download,
  Filter,
  ChevronRight,
  Sparkles,
  Timer,
  Hash,
  ArrowUp,
  ArrowDown,
  Minus,
  Star,
  CheckCircle,
  AlertCircle
} from 'lucide-react'
import { api } from '@/lib/api'

interface AnalyticsDashboardProps {
  userId: number
  onClose?: () => void
}

interface AnalyticsData {
  overview: {
    totalConversations: number
    totalMessages: number
    totalTokens: number
    avgResponseTime: number
    activeModels: number
    favoriteTemplates: number
  }
  trends: {
    conversationsOverTime: Array<{ date: string, count: number }>
    messagesOverTime: Array<{ date: string, count: number }>
    tokensOverTime: Array<{ date: string, count: number }>
    responseTimeOverTime: Array<{ date: string, avgTime: number }>
  }
  usage: {
    modelUsage: Array<{ model: string, count: number, percentage: number }>
    categoryUsage: Array<{ category: string, count: number, percentage: number }>
    templateUsage: Array<{ template: string, count: number, percentage: number }>
    hourlyActivity: Array<{ hour: number, count: number }>
    weeklyActivity: Array<{ day: string, count: number }>
  }
  performance: {
    averageResponseTime: number
    fastestResponse: number
    slowestResponse: number
    totalProcessingTime: number
    efficiency: number
  }
  insights: {
    mostProductiveHour: number
    mostProductiveDay: string
    favoriteModel: string
    favoriteCategory: string
    averageConversationLength: number
    longestConversation: number
    trends: Array<{ type: string, trend: 'up' | 'down' | 'stable', value: number, description: string }>
  }
}

export default function AnalyticsDashboard({ userId, onClose }: AnalyticsDashboardProps) {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [timeRange, setTimeRange] = useState('30')
  const [selectedMetric, setSelectedMetric] = useState('conversations')

  useEffect(() => {
    loadAnalytics()
  }, [userId, timeRange])

  const loadAnalytics = async () => {
    setIsLoading(true)
    try {
      const response = await api.getConversationAnalytics(undefined, parseInt(timeRange))
      if (response.data) {
        // Transform the data into our expected format
        const data = response.data as any
        
        // Create mock analytics data structure (in real implementation, this would come from the API)
        const analyticsData: AnalyticsData = {
          overview: {
            totalConversations: data.totalConversations || 127,
            totalMessages: data.totalMessages || 2846,
            totalTokens: data.totalTokens || 1250000,
            avgResponseTime: data.avgResponseTime || 2.3,
            activeModels: data.activeModels || 5,
            favoriteTemplates: data.favoriteTemplates || 12
          },
          trends: {
            conversationsOverTime: generateTrendData(parseInt(timeRange), 'conversations'),
            messagesOverTime: generateTrendData(parseInt(timeRange), 'messages'),
            tokensOverTime: generateTrendData(parseInt(timeRange), 'tokens'),
            responseTimeOverTime: generateTrendData(parseInt(timeRange), 'responseTime')
          },
          usage: {
            modelUsage: [
              { model: 'phi-3-mini', count: 67, percentage: 45 },
              { model: 'llama-3-8b', count: 34, percentage: 28 },
              { model: 'mistral-7b', count: 18, percentage: 15 },
              { model: 'gemma-2b', count: 8, percentage: 12 }
            ],
            categoryUsage: [
              { category: 'work', count: 45, percentage: 35 },
              { category: 'coding', count: 38, percentage: 30 },
              { category: 'creative', count: 25, percentage: 20 },
              { category: 'research', count: 19, percentage: 15 }
            ],
            templateUsage: [
              { template: 'Code Review Assistant', count: 23, percentage: 25 },
              { template: 'Meeting Minutes', count: 18, percentage: 20 },
              { template: 'Creative Writing', count: 15, percentage: 16 },
              { template: 'Research Helper', count: 12, percentage: 13 }
            ],
            hourlyActivity: generateHourlyActivity(),
            weeklyActivity: [
              { day: 'Mon', count: 18 },
              { day: 'Tue', count: 24 },
              { day: 'Wed', count: 31 },
              { day: 'Thu', count: 28 },
              { day: 'Fri', count: 22 },
              { day: 'Sat', count: 12 },
              { day: 'Sun', count: 8 }
            ]
          },
          performance: {
            averageResponseTime: 2.3,
            fastestResponse: 0.8,
            slowestResponse: 8.2,
            totalProcessingTime: 6847,
            efficiency: 92
          },
          insights: {
            mostProductiveHour: 14,
            mostProductiveDay: 'Wednesday',
            favoriteModel: 'phi-3-mini',
            favoriteCategory: 'work',
            averageConversationLength: 22,
            longestConversation: 156,
            trends: [
              { type: 'conversations', trend: 'up', value: 15, description: '15% increase in conversations this week' },
              { type: 'efficiency', trend: 'up', value: 8, description: '8% improvement in response efficiency' },
              { type: 'tokens', trend: 'down', value: 3, description: '3% decrease in token usage (more efficient)' }
            ]
          }
        }
        
        setAnalytics(analyticsData)
      }
    } catch (error) {
      console.error('Failed to load analytics:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const generateTrendData = (days: number, type: string) => {
    const data = []
    const now = new Date()
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(now)
      date.setDate(date.getDate() - i)
      
      let value = 0
      switch (type) {
        case 'conversations':
          value = Math.floor(Math.random() * 10) + 5
          break
        case 'messages':
          value = Math.floor(Math.random() * 50) + 20
          break
        case 'tokens':
          value = Math.floor(Math.random() * 5000) + 1000
          break
        case 'responseTime':
          value = Math.random() * 3 + 1
          break
      }
      
      data.push({
        date: date.toISOString().split('T')[0],
        count: type === 'responseTime' ? undefined : value,
        avgTime: type === 'responseTime' ? value : undefined
      })
    }
    
    return data
  }

  const generateHourlyActivity = () => {
    const hours = []
    for (let i = 0; i < 24; i++) {
      // Simulate typical work patterns
      let count = 0
      if (i >= 9 && i <= 17) {
        count = Math.floor(Math.random() * 15) + 5 // Higher activity during work hours
      } else if (i >= 19 && i <= 22) {
        count = Math.floor(Math.random() * 8) + 2 // Some evening activity
      } else {
        count = Math.floor(Math.random() * 3) // Lower activity otherwise
      }
      
      hours.push({ hour: i, count })
    }
    return hours
  }

  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M'
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K'
    }
    return num.toString()
  }

  const formatTime = (seconds: number): string => {
    if (seconds < 1) {
      return `${Math.round(seconds * 1000)}ms`
    } else if (seconds < 60) {
      return `${seconds.toFixed(1)}s`
    } else {
      const minutes = Math.floor(seconds / 60)
      const remainingSeconds = seconds % 60
      return `${minutes}m ${remainingSeconds.toFixed(0)}s`
    }
  }

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up':
        return <ArrowUp className="h-3 w-3 text-green-500" />
      case 'down':
        return <ArrowDown className="h-3 w-3 text-red-500" />
      default:
        return <Minus className="h-3 w-3 text-gray-500" />
    }
  }

  const getTrendColor = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up':
        return 'text-green-600'
      case 'down':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  if (isLoading) {
    return (
      <Card className="w-full max-w-7xl mx-auto">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <BarChart3 className="h-8 w-8 animate-pulse mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Loading analytics...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!analytics) {
    return (
      <Card className="w-full max-w-7xl mx-auto">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <AlertCircle className="h-8 w-8 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Failed to load analytics data</p>
            <Button onClick={loadAnalytics} className="mt-4">
              <RefreshCw className="h-4 w-4 mr-2" />
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="w-full max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <BarChart3 className="h-8 w-8" />
            Analytics & Insights
          </h1>
          <p className="text-muted-foreground mt-1">
            Analyze your AI interaction patterns and optimize your workflow
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7">Last 7 days</SelectItem>
              <SelectItem value="30">Last 30 days</SelectItem>
              <SelectItem value="90">Last 3 months</SelectItem>
              <SelectItem value="365">Last year</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={loadAnalytics}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          {onClose && (
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Conversations</p>
                <p className="text-2xl font-bold">{analytics.overview.totalConversations}</p>
              </div>
              <MessageSquare className="h-4 w-4 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Messages</p>
                <p className="text-2xl font-bold">{formatNumber(analytics.overview.totalMessages)}</p>
              </div>
              <Hash className="h-4 w-4 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Tokens</p>
                <p className="text-2xl font-bold">{formatNumber(analytics.overview.totalTokens)}</p>
              </div>
              <Zap className="h-4 w-4 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Avg Response</p>
                <p className="text-2xl font-bold">{formatTime(analytics.overview.avgResponseTime)}</p>
              </div>
              <Timer className="h-4 w-4 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Models</p>
                <p className="text-2xl font-bold">{analytics.overview.activeModels}</p>
              </div>
              <Brain className="h-4 w-4 text-indigo-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Templates</p>
                <p className="text-2xl font-bold">{analytics.overview.favoriteTemplates}</p>
              </div>
              <Star className="h-4 w-4 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="usage">Usage Patterns</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Activity Heatmap */}
            <Card>
              <CardHeader>
                <CardTitle>Daily Activity</CardTitle>
                <CardDescription>
                  Your conversation activity throughout the week
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {analytics.usage.weeklyActivity.map((day, index) => (
                    <div key={day.day} className="flex items-center gap-3">
                      <div className="w-12 text-sm font-medium">{day.day}</div>
                      <div className="flex-1">
                        <Progress 
                          value={(day.count / Math.max(...analytics.usage.weeklyActivity.map(d => d.count))) * 100} 
                          className="h-2" 
                        />
                      </div>
                      <div className="w-8 text-sm text-muted-foreground">{day.count}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Model Usage */}
            <Card>
              <CardHeader>
                <CardTitle>Model Usage</CardTitle>
                <CardDescription>
                  Distribution of conversations by AI model
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {analytics.usage.modelUsage.map((model, index) => (
                    <div key={model.model} className="flex items-center gap-3">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium">{model.model}</span>
                          <span className="text-sm text-muted-foreground">{model.percentage}%</span>
                        </div>
                        <Progress value={model.percentage} className="h-2" />
                      </div>
                      <div className="w-8 text-sm text-muted-foreground">{model.count}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Trends */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Trends</CardTitle>
              <CardDescription>
                Key changes in your AI usage patterns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {analytics.insights.trends.map((trend, index) => (
                  <div key={index} className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                    {getTrendIcon(trend.trend)}
                    <div className="flex-1">
                      <div className="font-medium capitalize">{trend.type}</div>
                      <div className={`text-sm ${getTrendColor(trend.trend)}`}>
                        {trend.description}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Usage Patterns Tab */}
        <TabsContent value="usage" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Category Usage */}
            <Card>
              <CardHeader>
                <CardTitle>Category Distribution</CardTitle>
                <CardDescription>
                  How you use different conversation categories
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analytics.usage.categoryUsage.map((category, index) => (
                    <div key={category.category} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-blue-500" />
                        <span className="text-sm capitalize">{category.category}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">{category.count}</span>
                        <Badge variant="outline">{category.percentage}%</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Template Usage */}
            <Card>
              <CardHeader>
                <CardTitle>Popular Templates</CardTitle>
                <CardDescription>
                  Your most frequently used conversation templates
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analytics.usage.templateUsage.map((template, index) => (
                    <div key={template.template} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Award className="h-4 w-4 text-yellow-500" />
                        <span className="text-sm">{template.template}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">{template.count}</span>
                        <Badge variant="outline">{template.percentage}%</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Hourly Activity */}
          <Card>
            <CardHeader>
              <CardTitle>Hourly Activity Pattern</CardTitle>
              <CardDescription>
                Your conversation activity throughout the day
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-12 gap-1">
                {analytics.usage.hourlyActivity.map((hour, index) => (
                  <div key={hour.hour} className="text-center">
                    <div 
                      className="h-16 bg-blue-500 rounded-sm mb-1 relative group"
                      style={{ 
                        opacity: Math.max(0.1, hour.count / Math.max(...analytics.usage.hourlyActivity.map(h => h.count)))
                      }}
                    >
                      <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-black text-white text-xs px-1 py-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                        {hour.count}
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {hour.hour.toString().padStart(2, '0')}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Timer className="h-8 w-8 text-blue-500 mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Average Response</p>
                  <p className="text-2xl font-bold">{formatTime(analytics.performance.averageResponseTime)}</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Zap className="h-8 w-8 text-green-500 mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Fastest Response</p>
                  <p className="text-2xl font-bold">{formatTime(analytics.performance.fastestResponse)}</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Clock className="h-8 w-8 text-red-500 mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Slowest Response</p>
                  <p className="text-2xl font-bold">{formatTime(analytics.performance.slowestResponse)}</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <div className="text-center">
                  <Target className="h-8 w-8 text-purple-500 mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">Efficiency</p>
                  <p className="text-2xl font-bold">{analytics.performance.efficiency}%</p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Performance Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Breakdown</CardTitle>
              <CardDescription>
                Detailed analysis of your AI interaction performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Response Speed</span>
                    <span className="text-sm text-muted-foreground">Excellent</span>
                  </div>
                  <Progress value={85} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Token Efficiency</span>
                    <span className="text-sm text-muted-foreground">Very Good</span>
                  </div>
                  <Progress value={78} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Resource Usage</span>
                    <span className="text-sm text-muted-foreground">Good</span>
                  </div>
                  <Progress value={72} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Overall Efficiency</span>
                    <span className="text-sm text-muted-foreground">Excellent</span>
                  </div>
                  <Progress value={analytics.performance.efficiency} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trends Tab */}
        <TabsContent value="trends" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Activity Trends</CardTitle>
              <CardDescription>
                Track your AI usage over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                  <SelectTrigger className="w-48">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="conversations">Conversations</SelectItem>
                    <SelectItem value="messages">Messages</SelectItem>
                    <SelectItem value="tokens">Tokens</SelectItem>
                    <SelectItem value="responseTime">Response Time</SelectItem>
                  </SelectContent>
                </Select>

                {/* Simple trend visualization */}
                <div className="h-64 border rounded-lg p-4 bg-muted/20">
                  <div className="h-full flex items-end justify-between gap-1">
                    {analytics.trends.conversationsOverTime.slice(-14).map((point, index) => (
                      <div key={index} className="flex-1 flex flex-col items-center">
                        <div 
                          className="w-full bg-blue-500 rounded-t-sm min-h-[4px]"
                          style={{ 
                            height: `${Math.max(4, (point.count || 0) * 8)}px` 
                          }}
                        />
                        <div className="text-xs text-muted-foreground mt-1 rotate-45 origin-left">
                          {new Date(point.date).getDate()}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Key Insights */}
            <Card>
              <CardHeader>
                <CardTitle>Key Insights</CardTitle>
                <CardDescription>
                  Personalized recommendations based on your usage
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-blue-50">
                    <Sparkles className="h-5 w-5 text-blue-500 mt-0.5" />
                    <div>
                      <p className="font-medium text-blue-900">Peak Productivity</p>
                      <p className="text-sm text-blue-700">
                        You're most active at {analytics.insights.mostProductiveHour}:00 on {analytics.insights.mostProductiveDay}s
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3 p-3 rounded-lg bg-green-50">
                    <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                    <div>
                      <p className="font-medium text-green-900">Efficiency Boost</p>
                      <p className="text-sm text-green-700">
                        Your {analytics.insights.favoriteModel} usage shows 15% faster response times
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3 p-3 rounded-lg bg-purple-50">
                    <Brain className="h-5 w-5 text-purple-500 mt-0.5" />
                    <div>
                      <p className="font-medium text-purple-900">Usage Pattern</p>
                      <p className="text-sm text-purple-700">
                        {analytics.insights.favoriteCategory} conversations average {analytics.insights.averageConversationLength} messages
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Recommendations */}
            <Card>
              <CardHeader>
                <CardTitle>Recommendations</CardTitle>
                <CardDescription>
                  Ways to optimize your AI workflow
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center font-bold">1</div>
                    <div>
                      <p className="font-medium">Try Templates</p>
                      <p className="text-sm text-muted-foreground">
                        Use conversation templates to save 30% time on repetitive tasks
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-green-500 text-white text-xs flex items-center justify-center font-bold">2</div>
                    <div>
                      <p className="font-medium">Model Optimization</p>
                      <p className="text-sm text-muted-foreground">
                        Consider using lighter models for simple tasks to improve speed
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-purple-500 text-white text-xs flex items-center justify-center font-bold">3</div>
                    <div>
                      <p className="font-medium">Schedule Optimization</p>
                      <p className="text-sm text-muted-foreground">
                        Your peak hours are {analytics.insights.mostProductiveHour}:00 - plan important tasks then
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Achievement Badges */}
          <Card>
            <CardHeader>
              <CardTitle>Achievements</CardTitle>
              <CardDescription>
                Milestones you've reached with Tektra
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 rounded-lg bg-yellow-50">
                  <Award className="h-8 w-8 text-yellow-500 mx-auto mb-2" />
                  <p className="font-medium text-yellow-900">Conversation Master</p>
                  <p className="text-xs text-yellow-700">100+ conversations</p>
                </div>

                <div className="text-center p-4 rounded-lg bg-blue-50">
                  <Target className="h-8 w-8 text-blue-500 mx-auto mb-2" />
                  <p className="font-medium text-blue-900">Efficiency Expert</p>
                  <p className="text-xs text-blue-700">90%+ efficiency rate</p>
                </div>

                <div className="text-center p-4 rounded-lg bg-green-50">
                  <Zap className="h-8 w-8 text-green-500 mx-auto mb-2" />
                  <p className="font-medium text-green-900">Speed Demon</p>
                  <p className="text-xs text-green-700">Sub-second responses</p>
                </div>

                <div className="text-center p-4 rounded-lg bg-purple-50">
                  <Brain className="h-8 w-8 text-purple-500 mx-auto mb-2" />
                  <p className="font-medium text-purple-900">Model Explorer</p>
                  <p className="text-xs text-purple-700">Used 5+ models</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}