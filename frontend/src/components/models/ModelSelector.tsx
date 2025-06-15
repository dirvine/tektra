"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  Brain,
  CheckCircle,
  Download,
  Settings,
  Zap,
  Globe,
  ChevronDown,
  Cpu,
  MemoryStick,
  Clock
} from 'lucide-react'
import { api, type ModelInfo } from '@/lib/api'

interface ModelSelectorProps {
  selectedModel?: string
  onModelChange: (modelName: string) => void
  showModelInfo?: boolean
  compact?: boolean
}

interface ModelOption {
  name: string
  displayName: string
  description: string
  isDownloaded: boolean
  isLocal: boolean
  size?: string
  parameters?: string
  performance?: {
    speed: string
    quality: string
  }
}

export default function ModelSelector({ 
  selectedModel, 
  onModelChange, 
  showModelInfo = true,
  compact = false 
}: ModelSelectorProps) {
  const [availableModels, setAvailableModels] = useState<ModelOption[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [loadingModel, setLoadingModel] = useState<string | null>(null)

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    setIsLoading(true)
    try {
      const response = await api.getAvailableModels()
      if (response.data) {
        const modelInfos = response.data as ModelInfo[]
        
        // Combine with known model configurations
        const models: ModelOption[] = [
          {
            name: 'phi-3-mini',
            displayName: 'Phi-3 Mini',
            description: 'Fast and efficient for quick responses',
            isDownloaded: modelInfos.some(m => m.name === 'phi-3-mini' && m.is_downloaded),
            isLocal: true,
            size: '2.3 GB',
            parameters: '3.8B',
            performance: { speed: 'fast', quality: 'good' }
          },
          {
            name: 'llama-3-8b',
            displayName: 'Llama 3 8B',
            description: 'Balanced performance with advanced reasoning',
            isDownloaded: modelInfos.some(m => m.name === 'llama-3-8b' && m.is_downloaded),
            isLocal: true,
            size: '4.7 GB',
            parameters: '8B',
            performance: { speed: 'medium', quality: 'high' }
          },
          {
            name: 'mistral-7b',
            displayName: 'Mistral 7B',
            description: 'Creative and multilingual capabilities',
            isDownloaded: modelInfos.some(m => m.name === 'mistral-7b' && m.is_downloaded),
            isLocal: true,
            size: '4.1 GB',
            parameters: '7.3B',
            performance: { speed: 'medium', quality: 'high' }
          },
          {
            name: 'gemma-2b',
            displayName: 'Gemma 2B',
            description: 'Lightweight and energy efficient',
            isDownloaded: modelInfos.some(m => m.name === 'gemma-2b' && m.is_downloaded),
            isLocal: true,
            size: '1.4 GB',
            parameters: '2.5B',
            performance: { speed: 'fast', quality: 'medium' }
          },
          {
            name: 'openai-gpt-4',
            displayName: 'OpenAI GPT-4',
            description: 'Advanced cloud-based reasoning',
            isDownloaded: true, // API models are always "available"
            isLocal: false,
            performance: { speed: 'medium', quality: 'excellent' }
          },
          {
            name: 'openai-gpt-3.5-turbo',
            displayName: 'OpenAI GPT-3.5 Turbo',
            description: 'Fast cloud-based conversations',
            isDownloaded: true,
            isLocal: false,
            performance: { speed: 'fast', quality: 'good' }
          }
        ]

        setAvailableModels(models)
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const downloadModel = async (modelName: string) => {
    setLoadingModel(modelName)
    try {
      await api.downloadModel(modelName)
      await loadModels() // Refresh model list
    } catch (error) {
      console.error('Failed to download model:', error)
    } finally {
      setLoadingModel(null)
    }
  }

  const getSelectedModelInfo = (): ModelOption | undefined => {
    return availableModels.find(m => m.name === selectedModel)
  }

  const getPerformanceColor = (performance: string): string => {
    switch (performance) {
      case 'excellent':
      case 'fast':
        return 'text-green-600'
      case 'high':
      case 'good':
        return 'text-blue-600'
      case 'medium':
        return 'text-yellow-600'
      default:
        return 'text-gray-600'
    }
  }

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <Select
          value={selectedModel}
          onValueChange={onModelChange}
          disabled={isLoading}
        >
          <SelectTrigger className="w-48">
            <div className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              <SelectValue placeholder="Select model" />
            </div>
          </SelectTrigger>
          <SelectContent>
            {availableModels.map(model => (
              <SelectItem key={model.name} value={model.name} disabled={!model.isDownloaded}>
                <div className="flex items-center gap-2">
                  <div className="flex-1">
                    <div className="font-medium">{model.displayName}</div>
                    <div className="text-xs text-muted-foreground">{model.description}</div>
                  </div>
                  {model.isDownloaded ? (
                    <CheckCircle className="h-3 w-3 text-green-500" />
                  ) : (
                    <Download className="h-3 w-3 text-muted-foreground" />
                  )}
                  {!model.isLocal && (
                    <Globe className="h-3 w-3 text-blue-500" />
                  )}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {showModelInfo && selectedModel && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            {getSelectedModelInfo()?.parameters && (
              <Badge variant="outline" className="text-xs">
                {getSelectedModelInfo()?.parameters}
              </Badge>
            )}
            {!getSelectedModelInfo()?.isLocal && (
              <Badge variant="outline" className="text-xs">
                <Globe className="h-2 w-2 mr-1" />
                Cloud
              </Badge>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg flex items-center gap-2">
          <Brain className="h-5 w-5" />
          AI Model Selection
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Model Selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Selected Model</label>
          <Select
            value={selectedModel}
            onValueChange={onModelChange}
            disabled={isLoading}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Choose an AI model" />
            </SelectTrigger>
            <SelectContent>
              {availableModels.map(model => (
                <SelectItem key={model.name} value={model.name} disabled={!model.isDownloaded}>
                  <div className="flex items-center justify-between w-full">
                    <div className="flex-1">
                      <div className="font-medium">{model.displayName}</div>
                      <div className="text-xs text-muted-foreground">{model.description}</div>
                    </div>
                    <div className="flex items-center gap-1 ml-2">
                      {model.isDownloaded ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <Download className="h-4 w-4 text-muted-foreground" />
                      )}
                      {!model.isLocal && (
                        <Globe className="h-4 w-4 text-blue-500" />
                      )}
                    </div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Selected Model Info */}
        {selectedModel && showModelInfo && (
          <div className="space-y-3">
            {(() => {
              const modelInfo = getSelectedModelInfo()
              if (!modelInfo) return null

              return (
                <div className="p-4 bg-muted/50 rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">{modelInfo.displayName}</h4>
                    <div className="flex items-center gap-2">
                      {modelInfo.isDownloaded ? (
                        <Badge variant="default" className="text-xs">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Ready
                        </Badge>
                      ) : (
                        <Badge variant="outline" className="text-xs">
                          <Download className="h-3 w-3 mr-1" />
                          Not Downloaded
                        </Badge>
                      )}
                      {!modelInfo.isLocal && (
                        <Badge variant="outline" className="text-xs">
                          <Globe className="h-3 w-3 mr-1" />
                          Cloud API
                        </Badge>
                      )}
                    </div>
                  </div>

                  <p className="text-sm text-muted-foreground">
                    {modelInfo.description}
                  </p>

                  {/* Model Specs */}
                  {(modelInfo.parameters || modelInfo.size) && (
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      {modelInfo.parameters && (
                        <div className="flex items-center gap-2">
                          <Cpu className="h-4 w-4 text-muted-foreground" />
                          <span className="text-muted-foreground">Parameters:</span>
                          <span className="font-medium">{modelInfo.parameters}</span>
                        </div>
                      )}
                      {modelInfo.size && (
                        <div className="flex items-center gap-2">
                          <MemoryStick className="h-4 w-4 text-muted-foreground" />
                          <span className="text-muted-foreground">Size:</span>
                          <span className="font-medium">{modelInfo.size}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Performance Indicators */}
                  {modelInfo.performance && (
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="flex items-center gap-2">
                        <Zap className="h-4 w-4 text-muted-foreground" />
                        <span className="text-muted-foreground">Speed:</span>
                        <span className={`font-medium capitalize ${getPerformanceColor(modelInfo.performance.speed)}`}>
                          {modelInfo.performance.speed}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Settings className="h-4 w-4 text-muted-foreground" />
                        <span className="text-muted-foreground">Quality:</span>
                        <span className={`font-medium capitalize ${getPerformanceColor(modelInfo.performance.quality)}`}>
                          {modelInfo.performance.quality}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Download Action */}
                  {!modelInfo.isDownloaded && modelInfo.isLocal && (
                    <Button
                      onClick={() => downloadModel(modelInfo.name)}
                      disabled={loadingModel === modelInfo.name}
                      size="sm"
                      className="w-full"
                    >
                      {loadingModel === modelInfo.name ? (
                        <>
                          <Clock className="h-4 w-4 mr-2 animate-spin" />
                          Downloading...
                        </>
                      ) : (
                        <>
                          <Download className="h-4 w-4 mr-2" />
                          Download Model
                        </>
                      )}
                    </Button>
                  )}
                </div>
              )
            })()}
          </div>
        )}

        {/* Available Models Summary */}
        <div className="pt-4 border-t">
          <div className="grid grid-cols-3 gap-4 text-center text-sm">
            <div>
              <div className="font-medium text-green-600">
                {availableModels.filter(m => m.isDownloaded && m.isLocal).length}
              </div>
              <div className="text-muted-foreground text-xs">Local Models</div>
            </div>
            <div>
              <div className="font-medium text-blue-600">
                {availableModels.filter(m => !m.isLocal).length}
              </div>
              <div className="text-muted-foreground text-xs">Cloud APIs</div>
            </div>
            <div>
              <div className="font-medium text-orange-600">
                {availableModels.filter(m => !m.isDownloaded && m.isLocal).length}
              </div>
              <div className="text-muted-foreground text-xs">Available</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}