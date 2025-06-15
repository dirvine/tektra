"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { 
  Brain,
  Download,
  Trash2,
  HardDrive,
  Zap,
  CheckCircle,
  AlertCircle,
  Clock,
  Settings,
  Search,
  Filter,
  MoreHorizontal,
  Play,
  Pause,
  RefreshCw,
  Info,
  ExternalLink,
  Cpu,
  MemoryStick,
  Gauge,
  ShieldCheck,
  Globe
} from 'lucide-react'
import { api, type ModelInfo } from '@/lib/api'

interface ModelManagerProps {
  onClose?: () => void
}

interface DownloadProgress {
  modelName: string
  progress: number
  status: 'downloading' | 'extracting' | 'complete' | 'error'
  speed?: string
  eta?: string
  error?: string
}

interface ModelConfig {
  name: string
  displayName: string
  description: string
  provider: string
  size: string
  parameters: string
  contextLength: number
  features: string[]
  requirements: {
    minRam: string
    recommendedRam: string
    platform: string[]
  }
  performance: {
    speed: 'fast' | 'medium' | 'slow'
    quality: 'high' | 'medium' | 'basic'
    efficiency: 'excellent' | 'good' | 'fair'
  }
  license: string
  homepage?: string
  isLocal: boolean
  isAppleSilicon?: boolean
}

const AVAILABLE_MODELS: ModelConfig[] = [
  {
    name: 'phi-3-mini',
    displayName: 'Microsoft Phi-3 Mini',
    description: 'Compact and efficient model optimized for speed and low resource usage',
    provider: 'Microsoft',
    size: '2.3 GB',
    parameters: '3.8B',
    contextLength: 4096,
    features: ['Fast Inference', 'Low Memory', 'Code Generation', 'Multilingual'],
    requirements: {
      minRam: '4 GB',
      recommendedRam: '8 GB',
      platform: ['macOS', 'Linux', 'Windows']
    },
    performance: {
      speed: 'fast',
      quality: 'medium',
      efficiency: 'excellent'
    },
    license: 'MIT',
    homepage: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct',
    isLocal: true
  },
  {
    name: 'llama-3-8b',
    displayName: 'Meta Llama 3 8B',
    description: 'Advanced reasoning model with excellent performance across diverse tasks',
    provider: 'Meta',
    size: '4.7 GB',
    parameters: '8B',
    contextLength: 8192,
    features: ['Advanced Reasoning', 'Creative Writing', 'Code Analysis', 'Math'],
    requirements: {
      minRam: '8 GB',
      recommendedRam: '16 GB',
      platform: ['macOS', 'Linux', 'Windows']
    },
    performance: {
      speed: 'medium',
      quality: 'high',
      efficiency: 'good'
    },
    license: 'Custom (Llama 3)',
    homepage: 'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct',
    isLocal: true
  },
  {
    name: 'mistral-7b',
    displayName: 'Mistral 7B Instruct',
    description: 'Versatile model with strong multilingual capabilities and creative output',
    provider: 'Mistral AI',
    size: '4.1 GB',
    parameters: '7.3B',
    contextLength: 8192,
    features: ['Multilingual', 'Creative Writing', 'Code Generation', 'Instruction Following'],
    requirements: {
      minRam: '8 GB',
      recommendedRam: '12 GB',
      platform: ['macOS', 'Linux', 'Windows']
    },
    performance: {
      speed: 'medium',
      quality: 'high',
      efficiency: 'good'
    },
    license: 'Apache 2.0',
    homepage: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2',
    isLocal: true
  },
  {
    name: 'gemma-2b',
    displayName: 'Google Gemma 2B',
    description: 'Lightweight model perfect for quick responses and resource-constrained environments',
    provider: 'Google',
    size: '1.4 GB',
    parameters: '2.5B',
    contextLength: 2048,
    features: ['Lightweight', 'Quick Responses', 'Energy Efficient', 'Mobile Friendly'],
    requirements: {
      minRam: '3 GB',
      recommendedRam: '6 GB',
      platform: ['macOS', 'Linux', 'Windows', 'Mobile']
    },
    performance: {
      speed: 'fast',
      quality: 'medium',
      efficiency: 'excellent'
    },
    license: 'Gemma Terms of Use',
    homepage: 'https://huggingface.co/google/gemma-2b-it',
    isLocal: true
  },
  {
    name: 'codellama-7b',
    displayName: 'Code Llama 7B',
    description: 'Specialized model for code generation, completion, and programming assistance',
    provider: 'Meta',
    size: '3.8 GB',
    parameters: '7B',
    contextLength: 4096,
    features: ['Code Generation', 'Code Completion', 'Bug Detection', 'Multiple Languages'],
    requirements: {
      minRam: '8 GB',
      recommendedRam: '12 GB',
      platform: ['macOS', 'Linux', 'Windows']
    },
    performance: {
      speed: 'medium',
      quality: 'high',
      efficiency: 'good'
    },
    license: 'Custom (Llama 2)',
    homepage: 'https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf',
    isLocal: true
  },
  {
    name: 'openai-gpt-4',
    displayName: 'OpenAI GPT-4',
    description: 'Advanced cloud-based model with superior reasoning and creative capabilities',
    provider: 'OpenAI',
    size: 'Cloud',
    parameters: 'Unknown',
    contextLength: 8192,
    features: ['Advanced Reasoning', 'Multimodal', 'Creative Writing', 'Code Generation'],
    requirements: {
      minRam: 'N/A',
      recommendedRam: 'N/A',
      platform: ['Cloud API']
    },
    performance: {
      speed: 'medium',
      quality: 'high',
      efficiency: 'fair'
    },
    license: 'Commercial',
    homepage: 'https://openai.com/gpt-4',
    isLocal: false
  }
]

export default function ModelManager({ onClose }: ModelManagerProps) {
  const [downloadedModels, setDownloadedModels] = useState<ModelInfo[]>([])
  const [downloadProgress, setDownloadProgress] = useState<Record<string, DownloadProgress>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedProvider, setSelectedProvider] = useState<string>('')
  const [showOnlyLocal, setShowOnlyLocal] = useState(true)
  const [cacheInfo, setCacheInfo] = useState<any>(null)

  useEffect(() => {
    loadModels()
    loadCacheInfo()
  }, [])

  const loadModels = async () => {
    setIsLoading(true)
    try {
      const response = await api.getAvailableModels()
      if (response.data) {
        setDownloadedModels(response.data as ModelInfo[])
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadCacheInfo = async () => {
    try {
      const response = await api.getCacheInfo()
      if (response.data) {
        setCacheInfo(response.data)
      }
    } catch (error) {
      console.error('Failed to load cache info:', error)
    }
  }

  const downloadModel = async (modelName: string) => {
    try {
      // Start download
      setDownloadProgress(prev => ({
        ...prev,
        [modelName]: {
          modelName,
          progress: 0,
          status: 'downloading'
        }
      }))

      await api.downloadModel(modelName)

      // Poll for progress
      const pollProgress = setInterval(async () => {
        try {
          const response = await api.getDownloadProgress(modelName)
          if (response.data) {
            const progressData = response.data as any
            
            setDownloadProgress(prev => ({
              ...prev,
              [modelName]: {
                modelName,
                progress: progressData.progress || 0,
                status: progressData.status || 'downloading',
                speed: progressData.speed,
                eta: progressData.eta,
                error: progressData.error
              }
            }))

            if (progressData.status === 'complete' || progressData.status === 'error') {
              clearInterval(pollProgress)
              if (progressData.status === 'complete') {
                loadModels() // Refresh model list
              }
            }
          }
        } catch (error) {
          console.error('Failed to get download progress:', error)
          clearInterval(pollProgress)
          setDownloadProgress(prev => ({
            ...prev,
            [modelName]: {
              ...prev[modelName],
              status: 'error',
              error: 'Failed to track progress'
            }
          }))
        }
      }, 1000)

    } catch (error) {
      console.error('Failed to start download:', error)
      setDownloadProgress(prev => ({
        ...prev,
        [modelName]: {
          ...prev[modelName],
          status: 'error',
          error: error instanceof Error ? error.message : 'Download failed'
        }
      }))
    }
  }

  const deleteModel = async (modelName: string) => {
    if (!confirm(`Are you sure you want to delete the ${modelName} model? This will free up disk space but you'll need to download it again to use it.`)) {
      return
    }

    try {
      await api.deleteModel(modelName)
      setDownloadedModels(prev => prev.filter(m => m.name !== modelName))
      loadCacheInfo() // Refresh cache info
    } catch (error) {
      console.error('Failed to delete model:', error)
    }
  }

  const clearCache = async () => {
    if (!confirm('Are you sure you want to clear the model cache? This will remove all cached data and temporary files.')) {
      return
    }

    try {
      await api.clearModelCache()
      loadCacheInfo()
    } catch (error) {
      console.error('Failed to clear cache:', error)
    }
  }

  const loadModel = async (modelName: string) => {
    try {
      await api.loadModel(modelName)
      loadModels() // Refresh to get updated status
    } catch (error) {
      console.error('Failed to load model:', error)
    }
  }

  const unloadModel = async (modelName: string) => {
    try {
      await api.unloadModel(modelName)
      loadModels() // Refresh to get updated status
    } catch (error) {
      console.error('Failed to unload model:', error)
    }
  }

  const isModelDownloaded = (modelName: string): boolean => {
    return downloadedModels.some(m => m.name === modelName && m.is_downloaded)
  }

  const getModelInfo = (modelName: string): ModelInfo | undefined => {
    return downloadedModels.find(m => m.name === modelName)
  }

  const getModelConfig = (modelName: string): ModelConfig | undefined => {
    return AVAILABLE_MODELS.find(m => m.name === modelName)
  }

  const filteredModels = AVAILABLE_MODELS.filter(model => {
    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      if (!model.displayName.toLowerCase().includes(query) &&
          !model.description.toLowerCase().includes(query) &&
          !model.provider.toLowerCase().includes(query) &&
          !model.features.some(f => f.toLowerCase().includes(query))) {
        return false
      }
    }

    // Provider filter
    if (selectedProvider && model.provider !== selectedProvider) {
      return false
    }

    // Local only filter
    if (showOnlyLocal && !model.isLocal) {
      return false
    }

    return true
  })

  const getProviders = (): string[] => {
    const providers = [...new Set(AVAILABLE_MODELS.map(m => m.provider))]
    return providers.sort()
  }

  const getPerformanceBadgeColor = (performance: string): string => {
    switch (performance) {
      case 'fast':
      case 'high':
      case 'excellent':
        return 'bg-green-500'
      case 'medium':
      case 'good':
        return 'bg-yellow-500'
      default:
        return 'bg-gray-500'
    }
  }

  if (isLoading) {
    return (
      <Card className="w-full max-w-6xl mx-auto">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Settings className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Loading model information...</p>
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
            <Brain className="h-8 w-8" />
            Model Management
          </h1>
          <p className="text-muted-foreground mt-1">
            Download and manage AI models for local inference
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={loadModels}>
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

      {/* Cache Information */}
      {cacheInfo && (
        <Card>
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="flex items-center gap-3">
                <HardDrive className="h-5 w-5 text-blue-500" />
                <div>
                  <p className="text-sm font-medium">Cache Size</p>
                  <p className="text-xs text-muted-foreground">
                    {cacheInfo.total_size || 'Unknown'}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Download className="h-5 w-5 text-green-500" />
                <div>
                  <p className="text-sm font-medium">Downloaded Models</p>
                  <p className="text-xs text-muted-foreground">
                    {downloadedModels.filter(m => m.is_downloaded).length} models
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <MemoryStick className="h-5 w-5 text-purple-500" />
                <div>
                  <p className="text-sm font-medium">Memory Usage</p>
                  <p className="text-xs text-muted-foreground">
                    {cacheInfo.memory_usage || 'Unknown'}
                  </p>
                </div>
              </div>
              <div className="flex justify-end">
                <Button variant="outline" size="sm" onClick={clearCache}>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear Cache
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="available" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="available">Available Models</TabsTrigger>
          <TabsTrigger value="downloaded">Downloaded Models</TabsTrigger>
          <TabsTrigger value="downloads">Active Downloads</TabsTrigger>
        </TabsList>

        {/* Available Models Tab */}
        <TabsContent value="available" className="space-y-6">
          {/* Filters */}
          <Card>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search models..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9"
                  />
                </div>

                {/* Provider Filter */}
                <select
                  value={selectedProvider}
                  onChange={(e) => setSelectedProvider(e.target.value)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="">All Providers</option>
                  {getProviders().map(provider => (
                    <option key={provider} value={provider}>{provider}</option>
                  ))}
                </select>

                {/* Local Only Toggle */}
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={showOnlyLocal}
                    onChange={(e) => setShowOnlyLocal(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm">Local models only</span>
                </label>

                {/* Clear Filters */}
                <Button
                  variant="outline"
                  onClick={() => {
                    setSearchQuery('')
                    setSelectedProvider('')
                    setShowOnlyLocal(true)
                  }}
                >
                  Clear Filters
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Models Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredModels.map(model => {
              const isDownloaded = isModelDownloaded(model.name)
              const modelInfo = getModelInfo(model.name)
              const downloadProg = downloadProgress[model.name]

              return (
                <Card key={model.name} className="relative">
                  <CardHeader className="pb-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-lg flex items-center gap-2">
                          {model.displayName}
                          {isDownloaded && (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          )}
                          {!model.isLocal && (
                            <Globe className="h-4 w-4 text-blue-500" />
                          )}
                        </CardTitle>
                        <div className="flex items-center gap-2 mt-1">
                          <Badge variant="outline">{model.provider}</Badge>
                          <Badge variant="outline">{model.parameters}</Badge>
                          <Badge variant="outline">{model.size}</Badge>
                        </div>
                      </div>
                      {model.homepage && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => window.open(model.homepage, '_blank')}
                          className="h-8 w-8 p-0"
                        >
                          <ExternalLink className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                    <CardDescription className="line-clamp-2">
                      {model.description}
                    </CardDescription>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {/* Features */}
                    <div className="flex flex-wrap gap-1">
                      {model.features.slice(0, 4).map(feature => (
                        <Badge key={feature} variant="secondary" className="text-xs">
                          {feature}
                        </Badge>
                      ))}
                      {model.features.length > 4 && (
                        <Badge variant="outline" className="text-xs">
                          +{model.features.length - 4}
                        </Badge>
                      )}
                    </div>

                    {/* Performance Indicators */}
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="text-center">
                        <div className="flex items-center justify-center gap-1 mb-1">
                          <Zap className="h-3 w-3" />
                          <span>Speed</span>
                        </div>
                        <Badge 
                          variant="outline" 
                          className={`text-white ${getPerformanceBadgeColor(model.performance.speed)}`}
                        >
                          {model.performance.speed}
                        </Badge>
                      </div>
                      <div className="text-center">
                        <div className="flex items-center justify-center gap-1 mb-1">
                          <Gauge className="h-3 w-3" />
                          <span>Quality</span>
                        </div>
                        <Badge 
                          variant="outline"
                          className={`text-white ${getPerformanceBadgeColor(model.performance.quality)}`}
                        >
                          {model.performance.quality}
                        </Badge>
                      </div>
                      <div className="text-center">
                        <div className="flex items-center justify-center gap-1 mb-1">
                          <Cpu className="h-3 w-3" />
                          <span>Efficiency</span>
                        </div>
                        <Badge 
                          variant="outline"
                          className={`text-white ${getPerformanceBadgeColor(model.performance.efficiency)}`}
                        >
                          {model.performance.efficiency}
                        </Badge>
                      </div>
                    </div>

                    {/* Requirements */}
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <div className="text-xs space-y-1">
                        <div className="flex justify-between">
                          <span>Min RAM:</span>
                          <span className="font-medium">{model.requirements.minRam}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Recommended:</span>
                          <span className="font-medium">{model.requirements.recommendedRam}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Context:</span>
                          <span className="font-medium">{model.contextLength.toLocaleString()} tokens</span>
                        </div>
                      </div>
                    </div>

                    {/* Download Progress */}
                    {downloadProg && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="capitalize">{downloadProg.status}</span>
                          <span>{downloadProg.progress}%</span>
                        </div>
                        <Progress value={downloadProg.progress} className="h-2" />
                        {downloadProg.speed && downloadProg.eta && (
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>{downloadProg.speed}</span>
                            <span>ETA: {downloadProg.eta}</span>
                          </div>
                        )}
                        {downloadProg.error && (
                          <div className="flex items-center gap-2 text-red-600 text-xs">
                            <AlertCircle className="h-3 w-3" />
                            <span>{downloadProg.error}</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Actions */}
                    <div className="space-y-2">
                      {!isDownloaded && !downloadProg && model.isLocal && (
                        <Button
                          onClick={() => downloadModel(model.name)}
                          className="w-full"
                          size="sm"
                        >
                          <Download className="h-4 w-4 mr-2" />
                          Download Model
                        </Button>
                      )}

                      {isDownloaded && (
                        <div className="flex gap-2">
                          <Button
                            onClick={() => loadModel(model.name)}
                            size="sm"
                            className="flex-1"
                          >
                            <Play className="h-4 w-4 mr-2" />
                            Load
                          </Button>
                          <Button
                            variant="outline"
                            onClick={() => deleteModel(model.name)}
                            size="sm"
                            className="flex-1"
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </Button>
                        </div>
                      )}

                      {!model.isLocal && (
                        <div className="text-center">
                          <Badge variant="outline" className="text-xs">
                            <Globe className="h-3 w-3 mr-1" />
                            Cloud API Required
                          </Badge>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </TabsContent>

        {/* Downloaded Models Tab */}
        <TabsContent value="downloaded" className="space-y-6">
          {downloadedModels.filter(m => m.is_downloaded).length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center">
                  <Brain className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-medium mb-2">No models downloaded</h3>
                  <p className="text-muted-foreground mb-4">
                    Download your first model to get started with local AI inference
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {downloadedModels.filter(m => m.is_downloaded).map(model => {
                const config = getModelConfig(model.name)

                return (
                  <Card key={model.name}>
                    <CardHeader className="pb-4">
                      <CardTitle className="text-lg flex items-center gap-2">
                        {model.display_name}
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      </CardTitle>
                      <CardDescription>{model.description}</CardDescription>
                    </CardHeader>

                    <CardContent className="space-y-4">
                      {/* Model Info */}
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Size:</span>
                          <span className="ml-2 font-medium">{(model.size_gb || 0).toFixed(1)} GB</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Parameters:</span>
                          <span className="ml-2 font-medium">{model.parameters}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Context:</span>
                          <span className="ml-2 font-medium">{model.context_length?.toLocaleString() || 'Unknown'}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Features:</span>
                          <span className="ml-2 font-medium">{model.supported_features?.length || 0}</span>
                        </div>
                      </div>

                      {/* Features */}
                      {model.supported_features && (
                        <div className="flex flex-wrap gap-1">
                          {model.supported_features.slice(0, 4).map((feature: string) => (
                            <Badge key={feature} variant="secondary" className="text-xs">
                              {feature}
                            </Badge>
                          ))}
                          {model.supported_features.length > 4 && (
                            <Badge variant="outline" className="text-xs">
                              +{model.supported_features.length - 4}
                            </Badge>
                          )}
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex gap-2">
                        <Button
                          onClick={() => loadModel(model.name)}
                          size="sm"
                          className="flex-1"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Load
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => deleteModel(model.name)}
                          size="sm"
                          className="flex-1"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </TabsContent>

        {/* Active Downloads Tab */}
        <TabsContent value="downloads" className="space-y-6">
          {Object.keys(downloadProgress).length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center">
                  <Download className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-medium mb-2">No active downloads</h3>
                  <p className="text-muted-foreground">
                    Model downloads will appear here when in progress
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {Object.values(downloadProgress).map(download => (
                <Card key={download.modelName}>
                  <CardContent className="pt-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-medium">{download.modelName}</h4>
                          <p className="text-sm text-muted-foreground capitalize">
                            {download.status}
                          </p>
                        </div>
                        <div className="text-right">
                          <div className="font-medium">{download.progress}%</div>
                          {download.eta && (
                            <div className="text-xs text-muted-foreground">
                              ETA: {download.eta}
                            </div>
                          )}
                        </div>
                      </div>

                      <Progress value={download.progress} className="h-3" />

                      {download.speed && (
                        <div className="flex justify-between text-xs text-muted-foreground">
                          <span>Speed: {download.speed}</span>
                          <span>Status: {download.status}</span>
                        </div>
                      )}

                      {download.error && (
                        <div className="flex items-center gap-2 text-red-600 text-sm">
                          <AlertCircle className="h-4 w-4" />
                          <span>{download.error}</span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}