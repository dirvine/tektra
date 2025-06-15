"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  Key,
  Plus,
  Trash2,
  Eye,
  EyeOff,
  Shield,
  AlertCircle,
  CheckCircle,
  Settings,
  ExternalLink
} from 'lucide-react'
import { api, type APIKey } from '@/lib/api'

interface APIKeyManagerProps {
  userId: number
}

interface ProviderInfo {
  name: string
  displayName: string
  description: string
  websiteUrl: string
  keyFormat: string
  features: string[]
}

const API_PROVIDERS: ProviderInfo[] = [
  {
    name: 'openai',
    displayName: 'OpenAI',
    description: 'GPT models, DALL-E, Whisper, and TTS',
    websiteUrl: 'https://platform.openai.com/api-keys',
    keyFormat: 'sk-...',
    features: ['Text Generation', 'Image Generation', 'Speech-to-Text', 'Text-to-Speech']
  },
  {
    name: 'anthropic',
    displayName: 'Anthropic',
    description: 'Claude models for advanced reasoning',
    websiteUrl: 'https://console.anthropic.com/',
    keyFormat: 'sk-ant-...',
    features: ['Text Generation', 'Code Analysis', 'Reasoning']
  },
  {
    name: 'elevenlabs',
    displayName: 'ElevenLabs',
    description: 'High-quality voice synthesis',
    websiteUrl: 'https://elevenlabs.io/',
    keyFormat: 'xxxxxxxxx',
    features: ['Voice Synthesis', 'Voice Cloning']
  },
  {
    name: 'huggingface',
    displayName: 'Hugging Face',
    description: 'Access to thousands of open-source models',
    websiteUrl: 'https://huggingface.co/settings/tokens',
    keyFormat: 'hf_...',
    features: ['Text Generation', 'Embeddings', 'Image Generation']
  },
  {
    name: 'google',
    displayName: 'Google Cloud',
    description: 'Gemini models and Google AI services',
    websiteUrl: 'https://console.cloud.google.com/',
    keyFormat: 'AIza...',
    features: ['Text Generation', 'Translation', 'Speech APIs']
  },
  {
    name: 'azure',
    displayName: 'Azure OpenAI',
    description: 'OpenAI models via Microsoft Azure',
    websiteUrl: 'https://portal.azure.com/',
    keyFormat: 'xxxxxxxxx',
    features: ['Text Generation', 'Embeddings', 'Enterprise']
  }
]

export default function APIKeyManager({ userId }: APIKeyManagerProps) {
  const [apiKeys, setApiKeys] = useState<APIKey[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [newKey, setNewKey] = useState({
    provider: '',
    key_name: '',
    api_key: '',
    usage_limit: ''
  })
  const [showKey, setShowKey] = useState<Record<number, boolean>>({})
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    loadAPIKeys()
  }, [userId])

  const loadAPIKeys = async () => {
    setIsLoading(true)
    try {
      const response = await api.getAPIKeys(userId)
      if (response.data) {
        setApiKeys(response.data as APIKey[])
      }
    } catch (error) {
      console.error('Failed to load API keys:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const addAPIKey = async () => {
    if (!newKey.provider || !newKey.key_name || !newKey.api_key) {
      return
    }

    setIsSaving(true)
    try {
      const keyData = {
        provider: newKey.provider,
        key_name: newKey.key_name,
        api_key: newKey.api_key,
        usage_limit: newKey.usage_limit ? parseInt(newKey.usage_limit) : undefined
      }

      await api.storeAPIKey(userId, keyData)
      await loadAPIKeys()
      
      // Reset form
      setNewKey({
        provider: '',
        key_name: '',
        api_key: '',
        usage_limit: ''
      })
      setShowAddForm(false)
    } catch (error) {
      console.error('Failed to add API key:', error)
    } finally {
      setIsSaving(false)
    }
  }

  const deleteAPIKey = async (keyId: number) => {
    if (!confirm('Are you sure you want to delete this API key?')) {
      return
    }

    try {
      await api.deleteAPIKey(userId, keyId)
      setApiKeys(prev => prev.filter(key => key.id !== keyId))
    } catch (error) {
      console.error('Failed to delete API key:', error)
    }
  }

  const toggleKeyVisibility = (keyId: number) => {
    setShowKey(prev => ({
      ...prev,
      [keyId]: !prev[keyId]
    }))
  }

  const maskKey = (key: string): string => {
    if (key.length <= 8) return '*'.repeat(key.length)
    return key.substring(0, 4) + '*'.repeat(key.length - 8) + key.substring(key.length - 4)
  }

  const getProviderInfo = (providerName: string): ProviderInfo | undefined => {
    return API_PROVIDERS.find(p => p.name === providerName)
  }

  const validateKeyFormat = (provider: string, key: string): boolean => {
    const providerInfo = getProviderInfo(provider)
    if (!providerInfo) return true // Allow unknown providers

    const format = providerInfo.keyFormat
    if (format.includes('...')) {
      const prefix = format.split('...')[0]
      return key.startsWith(prefix)
    }
    return true // Basic validation only
  }

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Settings className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Loading API keys...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Key className="h-6 w-6" />
            API Key Management
          </h2>
          <p className="text-muted-foreground">
            Securely store and manage your third-party API keys
          </p>
        </div>
        <Button
          onClick={() => setShowAddForm(!showAddForm)}
          className="flex items-center gap-2"
        >
          <Plus className="h-4 w-4" />
          Add API Key
        </Button>
      </div>

      {/* Security Notice */}
      <Card className="border-blue-200 bg-blue-50">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <Shield className="h-5 w-5 text-blue-600 mt-0.5" />
            <div>
              <h3 className="font-medium text-blue-900">Security Information</h3>
              <p className="text-sm text-blue-800 mt-1">
                Your API keys are encrypted before storage and never logged in plain text. 
                Keys are only decrypted when making API calls on your behalf.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Add New Key Form */}
      {showAddForm && (
        <Card>
          <CardHeader>
            <CardTitle>Add New API Key</CardTitle>
            <CardDescription>
              Configure a new API key for third-party services
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Provider Selection */}
            <div className="space-y-2">
              <Label>Provider</Label>
              <Select
                value={newKey.provider}
                onValueChange={(value) => setNewKey(prev => ({ ...prev, provider: value }))}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a provider" />
                </SelectTrigger>
                <SelectContent>
                  {API_PROVIDERS.map(provider => (
                    <SelectItem key={provider.name} value={provider.name}>
                      <div className="flex items-center justify-between w-full">
                        <div>
                          <div className="font-medium">{provider.displayName}</div>
                          <div className="text-xs text-muted-foreground">{provider.description}</div>
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Provider Info */}
              {newKey.provider && (
                <div className="mt-2">
                  {(() => {
                    const providerInfo = getProviderInfo(newKey.provider)
                    if (!providerInfo) return null
                    
                    return (
                      <div className="flex items-center gap-2 p-3 bg-muted/50 rounded-lg">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-medium">{providerInfo.displayName}</span>
                            <Button
                              size="sm"
                              variant="ghost"
                              className="h-6 w-6 p-0"
                              onClick={() => window.open(providerInfo.websiteUrl, '_blank')}
                            >
                              <ExternalLink className="h-3 w-3" />
                            </Button>
                          </div>
                          <p className="text-xs text-muted-foreground mb-2">{providerInfo.description}</p>
                          <div className="flex gap-1">
                            {providerInfo.features.map(feature => (
                              <Badge key={feature} variant="outline" className="text-xs">
                                {feature}
                              </Badge>
                            ))}
                          </div>
                          <p className="text-xs text-muted-foreground mt-2">
                            Expected format: <code className="bg-muted px-1 rounded">{providerInfo.keyFormat}</code>
                          </p>
                        </div>
                      </div>
                    )
                  })()}
                </div>
              )}
            </div>

            {/* Key Name */}
            <div className="space-y-2">
              <Label>Key Name</Label>
              <Input
                placeholder="e.g., Main API Key, Development Key"
                value={newKey.key_name}
                onChange={(e) => setNewKey(prev => ({ ...prev, key_name: e.target.value }))}
              />
            </div>

            {/* API Key */}
            <div className="space-y-2">
              <Label>API Key</Label>
              <Input
                type="password"
                placeholder="Enter your API key"
                value={newKey.api_key}
                onChange={(e) => setNewKey(prev => ({ ...prev, api_key: e.target.value }))}
              />
              {newKey.provider && newKey.api_key && !validateKeyFormat(newKey.provider, newKey.api_key) && (
                <div className="flex items-center gap-2 text-yellow-600">
                  <AlertCircle className="h-4 w-4" />
                  <span className="text-sm">Key format doesn't match expected pattern</span>
                </div>
              )}
            </div>

            {/* Usage Limit */}
            <div className="space-y-2">
              <Label>Usage Limit (Optional)</Label>
              <Input
                type="number"
                placeholder="Max API calls per month"
                value={newKey.usage_limit}
                onChange={(e) => setNewKey(prev => ({ ...prev, usage_limit: e.target.value }))}
              />
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-4">
              <Button
                onClick={addAPIKey}
                disabled={!newKey.provider || !newKey.key_name || !newKey.api_key || isSaving}
              >
                {isSaving ? 'Adding...' : 'Add Key'}
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setShowAddForm(false)
                  setNewKey({ provider: '', key_name: '', api_key: '', usage_limit: '' })
                }}
              >
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Existing Keys */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Your API Keys</h3>
        
        {apiKeys.length === 0 ? (
          <Card>
            <CardContent className="flex items-center justify-center py-12">
              <div className="text-center">
                <Key className="h-8 w-8 mx-auto mb-4 text-muted-foreground" />
                <p className="text-muted-foreground">No API keys configured</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Add your first API key to enable third-party integrations
                </p>
              </div>
            </CardContent>
          </Card>
        ) : (
          apiKeys.map(apiKey => {
            const providerInfo = getProviderInfo(apiKey.provider)
            const isVisible = showKey[apiKey.id] || false

            return (
              <Card key={apiKey.id}>
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-medium">{apiKey.key_name}</h4>
                        <Badge variant="outline">
                          {providerInfo?.displayName || apiKey.provider}
                        </Badge>
                        {!apiKey.is_active && (
                          <Badge variant="destructive">Inactive</Badge>
                        )}
                      </div>

                      <div className="space-y-2 text-sm text-muted-foreground">
                        <div className="flex items-center gap-2">
                          <span>API Key:</span>
                          <code className="bg-muted px-2 py-1 rounded text-xs">
                            {isVisible ? apiKey.key_name : maskKey(apiKey.key_name)}
                          </code>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 w-6 p-0"
                            onClick={() => toggleKeyVisibility(apiKey.id)}
                          >
                            {isVisible ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
                          </Button>
                        </div>

                        <div className="grid grid-cols-2 gap-4 text-xs">
                          <div>
                            <span className="font-medium">Usage:</span> {apiKey.usage_count} calls
                          </div>
                          {apiKey.usage_limit && (
                            <div>
                              <span className="font-medium">Limit:</span> {apiKey.usage_limit} calls/month
                            </div>
                          )}
                          <div>
                            <span className="font-medium">Added:</span> {new Date(apiKey.created_at).toLocaleDateString()}
                          </div>
                          {apiKey.last_used && (
                            <div>
                              <span className="font-medium">Last used:</span> {new Date(apiKey.last_used).toLocaleDateString()}
                            </div>
                          )}
                        </div>

                        {apiKey.usage_limit && apiKey.usage_count >= apiKey.usage_limit && (
                          <div className="flex items-center gap-2 text-red-600">
                            <AlertCircle className="h-4 w-4" />
                            <span>Usage limit reached</span>
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => deleteAPIKey(apiKey.id)}
                        className="text-destructive hover:text-destructive"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })
        )}
      </div>
    </div>
  )
}