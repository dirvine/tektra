"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  Settings,
  Brain,
  Sliders,
  Save,
  Trash2,
  Plus,
  Zap,
  Info,
  AlertCircle,
  CheckCircle
} from 'lucide-react'
import { api, type ModelSettings as ModelSettingsType } from '@/lib/api'

interface ModelSettingsProps {
  userId: number
  selectedModel?: string
  onModelChange?: (modelName: string) => void
}

interface ModelConfig {
  name: string
  displayName: string
  description: string
  maxTokens: number
  defaultTemperature: number
  supportsSystemPrompt: boolean
  features: string[]
}

const AVAILABLE_MODELS: ModelConfig[] = [
  {
    name: 'phi-3-mini',
    displayName: 'Phi-3 Mini',
    description: 'Fast and efficient model for general conversations',
    maxTokens: 4000,
    defaultTemperature: 0.7,
    supportsSystemPrompt: true,
    features: ['Fast', 'Efficient', 'Local']
  },
  {
    name: 'llama-3-8b',
    displayName: 'Llama 3 8B',
    description: 'Balanced model with good reasoning capabilities',
    maxTokens: 8000,
    defaultTemperature: 0.8,
    supportsSystemPrompt: true,
    features: ['Balanced', 'Reasoning', 'Creative']
  },
  {
    name: 'mistral-7b',
    displayName: 'Mistral 7B',
    description: 'Creative model optimized for various tasks',
    maxTokens: 8000,
    defaultTemperature: 1.0,
    supportsSystemPrompt: true,
    features: ['Creative', 'Versatile', 'Multilingual']
  },
  {
    name: 'gemma-2b',
    displayName: 'Gemma 2B',
    description: 'Compact model for quick responses',
    maxTokens: 2000,
    defaultTemperature: 0.6,
    supportsSystemPrompt: false,
    features: ['Compact', 'Quick', 'Lightweight']
  }
]

export default function ModelSettings({ userId, selectedModel, onModelChange }: ModelSettingsProps) {
  const [modelSettings, setModelSettings] = useState<Record<string, ModelSettingsType>>({})
  const [currentModel, setCurrentModel] = useState(selectedModel || 'phi-3-mini')
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => {
    loadModelSettings()
  }, [userId])

  const loadModelSettings = async () => {
    setIsLoading(true)
    try {
      const response = await api.getModelSettings(userId)
      if (response.data) {
        const settings: Record<string, ModelSettingsType> = {}
        response.data.forEach((setting: ModelSettingsType) => {
          settings[setting.model_name] = setting
        })
        setModelSettings(settings)
      }
    } catch (error) {
      console.error('Failed to load model settings:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const getCurrentSettings = (): Partial<ModelSettingsType> => {
    const modelConfig = AVAILABLE_MODELS.find(m => m.name === currentModel)
    const existingSettings = modelSettings[currentModel]
    
    return {
      temperature: existingSettings?.temperature ?? modelConfig?.defaultTemperature ?? 0.7,
      max_tokens: existingSettings?.max_tokens ?? modelConfig?.maxTokens ?? 2000,
      top_p: existingSettings?.top_p ?? 0.9,
      frequency_penalty: existingSettings?.frequency_penalty ?? 0.0,
      presence_penalty: existingSettings?.presence_penalty ?? 0.0,
      stop_sequences: existingSettings?.stop_sequences ?? [],
      system_prompt: existingSettings?.system_prompt ?? '',
      custom_parameters: existingSettings?.custom_parameters ?? {}
    }
  }

  const updateSetting = (key: keyof ModelSettingsType, value: any) => {
    const currentSettings = getCurrentSettings()
    const newSettings = { ...currentSettings, [key]: value }
    
    setModelSettings(prev => ({
      ...prev,
      [currentModel]: { ...prev[currentModel], ...newSettings } as ModelSettingsType
    }))
    setHasChanges(true)
  }

  const saveSettings = async () => {
    const settings = getCurrentSettings()
    if (!hasChanges) return

    setIsSaving(true)
    try {
      await api.updateModelSettings(userId, currentModel, settings)
      await loadModelSettings() // Reload to get the saved data
      setHasChanges(false)
    } catch (error) {
      console.error('Failed to save model settings:', error)
    } finally {
      setIsSaving(false)
    }
  }

  const resetSettings = async () => {
    if (!confirm(`Reset settings for ${currentModel} to defaults?`)) {
      return
    }

    setIsSaving(true)
    try {
      await api.deleteModelSettings(userId, currentModel)
      await loadModelSettings()
      setHasChanges(false)
    } catch (error) {
      console.error('Failed to reset model settings:', error)
    } finally {
      setIsSaving(false)
    }
  }

  const addStopSequence = () => {
    const currentSettings = getCurrentSettings()
    const newSequences = [...(currentSettings.stop_sequences || []), '']
    updateSetting('stop_sequences', newSequences)
  }

  const updateStopSequence = (index: number, value: string) => {
    const currentSettings = getCurrentSettings()
    const newSequences = [...(currentSettings.stop_sequences || [])]
    newSequences[index] = value
    updateSetting('stop_sequences', newSequences)
  }

  const removeStopSequence = (index: number) => {
    const currentSettings = getCurrentSettings()
    const newSequences = (currentSettings.stop_sequences || []).filter((_, i) => i !== index)
    updateSetting('stop_sequences', newSequences)
  }

  const modelConfig = AVAILABLE_MODELS.find(m => m.name === currentModel)
  const settings = getCurrentSettings()

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Settings className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Loading model settings...</p>
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
            <Brain className="h-6 w-6" />
            Model Settings
          </h2>
          <p className="text-muted-foreground">
            Configure parameters for AI models
          </p>
        </div>
        <div className="flex items-center gap-2">
          {hasChanges && (
            <Badge variant="secondary" className="animate-pulse">
              Unsaved changes
            </Badge>
          )}
          <Button
            variant="outline"
            onClick={resetSettings}
            disabled={isSaving}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button
            onClick={saveSettings}
            disabled={!hasChanges || isSaving}
          >
            <Save className="h-4 w-4 mr-2" />
            {isSaving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sliders className="h-5 w-5" />
            Model Configuration
          </CardTitle>
          <CardDescription>
            Select a model and configure its parameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Model Selection */}
          <div className="space-y-3">
            <Label>Select Model</Label>
            <Select
              value={currentModel}
              onValueChange={(value) => {
                setCurrentModel(value)
                onModelChange?.(value)
                setHasChanges(false) // Reset changes when switching models
              }}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {AVAILABLE_MODELS.map(model => (
                  <SelectItem key={model.name} value={model.name}>
                    <div className="flex items-center justify-between w-full">
                      <div>
                        <div className="font-medium">{model.displayName}</div>
                        <div className="text-xs text-muted-foreground">{model.description}</div>
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* Model Info */}
            {modelConfig && (
              <div className="flex items-center gap-2 p-3 bg-muted/50 rounded-lg">
                <Info className="h-4 w-4 text-blue-500" />
                <div className="flex-1">
                  <p className="text-sm font-medium">{modelConfig.displayName}</p>
                  <p className="text-xs text-muted-foreground">{modelConfig.description}</p>
                  <div className="flex gap-1 mt-1">
                    {modelConfig.features.map(feature => (
                      <Badge key={feature} variant="outline" className="text-xs">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <Separator />

          {/* Core Parameters */}
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Core Parameters</h3>

            {/* Temperature */}
            <div className="space-y-3">
              <Label>Temperature: {settings.temperature?.toFixed(2)}</Label>
              <Slider
                value={[settings.temperature || 0.7]}
                onValueChange={([value]) => updateSetting('temperature', value)}
                min={0}
                max={2}
                step={0.1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Focused (0.0)</span>
                <span>Balanced (1.0)</span>
                <span>Creative (2.0)</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Controls randomness. Lower values make responses more focused and deterministic.
              </p>
            </div>

            {/* Max Tokens */}
            <div className="space-y-3">
              <Label>Max Tokens: {settings.max_tokens}</Label>
              <Slider
                value={[settings.max_tokens || 2000]}
                onValueChange={([value]) => updateSetting('max_tokens', value)}
                min={100}
                max={modelConfig?.maxTokens || 4000}
                step={100}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Short (100)</span>
                <span>Medium ({Math.floor((modelConfig?.maxTokens || 4000) / 2)})</span>
                <span>Long ({modelConfig?.maxTokens || 4000})</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Maximum number of tokens in the response.
              </p>
            </div>

            {/* Top P */}
            <div className="space-y-3">
              <Label>Top P: {settings.top_p?.toFixed(2)}</Label>
              <Slider
                value={[settings.top_p || 0.9]}
                onValueChange={([value]) => updateSetting('top_p', value)}
                min={0.1}
                max={1.0}
                step={0.05}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Narrow (0.1)</span>
                <span>Balanced (0.5)</span>
                <span>Wide (1.0)</span>
              </div>
              <p className="text-xs text-muted-foreground">
                Controls diversity via nucleus sampling. Lower values focus on more likely tokens.
              </p>
            </div>
          </div>

          <Separator />

          {/* Advanced Parameters */}
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Advanced Parameters</h3>

            <div className="grid grid-cols-2 gap-6">
              {/* Frequency Penalty */}
              <div className="space-y-3">
                <Label>Frequency Penalty: {settings.frequency_penalty?.toFixed(2)}</Label>
                <Slider
                  value={[settings.frequency_penalty || 0.0]}
                  onValueChange={([value]) => updateSetting('frequency_penalty', value)}
                  min={-2.0}
                  max={2.0}
                  step={0.1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Reduces repetition of tokens based on frequency.
                </p>
              </div>

              {/* Presence Penalty */}
              <div className="space-y-3">
                <Label>Presence Penalty: {settings.presence_penalty?.toFixed(2)}</Label>
                <Slider
                  value={[settings.presence_penalty || 0.0]}
                  onValueChange={([value]) => updateSetting('presence_penalty', value)}
                  min={-2.0}
                  max={2.0}
                  step={0.1}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Encourages new topics by penalizing repeated tokens.
                </p>
              </div>
            </div>

            {/* Stop Sequences */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Stop Sequences</Label>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={addStopSequence}
                  className="h-8"
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Add
                </Button>
              </div>
              <div className="space-y-2">
                {(settings.stop_sequences || []).map((sequence, index) => (
                  <div key={index} className="flex gap-2">
                    <Input
                      placeholder="Stop sequence..."
                      value={sequence}
                      onChange={(e) => updateStopSequence(index, e.target.value)}
                      className="flex-1"
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => removeStopSequence(index)}
                      className="h-10 w-10 p-0"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
                {(settings.stop_sequences || []).length === 0 && (
                  <p className="text-xs text-muted-foreground">
                    No stop sequences defined. Add sequences to stop generation at specific tokens.
                  </p>
                )}
              </div>
            </div>
          </div>

          <Separator />

          {/* System Prompt */}
          {modelConfig?.supportsSystemPrompt && (
            <div className="space-y-3">
              <Label>System Prompt</Label>
              <Textarea
                placeholder="Enter system prompt to define the AI's behavior and role..."
                value={settings.system_prompt || ''}
                onChange={(e) => updateSetting('system_prompt', e.target.value)}
                rows={6}
                className="resize-none"
              />
              <p className="text-xs text-muted-foreground">
                Define the AI's role, personality, and behavior guidelines.
              </p>
            </div>
          )}

          {!modelConfig?.supportsSystemPrompt && (
            <div className="flex items-center gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <p className="text-sm text-yellow-800">
                This model does not support custom system prompts.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Save Status */}
      {!hasChanges && !isSaving && modelSettings[currentModel] && (
        <div className="flex items-center gap-2 text-green-600">
          <CheckCircle className="h-4 w-4" />
          <span className="text-sm">Settings saved</span>
        </div>
      )}
    </div>
  )
}