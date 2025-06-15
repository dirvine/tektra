"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  Settings,
  Palette,
  Mic,
  Bot,
  Shield,
  Bell,
  Monitor,
  Moon,
  Sun,
  Smartphone,
  Volume2,
  Eye,
  Keyboard,
  Save,
  RotateCcw,
  User,
  Sliders,
  Zap,
  Brain,
  MessageSquare,
  Code,
  Paintbrush
} from 'lucide-react'
import { api, type UserPreferences as UserPreferencesType } from '@/lib/api'

interface UserPreferencesProps {
  userId: number
  onClose?: () => void
}

export default function UserPreferences({ userId, onClose }: UserPreferencesProps) {
  const [preferences, setPreferences] = useState<UserPreferencesType | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  useEffect(() => {
    loadPreferences()
  }, [userId])

  const loadPreferences = async () => {
    setIsLoading(true)
    try {
      const response = await api.getUserPreferences(userId)
      if (response.data) {
        setPreferences(response.data as UserPreferencesType)
      }
    } catch (error) {
      console.error('Failed to load preferences:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const updatePreference = (key: keyof UserPreferencesType, value: any) => {
    if (!preferences) return
    
    setPreferences(prev => prev ? { ...prev, [key]: value } : null)
    setHasChanges(true)
  }

  const savePreferences = async () => {
    if (!preferences || !hasChanges) return

    setIsSaving(true)
    try {
      await api.updateUserPreferences(userId, preferences)
      setHasChanges(false)
    } catch (error) {
      console.error('Failed to save preferences:', error)
    } finally {
      setIsSaving(false)
    }
  }

  const resetPreferences = async () => {
    if (!confirm('Are you sure you want to reset all preferences to defaults?')) {
      return
    }

    setIsSaving(true)
    try {
      await api.resetUserPreferences(userId)
      await loadPreferences()
      setHasChanges(false)
    } catch (error) {
      console.error('Failed to reset preferences:', error)
    } finally {
      setIsSaving(false)
    }
  }

  if (isLoading) {
    return (
      <Card className="w-full max-w-4xl mx-auto">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Settings className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Loading preferences...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!preferences) {
    return (
      <Card className="w-full max-w-4xl mx-auto">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Settings className="h-8 w-8 mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Failed to load preferences</p>
            <Button onClick={loadPreferences} className="mt-4">
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Settings className="h-8 w-8" />
            User Preferences
          </h1>
          <p className="text-muted-foreground mt-1">
            Customize your Tektra AI Assistant experience
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
            onClick={resetPreferences}
            disabled={isSaving}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button
            onClick={savePreferences}
            disabled={!hasChanges || isSaving}
          >
            <Save className="h-4 w-4 mr-2" />
            {isSaving ? 'Saving...' : 'Save Changes'}
          </Button>
          {onClose && (
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>

      <Tabs defaultValue="appearance" className="w-full">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="appearance" className="flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Appearance
          </TabsTrigger>
          <TabsTrigger value="ai" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            AI & Models
          </TabsTrigger>
          <TabsTrigger value="voice" className="flex items-center gap-2">
            <Mic className="h-4 w-4" />
            Voice & Audio
          </TabsTrigger>
          <TabsTrigger value="avatar" className="flex items-center gap-2">
            <Bot className="h-4 w-4" />
            Avatar
          </TabsTrigger>
          <TabsTrigger value="privacy" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Privacy
          </TabsTrigger>
          <TabsTrigger value="notifications" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            Notifications
          </TabsTrigger>
        </TabsList>

        {/* Appearance Settings */}
        <TabsContent value="appearance" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Palette className="h-5 w-5" />
                Theme & Display
              </CardTitle>
              <CardDescription>
                Customize the visual appearance of your interface
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Theme Mode */}
              <div className="space-y-2">
                <Label>Theme Mode</Label>
                <Select
                  value={preferences.theme_mode}
                  onValueChange={(value) => updatePreference('theme_mode', value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">
                      <div className="flex items-center gap-2">
                        <Sun className="h-4 w-4" />
                        Light
                      </div>
                    </SelectItem>
                    <SelectItem value="dark">
                      <div className="flex items-center gap-2">
                        <Moon className="h-4 w-4" />
                        Dark
                      </div>
                    </SelectItem>
                    <SelectItem value="system">
                      <div className="flex items-center gap-2">
                        <Monitor className="h-4 w-4" />
                        System
                      </div>
                    </SelectItem>
                    <SelectItem value="auto">
                      <div className="flex items-center gap-2">
                        <Zap className="h-4 w-4" />
                        Auto (Scheduled)
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Font Size */}
              <div className="space-y-3">
                <Label>Font Size: {preferences.font_size}px</Label>
                <Slider
                  value={[preferences.font_size]}
                  onValueChange={([value]) => updatePreference('font_size', value)}
                  min={12}
                  max={20}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Small (12px)</span>
                  <span>Medium (16px)</span>
                  <span>Large (20px)</span>
                </div>
              </div>

              <Separator />

              {/* Interface Options */}
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Layout</h4>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="sidebar-collapsed">Collapse sidebar</Label>
                      <Switch
                        id="sidebar-collapsed"
                        checked={preferences.sidebar_collapsed}
                        onCheckedChange={(checked) => updatePreference('sidebar_collapsed', checked)}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="compact-mode">Compact mode</Label>
                      <Switch
                        id="compact-mode"
                        checked={preferences.compact_mode}
                        onCheckedChange={(checked) => updatePreference('compact_mode', checked)}
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Visual Elements</h4>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="show-avatars">Show avatars</Label>
                      <Switch
                        id="show-avatars"
                        checked={preferences.show_avatars}
                        onCheckedChange={(checked) => updatePreference('show_avatars', checked)}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="enable-animations">Enable animations</Label>
                      <Switch
                        id="enable-animations"
                        checked={preferences.enable_animations}
                        onCheckedChange={(checked) => updatePreference('enable_animations', checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Chat Display Options */}
              <div>
                <h4 className="text-sm font-medium mb-4">Chat Display</h4>
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="auto-scroll">Auto-scroll</Label>
                      <Switch
                        id="auto-scroll"
                        checked={preferences.auto_scroll}
                        onCheckedChange={(checked) => updatePreference('auto_scroll', checked)}
                      />
                    </div>
                    <div className="flex items-center justify-between">
                      <Label htmlFor="show-timestamps">Show timestamps</Label>
                      <Switch
                        id="show-timestamps"
                        checked={preferences.show_timestamps}
                        onCheckedChange={(checked) => updatePreference('show_timestamps', checked)}
                      />
                    </div>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="show-typing">Show typing indicators</Label>
                      <Switch
                        id="show-typing"
                        checked={preferences.show_typing_indicators}
                        onCheckedChange={(checked) => updatePreference('show_typing_indicators', checked)}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Custom CSS */}
              <div className="space-y-2">
                <Label htmlFor="custom-css">Custom CSS</Label>
                <Textarea
                  id="custom-css"
                  placeholder="/* Add your custom CSS here */"
                  value={preferences.custom_css || ''}
                  onChange={(e) => updatePreference('custom_css', e.target.value)}
                  rows={6}
                  className="font-mono text-sm"
                />
                <p className="text-xs text-muted-foreground">
                  Add custom CSS to personalize your interface appearance
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* AI & Models Settings */}
        <TabsContent value="ai" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                AI Model Settings
              </CardTitle>
              <CardDescription>
                Configure AI model behavior and preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Preferred Model */}
              <div className="space-y-2">
                <Label>Preferred Model</Label>
                <Select
                  value={preferences.preferred_model}
                  onValueChange={(value) => updatePreference('preferred_model', value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="phi-3-mini">Phi-3 Mini (Fast, efficient)</SelectItem>
                    <SelectItem value="llama-3-8b">Llama 3 8B (Balanced)</SelectItem>
                    <SelectItem value="mistral-7b">Mistral 7B (Creative)</SelectItem>
                    <SelectItem value="gemma-2b">Gemma 2B (Compact)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Default Temperature */}
              <div className="space-y-3">
                <Label>Default Temperature: {preferences.default_temperature}</Label>
                <Slider
                  value={[preferences.default_temperature]}
                  onValueChange={([value]) => updatePreference('default_temperature', value)}
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
              </div>

              {/* Max Tokens */}
              <div className="space-y-3">
                <Label>Default Max Tokens: {preferences.default_max_tokens}</Label>
                <Slider
                  value={[preferences.default_max_tokens]}
                  onValueChange={([value]) => updatePreference('default_max_tokens', value)}
                  min={100}
                  max={4000}
                  step={100}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Short (100)</span>
                  <span>Medium (2000)</span>
                  <span>Long (4000)</span>
                </div>
              </div>

              <Separator />

              {/* Response Options */}
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Response Format</h4>
                  <Select
                    value={preferences.response_format}
                    onValueChange={(value) => updatePreference('response_format', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="markdown">Markdown</SelectItem>
                      <SelectItem value="plain">Plain Text</SelectItem>
                      <SelectItem value="html">HTML</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Streaming</h4>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="enable-streaming">Enable streaming responses</Label>
                    <Switch
                      id="enable-streaming"
                      checked={preferences.enable_streaming}
                      onCheckedChange={(checked) => updatePreference('enable_streaming', checked)}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Voice & Audio Settings */}
        <TabsContent value="voice" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mic className="h-5 w-5" />
                Voice & Audio Settings
              </CardTitle>
              <CardDescription>
                Configure voice input and audio output preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Voice Provider */}
              <div className="space-y-2">
                <Label>Voice Provider</Label>
                <Select
                  value={preferences.voice_provider}
                  onValueChange={(value) => updatePreference('voice_provider', value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI</SelectItem>
                    <SelectItem value="elevenlabs">ElevenLabs</SelectItem>
                    <SelectItem value="azure">Azure Cognitive Services</SelectItem>
                    <SelectItem value="google">Google Cloud TTS</SelectItem>
                    <SelectItem value="system">System Default</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Voice Selection */}
              <div className="space-y-2">
                <Label>Voice</Label>
                <Select
                  value={preferences.voice_id}
                  onValueChange={(value) => updatePreference('voice_id', value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="alloy">Alloy (Neutral)</SelectItem>
                    <SelectItem value="echo">Echo (Male)</SelectItem>
                    <SelectItem value="fable">Fable (British)</SelectItem>
                    <SelectItem value="onyx">Onyx (Deep)</SelectItem>
                    <SelectItem value="nova">Nova (Female)</SelectItem>
                    <SelectItem value="shimmer">Shimmer (Soft)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Voice Speed */}
              <div className="space-y-3">
                <Label>Voice Speed: {preferences.voice_speed}x</Label>
                <Slider
                  value={[preferences.voice_speed]}
                  onValueChange={([value]) => updatePreference('voice_speed', value)}
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Slow (0.5x)</span>
                  <span>Normal (1.0x)</span>
                  <span>Fast (2.0x)</span>
                </div>
              </div>

              {/* Voice Pitch */}
              <div className="space-y-3">
                <Label>Voice Pitch: {preferences.voice_pitch}</Label>
                <Slider
                  value={[preferences.voice_pitch]}
                  onValueChange={([value]) => updatePreference('voice_pitch', value)}
                  min={-20}
                  max={20}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Lower (-20)</span>
                  <span>Normal (0)</span>
                  <span>Higher (+20)</span>
                </div>
              </div>

              <Separator />

              {/* Audio Options */}
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Speech Recognition</h4>
                  <div className="space-y-2">
                    <Label>Language</Label>
                    <Select
                      value={preferences.speech_recognition_language}
                      onValueChange={(value) => updatePreference('speech_recognition_language', value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="en-US">English (US)</SelectItem>
                        <SelectItem value="en-GB">English (UK)</SelectItem>
                        <SelectItem value="es-ES">Spanish</SelectItem>
                        <SelectItem value="fr-FR">French</SelectItem>
                        <SelectItem value="de-DE">German</SelectItem>
                        <SelectItem value="it-IT">Italian</SelectItem>
                        <SelectItem value="pt-BR">Portuguese</SelectItem>
                        <SelectItem value="ja-JP">Japanese</SelectItem>
                        <SelectItem value="ko-KR">Korean</SelectItem>
                        <SelectItem value="zh-CN">Chinese (Simplified)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-medium">Auto Speech</h4>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="auto-speech">Enable auto speech</Label>
                    <Switch
                      id="auto-speech"
                      checked={preferences.auto_speech}
                      onCheckedChange={(checked) => updatePreference('auto_speech', checked)}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Automatically speak AI responses
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Avatar Settings */}
        <TabsContent value="avatar" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bot className="h-5 w-5" />
                Avatar Settings
              </CardTitle>
              <CardDescription>
                Configure your AI assistant's visual avatar
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Enable Avatar */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="avatar-enabled">Enable Avatar</Label>
                  <p className="text-sm text-muted-foreground">
                    Show a visual avatar during conversations
                  </p>
                </div>
                <Switch
                  id="avatar-enabled"
                  checked={preferences.avatar_enabled}
                  onCheckedChange={(checked) => updatePreference('avatar_enabled', checked)}
                />
              </div>

              {preferences.avatar_enabled && (
                <>
                  <Separator />

                  {/* Avatar Selection */}
                  <div className="space-y-2">
                    <Label>Avatar Style</Label>
                    <Select
                      value={preferences.avatar_id}
                      onValueChange={(value) => updatePreference('avatar_id', value)}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="default">Default Assistant</SelectItem>
                        <SelectItem value="robot">Friendly Robot</SelectItem>
                        <SelectItem value="human-female">Human Female</SelectItem>
                        <SelectItem value="human-male">Human Male</SelectItem>
                        <SelectItem value="abstract">Abstract Form</SelectItem>
                        <SelectItem value="mascot">Mascot</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Gesture Sensitivity */}
                  <div className="space-y-3">
                    <Label>Gesture Sensitivity: {preferences.gesture_sensitivity}</Label>
                    <Slider
                      value={[preferences.gesture_sensitivity]}
                      onValueChange={([value]) => updatePreference('gesture_sensitivity', value)}
                      min={0}
                      max={100}
                      step={5}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Minimal (0)</span>
                      <span>Normal (50)</span>
                      <span>Expressive (100)</span>
                    </div>
                  </div>

                  {/* Expression Intensity */}
                  <div className="space-y-3">
                    <Label>Expression Intensity: {preferences.expression_intensity}</Label>
                    <Slider
                      value={[preferences.expression_intensity]}
                      onValueChange={([value]) => updatePreference('expression_intensity', value)}
                      min={0}
                      max={100}
                      step={5}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Subtle (0)</span>
                      <span>Balanced (50)</span>
                      <span>Dramatic (100)</span>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Privacy Settings */}
        <TabsContent value="privacy" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Privacy & Security
              </CardTitle>
              <CardDescription>
                Control your data privacy and security settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Privacy Mode */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="privacy-mode">Privacy Mode</Label>
                  <p className="text-sm text-muted-foreground">
                    Enhanced privacy with minimal data collection
                  </p>
                </div>
                <Switch
                  id="privacy-mode"
                  checked={preferences.privacy_mode}
                  onCheckedChange={(checked) => updatePreference('privacy_mode', checked)}
                />
              </div>

              <Separator />

              {/* Data Retention */}
              <div className="space-y-3">
                <Label>Data Retention Period: {preferences.data_retention_days} days</Label>
                <Slider
                  value={[preferences.data_retention_days]}
                  onValueChange={([value]) => updatePreference('data_retention_days', value)}
                  min={7}
                  max={365}
                  step={7}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>1 Week</span>
                  <span>6 Months</span>
                  <span>1 Year</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Automatically delete conversations older than this period
                </p>
              </div>

              <Separator />

              {/* Analytics Sharing */}
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="share-analytics">Share Anonymous Analytics</Label>
                  <p className="text-sm text-muted-foreground">
                    Help improve Tektra by sharing usage analytics
                  </p>
                </div>
                <Switch
                  id="share-analytics"
                  checked={preferences.share_analytics}
                  onCheckedChange={(checked) => updatePreference('share_analytics', checked)}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications Settings */}
        <TabsContent value="notifications" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" />
                Notifications
              </CardTitle>
              <CardDescription>
                Configure how you receive notifications and alerts
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Notification Level */}
              <div className="space-y-2">
                <Label>Notification Level</Label>
                <Select
                  value={preferences.notification_level}
                  onValueChange={(value) => updatePreference('notification_level', value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All notifications</SelectItem>
                    <SelectItem value="important">Important only</SelectItem>
                    <SelectItem value="minimal">Minimal</SelectItem>
                    <SelectItem value="none">None</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              {/* Notification Types */}
              <div className="space-y-4">
                <h4 className="text-sm font-medium">Notification Types</h4>
                <div className="grid grid-cols-1 gap-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label htmlFor="email-notifications">Email Notifications</Label>
                      <p className="text-sm text-muted-foreground">
                        Receive important updates via email
                      </p>
                    </div>
                    <Switch
                      id="email-notifications"
                      checked={preferences.email_notifications}
                      onCheckedChange={(checked) => updatePreference('email_notifications', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label htmlFor="push-notifications">Push Notifications</Label>
                      <p className="text-sm text-muted-foreground">
                        Browser push notifications for real-time updates
                      </p>
                    </div>
                    <Switch
                      id="push-notifications"
                      checked={preferences.push_notifications}
                      onCheckedChange={(checked) => updatePreference('push_notifications', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label htmlFor="sound-notifications">Sound Notifications</Label>
                      <p className="text-sm text-muted-foreground">
                        Play sounds for notifications and alerts
                      </p>
                    </div>
                    <Switch
                      id="sound-notifications"
                      checked={preferences.sound_notifications}
                      onCheckedChange={(checked) => updatePreference('sound_notifications', checked)}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}