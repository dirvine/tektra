"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  MessageSquare,
  Plus,
  Edit3,
  Trash2,
  Copy,
  Star,
  Users,
  Clock,
  Play,
  Settings,
  Tag,
  Search,
  Filter,
  MoreHorizontal,
  ChevronDown,
  ChevronUp,
  BookOpen,
  Zap,
  Code,
  Briefcase,
  Lightbulb,
  GraduationCap,
  Paintbrush
} from 'lucide-react'
import { api, type ConversationTemplate } from '@/lib/api'

interface ConversationTemplatesProps {
  userId: number
  onUseTemplate?: (template: ConversationTemplate) => void
  onClose?: () => void
}

interface TemplateFormData {
  name: string
  description: string
  system_prompt: string
  category: string
  initial_messages: Array<{role: string, content: string}>
  model_settings: Record<string, any>
  tags: string[]
  is_public: boolean
  is_favorite: boolean
}

const TEMPLATE_CATEGORIES = [
  { value: 'work', label: 'Work & Business', icon: Briefcase, color: 'bg-blue-500' },
  { value: 'coding', label: 'Programming', icon: Code, color: 'bg-green-500' },
  { value: 'creative', label: 'Creative Writing', icon: Paintbrush, color: 'bg-purple-500' },
  { value: 'education', label: 'Education', icon: GraduationCap, color: 'bg-orange-500' },
  { value: 'research', label: 'Research', icon: BookOpen, color: 'bg-red-500' },
  { value: 'productivity', label: 'Productivity', icon: Zap, color: 'bg-yellow-500' },
  { value: 'brainstorming', label: 'Brainstorming', icon: Lightbulb, color: 'bg-indigo-500' }
]

const PRESET_TEMPLATES = [
  {
    name: "Code Review Assistant",
    description: "AI assistant specialized in code review and suggestions",
    system_prompt: "You are an expert code reviewer. Analyze code for best practices, potential bugs, security issues, and suggest improvements. Be constructive and educational in your feedback.",
    category: "coding",
    tags: ["code-review", "programming", "best-practices"],
    initial_messages: [
      { role: "user", content: "Please review this code for any issues or improvements:" },
      { role: "assistant", content: "I'll review your code carefully. Please share the code you'd like me to examine, and I'll look for:\n\n• Logic errors and potential bugs\n• Security vulnerabilities\n• Performance optimizations\n• Code style and best practices\n• Documentation improvements\n\nGo ahead and paste your code!" }
    ]
  },
  {
    name: "Meeting Minutes Assistant",
    description: "Helps structure and organize meeting notes and action items",
    system_prompt: "You are a professional meeting assistant. Help organize meeting discussions into clear minutes with action items, decisions made, and follow-ups. Format everything clearly and professionally.",
    category: "work",
    tags: ["meetings", "organization", "productivity"],
    initial_messages: [
      { role: "user", content: "Help me organize these meeting notes:" },
      { role: "assistant", content: "I'll help you create professional meeting minutes. Please share your meeting notes, and I'll organize them into:\n\n📋 **Meeting Summary**\n✅ **Key Decisions**\n🎯 **Action Items** (with owners and deadlines)\n📌 **Follow-up Topics**\n\nShare your notes when ready!" }
    ]
  },
  {
    name: "Creative Writing Coach",
    description: "AI writing coach for creative projects and storytelling",
    system_prompt: "You are a supportive creative writing coach. Help develop characters, plot ideas, dialogue, and provide constructive feedback on creative writing. Encourage creativity while offering practical advice.",
    category: "creative",
    tags: ["writing", "creativity", "storytelling"],
    initial_messages: [
      { role: "user", content: "I need help with my creative writing project" },
      { role: "assistant", content: "I'm excited to help with your creative writing! I can assist with:\n\n✍️ **Character Development** - Building compelling characters\n📖 **Plot Structure** - Story arcs and pacing\n💬 **Dialogue** - Natural conversations\n🎨 **World Building** - Creating vivid settings\n📝 **Writing Feedback** - Constructive reviews\n\nWhat aspect of your project would you like to work on?" }
    ]
  },
  {
    name: "Research Assistant",
    description: "Helps with research methodology and academic writing",
    system_prompt: "You are a knowledgeable research assistant. Help with research methodology, source evaluation, academic writing, and organizing information. Maintain academic rigor and encourage critical thinking.",
    category: "research",
    tags: ["research", "academic", "methodology"],
    initial_messages: [
      { role: "user", content: "I need help with my research project" },
      { role: "assistant", content: "I'm here to support your research! I can help with:\n\n🔍 **Research Strategy** - Planning your approach\n📚 **Source Evaluation** - Assessing credibility\n📝 **Academic Writing** - Structure and style\n📊 **Data Analysis** - Interpreting findings\n🎯 **Thesis Development** - Refining your argument\n\nWhat's your research topic and how can I assist?" }
    ]
  },
  {
    name: "Learning Tutor",
    description: "Patient tutor for explaining complex concepts",
    system_prompt: "You are a patient and encouraging tutor. Break down complex concepts into understandable parts, use analogies and examples, and adapt your teaching style to the student's needs. Always encourage questions.",
    category: "education",
    tags: ["tutoring", "learning", "education"],
    initial_messages: [
      { role: "user", content: "I need help understanding this concept" },
      { role: "assistant", content: "I'm here to help you learn! I believe every concept can be understood with the right approach. I'll:\n\n🧩 **Break it down** into manageable pieces\n🌟 **Use examples** and analogies\n❓ **Encourage questions** - no question is too basic\n📈 **Build step by step** from basics to advanced\n✅ **Check understanding** along the way\n\nWhat would you like to learn about?" }
    ]
  }
]

export default function ConversationTemplates({ userId, onUseTemplate, onClose }: ConversationTemplatesProps) {
  const [templates, setTemplates] = useState<ConversationTemplate[]>([])
  const [filteredTemplates, setFilteredTemplates] = useState<ConversationTemplate[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [editingTemplate, setEditingTemplate] = useState<ConversationTemplate | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [showOnlyFavorites, setShowOnlyFavorites] = useState(false)
  const [showOnlyPublic, setShowOnlyPublic] = useState(false)
  const [expandedTemplate, setExpandedTemplate] = useState<number | null>(null)

  const [formData, setFormData] = useState<TemplateFormData>({
    name: '',
    description: '',
    system_prompt: '',
    category: '',
    initial_messages: [],
    model_settings: {},
    tags: [],
    is_public: false,
    is_favorite: false
  })

  useEffect(() => {
    loadTemplates()
  }, [userId])

  useEffect(() => {
    filterTemplates()
  }, [templates, searchQuery, selectedCategory, showOnlyFavorites, showOnlyPublic])

  const loadTemplates = async () => {
    setIsLoading(true)
    try {
      const response = await api.getConversationTemplates(userId, undefined, true)
      if (response.data) {
        setTemplates(response.data as ConversationTemplate[])
      }
    } catch (error) {
      console.error('Failed to load templates:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const filterTemplates = () => {
    let filtered = [...templates]

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(template => 
        template.name.toLowerCase().includes(query) ||
        template.description?.toLowerCase().includes(query) ||
        template.tags.some(tag => tag.toLowerCase().includes(query))
      )
    }

    // Category filter
    if (selectedCategory) {
      filtered = filtered.filter(template => template.category === selectedCategory)
    }

    // Favorites filter
    if (showOnlyFavorites) {
      filtered = filtered.filter(template => template.is_favorite)
    }

    // Public filter
    if (showOnlyPublic) {
      filtered = filtered.filter(template => template.is_public)
    }

    // Sort by usage count and favorites
    filtered.sort((a, b) => {
      if (a.is_favorite !== b.is_favorite) {
        return a.is_favorite ? -1 : 1
      }
      return b.usage_count - a.usage_count
    })

    setFilteredTemplates(filtered)
  }

  const createTemplate = async () => {
    if (!formData.name || !formData.system_prompt) return

    try {
      await api.createConversationTemplate(userId, formData)
      await loadTemplates()
      resetForm()
      setShowCreateForm(false)
    } catch (error) {
      console.error('Failed to create template:', error)
    }
  }

  const updateTemplate = async () => {
    if (!editingTemplate || !formData.name || !formData.system_prompt) return

    try {
      await api.updateConversationTemplate(userId, editingTemplate.id, formData)
      await loadTemplates()
      resetForm()
      setEditingTemplate(null)
    } catch (error) {
      console.error('Failed to update template:', error)
    }
  }

  const deleteTemplate = async (templateId: number) => {
    if (!confirm('Are you sure you want to delete this template?')) return

    try {
      await api.deleteConversationTemplate(userId, templateId)
      setTemplates(prev => prev.filter(t => t.id !== templateId))
    } catch (error) {
      console.error('Failed to delete template:', error)
    }
  }

  const useTemplate = async (template: ConversationTemplate) => {
    try {
      await api.useConversationTemplate(userId, template.id)
      onUseTemplate?.(template)
      
      // Update usage count locally
      setTemplates(prev => prev.map(t => 
        t.id === template.id 
          ? { ...t, usage_count: t.usage_count + 1 }
          : t
      ))
    } catch (error) {
      console.error('Failed to use template:', error)
    }
  }

  const toggleFavorite = async (template: ConversationTemplate) => {
    try {
      await api.updateConversationTemplate(userId, template.id, {
        is_favorite: !template.is_favorite
      })
      
      setTemplates(prev => prev.map(t => 
        t.id === template.id 
          ? { ...t, is_favorite: !t.is_favorite }
          : t
      ))
    } catch (error) {
      console.error('Failed to toggle favorite:', error)
    }
  }

  const loadPresetTemplate = (preset: any) => {
    setFormData({
      name: preset.name,
      description: preset.description,
      system_prompt: preset.system_prompt,
      category: preset.category,
      initial_messages: preset.initial_messages,
      model_settings: {},
      tags: preset.tags,
      is_public: false,
      is_favorite: false
    })
    setShowCreateForm(true)
  }

  const startEdit = (template: ConversationTemplate) => {
    setFormData({
      name: template.name,
      description: template.description || '',
      system_prompt: template.system_prompt,
      category: template.category || '',
      initial_messages: template.initial_messages,
      model_settings: template.model_settings,
      tags: template.tags,
      is_public: template.is_public,
      is_favorite: template.is_favorite
    })
    setEditingTemplate(template)
    setShowCreateForm(true)
  }

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      system_prompt: '',
      category: '',
      initial_messages: [],
      model_settings: {},
      tags: [],
      is_public: false,
      is_favorite: false
    })
    setEditingTemplate(null)
    setShowCreateForm(false)
  }

  const addInitialMessage = () => {
    setFormData(prev => ({
      ...prev,
      initial_messages: [...prev.initial_messages, { role: 'user', content: '' }]
    }))
  }

  const updateInitialMessage = (index: number, field: 'role' | 'content', value: string) => {
    setFormData(prev => ({
      ...prev,
      initial_messages: prev.initial_messages.map((msg, i) => 
        i === index ? { ...msg, [field]: value } : msg
      )
    }))
  }

  const removeInitialMessage = (index: number) => {
    setFormData(prev => ({
      ...prev,
      initial_messages: prev.initial_messages.filter((_, i) => i !== index)
    }))
  }

  const addTag = (tag: string) => {
    if (tag && !formData.tags.includes(tag)) {
      setFormData(prev => ({
        ...prev,
        tags: [...prev.tags, tag]
      }))
    }
  }

  const removeTag = (tagToRemove: string) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }))
  }

  const getCategoryInfo = (category: string) => {
    return TEMPLATE_CATEGORIES.find(c => c.value === category)
  }

  if (isLoading) {
    return (
      <Card className="w-full max-w-6xl mx-auto">
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <Settings className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
            <p className="text-muted-foreground">Loading templates...</p>
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
            <MessageSquare className="h-8 w-8" />
            Conversation Templates
          </h1>
          <p className="text-muted-foreground mt-1">
            Create and manage reusable conversation templates for consistent AI interactions
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => setShowCreateForm(!showCreateForm)}
          >
            <Plus className="h-4 w-4 mr-2" />
            Create Template
          </Button>
          {onClose && (
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </div>

      <Tabs defaultValue="templates" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="templates">My Templates</TabsTrigger>
          <TabsTrigger value="presets">Preset Templates</TabsTrigger>
          <TabsTrigger value="create" disabled={!showCreateForm}>
            {editingTemplate ? 'Edit Template' : 'Create Template'}
          </TabsTrigger>
        </TabsList>

        {/* My Templates Tab */}
        <TabsContent value="templates" className="space-y-6">
          {/* Filters */}
          <Card>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {/* Search */}
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search templates..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-9"
                  />
                </div>

                {/* Category Filter */}
                <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                  <SelectTrigger>
                    <SelectValue placeholder="All categories" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All categories</SelectItem>
                    {TEMPLATE_CATEGORIES.map(category => (
                      <SelectItem key={category.value} value={category.value}>
                        <div className="flex items-center gap-2">
                          <category.icon className="h-4 w-4" />
                          {category.label}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {/* Quick Filters */}
                <div className="flex items-center gap-4">
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="favorites"
                      checked={showOnlyFavorites}
                      onCheckedChange={setShowOnlyFavorites}
                    />
                    <Label htmlFor="favorites" className="text-sm">Favorites</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="public"
                      checked={showOnlyPublic}
                      onCheckedChange={setShowOnlyPublic}
                    />
                    <Label htmlFor="public" className="text-sm">Public</Label>
                  </div>
                </div>

                {/* Clear Filters */}
                <Button
                  variant="outline"
                  onClick={() => {
                    setSearchQuery('')
                    setSelectedCategory('')
                    setShowOnlyFavorites(false)
                    setShowOnlyPublic(false)
                  }}
                  className="w-full"
                >
                  Clear Filters
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Templates Grid */}
          {filteredTemplates.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center">
                  <MessageSquare className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-medium mb-2">No templates found</h3>
                  <p className="text-muted-foreground mb-4">
                    {searchQuery || selectedCategory || showOnlyFavorites || showOnlyPublic
                      ? 'Try adjusting your filters'
                      : 'Create your first template to get started'
                    }
                  </p>
                  <Button onClick={() => setShowCreateForm(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Create Template
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredTemplates.map(template => {
                const categoryInfo = getCategoryInfo(template.category || '')
                const isExpanded = expandedTemplate === template.id

                return (
                  <Card key={template.id} className="relative">
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <CardTitle className="text-lg">{template.name}</CardTitle>
                            {template.is_favorite && (
                              <Star className="h-4 w-4 text-yellow-500 fill-current" />
                            )}
                            {template.is_public && (
                              <Badge variant="outline" className="text-xs">
                                <Users className="h-3 w-3 mr-1" />
                                Public
                              </Badge>
                            )}
                          </div>
                          
                          {categoryInfo && (
                            <Badge variant="secondary" className="mb-2">
                              <categoryInfo.icon className="h-3 w-3 mr-1" />
                              {categoryInfo.label}
                            </Badge>
                          )}

                          <CardDescription className="line-clamp-2">
                            {template.description}
                          </CardDescription>
                        </div>

                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => toggleFavorite(template)}
                          className="h-8 w-8 p-0"
                        >
                          <Star className={`h-4 w-4 ${template.is_favorite ? 'text-yellow-500 fill-current' : ''}`} />
                        </Button>
                      </div>

                      {/* Tags */}
                      {template.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {template.tags.slice(0, 3).map(tag => (
                            <Badge key={tag} variant="outline" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                          {template.tags.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{template.tags.length - 3}
                            </Badge>
                          )}
                        </div>
                      )}
                    </CardHeader>

                    <CardContent className="space-y-4">
                      {/* Usage Stats */}
                      <div className="flex items-center justify-between text-sm text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          Used {template.usage_count} times
                        </div>
                        {template.last_used && (
                          <div>
                            Last: {new Date(template.last_used).toLocaleDateString()}
                          </div>
                        )}
                      </div>

                      {/* Expandable Content */}
                      <div className="space-y-3">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setExpandedTemplate(isExpanded ? null : template.id)}
                          className="w-full justify-between p-2"
                        >
                          <span>Preview</span>
                          {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </Button>

                        {isExpanded && (
                          <div className="space-y-3 p-3 bg-muted/50 rounded-lg">
                            <div>
                              <Label className="text-xs font-medium">System Prompt:</Label>
                              <p className="text-xs text-muted-foreground mt-1 line-clamp-3">
                                {template.system_prompt}
                              </p>
                            </div>
                            
                            {template.initial_messages.length > 0 && (
                              <div>
                                <Label className="text-xs font-medium">Initial Messages:</Label>
                                <div className="space-y-1 mt-1">
                                  {template.initial_messages.slice(0, 2).map((msg, idx) => (
                                    <div key={idx} className="text-xs">
                                      <span className="font-medium capitalize">{msg.role}:</span>
                                      <span className="text-muted-foreground ml-1 line-clamp-1">
                                        {msg.content}
                                      </span>
                                    </div>
                                  ))}
                                  {template.initial_messages.length > 2 && (
                                    <p className="text-xs text-muted-foreground">
                                      +{template.initial_messages.length - 2} more messages
                                    </p>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Actions */}
                      <div className="flex gap-2">
                        <Button
                          onClick={() => useTemplate(template)}
                          className="flex-1"
                          size="sm"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          Use Template
                        </Button>
                        
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => startEdit(template)}
                          className="h-8 w-8 p-0"
                        >
                          <Edit3 className="h-3 w-3" />
                        </Button>
                        
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => deleteTemplate(template.id)}
                          className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </TabsContent>

        {/* Preset Templates Tab */}
        <TabsContent value="presets" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Preset Templates</CardTitle>
              <CardDescription>
                Professional templates ready to use. Click "Use Preset" to customize and add to your collection.
              </CardDescription>
            </CardHeader>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {PRESET_TEMPLATES.map((preset, index) => {
              const categoryInfo = getCategoryInfo(preset.category)

              return (
                <Card key={index}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2 mb-2">
                      <CardTitle className="text-lg">{preset.name}</CardTitle>
                      {categoryInfo && (
                        <Badge variant="secondary">
                          <categoryInfo.icon className="h-3 w-3 mr-1" />
                          {categoryInfo.label}
                        </Badge>
                      )}
                    </div>
                    <CardDescription>{preset.description}</CardDescription>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {/* Tags */}
                    <div className="flex flex-wrap gap-1">
                      {preset.tags.map(tag => (
                        <Badge key={tag} variant="outline" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>

                    {/* Preview */}
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <Label className="text-xs font-medium">System Prompt Preview:</Label>
                      <p className="text-xs text-muted-foreground mt-1 line-clamp-3">
                        {preset.system_prompt}
                      </p>
                    </div>

                    {/* Actions */}
                    <Button
                      onClick={() => loadPresetTemplate(preset)}
                      className="w-full"
                      size="sm"
                    >
                      <Copy className="h-4 w-4 mr-2" />
                      Use Preset
                    </Button>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </TabsContent>

        {/* Create/Edit Template Tab */}
        <TabsContent value="create" className="space-y-6">
          {showCreateForm && (
            <Card>
              <CardHeader>
                <CardTitle>
                  {editingTemplate ? 'Edit Template' : 'Create New Template'}
                </CardTitle>
                <CardDescription>
                  {editingTemplate 
                    ? 'Update your template configuration'
                    : 'Define a new conversation template for consistent AI interactions'
                  }
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-6">
                {/* Basic Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="name">Template Name *</Label>
                    <Input
                      id="name"
                      placeholder="e.g., Code Review Assistant"
                      value={formData.name}
                      onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="category">Category</Label>
                    <Select
                      value={formData.category}
                      onValueChange={(value) => setFormData(prev => ({ ...prev, category: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select category" />
                      </SelectTrigger>
                      <SelectContent>
                        {TEMPLATE_CATEGORIES.map(category => (
                          <SelectItem key={category.value} value={category.value}>
                            <div className="flex items-center gap-2">
                              <category.icon className="h-4 w-4" />
                              {category.label}
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    placeholder="Brief description of what this template does..."
                    value={formData.description}
                    onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    rows={2}
                  />
                </div>

                <Separator />

                {/* System Prompt */}
                <div className="space-y-2">
                  <Label htmlFor="system-prompt">System Prompt *</Label>
                  <Textarea
                    id="system-prompt"
                    placeholder="Define the AI's role, behavior, and expertise..."
                    value={formData.system_prompt}
                    onChange={(e) => setFormData(prev => ({ ...prev, system_prompt: e.target.value }))}
                    rows={6}
                  />
                  <p className="text-xs text-muted-foreground">
                    This sets the AI's personality and expertise for conversations using this template.
                  </p>
                </div>

                <Separator />

                {/* Initial Messages */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>Initial Messages</Label>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={addInitialMessage}
                    >
                      <Plus className="h-4 w-4 mr-2" />
                      Add Message
                    </Button>
                  </div>

                  <div className="space-y-3">
                    {formData.initial_messages.map((message, index) => (
                      <div key={index} className="flex gap-3 p-3 border rounded-lg">
                        <div className="w-24">
                          <Select
                            value={message.role}
                            onValueChange={(value) => updateInitialMessage(index, 'role', value)}
                          >
                            <SelectTrigger className="h-8">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="user">User</SelectItem>
                              <SelectItem value="assistant">Assistant</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        
                        <Textarea
                          placeholder="Message content..."
                          value={message.content}
                          onChange={(e) => updateInitialMessage(index, 'content', e.target.value)}
                          rows={2}
                          className="flex-1"
                        />
                        
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => removeInitialMessage(index)}
                          className="h-8 w-8 p-0"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    ))}
                    
                    {formData.initial_messages.length === 0 && (
                      <p className="text-sm text-muted-foreground text-center py-4">
                        No initial messages. Add messages to start conversations with context.
                      </p>
                    )}
                  </div>
                </div>

                <Separator />

                {/* Tags */}
                <div className="space-y-4">
                  <Label>Tags</Label>
                  <div className="flex flex-wrap gap-2">
                    {formData.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="cursor-pointer">
                        {tag}
                        <button
                          type="button"
                          onClick={() => removeTag(tag)}
                          className="ml-1 hover:text-destructive"
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                  </div>
                  
                  <div className="flex gap-2">
                    <Input
                      placeholder="Add a tag..."
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault()
                          const input = e.target as HTMLInputElement
                          addTag(input.value.trim())
                          input.value = ''
                        }
                      }}
                    />
                  </div>
                </div>

                <Separator />

                {/* Settings */}
                <div className="grid grid-cols-2 gap-6">
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="is-public"
                      checked={formData.is_public}
                      onCheckedChange={(checked) => setFormData(prev => ({ ...prev, is_public: checked }))}
                    />
                    <Label htmlFor="is-public">Make Public</Label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      id="is-favorite"
                      checked={formData.is_favorite}
                      onCheckedChange={(checked) => setFormData(prev => ({ ...prev, is_favorite: checked }))}
                    />
                    <Label htmlFor="is-favorite">Add to Favorites</Label>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3 pt-6">
                  <Button
                    onClick={editingTemplate ? updateTemplate : createTemplate}
                    disabled={!formData.name || !formData.system_prompt}
                  >
                    {editingTemplate ? 'Update Template' : 'Create Template'}
                  </Button>
                  
                  <Button variant="outline" onClick={resetForm}>
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}