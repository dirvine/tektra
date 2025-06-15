"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { 
  MessageSquare,
  Search,
  Star,
  Play,
  Users,
  Clock,
  ChevronRight,
  Zap,
  Code,
  Briefcase,
  Lightbulb,
  GraduationCap,
  Paintbrush,
  BookOpen
} from 'lucide-react'
import { api, type ConversationTemplate } from '@/lib/api'

interface TemplateSelectorProps {
  userId: number
  onSelectTemplate: (template: ConversationTemplate) => void
  onClose?: () => void
  compact?: boolean
}

const CATEGORY_ICONS = {
  work: Briefcase,
  coding: Code,
  creative: Paintbrush,
  education: GraduationCap,
  research: BookOpen,
  productivity: Zap,
  brainstorming: Lightbulb
}

export default function TemplateSelector({ 
  userId, 
  onSelectTemplate, 
  onClose, 
  compact = false 
}: TemplateSelectorProps) {
  const [templates, setTemplates] = useState<ConversationTemplate[]>([])
  const [filteredTemplates, setFilteredTemplates] = useState<ConversationTemplate[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('')

  useEffect(() => {
    loadTemplates()
  }, [userId])

  useEffect(() => {
    filterTemplates()
  }, [templates, searchQuery, selectedCategory])

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

    // Sort by favorites first, then usage count
    filtered.sort((a, b) => {
      if (a.is_favorite !== b.is_favorite) {
        return a.is_favorite ? -1 : 1
      }
      return b.usage_count - a.usage_count
    })

    setFilteredTemplates(filtered)
  }

  const handleSelectTemplate = async (template: ConversationTemplate) => {
    try {
      // Mark template as used
      await api.useConversationTemplate(userId, template.id)
      
      // Update local usage count
      setTemplates(prev => prev.map(t => 
        t.id === template.id 
          ? { ...t, usage_count: t.usage_count + 1 }
          : t
      ))
      
      onSelectTemplate(template)
    } catch (error) {
      console.error('Failed to use template:', error)
      // Still allow selection even if tracking fails
      onSelectTemplate(template)
    }
  }

  const getCategoryIcon = (category: string) => {
    return CATEGORY_ICONS[category as keyof typeof CATEGORY_ICONS] || MessageSquare
  }

  const getCategories = () => {
    const categories = [...new Set(templates.map(t => t.category).filter(Boolean))]
    return categories.sort()
  }

  if (isLoading) {
    return (
      <Card className={compact ? "w-80" : "w-96"}>
        <CardContent className="flex items-center justify-center py-8">
          <div className="text-center">
            <MessageSquare className="h-6 w-6 animate-pulse mx-auto mb-2 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Loading templates...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (compact) {
    return (
      <Card className="w-80">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Quick Templates
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 h-9"
            />
          </div>

          {/* Quick Templates List */}
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {filteredTemplates.slice(0, 8).map(template => {
              const CategoryIcon = getCategoryIcon(template.category || '')
              
              return (
                <button
                  key={template.id}
                  onClick={() => handleSelectTemplate(template)}
                  className="w-full p-2 text-left rounded-lg hover:bg-accent transition-colors group"
                >
                  <div className="flex items-center gap-2">
                    <CategoryIcon className="h-4 w-4 text-muted-foreground" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1">
                        <span className="font-medium text-sm truncate">{template.name}</span>
                        {template.is_favorite && (
                          <Star className="h-3 w-3 text-yellow-500 fill-current" />
                        )}
                      </div>
                      {template.description && (
                        <p className="text-xs text-muted-foreground truncate">
                          {template.description}
                        </p>
                      )}
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </button>
              )
            })}
            
            {filteredTemplates.length === 0 && (
              <div className="text-center py-4">
                <p className="text-sm text-muted-foreground">No templates found</p>
              </div>
            )}
          </div>

          {filteredTemplates.length > 8 && (
            <div className="pt-2 border-t">
              <p className="text-xs text-muted-foreground text-center">
                +{filteredTemplates.length - 8} more templates available
              </p>
            </div>
          )}

          {onClose && (
            <Button variant="outline" onClick={onClose} className="w-full mt-3">
              Close
            </Button>
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-96">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Conversation Templates
          </div>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ×
            </Button>
          )}
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search templates..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Category Filter */}
        {getCategories().length > 0 && (
          <div className="flex flex-wrap gap-1">
            <Button
              variant={selectedCategory === '' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedCategory('')}
              className="h-7 px-2 text-xs"
            >
              All
            </Button>
            {getCategories().map(category => {
              const CategoryIcon = getCategoryIcon(category)
              return (
                <Button
                  key={category}
                  variant={selectedCategory === category ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedCategory(category)}
                  className="h-7 px-2 text-xs"
                >
                  <CategoryIcon className="h-3 w-3 mr-1" />
                  {category}
                </Button>
              )
            })}
          </div>
        )}

        <Separator />

        {/* Templates List */}
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {filteredTemplates.length === 0 ? (
            <div className="text-center py-8">
              <MessageSquare className="h-8 w-8 mx-auto mb-3 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">No templates found</p>
              <p className="text-xs text-muted-foreground mt-1">
                {searchQuery || selectedCategory ? 'Try adjusting your search' : 'Create your first template'}
              </p>
            </div>
          ) : (
            filteredTemplates.map(template => {
              const CategoryIcon = getCategoryIcon(template.category || '')
              
              return (
                <Card key={template.id} className="cursor-pointer hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      {/* Header */}
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h4 className="font-medium text-sm">{template.name}</h4>
                            {template.is_favorite && (
                              <Star className="h-3 w-3 text-yellow-500 fill-current" />
                            )}
                            {template.is_public && (
                              <Badge variant="outline" className="text-xs h-4">
                                <Users className="h-2 w-2 mr-1" />
                                Public
                              </Badge>
                            )}
                          </div>
                          
                          <div className="flex items-center gap-2 mb-2">
                            <CategoryIcon className="h-3 w-3 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground capitalize">
                              {template.category || 'Uncategorized'}
                            </span>
                          </div>

                          {template.description && (
                            <p className="text-xs text-muted-foreground line-clamp-2">
                              {template.description}
                            </p>
                          )}
                        </div>
                      </div>

                      {/* Tags */}
                      {template.tags.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {template.tags.slice(0, 3).map(tag => (
                            <Badge key={tag} variant="secondary" className="text-xs h-4">
                              {tag}
                            </Badge>
                          ))}
                          {template.tags.length > 3 && (
                            <Badge variant="outline" className="text-xs h-4">
                              +{template.tags.length - 3}
                            </Badge>
                          )}
                        </div>
                      )}

                      {/* Stats */}
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          Used {template.usage_count} times
                        </div>
                        {template.last_used && (
                          <div>
                            {new Date(template.last_used).toLocaleDateString()}
                          </div>
                        )}
                      </div>

                      {/* Action */}
                      <Button
                        onClick={() => handleSelectTemplate(template)}
                        size="sm"
                        className="w-full"
                      >
                        <Play className="h-3 w-3 mr-2" />
                        Use Template
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )
            })
          )}
        </div>
      </CardContent>
    </Card>
  )
}