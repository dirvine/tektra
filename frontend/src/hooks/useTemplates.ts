"use client"

import { useState, useEffect, useCallback } from 'react'
import { api, type ConversationTemplate } from '@/lib/api'

export interface UseTemplatesOptions {
  userId: number
  category?: string
  includePublic?: boolean
  autoLoad?: boolean
}

export interface UseTemplatesReturn {
  templates: ConversationTemplate[]
  isLoading: boolean
  error: string | null
  loadTemplates: () => Promise<void>
  createTemplate: (templateData: Partial<ConversationTemplate>) => Promise<ConversationTemplate | null>
  updateTemplate: (templateId: number, templateData: Partial<ConversationTemplate>) => Promise<ConversationTemplate | null>
  deleteTemplate: (templateId: number) => Promise<boolean>
  useTemplate: (templateId: number) => Promise<ConversationTemplate | null>
  toggleFavorite: (templateId: number) => Promise<ConversationTemplate | null>
  searchTemplates: (query: string, filters?: Partial<ConversationTemplate>) => ConversationTemplate[]
  getFavoriteTemplates: () => ConversationTemplate[]
  getTemplatesByCategory: (category: string) => ConversationTemplate[]
  getMostUsedTemplates: (limit?: number) => ConversationTemplate[]
  getRecentTemplates: (limit?: number) => ConversationTemplate[]
}

export function useTemplates({
  userId,
  category,
  includePublic = true,
  autoLoad = true
}: UseTemplatesOptions): UseTemplatesReturn {
  const [templates, setTemplates] = useState<ConversationTemplate[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadTemplates = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await api.getConversationTemplates(userId, category, includePublic)
      if (response.data) {
        setTemplates(response.data as ConversationTemplate[])
      } else {
        setError('Failed to load templates')
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load templates'
      setError(errorMessage)
      console.error('Failed to load templates:', err)
    } finally {
      setIsLoading(false)
    }
  }, [userId, category, includePublic])

  const createTemplate = useCallback(async (templateData: Partial<ConversationTemplate>): Promise<ConversationTemplate | null> => {
    try {
      const response = await api.createConversationTemplate(userId, templateData)
      if (response.data) {
        const newTemplate = response.data as ConversationTemplate
        setTemplates(prev => [...prev, newTemplate])
        return newTemplate
      }
      return null
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create template'
      setError(errorMessage)
      console.error('Failed to create template:', err)
      return null
    }
  }, [userId])

  const updateTemplate = useCallback(async (templateId: number, templateData: Partial<ConversationTemplate>): Promise<ConversationTemplate | null> => {
    try {
      const response = await api.updateConversationTemplate(userId, templateId, templateData)
      if (response.data) {
        const updatedTemplate = response.data as ConversationTemplate
        setTemplates(prev => prev.map(t => t.id === templateId ? updatedTemplate : t))
        return updatedTemplate
      }
      return null
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update template'
      setError(errorMessage)
      console.error('Failed to update template:', err)
      return null
    }
  }, [userId])

  const deleteTemplate = useCallback(async (templateId: number): Promise<boolean> => {
    try {
      const response = await api.deleteConversationTemplate(userId, templateId)
      if (response.status === 200) {
        setTemplates(prev => prev.filter(t => t.id !== templateId))
        return true
      }
      return false
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete template'
      setError(errorMessage)
      console.error('Failed to delete template:', err)
      return false
    }
  }, [userId])

  const useTemplate = useCallback(async (templateId: number): Promise<ConversationTemplate | null> => {
    try {
      const response = await api.useConversationTemplate(userId, templateId)
      if (response.data) {
        const usedTemplate = response.data as ConversationTemplate
        // Update local usage count
        setTemplates(prev => prev.map(t => 
          t.id === templateId 
            ? { ...t, usage_count: t.usage_count + 1, last_used: new Date().toISOString() }
            : t
        ))
        return usedTemplate
      }
      return null
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to use template'
      setError(errorMessage)
      console.error('Failed to use template:', err)
      return null
    }
  }, [userId])

  const toggleFavorite = useCallback(async (templateId: number): Promise<ConversationTemplate | null> => {
    try {
      const template = templates.find(t => t.id === templateId)
      if (!template) return null

      const response = await api.updateConversationTemplate(userId, templateId, {
        is_favorite: !template.is_favorite
      })
      
      if (response.data) {
        const updatedTemplate = response.data as ConversationTemplate
        setTemplates(prev => prev.map(t => t.id === templateId ? updatedTemplate : t))
        return updatedTemplate
      }
      return null
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to toggle favorite'
      setError(errorMessage)
      console.error('Failed to toggle favorite:', err)
      return null
    }
  }, [userId, templates])

  const searchTemplates = useCallback((query: string, filters?: Partial<ConversationTemplate>): ConversationTemplate[] => {
    let filtered = [...templates]

    // Text search
    if (query.trim()) {
      const searchQuery = query.toLowerCase()
      filtered = filtered.filter(template =>
        template.name.toLowerCase().includes(searchQuery) ||
        template.description?.toLowerCase().includes(searchQuery) ||
        template.system_prompt.toLowerCase().includes(searchQuery) ||
        template.tags.some(tag => tag.toLowerCase().includes(searchQuery))
      )
    }

    // Apply filters
    if (filters) {
      if (filters.category) {
        filtered = filtered.filter(t => t.category === filters.category)
      }
      if (filters.is_favorite !== undefined) {
        filtered = filtered.filter(t => t.is_favorite === filters.is_favorite)
      }
      if (filters.is_public !== undefined) {
        filtered = filtered.filter(t => t.is_public === filters.is_public)
      }
    }

    return filtered
  }, [templates])

  const getFavoriteTemplates = useCallback((): ConversationTemplate[] => {
    return templates
      .filter(template => template.is_favorite)
      .sort((a, b) => b.usage_count - a.usage_count)
  }, [templates])

  const getTemplatesByCategory = useCallback((category: string): ConversationTemplate[] => {
    return templates
      .filter(template => template.category === category)
      .sort((a, b) => b.usage_count - a.usage_count)
  }, [templates])

  const getMostUsedTemplates = useCallback((limit = 5): ConversationTemplate[] => {
    return templates
      .filter(template => template.usage_count > 0)
      .sort((a, b) => b.usage_count - a.usage_count)
      .slice(0, limit)
  }, [templates])

  const getRecentTemplates = useCallback((limit = 5): ConversationTemplate[] => {
    return templates
      .filter(template => template.last_used)
      .sort((a, b) => {
        const aDate = new Date(a.last_used || 0)
        const bDate = new Date(b.last_used || 0)
        return bDate.getTime() - aDate.getTime()
      })
      .slice(0, limit)
  }, [templates])

  // Auto-load templates on mount
  useEffect(() => {
    if (autoLoad) {
      loadTemplates()
    }
  }, [autoLoad, loadTemplates])

  return {
    templates,
    isLoading,
    error,
    loadTemplates,
    createTemplate,
    updateTemplate,
    deleteTemplate,
    useTemplate,
    toggleFavorite,
    searchTemplates,
    getFavoriteTemplates,
    getTemplatesByCategory,
    getMostUsedTemplates,
    getRecentTemplates
  }
}

// Utility function to apply a template to create a new conversation
export function applyTemplate(template: ConversationTemplate): {
  title: string
  systemPrompt: string
  initialMessages: Array<{role: string, content: string}>
  modelSettings: Record<string, any>
} {
  return {
    title: `${template.name} - ${new Date().toLocaleDateString()}`,
    systemPrompt: template.system_prompt,
    initialMessages: template.initial_messages,
    modelSettings: template.model_settings
  }
}

// Utility function to create a template from a conversation
export function createTemplateFromConversation(
  conversation: any,
  templateName: string,
  templateDescription?: string
): Partial<ConversationTemplate> {
  // Extract first few messages as initial messages
  const initialMessages = conversation.messages
    .slice(0, 4) // Take first 4 messages
    .map((msg: any) => ({
      role: msg.role,
      content: msg.content
    }))

  return {
    name: templateName,
    description: templateDescription || `Template created from conversation: ${conversation.title}`,
    system_prompt: conversation.system_prompt || "You are a helpful AI assistant.",
    category: 'custom',
    initial_messages: initialMessages,
    model_settings: conversation.model_settings || {},
    tags: ['custom', 'conversation-derived'],
    is_public: false,
    is_favorite: false
  }
}