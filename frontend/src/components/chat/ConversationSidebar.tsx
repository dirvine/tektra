"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { 
  Plus, 
  Search, 
  MessageSquare, 
  Trash2, 
  Edit3, 
  Clock,
  User,
  Bot,
  Filter,
  Pin,
  Archive,
  Tag,
  ChevronDown,
  X,
  Star,
  MoreHorizontal,
  SortAsc,
  SortDesc,
  Calendar,
  Hash
} from 'lucide-react'
import { api, type Conversation, type Tag as TagType, type SearchParams } from '@/lib/api'

export interface ConversationWithMessages extends Conversation {
  messages: Array<{
    id: number
    conversation_id: number
    role: string
    content: string
    message_type: string
    created_at: string
    metadata: Record<string, unknown>
  }>
}

interface FilterState {
  category?: string
  tags: string[]
  isPinned?: boolean
  isArchived?: boolean
  modelName?: string
  dateFrom?: string
  dateTo?: string
  minMessages?: number
  maxMessages?: number
}

interface SortState {
  field: string
  order: 'asc' | 'desc'
}

interface ConversationSidebarProps {
  currentConversation: number | null
  onConversationSelect: (conversation: ConversationWithMessages) => void
  onNewConversation: () => void
}

export default function ConversationSidebar({ 
  currentConversation, 
  onConversationSelect, 
  onNewConversation 
}: ConversationSidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [editTitle, setEditTitle] = useState('')
  const [tags, setTags] = useState<TagType[]>([])
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [showFilters, setShowFilters] = useState(false)
  const [showArchived, setShowArchived] = useState(false)
  const [filters, setFilters] = useState<FilterState>({
    tags: [],
    isArchived: false
  })
  const [sort, setSort] = useState<SortState>({
    field: 'updated_at',
    order: 'desc'
  })

  useEffect(() => {
    loadConversations()
    loadTags()
  }, [])

  useEffect(() => {
    performSearch()
  }, [searchQuery, filters, sort])

  const loadConversations = async () => {
    setIsLoading(true)
    try {
      const response = await api.getConversations(50, 0)
      if (response.data) {
        setConversations(response.data as Conversation[])
      }
    } catch (error) {
      console.error('Failed to load conversations:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadTags = async () => {
    try {
      const response = await api.getTags()
      if (response.data) {
        setTags(response.data as TagType[])
      }
    } catch (error) {
      console.error('Failed to load tags:', error)
    }
  }

  const performSearch = async () => {
    setIsLoading(true)
    try {
      const searchParams: SearchParams = {
        query: searchQuery || undefined,
        tags: filters.tags.length > 0 ? filters.tags : undefined,
        category: filters.category,
        model_name: filters.modelName,
        date_from: filters.dateFrom,
        date_to: filters.dateTo,
        is_pinned: filters.isPinned,
        is_archived: filters.isArchived,
        min_messages: filters.minMessages,
        max_messages: filters.maxMessages,
        sort_by: sort.field,
        sort_order: sort.order,
        limit: 50,
        offset: 0
      }

      const response = await api.searchConversations(searchParams)
      if (response.data) {
        setConversations(response.data as Conversation[])
      }
    } catch (error) {
      console.error('Failed to search conversations:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleConversationClick = async (conversation: Conversation) => {
    try {
      const response = await api.getConversation(conversation.id)
      if (response.data) {
        onConversationSelect(response.data as ConversationWithMessages)
      }
    } catch (error) {
      console.error('Failed to load conversation:', error)
    }
  }

  const handleDeleteConversation = async (conversationId: number, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this conversation?')) {
      return
    }

    try {
      await api.deleteConversation(conversationId)
      setConversations(prev => prev.filter(c => c.id !== conversationId))
      
      // If we deleted the current conversation, clear it
      if (currentConversation === conversationId) {
        onNewConversation()
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error)
    }
  }

  const handleEditTitle = async (conversationId: number, newTitle: string) => {
    if (!newTitle.trim()) return

    try {
      await api.updateConversationTitle(conversationId, newTitle)
      setConversations(prev => 
        prev.map(c => 
          c.id === conversationId 
            ? { ...c, title: newTitle }
            : c
        )
      )
      setEditingId(null)
      setEditTitle('')
    } catch (error) {
      console.error('Failed to update conversation title:', error)
    }
  }

  const handlePinConversation = async (conversationId: number, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const conversation = conversations.find(c => c.id === conversationId)
      if (!conversation) return

      await api.updateConversationMetadata(conversationId, {
        is_pinned: !conversation.is_pinned
      })
      
      setConversations(prev => 
        prev.map(c => 
          c.id === conversationId 
            ? { ...c, is_pinned: !c.is_pinned }
            : c
        )
      )
    } catch (error) {
      console.error('Failed to pin conversation:', error)
    }
  }

  const handleArchiveConversation = async (conversationId: number, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const conversation = conversations.find(c => c.id === conversationId)
      if (!conversation) return

      await api.updateConversationMetadata(conversationId, {
        is_archived: !conversation.is_archived
      })
      
      // Remove from current view if we're not showing archived
      if (!showArchived) {
        setConversations(prev => prev.filter(c => c.id !== conversationId))
      } else {
        setConversations(prev => 
          prev.map(c => 
            c.id === conversationId 
              ? { ...c, is_archived: !c.is_archived }
              : c
          )
        )
      }
    } catch (error) {
      console.error('Failed to archive conversation:', error)
    }
  }

  const clearFilters = () => {
    setFilters({
      tags: [],
      isArchived: false
    })
    setSelectedTags([])
    setSearchQuery('')
  }

  const toggleTag = (tagName: string) => {
    const newTags = selectedTags.includes(tagName)
      ? selectedTags.filter(t => t !== tagName)
      : [...selectedTags, tagName]
    
    setSelectedTags(newTags)
    setFilters(prev => ({ ...prev, tags: newTags }))
  }

  const toggleSort = (field: string) => {
    setSort(prev => ({
      field,
      order: prev.field === field && prev.order === 'desc' ? 'asc' : 'desc'
    }))
  }

  const startEditing = (conversation: Conversation, e: React.MouseEvent) => {
    e.stopPropagation()
    setEditingId(conversation.id)
    setEditTitle(conversation.title)
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))
    
    if (diffInDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } else if (diffInDays === 1) {
      return 'Yesterday'
    } else if (diffInDays < 7) {
      return `${diffInDays} days ago`
    } else {
      return date.toLocaleDateString()
    }
  }

  const getMessageCountDisplay = (count?: number) => {
    if (!count) return ''
    return count === 1 ? '1 message' : `${count} messages`
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Conversations
          </span>
          <div className="flex items-center gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setShowFilters(!showFilters)}
              className="h-8 w-8 p-0"
            >
              <Filter className="h-4 w-4" />
            </Button>
            <Button
              size="sm"
              onClick={onNewConversation}
              className="h-8 w-8 p-0"
            >
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </CardTitle>
        
        <div className="space-y-3">
          <div className="relative">
            <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search conversations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Quick Filters */}
          <div className="flex items-center gap-2 text-xs">
            <Button
              size="sm"
              variant={showArchived ? "default" : "outline"}
              onClick={() => {
                setShowArchived(!showArchived)
                setFilters(prev => ({ ...prev, isArchived: !showArchived }))
              }}
              className="h-6 px-2"
            >
              <Archive className="h-3 w-3 mr-1" />
              Archived
            </Button>
            <Button
              size="sm"
              variant={filters.isPinned ? "default" : "outline"}
              onClick={() => setFilters(prev => ({ ...prev, isPinned: !prev.isPinned }))}
              className="h-6 px-2"
            >
              <Pin className="h-3 w-3 mr-1" />
              Pinned
            </Button>
          </div>

          {/* Selected Tags */}
          {selectedTags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {selectedTags.map(tagName => (
                <Badge
                  key={tagName}
                  variant="secondary"
                  className="text-xs h-5 px-2 cursor-pointer"
                  onClick={() => toggleTag(tagName)}
                >
                  {tagName}
                  <X className="h-3 w-3 ml-1" />
                </Badge>
              ))}
              <Button
                size="sm"
                variant="ghost"
                onClick={clearFilters}
                className="h-5 px-2 text-xs"
              >
                Clear all
              </Button>
            </div>
          )}

          {/* Advanced Filters */}
          <Collapsible open={showFilters} onOpenChange={setShowFilters}>
            <CollapsibleContent className="space-y-3 pt-2 border-t">
              {/* Sort Options */}
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-2 block">
                  Sort by
                </label>
                <div className="flex gap-1">
                  {[
                    { field: 'updated_at', label: 'Recent', icon: Clock },
                    { field: 'created_at', label: 'Created', icon: Calendar },
                    { field: 'message_count', label: 'Messages', icon: Hash },
                    { field: 'title', label: 'Title', icon: SortAsc }
                  ].map(({ field, label, icon: Icon }) => (
                    <Button
                      key={field}
                      size="sm"
                      variant={sort.field === field ? "default" : "outline"}
                      onClick={() => toggleSort(field)}
                      className="h-6 px-2 text-xs flex-1"
                    >
                      <Icon className="h-3 w-3 mr-1" />
                      {label}
                      {sort.field === field && (
                        sort.order === 'desc' ? 
                          <SortDesc className="h-3 w-3 ml-1" /> : 
                          <SortAsc className="h-3 w-3 ml-1" />
                      )}
                    </Button>
                  ))}
                </div>
              </div>

              {/* Category Filter */}
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-2 block">
                  Category
                </label>
                <Select
                  value={filters.category || ""}
                  onValueChange={(value) => 
                    setFilters(prev => ({ ...prev, category: value || undefined }))
                  }
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue placeholder="All categories" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All categories</SelectItem>
                    <SelectItem value="work">Work</SelectItem>
                    <SelectItem value="personal">Personal</SelectItem>
                    <SelectItem value="research">Research</SelectItem>
                    <SelectItem value="coding">Coding</SelectItem>
                    <SelectItem value="creative">Creative</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Tags Filter */}
              {tags.length > 0 && (
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-2 block">
                    Tags
                  </label>
                  <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                    {tags.map(tag => (
                      <Badge
                        key={tag.id}
                        variant={selectedTags.includes(tag.name) ? "default" : "outline"}
                        className="text-xs h-5 px-2 cursor-pointer"
                        onClick={() => toggleTag(tag.name)}
                        style={{ backgroundColor: selectedTags.includes(tag.name) ? tag.color : undefined }}
                      >
                        <Tag className="h-3 w-3 mr-1" />
                        {tag.name}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Model Filter */}
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-2 block">
                  Model
                </label>
                <Select
                  value={filters.modelName || ""}
                  onValueChange={(value) => 
                    setFilters(prev => ({ ...prev, modelName: value || undefined }))
                  }
                >
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue placeholder="All models" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="">All models</SelectItem>
                    <SelectItem value="phi-3-mini">Phi-3 Mini</SelectItem>
                    <SelectItem value="llama-3-8b">Llama 3 8B</SelectItem>
                    <SelectItem value="mistral-7b">Mistral 7B</SelectItem>
                    <SelectItem value="gemma-2b">Gemma 2B</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Clear Filters Button */}
              <Button
                size="sm"
                variant="outline"
                onClick={clearFilters}
                className="w-full h-6 text-xs"
              >
                Clear All Filters
              </Button>
            </CollapsibleContent>
          </Collapsible>
        </div>
      </CardHeader>

      <CardContent className="flex-1 overflow-y-auto space-y-2 px-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-muted-foreground">Loading conversations...</div>
          </div>
        ) : conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <MessageSquare className="h-12 w-12 text-muted-foreground/50 mb-3" />
            <p className="text-sm text-muted-foreground">
              {searchQuery ? 'No conversations found' : 'No conversations yet'}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {searchQuery ? 'Try a different search term' : 'Start a new conversation to get started'}
            </p>
          </div>
        ) : (
          conversations.map((conversation) => (
            <div
              key={conversation.id}
              className={`group cursor-pointer rounded-lg border p-3 hover:bg-accent transition-colors ${
                currentConversation === conversation.id ? 'bg-accent border-primary' : ''
              } ${conversation.is_pinned ? 'border-yellow-300 bg-yellow-50/50' : ''} ${
                conversation.is_archived ? 'opacity-60' : ''
              }`}
              onClick={() => handleConversationClick(conversation)}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {conversation.is_pinned && (
                      <Pin className="h-3 w-3 text-yellow-600" />
                    )}
                    {conversation.is_archived && (
                      <Archive className="h-3 w-3 text-gray-500" />
                    )}
                    {conversation.priority > 0 && (
                      <Star className="h-3 w-3 text-orange-500" />
                    )}
                    {conversation.category && (
                      <Badge variant="outline" className="text-xs h-4 px-1">
                        {conversation.category}
                      </Badge>
                    )}
                  </div>

                  {editingId === conversation.id ? (
                    <Input
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onBlur={() => handleEditTitle(conversation.id, editTitle)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          handleEditTitle(conversation.id, editTitle)
                        } else if (e.key === 'Escape') {
                          setEditingId(null)
                          setEditTitle('')
                        }
                      }}
                      className="h-6 text-sm"
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <h4 className="font-medium text-sm truncate">
                      {conversation.title}
                    </h4>
                  )}

                  {conversation.description && (
                    <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                      {conversation.description}
                    </p>
                  )}
                  
                  <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatDate(conversation.updated_at)}
                    </span>
                    {conversation.message_count !== undefined && (
                      <span className="flex items-center gap-1">
                        <div className="flex">
                          <User className="h-3 w-3" />
                          <Bot className="h-3 w-3 -ml-1" />
                        </div>
                        {getMessageCountDisplay(conversation.message_count)}
                      </span>
                    )}
                    {conversation.avg_response_time > 0 && (
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {conversation.avg_response_time.toFixed(1)}s
                      </span>
                    )}
                  </div>
                  
                  <div className="flex items-center justify-between mt-1">
                    <div className="text-xs text-muted-foreground">
                      Model: {conversation.model_name}
                    </div>
                    {conversation.color && (
                      <div 
                        className="w-3 h-3 rounded-full border"
                        style={{ backgroundColor: conversation.color }}
                      />
                    )}
                  </div>

                  {/* Tags */}
                  {conversation.tags && conversation.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {conversation.tags.slice(0, 3).map(tag => (
                        <Badge
                          key={tag.id}
                          variant="secondary"
                          className="text-xs h-4 px-1"
                          style={{ backgroundColor: tag.color }}
                        >
                          {tag.name}
                        </Badge>
                      ))}
                      {conversation.tags.length > 3 && (
                        <Badge variant="outline" className="text-xs h-4 px-1">
                          +{conversation.tags.length - 3}
                        </Badge>
                      )}
                    </div>
                  )}
                </div>

                <div className="flex flex-col opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0"
                    onClick={(e) => handlePinConversation(conversation.id, e)}
                    title={conversation.is_pinned ? 'Unpin' : 'Pin'}
                  >
                    <Pin className={`h-3 w-3 ${conversation.is_pinned ? 'text-yellow-600' : ''}`} />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0"
                    onClick={(e) => handleArchiveConversation(conversation.id, e)}
                    title={conversation.is_archived ? 'Unarchive' : 'Archive'}
                  >
                    <Archive className={`h-3 w-3 ${conversation.is_archived ? 'text-gray-500' : ''}`} />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0"
                    onClick={(e) => startEditing(conversation, e)}
                    title="Edit title"
                  >
                    <Edit3 className="h-3 w-3" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                    onClick={(e) => handleDeleteConversation(conversation.id, e)}
                    title="Delete"
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  )
}