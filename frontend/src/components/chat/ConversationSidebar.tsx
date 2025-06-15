"use client"

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { 
  Plus, 
  Search, 
  MessageSquare, 
  Trash2, 
  Edit3, 
  Clock,
  User,
  Bot
} from 'lucide-react'
import { api } from '@/lib/api'

export interface Conversation {
  id: number
  title: string
  model_name: string
  created_at: string
  updated_at: string
  message_count?: number
}

export interface ConversationWithMessages extends Conversation {
  messages: Array<{
    id: number
    conversation_id: number
    role: string
    content: string
    message_type: string
    created_at: string
    metadata: Record<string, any>
  }>
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

  useEffect(() => {
    loadConversations()
  }, [])

  const loadConversations = async () => {
    setIsLoading(true)
    try {
      const response = await api.getConversations()
      if (response.data) {
        setConversations(response.data as Conversation[])
      }
    } catch (error) {
      console.error('Failed to load conversations:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSearch = async (query: string) => {
    setSearchQuery(query)
    if (!query.trim()) {
      loadConversations()
      return
    }

    try {
      const response = await api.searchConversations(query)
      if (response.data) {
        setConversations(response.data as Conversation[])
      }
    } catch (error) {
      console.error('Failed to search conversations:', error)
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
          <Button
            size="sm"
            onClick={onNewConversation}
            className="h-8 w-8 p-0"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </CardTitle>
        
        <div className="relative">
          <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="pl-9"
          />
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
              }`}
              onClick={() => handleConversationClick(conversation)}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
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
                  </div>
                  
                  <div className="text-xs text-muted-foreground mt-1">
                    Model: {conversation.model_name}
                  </div>
                </div>

                <div className="flex opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0"
                    onClick={(e) => startEditing(conversation, e)}
                  >
                    <Edit3 className="h-3 w-3" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                    onClick={(e) => handleDeleteConversation(conversation.id, e)}
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