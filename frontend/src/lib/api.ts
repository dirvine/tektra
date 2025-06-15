/**
 * API client for Tektra backend communication
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ApiResponse<T = unknown> {
  data?: T
  error?: string
  status: number
}

// Type definitions for API responses
export interface Conversation {
  id: number
  title: string
  description?: string
  model_name: string
  category?: string
  is_pinned: boolean
  is_archived: boolean
  priority: number
  color?: string
  message_count: number
  total_tokens: number
  avg_response_time: number
  created_at: string
  updated_at: string
  last_message_at?: string
  tags: Tag[]
}

export interface Message {
  id: number
  conversation_id: number
  role: string
  content: string
  message_type: string
  is_important: boolean
  is_favorite: boolean
  tokens: number
  response_time: number
  created_at: string
  updated_at: string
}

export interface Tag {
  id: number
  name: string
  color?: string
  description?: string
  usage_count: number
  created_at: string
}

export interface UserPreferences {
  id: number
  user_id: number
  theme_mode: string
  sidebar_collapsed: boolean
  show_avatars: boolean
  enable_animations: boolean
  compact_mode: boolean
  auto_scroll: boolean
  show_timestamps: boolean
  show_typing_indicators: boolean
  dark_mode_schedule: Record<string, any>
  font_size: number
  preferred_model: string
  default_temperature: number
  default_max_tokens: number
  response_format: string
  enable_streaming: boolean
  voice_provider: string
  voice_id: string
  voice_speed: number
  voice_pitch: number
  auto_speech: boolean
  speech_recognition_language: string
  avatar_enabled: boolean
  avatar_id: string
  gesture_sensitivity: number
  expression_intensity: number
  privacy_mode: boolean
  data_retention_days: number
  share_analytics: boolean
  notification_level: string
  email_notifications: boolean
  push_notifications: boolean
  sound_notifications: boolean
  custom_css: string
  shortcuts: Record<string, any>
  created_at: string
  updated_at: string
}

export interface ModelSettings {
  id: number
  user_preferences_id: number
  model_name: string
  temperature: number
  max_tokens: number
  top_p: number
  frequency_penalty: number
  presence_penalty: number
  stop_sequences: string[]
  system_prompt?: string
  custom_parameters: Record<string, any>
  created_at: string
  updated_at: string
}

export interface ConversationTemplate {
  id: number
  user_id: number
  name: string
  description?: string
  system_prompt: string
  category?: string
  initial_messages: Array<{role: string, content: string}>
  model_settings: Record<string, any>
  tags: string[]
  is_public: boolean
  is_favorite: boolean
  usage_count: number
  last_used?: string
  created_at: string
  updated_at: string
}

export interface APIKey {
  id: number
  user_id: number
  provider: string
  key_name: string
  is_active: boolean
  usage_count: number
  usage_limit?: number
  last_used?: string
  created_at: string
  updated_at: string
}

export interface ModelInfo {
  name: string
  display_name: string
  description: string
  size_gb: number
  parameters: string
  context_length: number
  is_downloaded: boolean
  download_progress?: number
  supported_features: string[]
}

export interface SearchParams {
  query?: string
  tags?: string[]
  category?: string
  model_name?: string
  date_from?: string
  date_to?: string
  is_pinned?: boolean
  is_archived?: boolean
  min_messages?: number
  max_messages?: number
  sort_by?: string
  sort_order?: string
  limit?: number
  offset?: number
}

export interface MessageSearchParams {
  query: string
  conversation_id?: number
  role?: string
  message_type?: string
  date_from?: string
  date_to?: string
  is_favorite?: boolean
  is_important?: boolean
  limit?: number
  offset?: number
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseUrl}${endpoint}`
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      const data = await response.json()

      return {
        data,
        status: response.status,
      }
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : 'Unknown error',
        status: 500,
      }
    }
  }

  // AI endpoints
  async sendChatMessage(message: string, model: string = 'default') {
    return this.request('/api/v1/ai/chat', {
      method: 'POST',
      body: JSON.stringify({
        message,
        model,
        stream: false,
      }),
    })
  }

  async getModels() {
    return this.request('/api/v1/ai/models')
  }

  async loadModel(modelName: string) {
    return this.request(`/api/v1/ai/models/${modelName}/load`, {
      method: 'POST',
    })
  }

  async unloadModel(modelName: string) {
    return this.request(`/api/v1/ai/models/${modelName}`, {
      method: 'DELETE',
    })
  }

  // Audio endpoints
  async startRecording() {
    return this.request('/api/v1/audio/record/start', {
      method: 'POST',
    })
  }

  async stopRecording() {
    return this.request('/api/v1/audio/record/stop', {
      method: 'POST',
    })
  }

  async transcribeAudio(audioFile: File) {
    const formData = new FormData()
    formData.append('audio_file', audioFile)
    
    return this.request('/api/v1/audio/transcribe', {
      method: 'POST',
      body: formData,
      headers: {}, // Remove Content-Type to let browser set it for FormData
    })
  }

  async synthesizeSpeech(text: string, voice: string = 'default') {
    return this.request('/api/v1/audio/synthesize', {
      method: 'POST',
      body: JSON.stringify({
        text,
        voice,
        speed: 1.0,
        language: 'en',
      }),
    })
  }

  async getVoices() {
    return this.request('/api/v1/audio/voices')
  }

  // Avatar endpoints
  async getAvatarStatus() {
    return this.request('/api/v1/avatar/status')
  }

  async setAvatarExpression(expression: string, intensity: number = 1.0) {
    return this.request('/api/v1/avatar/expression', {
      method: 'POST',
      body: JSON.stringify({
        expression,
        intensity,
        duration: 2.0,
      }),
    })
  }

  async triggerAvatarGesture(gesture: string, speed: number = 1.0) {
    return this.request('/api/v1/avatar/gesture', {
      method: 'POST',
      body: JSON.stringify({
        gesture,
        speed,
        repeat: 1,
      }),
    })
  }

  async makeAvatarSpeak(text: string, expression: string = 'neutral') {
    return this.request('/api/v1/avatar/speak', {
      method: 'POST',
      body: JSON.stringify({
        text,
        lip_sync: true,
        expression,
      }),
    })
  }

  // Camera endpoints
  async getCameraStatus() {
    return this.request('/api/v1/camera/status')
  }

  async captureImage() {
    return this.request('/api/v1/camera/capture', {
      method: 'POST',
    })
  }

  async analyzeImage(imageFile: File) {
    const formData = new FormData()
    formData.append('image_file', imageFile)
    
    return this.request('/api/v1/camera/analyze', {
      method: 'POST',
      body: formData,
      headers: {},
    })
  }

  // Robot endpoints
  async getRobots() {
    return this.request('/api/v1/robots')
  }

  async sendRobotCommand(robotId: string, action: string, parameters: Record<string, unknown> = {}) {
    return this.request(`/api/v1/robots/${robotId}/command`, {
      method: 'POST',
      body: JSON.stringify({
        action,
        parameters,
        priority: 'normal',
      }),
    })
  }

  async getRobotStatus(robotId: string) {
    return this.request(`/api/v1/robots/${robotId}/status`)
  }

  async emergencyStop(robotId: string) {
    return this.request(`/api/v1/robots/${robotId}/emergency`, {
      method: 'POST',
    })
  }

  // Enhanced Conversation endpoints
  async getConversations(limit = 50, offset = 0) {
    return this.request(`/api/v1/conversations?limit=${limit}&offset=${offset}`)
  }

  async createConversation(title?: string, modelName = 'phi-3-mini') {
    return this.request('/api/v1/conversations', {
      method: 'POST',
      body: JSON.stringify({
        title,
        model_name: modelName,
      }),
    })
  }

  async getConversation(conversationId: number) {
    return this.request(`/api/v1/conversations/${conversationId}`)
  }

  async addMessageToConversation(conversationId: number, content: string, role = 'user', messageType = 'text') {
    return this.request(`/api/v1/conversations/${conversationId}/messages`, {
      method: 'POST',
      body: JSON.stringify({
        content,
        role,
        message_type: messageType,
      }),
    })
  }

  async getConversationMessages(conversationId: number, limit?: number, includeSystem = true) {
    const params = new URLSearchParams()
    if (limit) params.set('limit', limit.toString())
    params.set('include_system', includeSystem.toString())
    
    return this.request(`/api/v1/conversations/${conversationId}/messages?${params}`)
  }

  async updateConversationTitle(conversationId: number, title: string) {
    return this.request(`/api/v1/conversations/${conversationId}/title`, {
      method: 'PUT',
      body: JSON.stringify({ title }),
    })
  }

  async deleteConversation(conversationId: number) {
    return this.request(`/api/v1/conversations/${conversationId}`, {
      method: 'DELETE',
    })
  }

  // Advanced Search & Organization
  async searchConversations(searchParams: {
    query?: string
    tags?: string[]
    category?: string
    model_name?: string
    date_from?: string
    date_to?: string
    is_pinned?: boolean
    is_archived?: boolean
    min_messages?: number
    max_messages?: number
    sort_by?: string
    sort_order?: string
    limit?: number
    offset?: number
  }) {
    return this.request('/api/v1/conversations/search', {
      method: 'POST',
      body: JSON.stringify(searchParams),
    })
  }

  async searchMessages(searchParams: {
    query: string
    conversation_id?: number
    role?: string
    message_type?: string
    date_from?: string
    date_to?: string
    is_favorite?: boolean
    is_important?: boolean
    limit?: number
    offset?: number
  }) {
    return this.request('/api/v1/conversations/messages/search', {
      method: 'POST',
      body: JSON.stringify(searchParams),
    })
  }

  async updateConversationMetadata(conversationId: number, metadata: {
    title?: string
    description?: string
    category?: string
    is_pinned?: boolean
    is_archived?: boolean
    priority?: number
    color?: string
  }) {
    return this.request(`/api/v1/conversations/${conversationId}/metadata`, {
      method: 'PUT',
      body: JSON.stringify(metadata),
    })
  }

  async updateMessageMetadata(messageId: number, metadata: {
    is_important?: boolean
    is_favorite?: boolean
  }) {
    return this.request(`/api/v1/conversations/messages/${messageId}/metadata`, {
      method: 'PUT',
      body: JSON.stringify(metadata),
    })
  }

  async bulkArchiveConversations(conversationIds: number[]) {
    return this.request('/api/v1/conversations/bulk/archive', {
      method: 'POST',
      body: JSON.stringify(conversationIds),
    })
  }

  async bulkDeleteConversations(conversationIds: number[], permanent = false) {
    return this.request('/api/v1/conversations/bulk/delete', {
      method: 'POST',
      body: JSON.stringify({ conversation_ids: conversationIds, permanent }),
    })
  }

  async exportConversations(exportParams: {
    conversation_ids?: number[]
    format?: string
    include_metadata?: boolean
  }) {
    return this.request('/api/v1/conversations/export', {
      method: 'POST',
      body: JSON.stringify(exportParams),
    })
  }

  async getConversationAnalytics(conversationId?: number, days = 30) {
    const params = new URLSearchParams()
    if (conversationId) params.set('conversation_id', conversationId.toString())
    params.set('days', days.toString())
    
    return this.request(`/api/v1/conversations/analytics?${params}`)
  }

  // Tag Management
  async getTags() {
    return this.request('/api/v1/conversations/tags')
  }

  async createTag(tagData: {
    name: string
    color?: string
    description?: string
  }) {
    return this.request('/api/v1/conversations/tags', {
      method: 'POST',
      body: JSON.stringify(tagData),
    })
  }

  async assignTagsToConversation(conversationId: number, tagNames: string[]) {
    return this.request('/api/v1/conversations/tags/assign', {
      method: 'POST',
      body: JSON.stringify({
        conversation_id: conversationId,
        tag_names: tagNames,
      }),
    })
  }

  async deleteTag(tagId: number) {
    return this.request(`/api/v1/conversations/tags/${tagId}`, {
      method: 'DELETE',
    })
  }

  // User Preferences endpoints
  async getUserPreferences(userId: number) {
    return this.request(`/api/v1/users/${userId}/preferences`)
  }

  async updateUserPreferences(userId: number, preferences: {
    theme_mode?: string
    sidebar_collapsed?: boolean
    show_avatars?: boolean
    enable_animations?: boolean
    compact_mode?: boolean
    auto_scroll?: boolean
    show_timestamps?: boolean
    show_typing_indicators?: boolean
    dark_mode_schedule?: Record<string, any>
    font_size?: number
    preferred_model?: string
    default_temperature?: number
    default_max_tokens?: number
    response_format?: string
    enable_streaming?: boolean
    voice_provider?: string
    voice_id?: string
    voice_speed?: number
    voice_pitch?: number
    auto_speech?: boolean
    speech_recognition_language?: string
    avatar_enabled?: boolean
    avatar_id?: string
    gesture_sensitivity?: number
    expression_intensity?: number
    privacy_mode?: boolean
    data_retention_days?: number
    share_analytics?: boolean
    notification_level?: string
    email_notifications?: boolean
    push_notifications?: boolean
    sound_notifications?: boolean
    custom_css?: string
    shortcuts?: Record<string, any>
  }) {
    return this.request(`/api/v1/users/${userId}/preferences`, {
      method: 'PUT',
      body: JSON.stringify(preferences),
    })
  }

  async resetUserPreferences(userId: number) {
    return this.request(`/api/v1/users/${userId}/preferences/reset`, {
      method: 'POST',
    })
  }

  // Model Settings endpoints
  async getModelSettings(userId: number, modelName?: string) {
    const params = modelName ? `?model_name=${modelName}` : ''
    return this.request(`/api/v1/users/${userId}/model-settings${params}`)
  }

  async updateModelSettings(userId: number, modelName: string, settings: {
    temperature?: number
    max_tokens?: number
    top_p?: number
    frequency_penalty?: number
    presence_penalty?: number
    stop_sequences?: string[]
    system_prompt?: string
    custom_parameters?: Record<string, any>
  }) {
    return this.request(`/api/v1/users/${userId}/model-settings/${modelName}`, {
      method: 'PUT',
      body: JSON.stringify(settings),
    })
  }

  async deleteModelSettings(userId: number, modelName: string) {
    return this.request(`/api/v1/users/${userId}/model-settings/${modelName}`, {
      method: 'DELETE',
    })
  }

  // Conversation Templates endpoints
  async getConversationTemplates(userId: number, category?: string, includePublic = true) {
    const params = new URLSearchParams()
    if (category) params.set('category', category)
    params.set('include_public', includePublic.toString())
    
    return this.request(`/api/v1/users/${userId}/templates?${params}`)
  }

  async createConversationTemplate(userId: number, template: {
    name: string
    description?: string
    system_prompt: string
    category?: string
    initial_messages?: Array<{role: string, content: string}>
    model_settings?: Record<string, any>
    tags?: string[]
    is_public?: boolean
    is_favorite?: boolean
  }) {
    return this.request(`/api/v1/users/${userId}/templates`, {
      method: 'POST',
      body: JSON.stringify(template),
    })
  }

  async updateConversationTemplate(userId: number, templateId: number, template: {
    name?: string
    description?: string
    system_prompt?: string
    category?: string
    initial_messages?: Array<{role: string, content: string}>
    model_settings?: Record<string, any>
    tags?: string[]
    is_public?: boolean
    is_favorite?: boolean
  }) {
    return this.request(`/api/v1/users/${userId}/templates/${templateId}`, {
      method: 'PUT',
      body: JSON.stringify(template),
    })
  }

  async deleteConversationTemplate(userId: number, templateId: number) {
    return this.request(`/api/v1/users/${userId}/templates/${templateId}`, {
      method: 'DELETE',
    })
  }

  async useConversationTemplate(userId: number, templateId: number) {
    return this.request(`/api/v1/users/${userId}/templates/${templateId}/use`, {
      method: 'POST',
    })
  }

  // API Key Management endpoints
  async getAPIKeys(userId: number) {
    return this.request(`/api/v1/users/${userId}/api-keys`)
  }

  async storeAPIKey(userId: number, keyData: {
    provider: string
    key_name: string
    api_key: string
    usage_limit?: number
  }) {
    return this.request(`/api/v1/users/${userId}/api-keys`, {
      method: 'POST',
      body: JSON.stringify(keyData),
    })
  }

  async deleteAPIKey(userId: number, keyId: number) {
    return this.request(`/api/v1/users/${userId}/api-keys/${keyId}`, {
      method: 'DELETE',
    })
  }

  // Model Management endpoints
  async getAvailableModels() {
    return this.request('/api/v1/models/available')
  }

  async getModelInfo(modelName: string) {
    return this.request(`/api/v1/models/${modelName}/info`)
  }

  async downloadModel(modelName: string) {
    return this.request(`/api/v1/models/${modelName}/download`, {
      method: 'POST',
    })
  }

  async getDownloadProgress(modelName: string) {
    return this.request(`/api/v1/models/${modelName}/download/progress`)
  }

  async deleteModel(modelName: string) {
    return this.request(`/api/v1/models/${modelName}`, {
      method: 'DELETE',
    })
  }

  async getCacheInfo() {
    return this.request('/api/v1/models/cache/info')
  }

  async clearModelCache() {
    return this.request('/api/v1/models/cache/clear', {
      method: 'POST',
    })
  }

  // Health check
  async healthCheck() {
    return this.request('/health')
  }

  // WebSocket connection helper
  createWebSocket(endpoint: string): WebSocket {
    const wsUrl = this.baseUrl.replace('http://', 'ws://').replace('https://', 'wss://')
    return new WebSocket(`${wsUrl}${endpoint}`)
  }

  // Streaming chat helper
  async createStreamingChat(conversationId: number, message: string, modelName = 'phi-3-mini'): Promise<WebSocket> {
    const ws = this.createWebSocket(`/ws/chat/${conversationId}`)
    
    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'message',
        content: message,
        model: modelName,
      }))
    }
    
    return ws
  }
}

export const api = new ApiClient()
export default api