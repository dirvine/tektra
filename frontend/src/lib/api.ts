/**
 * API client for Tektra backend communication
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ApiResponse<T = unknown> {
  data?: T
  error?: string
  status: number
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

  // Conversation endpoints
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

  async searchConversations(query: string, limit = 20) {
    return this.request(`/api/v1/conversations/search/${encodeURIComponent(query)}?limit=${limit}`)
  }

  async getConversationStats(conversationId: number) {
    return this.request(`/api/v1/conversations/${conversationId}/stats`)
  }

  async getConversationContext(conversationId: number, maxMessages = 50) {
    return this.request(`/api/v1/conversations/${conversationId}/context?max_messages=${maxMessages}`)
  }

  // Health check
  async healthCheck() {
    return this.request('/health')
  }
}

export const api = new ApiClient()
export default api