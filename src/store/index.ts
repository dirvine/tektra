import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// Types
export interface ModelStatus {
  isLoaded: boolean;
  modelName: string;
  backend: string;
  isLoading: boolean;
  whisperReady: boolean;
  cameraEnabled: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    hasImage?: boolean;
    hasAudio?: boolean;
    tokenCount?: number;
    processingTime?: number;
  };
}

export interface AvatarState {
  isVisible: boolean;
  isMinimized: boolean;
  expression: 'neutral' | 'happy' | 'thinking' | 'surprised' | 'concerned' | 'excited' | 'friendly';
  isSpeaking: boolean;
  isListening: boolean;
  voiceSettings: {
    voice: string;
    pitch: number;
    speed: number;
    emotionalTone: boolean;
  };
  appearance: {
    style: 'realistic' | 'stylized' | 'minimal';
    age: number;
    clothing: 'business' | 'casual' | 'lab_coat';
  };
  animation: {
    idleAnimations: boolean;
    lipSyncSensitivity: number;
    eyeTracking: boolean;
  };
}

export interface UIState {
  leftSidebarCollapsed: boolean;
  rightSidebarVisible: boolean;
  activeTab: 'analytics' | 'session' | 'files' | 'knowledge' | 'tasks';
  theme: 'dark' | 'light';
  currentProject: string;
  notifications: Array<{
    id: string;
    type: 'info' | 'success' | 'warning' | 'error';
    message: string;
    timestamp: Date;
  }>;
}

export interface SessionState {
  duration: number;
  tokenUsage: number;
  costEstimate: number;
  conversationSummary: string;
  engagementMetrics: {
    eyeContactTime: number;
    expressionChanges: number;
    userSentiment: 'positive' | 'neutral' | 'negative';
  };
}

// Store interface
interface TektraStore {
  // State
  modelStatus: ModelStatus;
  messages: ChatMessage[];
  avatarState: AvatarState;
  uiState: UIState;
  sessionState: SessionState;
  isRecording: boolean;
  
  // Actions
  setModelStatus: (status: Partial<ModelStatus>) => void;
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  updateLastMessage: (content: string) => void;
  clearMessages: () => void;
  setAvatarState: (state: Partial<AvatarState>) => void;
  setAvatarExpression: (expression: AvatarState['expression']) => void;
  setAvatarSpeaking: (speaking: boolean) => void;
  setAvatarListening: (listening: boolean) => void;
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setActiveTab: (tab: UIState['activeTab']) => void;
  addNotification: (notification: Omit<UIState['notifications'][0], 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  setRecording: (recording: boolean) => void;
  updateSessionMetrics: (metrics: Partial<SessionState>) => void;
}

// Default states
const defaultModelStatus: ModelStatus = {
  isLoaded: false,
  modelName: 'gemma2:2b',
  backend: 'Ollama',
  isLoading: false,
  whisperReady: false,
  cameraEnabled: false,
};

const defaultAvatarState: AvatarState = {
  isVisible: true,
  isMinimized: false,
  expression: 'neutral',
  isSpeaking: false,
  isListening: false,
  voiceSettings: {
    voice: 'default',
    pitch: 1.0,
    speed: 1.0,
    emotionalTone: true,
  },
  appearance: {
    style: 'realistic',
    age: 30,
    clothing: 'business',
  },
  animation: {
    idleAnimations: true,
    lipSyncSensitivity: 0.8,
    eyeTracking: true,
  },
};

const defaultUIState: UIState = {
  leftSidebarCollapsed: false,
  rightSidebarVisible: true,
  activeTab: 'analytics',
  theme: 'dark',
  currentProject: 'My Project',
  notifications: [],
};

const defaultSessionState: SessionState = {
  duration: 0,
  tokenUsage: 0,
  costEstimate: 0,
  conversationSummary: '',
  engagementMetrics: {
    eyeContactTime: 0,
    expressionChanges: 0,
    userSentiment: 'neutral',
  },
};

// Create store
export const useTektraStore = create<TektraStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      modelStatus: defaultModelStatus,
      messages: [],
      avatarState: defaultAvatarState,
      uiState: defaultUIState,
      sessionState: defaultSessionState,
      isRecording: false,

      // Actions
      setModelStatus: (status) =>
        set((state) => ({
          modelStatus: { ...state.modelStatus, ...status },
        })),

      addMessage: (message) =>
        set((state) => ({
          messages: [
            ...state.messages,
            {
              ...message,
              id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
              timestamp: new Date(),
            },
          ],
        })),

      updateLastMessage: (content) =>
        set((state) => {
          const messages = [...state.messages];
          if (messages.length > 0) {
            messages[messages.length - 1] = {
              ...messages[messages.length - 1],
              content,
            };
          }
          return { messages };
        }),

      clearMessages: () => set({ messages: [] }),

      setAvatarState: (state) =>
        set((current) => ({
          avatarState: { ...current.avatarState, ...state },
        })),

      setAvatarExpression: (expression) =>
        set((state) => ({
          avatarState: { ...state.avatarState, expression },
        })),

      setAvatarSpeaking: (speaking) =>
        set((state) => ({
          avatarState: { ...state.avatarState, isSpeaking: speaking },
        })),

      setAvatarListening: (listening) =>
        set((state) => ({
          avatarState: { ...state.avatarState, isListening: listening },
        })),

      toggleLeftSidebar: () =>
        set((state) => ({
          uiState: {
            ...state.uiState,
            leftSidebarCollapsed: !state.uiState.leftSidebarCollapsed,
          },
        })),

      toggleRightSidebar: () =>
        set((state) => ({
          uiState: {
            ...state.uiState,
            rightSidebarVisible: !state.uiState.rightSidebarVisible,
          },
        })),

      setActiveTab: (tab) =>
        set((state) => ({
          uiState: { ...state.uiState, activeTab: tab },
        })),

      addNotification: (notification) =>
        set((state) => ({
          uiState: {
            ...state.uiState,
            notifications: [
              ...state.uiState.notifications,
              {
                ...notification,
                id: Date.now().toString(),
                timestamp: new Date(),
              },
            ],
          },
        })),

      removeNotification: (id) =>
        set((state) => ({
          uiState: {
            ...state.uiState,
            notifications: state.uiState.notifications.filter((n) => n.id !== id),
          },
        })),

      setRecording: (recording) => set({ isRecording: recording }),

      updateSessionMetrics: (metrics) =>
        set((state) => ({
          sessionState: { ...state.sessionState, ...metrics },
        })),
    }),
    {
      name: 'tektra-store',
    }
  )
);

// Selectors
export const useModelStatus = () => useTektraStore((state) => state.modelStatus);
export const useMessages = () => useTektraStore((state) => state.messages);
export const useAvatarState = () => useTektraStore((state) => state.avatarState);
export const useUIState = () => useTektraStore((state) => state.uiState);
export const useSessionState = () => useTektraStore((state) => state.sessionState);
export const useIsRecording = () => useTektraStore((state) => state.isRecording);