import { ChatMessage } from '../store';

const CHAT_STORAGE_KEY = 'tektra_chat_history';
const MAX_STORED_MESSAGES = 100;

export const saveChatHistory = (messages: ChatMessage[]) => {
  try {
    // Only save the most recent messages to avoid storage bloat
    const messagesToSave = messages.slice(-MAX_STORED_MESSAGES);
    
    // Convert Date objects to ISO strings for storage
    const serializedMessages = messagesToSave.map(msg => ({
      ...msg,
      timestamp: msg.timestamp.toISOString()
    }));
    
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(serializedMessages));
  } catch (error) {
    console.error('Failed to save chat history:', error);
  }
};

export const loadChatHistory = (): ChatMessage[] => {
  try {
    const stored = localStorage.getItem(CHAT_STORAGE_KEY);
    if (!stored) return [];
    
    const parsed = JSON.parse(stored);
    
    // Convert ISO strings back to Date objects
    return parsed.map((msg: any) => ({
      ...msg,
      timestamp: new Date(msg.timestamp)
    }));
  } catch (error) {
    console.error('Failed to load chat history:', error);
    return [];
  }
};

export const clearChatHistory = () => {
  try {
    localStorage.removeItem(CHAT_STORAGE_KEY);
  } catch (error) {
    console.error('Failed to clear chat history:', error);
  }
};