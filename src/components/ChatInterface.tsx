import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';
import { ChatMessage } from '../types';
import { formatResponse } from '../utils/formatting';

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  disabled: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  disabled
}) => {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !disabled) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const getMessageIcon = (role: ChatMessage['role']) => {
    switch (role) {
      case 'assistant':
        return <Bot className="w-5 h-5 text-blue-500" />;
      case 'user':
        return <User className="w-5 h-5 text-green-500" />;
      case 'system':
        return <div className="w-2 h-2 bg-gray-400 rounded-full" />;
      default:
        return null;
    }
  };

  const getMessageClassName = (role: ChatMessage['role']) => {
    const baseClasses = 'message';
    switch (role) {
      case 'assistant':
        return `${baseClasses} assistant-message`;
      case 'user':
        return `${baseClasses} user-message`;
      case 'system':
        return `${baseClasses} system-message`;
      default:
        return baseClasses;
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>Conversation</h2>
        <div className="chat-status">
          <div className={`status-indicator ${disabled ? 'offline' : 'online'}`} />
          <span>{disabled ? 'Model Loading...' : 'Ready'}</span>
        </div>
      </div>

      <div className="messages-container">
        <div className="messages-list">
          {messages.map((message) => (
            <div key={message.id} className={getMessageClassName(message.role)}>
              <div className="message-icon">
                {getMessageIcon(message.role)}
              </div>
              
              <div className="message-content">
                {message.role === 'assistant' ? (
                  <div dangerouslySetInnerHTML={{ __html: formatResponse(message.content) }} />
                ) : (
                  <div className="message-text">{message.content}</div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <form onSubmit={handleSubmit} className="message-input-form">
        <div className="input-container">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={disabled ? "Model is loading..." : "Type your message..."}
            className="message-input"
            disabled={disabled}
          />
          <button
            type="submit"
            disabled={disabled || !inputValue.trim()}
            className="send-button"
            title="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;