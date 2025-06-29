import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Bot,
  User,
  Copy,
  Play,
  Code,
  FileText,
  Image,
  Mic,
  MicOff,
  Paperclip,
  MoreVertical,
  ChevronDown,
  ChevronUp,
  Eye,
  EyeOff,
} from 'lucide-react';
import { useTektraStore } from '../store';
import { formatResponse } from '../utils/formatting';

interface MessageProps {
  message: any;
  isLatest: boolean;
}

const Message: React.FC<MessageProps> = ({ message, isLatest }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [showActions, setShowActions] = useState(false);
  
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // Could add toast notification here
  };

  const isLongMessage = message.content.length > 500;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`
        group flex space-x-3 p-4 rounded-card transition-colors
        ${message.role === 'assistant' 
          ? 'bg-secondary-bg border border-border-primary' 
          : 'bg-surface/50'
        }
        ${message.role === 'system' ? 'bg-warning/10 border border-warning/20' : ''}
      `}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* Avatar */}
      <div className={`
        w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0
        ${message.role === 'assistant' 
          ? 'bg-gradient-to-br from-accent to-accent-light' 
          : message.role === 'user'
          ? 'bg-gradient-to-br from-success to-green-600'
          : 'bg-gradient-to-br from-warning to-orange-600'
        }
      `}>
        {message.role === 'assistant' ? (
          <Bot className="w-4 h-4 text-white" />
        ) : message.role === 'user' ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <div className="w-2 h-2 bg-white rounded-full" />
        )}
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        {/* Message Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-text-primary capitalize">
              {message.role}
            </span>
            <span className="text-xs text-text-tertiary">
              {new Date(message.timestamp).toLocaleTimeString()}
            </span>
            {message.metadata?.tokenCount && (
              <span className="text-xs text-text-tertiary font-mono">
                {message.metadata.tokenCount} tokens
              </span>
            )}
          </div>
          
          {/* Actions */}
          <AnimatePresence>
            {showActions && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center space-x-1"
              >
                <button
                  onClick={() => copyToClipboard(message.content)}
                  className="p-1.5 rounded-button hover:bg-surface-hover transition-colors"
                  title="Copy message"
                >
                  <Copy className="w-3 h-3 text-text-secondary" />
                </button>
                
                {isLongMessage && (
                  <button
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    className="p-1.5 rounded-button hover:bg-surface-hover transition-colors"
                    title={isCollapsed ? "Expand" : "Collapse"}
                  >
                    {isCollapsed ? (
                      <ChevronDown className="w-3 h-3 text-text-secondary" />
                    ) : (
                      <ChevronUp className="w-3 h-3 text-text-secondary" />
                    )}
                  </button>
                )}
                
                <button className="p-1.5 rounded-button hover:bg-surface-hover transition-colors">
                  <MoreVertical className="w-3 h-3 text-text-secondary" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Message Body */}
        <div className={`
          text-text-primary transition-all duration-300
          ${isCollapsed ? 'max-h-20 overflow-hidden' : ''}
        `}>
          {message.role === 'assistant' ? (
            <div 
              dangerouslySetInnerHTML={{ 
                __html: formatResponse(
                  isCollapsed && isLongMessage 
                    ? message.content.substring(0, 200) + '...' 
                    : message.content
                ) 
              }} 
              className="prose prose-invert prose-sm max-w-none"
            />
          ) : (
            <div className="whitespace-pre-wrap">
              {isCollapsed && isLongMessage 
                ? message.content.substring(0, 200) + '...' 
                : message.content}
            </div>
          )}
        </div>

        {/* Attachments */}
        {message.metadata?.hasImage && (
          <div className="mt-3 flex items-center space-x-2 text-sm text-text-secondary">
            <Image className="w-4 h-4" />
            <span>Image attached</span>
          </div>
        )}
        
        {message.metadata?.hasAudio && (
          <div className="mt-3 flex items-center space-x-2 text-sm text-text-secondary">
            <Mic className="w-4 h-4" />
            <span>Audio attached</span>
          </div>
        )}

        {/* Code blocks with run button */}
        {message.content.includes('```') && message.role === 'assistant' && (
          <button className="mt-2 inline-flex items-center space-x-2 px-3 py-1.5 bg-accent text-white text-sm rounded-button hover:bg-accent-hover transition-colors">
            <Play className="w-3 h-3" />
            <span>Run Code</span>
          </button>
        )}

        {/* Processing time */}
        {message.metadata?.processingTime && (
          <div className="mt-2 text-xs text-text-tertiary">
            Processed in {message.metadata.processingTime}ms
          </div>
        )}
      </div>
    </motion.div>
  );
};

// Project Context Bar
const ProjectContextBar: React.FC = () => {
  const { uiState } = useTektraStore();
  
  return (
    <div className="border-b border-border-primary bg-surface/30 px-6 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <h3 className="font-medium text-text-primary">{uiState.currentProject}</h3>
            <button className="text-text-tertiary hover:text-text-secondary">
              <Code className="w-4 h-4" />
            </button>
          </div>
          
          <div className="flex space-x-2">
            <span className="px-2 py-1 text-xs bg-accent/20 text-accent rounded-full">
              Development
            </span>
            <span className="px-2 py-1 text-xs bg-success/20 text-success rounded-full">
              Active
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <button className="flex items-center space-x-2 px-3 py-1.5 text-sm text-text-secondary hover:text-text-primary transition-colors">
            <FileText className="w-4 h-4" />
            <span>Files</span>
          </button>
          <button className="flex items-center space-x-2 px-3 py-1.5 text-sm text-text-secondary hover:text-text-primary transition-colors">
            <Eye className="w-4 h-4" />
            <span>Docs</span>
          </button>
          <div className="w-px h-4 bg-border-primary" />
          <div className="flex items-center space-x-2">
            <div className="w-16 h-1.5 bg-surface rounded-full overflow-hidden">
              <div className="w-3/4 h-full bg-accent" />
            </div>
            <span className="text-xs text-text-tertiary">75%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Enhanced Input Area
const EnhancedInputArea: React.FC<{
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  isRecording: boolean;
  onToggleRecording: () => void;
}> = ({ value, onChange, onSubmit, disabled, isRecording, onToggleRecording }) => {
  const [showPreview, setShowPreview] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setAttachments(Array.from(e.target.files));
    }
  };

  return (
    <div className="border-t border-border-primary bg-secondary-bg p-4">
      {/* Attachments Preview */}
      {attachments.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {attachments.map((file, index) => (
            <div key={index} className="flex items-center space-x-2 px-3 py-1.5 bg-surface rounded-card border border-border-primary">
              <FileText className="w-4 h-4 text-text-secondary" />
              <span className="text-sm text-text-primary">{file.name}</span>
              <button 
                onClick={() => setAttachments(attachments.filter((_, i) => i !== index))}
                className="text-text-tertiary hover:text-error transition-colors"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex items-end space-x-3">
        {/* Input Area */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={disabled ? "Model is loading..." : "Type your message... (Shift+Enter for new line)"}
            disabled={disabled}
            className={`
              w-full min-h-[44px] max-h-32 px-4 py-3 pr-12
              bg-surface border border-border-primary rounded-card
              text-text-primary placeholder-text-tertiary
              resize-none outline-none
              focus:border-accent transition-colors
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            rows={1}
          />
          
          {/* Character Count */}
          <div className="absolute bottom-2 right-3 text-xs text-text-tertiary">
            {value.length}/2000
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2">
          {/* File Upload */}
          <label className="p-3 rounded-button hover:bg-surface-hover transition-colors cursor-pointer">
            <Paperclip className="w-4 h-4 text-text-secondary" />
            <input
              type="file"
              multiple
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>

          {/* Voice Recording */}
          <button
            onClick={onToggleRecording}
            className={`
              p-3 rounded-button transition-colors
              ${isRecording 
                ? 'bg-error text-white animate-pulse' 
                : 'hover:bg-surface-hover text-text-secondary'
              }
            `}
            title={isRecording ? 'Stop recording' : 'Start recording'}
          >
            {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
          </button>

          {/* Markdown Preview Toggle */}
          <button
            onClick={() => setShowPreview(!showPreview)}
            className={`
              p-3 rounded-button transition-colors
              ${showPreview ? 'bg-accent text-white' : 'hover:bg-surface-hover text-text-secondary'}
            `}
            title="Toggle markdown preview"
          >
            {showPreview ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>

          {/* Send Button */}
          <button
            onClick={onSubmit}
            disabled={disabled || !value.trim()}
            className={`
              p-3 rounded-button transition-colors
              ${disabled || !value.trim()
                ? 'bg-surface text-text-tertiary cursor-not-allowed'
                : 'bg-accent text-white hover:bg-accent-hover'
              }
            `}
            title="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Markdown Preview */}
      {showPreview && value && (
        <div className="mt-3 p-3 bg-surface border border-border-primary rounded-card">
          <div className="text-xs text-text-secondary mb-2">Preview:</div>
          <div 
            className="prose prose-invert prose-sm max-w-none"
            dangerouslySetInnerHTML={{ __html: formatResponse(value) }}
          />
        </div>
      )}

      {/* Input Hints */}
      <div className="mt-2 flex items-center justify-between text-xs text-text-tertiary">
        <div className="flex items-center space-x-4">
          <span>Shift+Enter for new line</span>
          <span>Supports Markdown</span>
        </div>
        <div className="flex items-center space-x-2">
          <span>Push-to-talk: Space</span>
        </div>
      </div>
    </div>
  );
};

// Main Enhanced Chat Interface
interface EnhancedChatInterfaceProps {
  className?: string;
}

const EnhancedChatInterface: React.FC<EnhancedChatInterfaceProps> = ({ className = '' }) => {
  const { messages, modelStatus, isRecording, addMessage, setRecording } = useTektraStore();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !modelStatus.isLoaded) return;

    const userMessage = {
      role: 'user' as const,
      content: inputValue.trim(),
    };

    addMessage(userMessage);
    setInputValue('');

    // TODO: Integrate with backend
    // For now, simulate typing indicator
    setTimeout(() => {
      addMessage({
        role: 'assistant',
        content: 'I received your message and I\'m processing it...',
      });
    }, 1000);
  };

  const handleToggleRecording = () => {
    setRecording(!isRecording);
    // TODO: Integrate with backend audio recording
  };

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Project Context Bar */}
      <ProjectContextBar />

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4">
        <AnimatePresence>
          {messages.map((message, index) => (
            <Message
              key={message.id}
              message={message}
              isLatest={index === messages.length - 1}
            />
          ))}
        </AnimatePresence>
        
        {/* Empty State */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="w-16 h-16 bg-gradient-to-br from-accent to-accent-light rounded-2xl flex items-center justify-center mb-4">
              <Bot className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-lg font-semibold text-text-primary mb-2">
              Welcome to Tektra
            </h3>
            <p className="text-text-secondary max-w-md">
              Start a conversation with your AI assistant. You can type, speak, or share images to get started.
            </p>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <EnhancedInputArea
        value={inputValue}
        onChange={setInputValue}
        onSubmit={handleSendMessage}
        disabled={!modelStatus.isLoaded}
        isRecording={isRecording}
        onToggleRecording={handleToggleRecording}
      />
    </div>
  );
};

export default EnhancedChatInterface;