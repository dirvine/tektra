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
  Upload,
  X,
  Film,
  File,
} from 'lucide-react';
import { useTektraStore } from '../store';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { analyzeError, attemptOllamaRestart } from '../utils/errorHandling';
import { formatResponse } from '../utils/formatting';
import '../styles/react-markdown.css';

interface MessageProps {
  message: any;
  isLatest: boolean;
}

const Message: React.FC<MessageProps> = ({ message, isLatest }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [showActions, setShowActions] = useState(false);
  const [isRestarting, setIsRestarting] = useState(false);
  const { addMessage } = useTektraStore();
  
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // Could add toast notification here
  };
  
  const handleRestartOllama = async () => {
    setIsRestarting(true);
    const result = await attemptOllamaRestart();
    
    addMessage({
      role: 'system',
      content: result.message,
    });
    
    setIsRestarting(false);
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
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              className="prose prose-sm max-w-none"
              components={{
                p: ({children}) => <p className="mb-2 text-text-secondary">{children}</p>,
                strong: ({children}) => <strong className="text-white font-bold">{children}</strong>,
                em: ({children}) => <em className="text-text-secondary italic">{children}</em>,
                ul: ({children}) => <ul className="list-disc list-inside mb-2 text-text-secondary">{children}</ul>,
                ol: ({children}) => <ol className="list-decimal list-inside mb-2 text-text-secondary">{children}</ol>,
                li: ({children}) => <li className="mb-1">{children}</li>,
                code: ({inline, className, children}) => {
                  return inline ? (
                    <code className="bg-surface px-1 py-0.5 rounded text-accent-light text-sm">{children}</code>
                  ) : (
                    <code className="block bg-surface p-3 rounded-md text-sm overflow-x-auto">{children}</code>
                  );
                },
                pre: ({children}) => <pre className="bg-surface p-3 rounded-md overflow-x-auto mb-2">{children}</pre>,
                blockquote: ({children}) => (
                  <blockquote className="border-l-4 border-accent pl-4 my-2 text-text-secondary italic">
                    {children}
                  </blockquote>
                ),
                a: ({href, children}) => (
                  <a href={href} className="text-accent hover:text-accent-light underline" target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
                h1: ({children}) => <h1 className="text-xl font-bold text-white mb-2">{children}</h1>,
                h2: ({children}) => <h2 className="text-lg font-bold text-white mb-2">{children}</h2>,
                h3: ({children}) => <h3 className="text-base font-bold text-white mb-2">{children}</h3>,
              }}
            >
              {isCollapsed && isLongMessage 
                ? message.content.substring(0, 200) + '...' 
                : message.content}
            </ReactMarkdown>
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
        
        {/* Restart button for Ollama errors */}
        {message.metadata?.canRestart && (
          <div className="mt-4">
            <button
              onClick={handleRestartOllama}
              disabled={isRestarting}
              className={`
                px-4 py-2 rounded-button transition-colors
                flex items-center space-x-2
                ${isRestarting 
                  ? 'bg-surface text-text-tertiary cursor-not-allowed' 
                  : 'bg-accent text-white hover:bg-accent-hover'
                }
              `}
            >
              {isRestarting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Restarting...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span>Restart AI Server</span>
                </>
              )}
            </button>
            
            <p className="mt-2 text-xs text-text-tertiary">
              Or manually restart Tektra if the issue persists
            </p>
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
  onSubmit: (attachments?: File[]) => void;
  disabled: boolean;
  isRecording: boolean;
  onToggleRecording: () => void;
}> = ({ value, onChange, onSubmit, disabled, isRecording, onToggleRecording }) => {
  const [showPreview, setShowPreview] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    console.log('Submit triggered with attachments:', attachments);
    onSubmit(attachments.length > 0 ? attachments : undefined);
    setAttachments([]); // Clear attachments after sending
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log('File upload triggered', e.target.files);
    if (e.target.files) {
      const files = Array.from(e.target.files);
      console.log('Files selected:', files);
      setAttachments(prev => [...prev, ...files]);
    }
    // Reset input value to allow selecting the same file again
    e.target.value = '';
  };


  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) return <Image className="w-4 h-4" />;
    if (file.type.startsWith('video/')) return <Film className="w-4 h-4" />;
    if (file.type.startsWith('audio/')) return <Mic className="w-4 h-4" />;
    if (file.type.includes('text') || file.name.endsWith('.md') || file.name.endsWith('.txt')) return <FileText className="w-4 h-4" />;
    return <File className="w-4 h-4" />;
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const triggerFileUpload = () => {
    console.log('Triggering file upload', fileInputRef.current);
    if (fileInputRef.current) {
      fileInputRef.current.click();
    } else {
      console.error('File input ref is null');
    }
  };

  return (
    <div 
      ref={dropZoneRef}
      className="border-t border-border-primary bg-secondary-bg p-4"
    >

      {/* Attachments Preview */}
      {attachments.length > 0 && (
        <div className="mb-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-text-primary">
              Attached Files ({attachments.length})
            </span>
            <button
              onClick={() => setAttachments([])}
              className="text-xs text-text-tertiary hover:text-error transition-colors"
            >
              Clear all
            </button>
          </div>
          <div className="flex flex-wrap gap-2">
            {attachments.map((file, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center space-x-2 px-3 py-2 bg-surface rounded-card border border-border-primary group hover:border-accent/30 transition-colors"
              >
                <div className="text-text-secondary">
                  {getFileIcon(file)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-text-primary truncate max-w-32" title={file.name}>
                    {file.name}
                  </div>
                  <div className="text-xs text-text-tertiary">
                    {formatFileSize(file.size)}
                  </div>
                </div>
                <button 
                  onClick={() => removeAttachment(index)}
                  className="opacity-0 group-hover:opacity-100 text-text-tertiary hover:text-error transition-all p-1 rounded"
                  title="Remove file"
                >
                  <X className="w-3 h-3" />
                </button>
              </motion.div>
            ))}
          </div>
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
          <label
            htmlFor="file-upload-input"
            className="p-3 rounded-button hover:bg-surface-hover transition-colors cursor-pointer inline-block"
            title="Upload files"
          >
            <Paperclip className="w-4 h-4 text-text-secondary" />
          </label>
          <input
            id="file-upload-input"
            ref={fileInputRef}
            type="file"
            multiple
            accept=".txt,.md,.json,text/*"
            onChange={handleFileUpload}
            className="hidden"
          />

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
            onClick={handleSubmit}
            disabled={disabled || (!value.trim() && attachments.length === 0)}
            className={`
              p-3 rounded-button transition-colors
              ${disabled || (!value.trim() && attachments.length === 0)
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
          <ReactMarkdown 
            remarkPlugins={[remarkGfm]}
            className="prose prose-sm max-w-none"
            components={{
              p: ({children}) => <p className="mb-2 text-text-secondary">{children}</p>,
              strong: ({children}) => <strong className="text-white font-bold">{children}</strong>,
              em: ({children}) => <em className="text-text-secondary italic">{children}</em>,
              ul: ({children}) => <ul className="list-disc list-inside mb-2 text-text-secondary">{children}</ul>,
              ol: ({children}) => <ol className="list-decimal list-inside mb-2 text-text-secondary">{children}</ol>,
              li: ({children}) => <li className="mb-1">{children}</li>,
              code: ({inline, className, children}) => {
                return inline ? (
                  <code className="bg-surface px-1 py-0.5 rounded text-accent-light text-sm">{children}</code>
                ) : (
                  <code className="block bg-surface p-3 rounded-md text-sm overflow-x-auto">{children}</code>
                );
              },
              pre: ({children}) => <pre className="bg-surface p-3 rounded-md overflow-x-auto mb-2">{children}</pre>,
              blockquote: ({children}) => (
                <blockquote className="border-l-4 border-accent pl-4 my-2 text-text-secondary italic">
                  {children}
                </blockquote>
              ),
              a: ({href, children}) => (
                <a href={href} className="text-accent hover:text-accent-light underline" target="_blank" rel="noopener noreferrer">
                  {children}
                </a>
              ),
              h1: ({children}) => <h1 className="text-xl font-bold text-white mb-2">{children}</h1>,
              h2: ({children}) => <h2 className="text-lg font-bold text-white mb-2">{children}</h2>,
              h3: ({children}) => <h3 className="text-base font-bold text-white mb-2">{children}</h3>,
            }}
          >
            {value}
          </ReactMarkdown>
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
  const [globalDragOver, setGlobalDragOver] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (attachments?: File[]) => {
    if ((!inputValue.trim() && !attachments?.length) || !modelStatus.isLoaded) return;

    try {
      // Process file attachments first if any
      if (attachments && attachments.length > 0) {
        addMessage({
          role: 'system',
          content: `📎 Processing ${attachments.length} file(s): ${attachments.map(f => f.name).join(', ')}`,
        });

        // Show file attachment status but don't process separately
        // Files will be processed together with the user's message
        addMessage({
          role: 'system', 
          content: `📎 ${attachments.length} file(s) attached and ready to analyze with your message`,
        });
      }

      // Send the text message if any
      if (inputValue.trim() || (attachments && attachments.length > 0)) {
        let combinedMessageContent = inputValue.trim();

        if (attachments && attachments.length > 0) {
          for (const file of attachments) {
            // Check if it's a text file
            if (file.type.startsWith('text/') || file.name.endsWith('.txt') || file.name.endsWith('.md') || file.name.endsWith('.json')) {
              try {
                const textContent = await file.text();
                combinedMessageContent += `

--- Start of ${file.name} ---
${textContent}
--- End of ${file.name} ---`;
              } catch (error) {
                console.error(`Error reading text file ${file.name}:`, error);
                addMessage({
                  role: 'system',
                  content: `❌ Error reading text file ${file.name}: ${error}`,
                });
              }
            }
          }
        }

        const userMessage = {
          role: 'user' as const,
          content: combinedMessageContent,
        };

        addMessage(userMessage);
        setInputValue('');
        // @ts-ignore - setAttachments is defined in EnhancedInputArea
        // setAttachments([]); // Clear attachments after sending

        // Send combined message to backend using comprehensive multimodal processing
        try {
          const { invoke } = await import('@tauri-apps/api/core');
          
          // Prepare multimodal data for comprehensive processing
          let imageData: number[] | null = null;
          let audioData: number[] | null = null;
          let videoData: number[] | null = null;
          
          // Process all attachments for multimodal input
          if (attachments && attachments.length > 0) {
            for (const file of attachments) {
              try {
                if (file.type.startsWith('image/') || file.name.match(/\.(jpg|jpeg|png|gif|webp|bmp)$/i)) {
                  // Process first image attachment
                  if (!imageData) {
                    const arrayBuffer = await file.arrayBuffer();
                    imageData = Array.from(new Uint8Array(arrayBuffer));
                    console.log(`Prepared image data: ${file.name} (${imageData.length} bytes)`);
                  }
                } else if (file.type.startsWith('audio/') || file.name.match(/\.(mp3|wav|m4a|ogg|flac)$/i)) {
                  // Process first audio attachment
                  if (!audioData) {
                    const arrayBuffer = await file.arrayBuffer();
                    audioData = Array.from(new Uint8Array(arrayBuffer));
                    console.log(`Prepared audio data: ${file.name} (${audioData.length} bytes)`);
                  }
                } else if (file.type.startsWith('video/') || file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
                  // Process first video attachment
                  if (!videoData) {
                    const arrayBuffer = await file.arrayBuffer();
                    videoData = Array.from(new Uint8Array(arrayBuffer));
                    console.log(`Prepared video data: ${file.name} (${videoData.length} bytes)`);
                  }
                }
                // Text files are already combined into combinedMessageContent above
              } catch (error) {
                console.error(`Error processing attachment ${file.name}:`, error);
                addMessage({
                  role: 'system',
                  content: `❌ Error processing ${file.name}: ${error}`,
                });
              }
            }
          }
          
          // Use comprehensive multimodal processing for all cases
          console.log('Sending multimodal request:', {
            message: combinedMessageContent.substring(0, 100) + '...',
            hasImage: !!imageData,
            hasAudio: !!audioData,
            hasVideo: !!videoData
          });
          
          const response = await invoke<string>('process_multimodal_input', {
            message: combinedMessageContent,
            imageData: imageData,
            audioData: audioData,
            videoData: videoData,
          });
          
          addMessage({
            role: 'assistant',
            content: response,
          });
        } catch (error) {
          console.error('Backend call error:', error);
          
          // Analyze the error
          const errorInfo = analyzeError(error);
          
          if (errorInfo.needsRestart) {
            // Add a system message with restart option
            addMessage({
              role: 'system',
              content: errorInfo.userMessage,
              metadata: {
                isError: true,
                canRestart: true,
                errorType: 'ollama_connection'
              }
            });
          } else {
            // Regular error message
            addMessage({
              role: 'assistant',
              content: errorInfo.userMessage,
            });
          }
        }
      }
    } catch (error) {
      console.error('Message send error:', error);
      addMessage({
        role: 'system',
        content: `❌ Error sending message: ${error}`,
      });
    }
  };

  const handleToggleRecording = () => {
    setRecording(!isRecording);
    // TODO: Integrate with backend audio recording
  };

  // Global drag and drop handlers
  const handleGlobalDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Global drag over detected');
    if (!globalDragOver) setGlobalDragOver(true);
  };

  const handleGlobalDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only set to false if leaving the main container
    const relatedTarget = e.relatedTarget as Element | null;
    if (!relatedTarget || !e.currentTarget.contains(relatedTarget)) {
      setGlobalDragOver(false);
    }
  };

  const handleGlobalDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Global drop detected', e.dataTransfer.files);
    setGlobalDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    console.log('Globally dropped files:', files);
    if (files.length > 0) {
      handleSendMessage(files);
    }
  };

  return (
    <div 
      className={`flex flex-col h-full ${className} relative`}
      onDragOver={handleGlobalDragOver}
      onDragLeave={handleGlobalDragLeave}
      onDrop={handleGlobalDrop}
    >
      {/* Global Drag Overlay */}
      {globalDragOver && (
        <div className="absolute inset-0 bg-accent/10 border-4 border-dashed border-accent rounded-lg flex items-center justify-center z-50">
          <div className="text-center">
            <div className="w-20 h-20 bg-accent/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <Upload className="w-10 h-10 text-accent" />
            </div>
            <p className="text-xl font-medium text-accent mb-2">Drop files here to upload</p>
            <p className="text-text-secondary">Supports text files (.txt, .md, .json)</p>
          </div>
        </div>
      )}

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