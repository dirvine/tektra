import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3,
  Clock,
  FileText,
  Database,
  List,
  Download,
  Search,
  Eye,
  Heart,
  Smile,
  Brain,
  X,
  ChevronRight,
  Folder,
  File,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  PanelRight,
  PanelRightClose,
} from 'lucide-react';
import { useTektraStore } from '../store';

// Tab Navigation
const TabNavigation: React.FC<{
  activeTab: string;
  onTabChange: (tab: string) => void;
}> = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'analytics', label: 'Analytics', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'session', label: 'Session', icon: <Clock className="w-4 h-4" /> },
    { id: 'files', label: 'Files', icon: <FileText className="w-4 h-4" /> },
    { id: 'knowledge', label: 'Knowledge', icon: <Database className="w-4 h-4" /> },
    { id: 'tasks', label: 'Tasks', icon: <List className="w-4 h-4" /> },
  ];

  return (
    <div className="flex border-b border-border-primary">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          className={`
            flex-1 flex items-center justify-center space-x-1 px-3 py-3 text-sm transition-colors
            ${activeTab === tab.id
              ? 'text-accent border-b-2 border-accent bg-accent/5'
              : 'text-text-secondary hover:text-text-primary hover:bg-surface/50'
            }
          `}
        >
          {tab.icon}
          <span className="hidden sm:inline">{tab.label}</span>
        </button>
      ))}
    </div>
  );
};

// Analytics Tab
const AnalyticsTab: React.FC = () => {
  const { sessionState } = useTektraStore();

  const metrics = [
    {
      label: 'Eye Contact Time',
      value: `${sessionState.engagementMetrics.eyeContactTime}s`,
      change: '+12%',
      positive: true,
    },
    {
      label: 'Expression Changes',
      value: sessionState.engagementMetrics.expressionChanges.toString(),
      change: '+5',
      positive: true,
    },
    {
      label: 'Response Quality',
      value: '94%',
      change: '+2%',
      positive: true,
    },
    {
      label: 'User Satisfaction',
      value: '4.8/5',
      change: '+0.3',
      positive: true,
    },
  ];

  return (
    <div className="p-4 space-y-6">
      {/* Engagement Overview */}
      <div>
        <h3 className="text-sm font-medium text-text-primary mb-3">Avatar Analytics</h3>
        <div className="grid grid-cols-2 gap-3">
          {metrics.map((metric, index) => (
            <div key={index} className="p-3 bg-surface rounded-card border border-border-primary">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-text-tertiary">{metric.label}</span>
                <span className={`text-xs ${metric.positive ? 'text-success' : 'text-error'}`}>
                  {metric.change}
                </span>
              </div>
              <div className="text-lg font-semibold text-text-primary">{metric.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Sentiment Graph */}
      <div>
        <h3 className="text-sm font-medium text-text-primary mb-3">Conversation Mood</h3>
        <div className="h-32 bg-surface rounded-card border border-border-primary p-3 flex items-end justify-between space-x-1">
          {[0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 0.9, 0.8].map((height, index) => (
            <div
              key={index}
              className="bg-accent rounded-sm flex-1"
              style={{ height: `${height * 100}%` }}
            />
          ))}
        </div>
        <div className="flex items-center justify-center mt-2 space-x-4">
          <div className="flex items-center space-x-1">
            <Smile className="w-3 h-3 text-success" />
            <span className="text-xs text-text-secondary">Positive</span>
          </div>
          <div className="flex items-center space-x-1">
            <Heart className="w-3 h-3 text-accent" />
            <span className="text-xs text-text-secondary">
              {sessionState.engagementMetrics.userSentiment}
            </span>
          </div>
        </div>
      </div>

      {/* Interaction Stats */}
      <div>
        <h3 className="text-sm font-medium text-text-primary mb-3">Interaction Stats</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-secondary">Average Response Time</span>
            <span className="text-sm text-text-primary font-mono">1.2s</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-secondary">Messages Exchanged</span>
            <span className="text-sm text-text-primary font-mono">24</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-text-secondary">Questions Asked</span>
            <span className="text-sm text-text-primary font-mono">8</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Session Tab
const SessionTab: React.FC = () => {
  const { sessionState } = useTektraStore();

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return hours > 0 ? `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}` 
                     : `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="p-4 space-y-6">
      {/* Session Overview */}
      <div>
        <h3 className="text-sm font-medium text-text-primary mb-3">Current Session</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-surface rounded-card border border-border-primary">
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4 text-accent" />
              <span className="text-sm text-text-secondary">Duration</span>
            </div>
            <span className="text-sm text-text-primary font-mono">
              {formatDuration(sessionState.duration)}
            </span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-surface rounded-card border border-border-primary">
            <div className="flex items-center space-x-2">
              <Brain className="w-4 h-4 text-accent" />
              <span className="text-sm text-text-secondary">Tokens Used</span>
            </div>
            <span className="text-sm text-text-primary font-mono">
              {sessionState.tokenUsage.toLocaleString()}
            </span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-surface rounded-card border border-border-primary">
            <div className="flex items-center space-x-2">
              <span className="text-sm text-text-secondary">Est. Cost</span>
            </div>
            <span className="text-sm text-text-primary font-mono">
              ${sessionState.costEstimate.toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      {/* Conversation Summary */}
      <div>
        <h3 className="text-sm font-medium text-text-primary mb-3">Conversation Summary</h3>
        <div className="p-3 bg-surface rounded-card border border-border-primary">
          <p className="text-sm text-text-secondary leading-relaxed">
            {sessionState.conversationSummary || 
             'This session focused on AI model configuration and avatar customization. The user explored various settings and tested multimodal capabilities.'}
          </p>
        </div>
      </div>

      {/* Export Options */}
      <div>
        <h3 className="text-sm font-medium text-text-primary mb-3">Export Session</h3>
        <div className="space-y-2">
          {['PDF Report', 'Markdown', 'JSON Data'].map((format) => (
            <button
              key={format}
              className="w-full flex items-center justify-between p-3 bg-surface rounded-card border border-border-primary hover:border-accent transition-colors"
            >
              <span className="text-sm text-text-primary">{format}</span>
              <Download className="w-4 h-4 text-text-secondary" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

// Files Tab
const FilesTab: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');

  const files = [
    { name: 'project-notes.md', type: 'file', size: '2.3 KB', modified: '2 hours ago' },
    { name: 'src', type: 'folder', items: 12, modified: '1 hour ago' },
    { name: 'README.md', type: 'file', size: '1.1 KB', modified: '3 hours ago' },
    { name: 'package.json', type: 'file', size: '856 B', modified: '1 day ago' },
    { name: 'docs', type: 'folder', items: 5, modified: '2 days ago' },
  ];

  const filteredFiles = files.filter(file =>
    file.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="p-4 space-y-4">
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-text-tertiary" />
        <input
          type="text"
          placeholder="Search files..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-3 py-2 bg-surface border border-border-primary rounded-button text-text-primary placeholder-text-tertiary focus:border-accent outline-none"
        />
      </div>

      {/* Recent Files */}
      <div>
        <h3 className="text-sm font-medium text-text-primary mb-3">Recent Files</h3>
        <div className="space-y-1">
          {filteredFiles.map((file, index) => (
            <div
              key={index}
              className="flex items-center space-x-3 p-2 rounded-button hover:bg-surface/50 transition-colors cursor-pointer group"
            >
              {file.type === 'folder' ? (
                <Folder className="w-4 h-4 text-accent flex-shrink-0" />
              ) : (
                <File className="w-4 h-4 text-text-secondary flex-shrink-0" />
              )}
              
              <div className="flex-1 min-w-0">
                <p className="text-sm text-text-primary truncate">{file.name}</p>
                <p className="text-xs text-text-tertiary">
                  {file.type === 'folder' ? `${file.items} items` : file.size} • {file.modified}
                </p>
              </div>
              
              <ChevronRight className="w-3 h-3 text-text-tertiary opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
          ))}
        </div>
      </div>

      {/* File Tree Toggle */}
      <div className="pt-4 border-t border-border-primary">
        <button className="w-full flex items-center justify-center space-x-2 p-2 text-sm text-accent hover:text-accent-light transition-colors">
          <Eye className="w-4 h-4" />
          <span>Show Full Tree</span>
        </button>
      </div>
    </div>
  );
};

// Knowledge Tab
const KnowledgeTab: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');

  const knowledgeItems = [
    { title: 'React Best Practices', type: 'Document', lastAccessed: '1 hour ago', relevance: 95 },
    { title: 'TypeScript Handbook', type: 'Reference', lastAccessed: '2 hours ago', relevance: 88 },
    { title: 'API Documentation', type: 'Document', lastAccessed: '3 hours ago', relevance: 76 },
    { title: 'Design System Guide', type: 'Document', lastAccessed: '1 day ago', relevance: 82 },
  ];

  return (
    <div className="p-4 space-y-4">
      {/* Search Knowledge Base */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-text-tertiary" />
        <input
          type="text"
          placeholder="Search knowledge base..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-3 py-2 bg-surface border border-border-primary rounded-button text-text-primary placeholder-text-tertiary focus:border-accent outline-none"
        />
      </div>

      {/* Connected Documents */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-text-primary">Connected Documents</h3>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-success rounded-full" />
            <span className="text-xs text-text-tertiary">4 sources</span>
          </div>
        </div>
        
        <div className="space-y-2">
          {knowledgeItems.map((item, index) => (
            <div
              key={index}
              className="p-3 bg-surface rounded-card border border-border-primary hover:border-accent transition-colors cursor-pointer"
            >
              <div className="flex items-start justify-between mb-2">
                <h4 className="text-sm font-medium text-text-primary">{item.title}</h4>
                <span className="text-xs text-success">{item.relevance}%</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-text-tertiary">{item.type}</span>
                <span className="text-xs text-text-tertiary">{item.lastAccessed}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="pt-4 border-t border-border-primary">
        <button className="w-full flex items-center justify-center space-x-2 p-2 text-sm text-accent hover:text-accent-light transition-colors">
          <Database className="w-4 h-4" />
          <span>Add Knowledge Source</span>
        </button>
      </div>
    </div>
  );
};

// Tasks Tab
const TasksTab: React.FC = () => {
  const tasks = [
    {
      id: 1,
      title: 'Model initialization',
      status: 'completed',
      progress: 100,
      duration: '2.3s',
    },
    {
      id: 2,
      title: 'Processing user query',
      status: 'running',
      progress: 75,
      duration: '1.2s',
    },
    {
      id: 3,
      title: 'Generating response',
      status: 'queued',
      progress: 0,
      duration: null,
    },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-success" />;
      case 'running':
        return <Play className="w-4 h-4 text-accent animate-pulse" />;
      case 'queued':
        return <Pause className="w-4 h-4 text-text-tertiary" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-error" />;
      default:
        return <div className="w-4 h-4" />;
    }
  };

  return (
    <div className="p-4 space-y-4">
      {/* Active Tasks */}
      <div>
        <h3 className="text-sm font-medium text-text-primary mb-3">Active Tasks</h3>
        <div className="space-y-3">
          {tasks.map((task) => (
            <div key={task.id} className="p-3 bg-surface rounded-card border border-border-primary">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {getStatusIcon(task.status)}
                  <span className="text-sm text-text-primary">{task.title}</span>
                </div>
                {task.duration && (
                  <span className="text-xs text-text-tertiary font-mono">{task.duration}</span>
                )}
              </div>
              
              {task.status === 'running' && (
                <div className="w-full bg-surface-hover rounded-full h-1.5">
                  <div
                    className="bg-accent h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${task.progress}%` }}
                  />
                </div>
              )}
              
              <div className="flex items-center justify-between mt-2">
                <span className="text-xs text-text-tertiary capitalize">{task.status}</span>
                {task.status === 'running' && (
                  <button className="text-xs text-error hover:text-error/80 transition-colors">
                    Cancel
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Task History */}
      <div className="pt-4 border-t border-border-primary">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-text-primary">Completed Tasks</h3>
          <button className="text-xs text-text-tertiary hover:text-text-secondary transition-colors">
            Clear All
          </button>
        </div>
        
        <div className="text-center py-6">
          <CheckCircle2 className="w-8 h-8 text-success mx-auto mb-2" />
          <p className="text-sm text-text-secondary">All tasks completed</p>
          <p className="text-xs text-text-tertiary mt-1">No background processes running</p>
        </div>
      </div>
    </div>
  );
};

// Main Right Sidebar Component
const RightSidebar: React.FC = () => {
  const { uiState, toggleRightSidebar, setActiveTab } = useTektraStore();

  const renderTabContent = () => {
    switch (uiState.activeTab) {
      case 'analytics':
        return <AnalyticsTab />;
      case 'session':
        return <SessionTab />;
      case 'files':
        return <FilesTab />;
      case 'knowledge':
        return <KnowledgeTab />;
      case 'tasks':
        return <TasksTab />;
      default:
        return <AnalyticsTab />;
    }
  };

  if (!uiState.rightSidebarVisible) {
    return (
      <button
        onClick={toggleRightSidebar}
        className="fixed right-4 top-1/2 transform -translate-y-1/2 z-40 p-3 bg-secondary-bg border border-border-primary rounded-l-card hover:bg-surface transition-colors"
      >
        <PanelRight className="w-4 h-4 text-text-secondary" />
      </button>
    );
  }

  return (
    <motion.aside
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 320, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="fixed right-0 top-16 bottom-8 z-40 w-80 bg-secondary-bg border-l border-border-primary flex flex-col overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border-primary">
        <h2 className="font-semibold text-text-primary">Context & Resources</h2>
        <button
          onClick={toggleRightSidebar}
          className="p-2 rounded-button hover:bg-surface-hover transition-colors"
        >
          <PanelRightClose className="w-4 h-4 text-text-secondary" />
        </button>
      </div>

      {/* Tab Navigation */}
      <TabNavigation
        activeTab={uiState.activeTab}
        onTabChange={setActiveTab}
      />

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <AnimatePresence mode="wait">
          <motion.div
            key={uiState.activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
          >
            {renderTabContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.aside>
  );
};

export default RightSidebar;