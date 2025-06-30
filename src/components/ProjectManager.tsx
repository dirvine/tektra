import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FolderOpen,
  FolderPlus,
  FileText,
  Image,
  Film,
  Mic,
  File,
  Search,
  Filter,
  MoreVertical,
  Trash2,
  Edit3,
  Eye,
  Download,
  Upload,
  Settings,
  Grid,
  List,
  Calendar,
  Tag,
  Star,
  StarOff,
  ChevronDown,
  ChevronRight,
  X,
  Plus,
} from 'lucide-react';
import { useTektraStore } from '../store';

interface Project {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  documentCount: number;
  tags: string[];
  isStarred: boolean;
}

interface Document {
  id: string;
  projectId: string;
  name: string;
  type: 'text' | 'image' | 'audio' | 'video' | 'other';
  size: number;
  path: string;
  uploadedAt: string;
  tags: string[];
  metadata?: {
    dimensions?: { width: number; height: number };
    duration?: number;
    mimeType?: string;
  };
}

interface ProjectManagerProps {
  className?: string;
  onDocumentSelect?: (document: Document) => void;
}

const ProjectManager: React.FC<ProjectManagerProps> = ({ className = '', onDocumentSelect }) => {
  const { uiState, setUIState } = useTektraStore();
  const [projects, setProjects] = useState<Project[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string | null>(null);
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Load projects on mount
  useEffect(() => {
    loadProjects();
  }, []);

  // Load documents when project changes
  useEffect(() => {
    if (selectedProject) {
      loadProjectDocuments(selectedProject.id);
    }
  }, [selectedProject]);

  const loadProjects = async () => {
    setIsLoading(true);
    try {
      const projectList = await invoke<Project[]>('get_projects');
      setProjects(projectList);
      
      // Select current project if set
      if (uiState.currentProject) {
        const currentProj = projectList.find(p => p.name === uiState.currentProject);
        if (currentProj) setSelectedProject(currentProj);
      }
    } catch (error) {
      console.error('Failed to load projects:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadProjectDocuments = async (projectId: string) => {
    try {
      const docs = await invoke<Document[]>('get_project_documents', { projectId });
      setDocuments(docs);
    } catch (error) {
      console.error('Failed to load project documents:', error);
    }
  };

  const createProject = async (name: string, description?: string) => {
    try {
      const project = await invoke<Project>('create_project', { name, description });
      setProjects(prev => [...prev, project]);
      setSelectedProject(project);
      setUIState({ ...uiState, currentProject: project.name });
      setShowNewProjectModal(false);
    } catch (error) {
      console.error('Failed to create project:', error);
    }
  };

  const deleteProject = async (projectId: string) => {
    try {
      await invoke('delete_project', { projectId });
      setProjects(prev => prev.filter(p => p.id !== projectId));
      if (selectedProject?.id === projectId) {
        setSelectedProject(null);
        setDocuments([]);
      }
    } catch (error) {
      console.error('Failed to delete project:', error);
    }
  };

  const toggleProjectStar = async (projectId: string) => {
    try {
      await invoke('toggle_project_star', { projectId });
      setProjects(prev => prev.map(p => 
        p.id === projectId ? { ...p, isStarred: !p.isStarred } : p
      ));
    } catch (error) {
      console.error('Failed to toggle project star:', error);
    }
  };

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'image': return <Image className="w-4 h-4" />;
      case 'video': return <Film className="w-4 h-4" />;
      case 'audio': return <Mic className="w-4 h-4" />;
      case 'text': return <FileText className="w-4 h-4" />;
      default: return <File className="w-4 h-4" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = !filterType || doc.type === filterType;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className={`flex flex-col h-full bg-surface border border-border-primary rounded-card ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border-primary">
        <div className="flex items-center space-x-3">
          <FolderOpen className="w-5 h-5 text-accent" />
          <h2 className="font-semibold text-text-primary">Projects</h2>
          {selectedProject && (
            <>
              <ChevronRight className="w-4 h-4 text-text-tertiary" />
              <span className="text-text-secondary">{selectedProject.name}</span>
            </>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
            className="p-2 hover:bg-surface-hover rounded-button transition-colors"
            title={`Switch to ${viewMode === 'grid' ? 'list' : 'grid'} view`}
          >
            {viewMode === 'grid' ? <List className="w-4 h-4" /> : <Grid className="w-4 h-4" />}
          </button>
          
          <button
            onClick={() => setShowNewProjectModal(true)}
            className="p-2 bg-accent text-white hover:bg-accent-hover rounded-button transition-colors"
            title="New project"
          >
            <FolderPlus className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* Projects Sidebar */}
        <div className="w-64 border-r border-border-primary flex flex-col">
          {/* Search */}
          <div className="p-3 border-b border-border-primary">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-text-tertiary" />
              <input
                type="text"
                placeholder="Search projects..."
                className="w-full pl-9 pr-3 py-2 bg-secondary-bg border border-border-secondary rounded-button text-sm focus:border-accent outline-none"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>

          {/* Projects List */}
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
              </div>
            ) : (
              projects.map(project => (
                <motion.div
                  key={project.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`
                    group p-3 rounded-card cursor-pointer transition-colors border
                    ${selectedProject?.id === project.id 
                      ? 'bg-accent/10 border-accent/20' 
                      : 'border-transparent hover:bg-surface-hover'
                    }
                  `}
                  onClick={() => setSelectedProject(project)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-text-primary truncate">
                          {project.name}
                        </span>
                        {project.isStarred && (
                          <Star className="w-3 h-3 text-warning fill-current" />
                        )}
                      </div>
                      <p className="text-xs text-text-tertiary mt-1 line-clamp-2">
                        {project.description || 'No description'}
                      </p>
                      <div className="flex items-center space-x-3 mt-2 text-xs text-text-tertiary">
                        <span>{project.documentCount} files</span>
                        <span>{new Date(project.updatedAt).toLocaleDateString()}</span>
                      </div>
                    </div>
                    
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity ml-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleProjectStar(project.id);
                        }}
                        className="p-1 hover:bg-surface-hover rounded text-text-tertiary hover:text-warning"
                      >
                        {project.isStarred ? <StarOff className="w-3 h-3" /> : <Star className="w-3 h-3" />}
                      </button>
                    </div>
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </div>

        {/* Documents Area */}
        <div className="flex-1 flex flex-col">
          {selectedProject ? (
            <>
              {/* Documents Header */}
              <div className="flex items-center justify-between p-4 border-b border-border-primary">
                <div className="flex items-center space-x-4">
                  <div>
                    <h3 className="font-medium text-text-primary">{selectedProject.name}</h3>
                    <p className="text-sm text-text-secondary">{documents.length} documents</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {/* Filter */}
                  <select
                    value={filterType || ''}
                    onChange={(e) => setFilterType(e.target.value || null)}
                    className="px-3 py-1 bg-secondary-bg border border-border-secondary rounded-button text-sm"
                  >
                    <option value="">All types</option>
                    <option value="text">Text</option>
                    <option value="image">Images</option>
                    <option value="audio">Audio</option>
                    <option value="video">Video</option>
                  </select>
                  
                  <button className="p-2 hover:bg-surface-hover rounded-button transition-colors">
                    <Upload className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Documents Grid/List */}
              <div className="flex-1 overflow-y-auto p-4">
                {filteredDocuments.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center py-12">
                    <FolderOpen className="w-16 h-16 text-text-tertiary mb-4" />
                    <h3 className="text-lg font-medium text-text-primary mb-2">No documents</h3>
                    <p className="text-text-secondary max-w-md">
                      {searchQuery ? 'No documents match your search.' : 'Start by uploading documents to this project.'}
                    </p>
                  </div>
                ) : (
                  <div className={`
                    ${viewMode === 'grid' 
                      ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4' 
                      : 'space-y-2'
                    }
                  `}>
                    {filteredDocuments.map(doc => (
                      <motion.div
                        key={doc.id}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className={`
                          group cursor-pointer border border-border-primary rounded-card overflow-hidden
                          hover:border-accent/30 transition-all duration-200
                          ${viewMode === 'grid' ? 'aspect-square' : 'flex items-center p-3'}
                        `}
                        onClick={() => onDocumentSelect?.(doc)}
                      >
                        {viewMode === 'grid' ? (
                          <div className="h-full flex flex-col">
                            <div className="flex-1 bg-secondary-bg flex items-center justify-center">
                              {doc.type === 'image' ? (
                                <div className="w-full h-full bg-cover bg-center" 
                                     style={{ backgroundImage: `url(${doc.path})` }} />
                              ) : (
                                <div className="w-12 h-12 text-text-tertiary">
                                  {getFileIcon(doc.type)}
                                </div>
                              )}
                            </div>
                            <div className="p-3">
                              <h4 className="font-medium text-text-primary text-sm truncate" title={doc.name}>
                                {doc.name}
                              </h4>
                              <div className="flex items-center justify-between mt-1">
                                <span className="text-xs text-text-tertiary">
                                  {formatFileSize(doc.size)}
                                </span>
                                <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                                  <button className="p-1 hover:bg-surface-hover rounded">
                                    <MoreVertical className="w-3 h-3" />
                                  </button>
                                </div>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <>
                            <div className="w-8 h-8 bg-secondary-bg rounded flex items-center justify-center mr-3">
                              {getFileIcon(doc.type)}
                            </div>
                            <div className="flex-1 min-w-0">
                              <h4 className="font-medium text-text-primary truncate">{doc.name}</h4>
                              <div className="flex items-center space-x-3 text-xs text-text-tertiary">
                                <span>{formatFileSize(doc.size)}</span>
                                <span>{new Date(doc.uploadedAt).toLocaleDateString()}</span>
                              </div>
                            </div>
                            <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                              <button className="p-2 hover:bg-surface-hover rounded">
                                <MoreVertical className="w-4 h-4" />
                              </button>
                            </div>
                          </>
                        )}
                      </motion.div>
                    ))}
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-center">
              <div>
                <FolderOpen className="w-16 h-16 text-text-tertiary mx-auto mb-4" />
                <h3 className="text-lg font-medium text-text-primary mb-2">Select a Project</h3>
                <p className="text-text-secondary">Choose a project to view its documents</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* New Project Modal */}
      <AnimatePresence>
        {showNewProjectModal && (
          <NewProjectModal
            onClose={() => setShowNewProjectModal(false)}
            onCreate={createProject}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

// New Project Modal Component
interface NewProjectModalProps {
  onClose: () => void;
  onCreate: (name: string, description?: string) => void;
}

const NewProjectModal: React.FC<NewProjectModalProps> = ({ onClose, onCreate }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) {
      onCreate(name.trim(), description.trim() || undefined);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-surface border border-border-primary rounded-card p-6 w-full max-w-md"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary">New Project</h3>
          <button
            onClick={onClose}
            className="p-1 hover:bg-surface-hover rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">
              Project Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 bg-secondary-bg border border-border-secondary rounded-button focus:border-accent outline-none"
              placeholder="Enter project name..."
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">
              Description (optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 bg-secondary-bg border border-border-secondary rounded-button focus:border-accent outline-none resize-none"
              rows={3}
              placeholder="Enter project description..."
            />
          </div>

          <div className="flex items-center justify-end space-x-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-text-secondary hover:text-text-primary transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim()}
              className="px-4 py-2 bg-accent text-white hover:bg-accent-hover disabled:bg-text-tertiary disabled:cursor-not-allowed rounded-button transition-colors"
            >
              Create Project
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
};

export default ProjectManager;