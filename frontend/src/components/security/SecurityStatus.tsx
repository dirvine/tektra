"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { 
  Shield, 
  User, 
  Database, 
  Eye,
  EyeOff,
  LogOut,
  Settings,
  CheckCircle,
  AlertTriangle,
  Clock
} from 'lucide-react'

interface SecurityStatusProps {
  sessionToken?: string
  userData?: any
  onLogout?: () => void
  className?: string
}

interface VaultStats {
  total_conversations: number
  total_messages: number
  last_access: string | null
}

interface SecurityInfo {
  biometric_capabilities: {
    face_recognition: boolean
    voice_recognition: boolean
    registration: boolean
    authentication: boolean
  }
  active_sessions: number
  registered_users: number
  security_features: {
    biometric_auth: boolean
    encrypted_vaults: boolean
    query_anonymization: boolean
    session_management: boolean
  }
}

export default function SecurityStatus({ 
  sessionToken, 
  userData, 
  onLogout,
  className 
}: SecurityStatusProps) {
  const [vaultStats, setVaultStats] = useState<VaultStats | null>(null)
  const [securityInfo, setSecurityInfo] = useState<SecurityInfo | null>(null)
  const [showDetails, setShowDetails] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  
  // Fetch vault statistics
  const fetchVaultStats = async () => {
    if (!sessionToken) return
    
    try {
      const response = await fetch('/api/v1/security/vault/stats', {
        headers: {
          'Authorization': `Bearer ${sessionToken}`
        }
      })
      
      if (response.ok) {
        const stats = await response.json()
        setVaultStats(stats)
      }
    } catch (error) {
      console.error('Failed to fetch vault stats:', error)
    }
  }
  
  // Fetch security system info
  const fetchSecurityInfo = async () => {
    try {
      const response = await fetch('/api/v1/security/status')
      
      if (response.ok) {
        const info = await response.json()
        setSecurityInfo(info)
      }
    } catch (error) {
      console.error('Failed to fetch security info:', error)
    }
  }
  
  // Handle logout
  const handleLogout = async () => {
    if (!sessionToken) return
    
    try {
      setIsLoading(true)
      
      const response = await fetch('/api/v1/security/logout', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${sessionToken}`
        }
      })
      
      if (onLogout) {
        onLogout()
      }
    } catch (error) {
      console.error('Logout failed:', error)
    } finally {
      setIsLoading(false)
    }
  }
  
  // Format time
  const formatTime = (isoString: string | null) => {
    if (!isoString) return 'Never'
    
    try {
      return new Date(isoString).toLocaleString()
    } catch {
      return 'Invalid date'
    }
  }
  
  // Get security level
  const getSecurityLevel = () => {
    if (!securityInfo) return { level: 'Unknown', color: 'gray' }
    
    const features = securityInfo.security_features
    const capabilities = securityInfo.biometric_capabilities
    
    if (features.biometric_auth && 
        features.encrypted_vaults && 
        features.query_anonymization &&
        capabilities.authentication) {
      return { level: 'Maximum', color: 'green' }
    } else if (features.encrypted_vaults && features.session_management) {
      return { level: 'High', color: 'blue' }
    } else {
      return { level: 'Basic', color: 'yellow' }
    }
  }
  
  useEffect(() => {
    fetchSecurityInfo()
    if (sessionToken) {
      fetchVaultStats()
    }
  }, [sessionToken])
  
  const securityLevel = getSecurityLevel()
  
  return (
    <div className={`space-y-4 ${className}`}>
      {/* Main Security Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Shield className="h-5 w-5 text-blue-600" />
              <CardTitle className="text-lg">Security Status</CardTitle>
            </div>
            
            <Badge 
              variant={securityLevel.color === 'green' ? 'default' : 'secondary'}
              className={`
                ${securityLevel.color === 'green' ? 'bg-green-600' : ''}
                ${securityLevel.color === 'blue' ? 'bg-blue-600' : ''}
                ${securityLevel.color === 'yellow' ? 'bg-yellow-600' : ''}
              `}
            >
              {securityLevel.level} Security
            </Badge>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* User Info */}
          {sessionToken && userData && (
            <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
                  <User className="h-5 w-5 text-white" />
                </div>
                <div>
                  <p className="font-medium text-green-900">{userData.user_id}</p>
                  <p className="text-sm text-green-700">Authenticated</p>
                </div>
              </div>
              
              <Button
                onClick={handleLogout}
                variant="outline"
                size="sm"
                disabled={isLoading}
                className="border-green-300 text-green-700 hover:bg-green-100"
              >
                <LogOut className="h-4 w-4 mr-2" />
                {isLoading ? 'Logging out...' : 'Logout'}
              </Button>
            </div>
          )}
          
          {/* Security Features */}
          {securityInfo && (
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium text-sm">Authentication</h4>
                <div className="space-y-1">
                  <div className="flex items-center space-x-2 text-sm">
                    {securityInfo.biometric_capabilities.face_recognition ? (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    )}
                    <span>Face Recognition</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm">
                    {securityInfo.biometric_capabilities.voice_recognition ? (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    )}
                    <span>Voice Recognition</span>
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium text-sm">Data Protection</h4>
                <div className="space-y-1">
                  <div className="flex items-center space-x-2 text-sm">
                    {securityInfo.security_features.encrypted_vaults ? (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    )}
                    <span>Encrypted Vaults</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm">
                    {securityInfo.security_features.query_anonymization ? (
                      <CheckCircle className="h-4 w-4 text-green-600" />
                    ) : (
                      <AlertTriangle className="h-4 w-4 text-yellow-600" />
                    )}
                    <span>Query Anonymization</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Toggle Details */}
          <Button
            onClick={() => setShowDetails(!showDetails)}
            variant="ghost"
            size="sm"
            className="w-full"
          >
            {showDetails ? (
              <>
                <EyeOff className="h-4 w-4 mr-2" />
                Hide Details
              </>
            ) : (
              <>
                <Eye className="h-4 w-4 mr-2" />
                Show Details
              </>
            )}
          </Button>
        </CardContent>
      </Card>
      
      {/* Detailed Information */}
      {showDetails && (
        <>
          {/* Vault Statistics */}
          {sessionToken && vaultStats && (
            <Card>
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Database className="h-5 w-5 text-purple-600" />
                  <CardTitle className="text-lg">Vault Statistics</CardTitle>
                </div>
              </CardHeader>
              
              <CardContent>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-2xl font-bold text-purple-600">
                      {vaultStats.total_conversations}
                    </p>
                    <p className="text-sm text-gray-600">Conversations</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-blue-600">
                      {vaultStats.total_messages}
                    </p>
                    <p className="text-sm text-gray-600">Messages</p>
                  </div>
                  <div>
                    <div className="flex items-center justify-center space-x-1">
                      <Clock className="h-4 w-4 text-gray-600" />
                      <p className="text-xs text-gray-600">Last Access</p>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {formatTime(vaultStats.last_access)}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* System Information */}
          {securityInfo && (
            <Card>
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Settings className="h-5 w-5 text-gray-600" />
                  <CardTitle className="text-lg">System Information</CardTitle>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Active Sessions:</span>
                    <span className="ml-2">{securityInfo.active_sessions}</span>
                  </div>
                  <div>
                    <span className="font-medium">Registered Users:</span>
                    <span className="ml-2">{securityInfo.registered_users}</span>
                  </div>
                </div>
                
                <div className="border-t pt-4">
                  <h4 className="font-medium mb-2">Capabilities Status</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span>Biometric Registration:</span>
                      <Badge variant={securityInfo.biometric_capabilities.registration ? "default" : "secondary"}>
                        {securityInfo.biometric_capabilities.registration ? "Available" : "Unavailable"}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Biometric Authentication:</span>
                      <Badge variant={securityInfo.biometric_capabilities.authentication ? "default" : "secondary"}>
                        {securityInfo.biometric_capabilities.authentication ? "Available" : "Unavailable"}
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Session Management:</span>
                      <Badge variant={securityInfo.security_features.session_management ? "default" : "secondary"}>
                        {securityInfo.security_features.session_management ? "Active" : "Inactive"}
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}