-- Tektra AI Assistant - PostgreSQL Initialization Script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create tektra user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'tektra') THEN
        CREATE ROLE tektra LOGIN PASSWORD 'your_password_here';
    END IF;
END
$$;

-- Create tektra database if it doesn't exist
SELECT 'CREATE DATABASE tektra OWNER tektra'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'tektra')\gexec

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE tektra TO tektra;

-- Connect to tektra database
\c tektra;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS tektra_core AUTHORIZATION tektra;
CREATE SCHEMA IF NOT EXISTS tektra_agents AUTHORIZATION tektra;
CREATE SCHEMA IF NOT EXISTS tektra_security AUTHORIZATION tektra;
CREATE SCHEMA IF NOT EXISTS tektra_performance AUTHORIZATION tektra;
CREATE SCHEMA IF NOT EXISTS tektra_audit AUTHORIZATION tektra;

-- Grant schema permissions
GRANT ALL ON SCHEMA tektra_core TO tektra;
GRANT ALL ON SCHEMA tektra_agents TO tektra;
GRANT ALL ON SCHEMA tektra_security TO tektra;
GRANT ALL ON SCHEMA tektra_performance TO tektra;
GRANT ALL ON SCHEMA tektra_audit TO tektra;

-- Core system tables
CREATE TABLE IF NOT EXISTS tektra_core.system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    environment VARCHAR(50) NOT NULL DEFAULT 'production',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tektra_core.system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    system_id VARCHAR(255) NOT NULL,
    overall_status VARCHAR(50) NOT NULL,
    components JSONB NOT NULL,
    metrics JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX (system_id, timestamp)
);

-- Agent tables
CREATE TABLE IF NOT EXISTS tektra_agents.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) UNIQUE NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'inactive',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tektra_agents.agent_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    agent_id VARCHAR(255) NOT NULL REFERENCES tektra_agents.agents(agent_id),
    user_id VARCHAR(255),
    security_context JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    INDEX (agent_id, status),
    INDEX (user_id, status)
);

CREATE TABLE IF NOT EXISTS tektra_agents.agent_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    agent_id VARCHAR(255) NOT NULL REFERENCES tektra_agents.agents(agent_id),
    session_id VARCHAR(255) REFERENCES tektra_agents.agent_sessions(session_id),
    task_description TEXT NOT NULL,
    task_context JSONB,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    result JSONB,
    error_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    INDEX (agent_id, status),
    INDEX (session_id, status),
    INDEX (created_at DESC)
);

-- Security tables
CREATE TABLE IF NOT EXISTS tektra_security.security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    agent_id VARCHAR(255),
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    event_data JSONB NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX (event_type, timestamp),
    INDEX (severity, timestamp),
    INDEX (agent_id, timestamp),
    INDEX (user_id, timestamp)
);

CREATE TABLE IF NOT EXISTS tektra_security.permission_grants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    permission_id VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    granted_to VARCHAR(255) NOT NULL, -- agent_id, user_id, etc.
    permission_level VARCHAR(50) NOT NULL,
    conditions JSONB,
    granted_by VARCHAR(255),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    revoked_at TIMESTAMP WITH TIME ZONE,
    INDEX (granted_to, resource_type),
    INDEX (resource_type, resource_id)
);

CREATE TABLE IF NOT EXISTS tektra_security.tool_validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    validation_id VARCHAR(255) UNIQUE NOT NULL,
    tool_id VARCHAR(255) NOT NULL,
    tool_code_hash VARCHAR(255) NOT NULL,
    agent_id VARCHAR(255),
    validation_result JSONB NOT NULL,
    is_safe BOOLEAN NOT NULL,
    confidence_score FLOAT,
    validation_method VARCHAR(100) NOT NULL,
    validated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX (tool_id, is_safe),
    INDEX (agent_id, validated_at),
    INDEX (tool_code_hash)
);

-- Performance tables
CREATE TABLE IF NOT EXISTS tektra_performance.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_id VARCHAR(255) NOT NULL,
    component VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(50),
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX (component, metric_name, timestamp),
    INDEX (timestamp DESC)
);

CREATE TABLE IF NOT EXISTS tektra_performance.resource_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    usage_data JSONB NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX (resource_type, resource_id, timestamp)
);

-- Audit tables
CREATE TABLE IF NOT EXISTS tektra_audit.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    actor_type VARCHAR(100), -- user, agent, system
    actor_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    old_values JSONB,
    new_values JSONB,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX (entity_type, entity_id, timestamp),
    INDEX (actor_type, actor_id, timestamp),
    INDEX (event_type, timestamp)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON tektra_core.system_health(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_agents_status ON tektra_agents.agents(status);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON tektra_agents.agent_tasks(status);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON tektra_security.security_events(severity, timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_component ON tektra_performance.performance_metrics(component, timestamp);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_agents_updated_at 
    BEFORE UPDATE ON tektra_agents.agents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant all permissions to tektra user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tektra_core TO tektra;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tektra_agents TO tektra;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tektra_security TO tektra;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tektra_performance TO tektra;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA tektra_audit TO tektra;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA tektra_core TO tektra;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA tektra_agents TO tektra;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA tektra_security TO tektra;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA tektra_performance TO tektra;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA tektra_audit TO tektra;

-- Insert initial configuration
INSERT INTO tektra_core.system_config (config_key, config_value, environment) 
VALUES 
    ('system_initialized', '{"initialized": true, "version": "1.0.0"}', 'production'),
    ('default_security_level', '{"level": "medium"}', 'production'),
    ('performance_thresholds', '{"memory_warning": 0.8, "memory_critical": 0.9}', 'production')
ON CONFLICT (config_key) DO NOTHING;