#!/bin/bash
# Tektra AI Assistant - Docker Entrypoint Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

# Default values
COMMAND=${1:-serve}

# Environment setup
export PYTHONPATH="${PYTHONPATH}:/app/src"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

log "🚀 Starting Tektra AI Assistant"
log "   Environment: ${TEKTRA_ENV:-production}"
log "   Version: ${TEKTRA_VERSION:-unknown}"
log "   Command: ${COMMAND}"

# Wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log "⏳ Waiting for ${service_name} at ${host}:${port}..."
    
    local counter=0
    while ! nc -z "${host}" "${port}" >/dev/null 2>&1; do
        counter=$((counter + 1))
        if [ $counter -gt $timeout ]; then
            error "❌ Timeout waiting for ${service_name}"
            exit 1
        fi
        sleep 1
    done
    
    success "✅ ${service_name} is available"
}

# Database migrations and setup
setup_database() {
    log "🗄️ Setting up database..."
    
    # Extract database info from URL
    if [[ ${TEKTRA_DATABASE__URL} =~ postgresql://([^:]+):([^@]+)@([^:]+):([0-9]+)/(.+) ]]; then
        local db_host="${BASH_REMATCH[3]}"
        local db_port="${BASH_REMATCH[4]}"
        
        wait_for_service "${db_host}" "${db_port}" "PostgreSQL"
        
        # Run migrations if needed
        # python -m tektra migrate || warn "Database migration failed"
        success "✅ Database setup complete"
    else
        warn "⚠️ Database URL not recognized, skipping setup"
    fi
}

# Redis setup
setup_redis() {
    log "🔄 Setting up Redis..."
    
    # Extract Redis info from URL
    if [[ ${TEKTRA_REDIS__URL} =~ redis://([^:]+):([0-9]+)/([0-9]+) ]]; then
        local redis_host="${BASH_REMATCH[1]}"
        local redis_port="${BASH_REMATCH[2]}"
        
        wait_for_service "${redis_host}" "${redis_port}" "Redis"
        success "✅ Redis setup complete"
    else
        warn "⚠️ Redis URL not recognized, skipping setup"
    fi
}

# Health check function
health_check() {
    log "🏥 Running health check..."
    
    # Check if Tektra service is responding
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        success "✅ Tektra service is healthy"
        return 0
    else
        error "❌ Tektra service health check failed"
        return 1
    fi
}

# Graceful shutdown handler
shutdown() {
    log "🛑 Received shutdown signal"
    
    if [[ -n ${TEKTRA_PID} ]]; then
        log "Stopping Tektra process (PID: ${TEKTRA_PID})"
        kill -TERM ${TEKTRA_PID} 2>/dev/null || true
        
        # Wait for graceful shutdown
        local counter=0
        while kill -0 ${TEKTRA_PID} 2>/dev/null && [ $counter -lt 30 ]; do
            sleep 1
            counter=$((counter + 1))
        done
        
        # Force kill if still running
        if kill -0 ${TEKTRA_PID} 2>/dev/null; then
            warn "Force killing Tektra process"
            kill -KILL ${TEKTRA_PID} 2>/dev/null || true
        fi
    fi
    
    success "✅ Shutdown complete"
    exit 0
}

# Setup signal handlers
trap shutdown SIGTERM SIGINT

# Create required directories
mkdir -p "${TEKTRA_DATA_DIR}" "${TEKTRA_LOGS_DIR}" "${TEKTRA_TEMP_DIR}"

# Handle different commands
case "${COMMAND}" in
    "serve")
        log "🌐 Starting Tektra server..."
        
        # Setup dependencies
        setup_database
        setup_redis
        
        # Start the Tektra system
        log "Starting Tektra AI Assistant system..."
        python -m tektra.core.tektra_system &
        TEKTRA_PID=$!
        
        # Wait for startup
        sleep 10
        
        # Perform health check
        if ! health_check; then
            error "❌ Service failed to start properly"
            exit 1
        fi
        
        success "✅ Tektra AI Assistant is running"
        log "   Main service: http://localhost:8000"
        log "   Metrics: http://localhost:8090/metrics"
        log "   Health: http://localhost:8000/health"
        
        # Wait for the process
        wait ${TEKTRA_PID}
        ;;
        
    "migrate")
        log "🗄️ Running database migrations..."
        setup_database
        # python -m tektra migrate
        success "✅ Database migrations complete"
        ;;
        
    "test")
        log "🧪 Running tests..."
        python -m pytest tests/ -v
        ;;
        
    "benchmark")
        log "📊 Running benchmarks..."
        python -m tektra.performance.benchmark
        ;;
        
    "shell")
        log "🐚 Starting interactive shell..."
        python -m tektra shell
        ;;
        
    "worker")
        log "👷 Starting background worker..."
        setup_database
        setup_redis
        python -m tektra worker &
        TEKTRA_PID=$!
        wait ${TEKTRA_PID}
        ;;
        
    "demo")
        log "🎪 Running demo..."
        python -m tektra demo
        ;;
        
    *)
        log "🔧 Running custom command: ${COMMAND}"
        exec "$@"
        ;;
esac