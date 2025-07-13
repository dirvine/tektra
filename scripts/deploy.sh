#!/bin/bash
# Tektra AI Assistant - Deployment Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT=${1:-development}
DEPLOYMENT_TYPE=${2:-docker-compose}
BUILD_IMAGE=${3:-true}
PUSH_IMAGE=${4:-false}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
IMAGE_NAME="tektra"
IMAGE_TAG="${TEKTRA_VERSION:-1.0.0}"
REGISTRY="${DOCKER_REGISTRY:-}"

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

# Usage information
usage() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [DEPLOYMENT_TYPE] [BUILD_IMAGE] [PUSH_IMAGE]

Arguments:
  ENVIRONMENT      Target environment (development|staging|production) [default: development]
  DEPLOYMENT_TYPE  Deployment method (docker-compose|kubernetes) [default: docker-compose]
  BUILD_IMAGE      Whether to build Docker image (true|false) [default: true]
  PUSH_IMAGE       Whether to push to registry (true|false) [default: false]

Examples:
  $0 development docker-compose
  $0 production kubernetes true true
  $0 staging docker-compose false false

Environment variables:
  TEKTRA_VERSION   Version tag for the image [default: 1.0.0]
  DOCKER_REGISTRY  Docker registry URL for pushing images
  POSTGRES_PASSWORD Database password
  REDIS_PASSWORD   Redis password
  JWT_SECRET       JWT signing secret
  SECRET_KEY       Application secret key
EOF
}

# Validate environment
validate_environment() {
    case "${ENVIRONMENT}" in
        development|staging|production)
            log "✅ Environment: ${ENVIRONMENT}"
            ;;
        *)
            error "❌ Invalid environment: ${ENVIRONMENT}"
            echo "Valid options: development, staging, production"
            exit 1
            ;;
    esac
}

# Validate deployment type
validate_deployment_type() {
    case "${DEPLOYMENT_TYPE}" in
        docker-compose|kubernetes)
            log "✅ Deployment type: ${DEPLOYMENT_TYPE}"
            ;;
        *)
            error "❌ Invalid deployment type: ${DEPLOYMENT_TYPE}"
            echo "Valid options: docker-compose, kubernetes"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "❌ Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose for docker-compose deployment
    if [[ "${DEPLOYMENT_TYPE}" == "docker-compose" ]] && ! command -v docker-compose &> /dev/null; then
        error "❌ Docker Compose is not installed"
        exit 1
    fi
    
    # Check kubectl for kubernetes deployment
    if [[ "${DEPLOYMENT_TYPE}" == "kubernetes" ]] && ! command -v kubectl &> /dev/null; then
        error "❌ kubectl is not installed"
        exit 1
    fi
    
    # Check required environment variables for production
    if [[ "${ENVIRONMENT}" == "production" ]]; then
        local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "JWT_SECRET" "SECRET_KEY")
        for var in "${required_vars[@]}"; do
            if [[ -z "${!var}" ]]; then
                error "❌ Required environment variable ${var} is not set for production"
                exit 1
            fi
        done
    fi
    
    success "✅ Prerequisites check passed"
}

# Generate secrets
generate_secrets() {
    log "🔐 Generating secrets..."
    
    local env_file="${PROJECT_ROOT}/.env"
    
    if [[ ! -f "${env_file}" ]]; then
        log "Creating .env file from template..."
        cp "${PROJECT_ROOT}/.env.example" "${env_file}"
    fi
    
    # Generate secrets if not set
    if [[ -z "${JWT_SECRET}" ]]; then
        export JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        sed -i.bak "s/JWT_SECRET=.*/JWT_SECRET=${JWT_SECRET}/" "${env_file}"
    fi
    
    if [[ -z "${SECRET_KEY}" ]]; then
        export SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        sed -i.bak "s/SECRET_KEY=.*/SECRET_KEY=${SECRET_KEY}/" "${env_file}"
    fi
    
    if [[ -z "${POSTGRES_PASSWORD}" ]]; then
        export POSTGRES_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")
        sed -i.bak "s/POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=${POSTGRES_PASSWORD}/" "${env_file}"
    fi
    
    if [[ -z "${REDIS_PASSWORD}" ]]; then
        export REDIS_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(16))")
        sed -i.bak "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=${REDIS_PASSWORD}/" "${env_file}"
    fi
    
    success "✅ Secrets generated"
}

# Build Docker image
build_image() {
    if [[ "${BUILD_IMAGE}" != "true" ]]; then
        log "⏭️ Skipping image build"
        return
    fi
    
    log "🏗️ Building Docker image..."
    
    cd "${PROJECT_ROOT}"
    
    # Build arguments
    local build_args=(
        "--build-arg" "TEKTRA_VERSION=${IMAGE_TAG}"
        "--build-arg" "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        "--build-arg" "VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
        "--tag" "${IMAGE_NAME}:${IMAGE_TAG}"
        "--tag" "${IMAGE_NAME}:latest"
    )
    
    # Add registry prefix if specified
    if [[ -n "${REGISTRY}" ]]; then
        build_args+=(
            "--tag" "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
            "--tag" "${REGISTRY}/${IMAGE_NAME}:latest"
        )
    fi
    
    if ! docker build "${build_args[@]}" .; then
        error "❌ Docker build failed"
        exit 1
    fi
    
    success "✅ Docker image built: ${IMAGE_NAME}:${IMAGE_TAG}"
}

# Push Docker image
push_image() {
    if [[ "${PUSH_IMAGE}" != "true" ]]; then
        log "⏭️ Skipping image push"
        return
    fi
    
    if [[ -z "${REGISTRY}" ]]; then
        warn "⚠️ No registry specified, skipping push"
        return
    fi
    
    log "📤 Pushing Docker image to registry..."
    
    # Push both tags
    docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    
    success "✅ Docker image pushed to ${REGISTRY}"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log "🚀 Deploying with Docker Compose..."
    
    cd "${PROJECT_ROOT}"
    
    # Choose compose file based on environment
    local compose_file="docker-compose.yml"
    local compose_args=("-f" "${compose_file}")
    
    if [[ -f "docker-compose.${ENVIRONMENT}.yml" ]]; then
        compose_args+=("-f" "docker-compose.${ENVIRONMENT}.yml")
    fi
    
    # Set environment variables
    export TEKTRA_ENV="${ENVIRONMENT}"
    export TEKTRA_VERSION="${IMAGE_TAG}"
    
    # Stop existing services
    log "Stopping existing services..."
    docker-compose "${compose_args[@]}" down --remove-orphans || true
    
    # Start services
    log "Starting services..."
    docker-compose "${compose_args[@]}" up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    local timeout=120
    local counter=0
    
    while ! docker-compose "${compose_args[@]}" ps | grep -q "healthy" && [ $counter -lt $timeout ]; do
        sleep 2
        counter=$((counter + 2))
        echo -n "."
    done
    echo
    
    if [ $counter -ge $timeout ]; then
        error "❌ Services failed to become healthy within ${timeout} seconds"
        docker-compose "${compose_args[@]}" logs --tail=50
        exit 1
    fi
    
    success "✅ Docker Compose deployment complete"
    
    # Show status
    docker-compose "${compose_args[@]}" ps
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log "☸️ Deploying to Kubernetes..."
    
    cd "${PROJECT_ROOT}"
    
    # Apply base manifests
    kubectl apply -f k8s/base/
    
    # Apply environment-specific overlays if they exist
    if [[ -d "k8s/overlays/${ENVIRONMENT}" ]]; then
        kubectl apply -k "k8s/overlays/${ENVIRONMENT}"
    fi
    
    # Wait for deployment to be ready
    log "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/tektra-app -n tektra
    
    # Check pod status
    kubectl get pods -n tektra -l app.kubernetes.io/name=tektra
    
    success "✅ Kubernetes deployment complete"
}

# Post-deployment verification
verify_deployment() {
    log "🔍 Verifying deployment..."
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            # Check if Tektra service is responding
            local health_url="http://localhost:8000/health"
            ;;
        kubernetes)
            # Port forward to check service
            kubectl port-forward -n tektra svc/tektra-service 8000:8000 &
            local port_forward_pid=$!
            sleep 5
            local health_url="http://localhost:8000/health"
            ;;
    esac
    
    # Health check
    local attempts=0
    local max_attempts=30
    
    while [ $attempts -lt $max_attempts ]; do
        if curl -f -s "${health_url}" > /dev/null; then
            success "✅ Tektra service is responding"
            break
        fi
        
        attempts=$((attempts + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempts -eq $max_attempts ]; then
        error "❌ Tektra service is not responding after verification"
        exit 1
    fi
    
    # Clean up port forward if kubernetes
    if [[ "${DEPLOYMENT_TYPE}" == "kubernetes" ]]; then
        kill ${port_forward_pid} 2>/dev/null || true
    fi
    
    success "✅ Deployment verification complete"
}

# Main deployment function
main() {
    log "🌟 Starting Tektra AI Assistant Deployment"
    log "   Environment: ${ENVIRONMENT}"
    log "   Deployment Type: ${DEPLOYMENT_TYPE}"
    log "   Image Tag: ${IMAGE_TAG}"
    
    validate_environment
    validate_deployment_type
    check_prerequisites
    generate_secrets
    build_image
    push_image
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            deploy_docker_compose
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
    esac
    
    verify_deployment
    
    success "🎉 Tektra AI Assistant deployment complete!"
    log "   Access the application at:"
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            log "   - Main service: http://localhost:8000"
            log "   - Metrics: http://localhost:8090/metrics"
            log "   - Grafana: http://localhost:3000"
            log "   - Prometheus: http://localhost:9090"
            ;;
        kubernetes)
            log "   - Use 'kubectl port-forward -n tektra svc/tektra-service 8000:8000' to access locally"
            log "   - Or configure ingress for external access"
            ;;
    esac
}

# Handle command line arguments
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"