#!/bin/bash
# Tektra AI Assistant - Scaling Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ACTION=${1:-status}
DEPLOYMENT_TYPE=${2:-docker-compose}
SCALE_TARGET=${3:-3}

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

usage() {
    cat << EOF
Usage: $0 [ACTION] [DEPLOYMENT_TYPE] [SCALE_TARGET]

Actions:
  status      Show current scaling status [default]
  scale       Scale to specified number of replicas
  scale-up    Scale up by specified amount
  scale-down  Scale down by specified amount
  auto        Enable auto-scaling

Arguments:
  DEPLOYMENT_TYPE  Deployment method (docker-compose|kubernetes) [default: docker-compose]
  SCALE_TARGET     Target number of replicas or amount to scale [default: 3]

Examples:
  $0 status kubernetes
  $0 scale docker-compose 5
  $0 scale-up kubernetes 2
  $0 auto kubernetes

Environment variables:
  TEKTRA_NAMESPACE  Kubernetes namespace [default: tektra]
EOF
}

# Show current status
show_status() {
    log "📊 Current Scaling Status"
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            log "Docker Compose Services:"
            docker-compose ps --format "table {{.Name}}\t{{.State}}\t{{.Ports}}"
            ;;
        kubernetes)
            local namespace="${TEKTRA_NAMESPACE:-tektra}"
            
            log "Kubernetes Deployment Status:"
            kubectl get deployment tektra-app -n "${namespace}" -o wide
            
            log "Pod Status:"
            kubectl get pods -n "${namespace}" -l app.kubernetes.io/name=tektra -o wide
            
            log "HPA Status:"
            kubectl get hpa tektra-hpa -n "${namespace}" -o wide || warn "HPA not found"
            
            log "Resource Usage:"
            kubectl top pods -n "${namespace}" -l app.kubernetes.io/name=tektra || warn "Metrics not available"
            ;;
    esac
}

# Scale to specific number
scale_to() {
    local target=$1
    
    log "🎯 Scaling to ${target} replicas"
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            docker-compose up -d --scale tektra="${target}"
            ;;
        kubernetes)
            local namespace="${TEKTRA_NAMESPACE:-tektra}"
            kubectl scale deployment tektra-app --replicas="${target}" -n "${namespace}"
            
            # Wait for rollout
            kubectl rollout status deployment/tektra-app -n "${namespace}" --timeout=300s
            ;;
    esac
    
    success "✅ Scaled to ${target} replicas"
}

# Scale up by amount
scale_up() {
    local amount=$1
    
    log "📈 Scaling up by ${amount}"
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            # Get current count
            local current=$(docker-compose ps -q tektra | wc -l)
            local target=$((current + amount))
            scale_to "${target}"
            ;;
        kubernetes)
            local namespace="${TEKTRA_NAMESPACE:-tektra}"
            local current=$(kubectl get deployment tektra-app -n "${namespace}" -o jsonpath='{.spec.replicas}')
            local target=$((current + amount))
            scale_to "${target}"
            ;;
    esac
}

# Scale down by amount
scale_down() {
    local amount=$1
    
    log "📉 Scaling down by ${amount}"
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            local current=$(docker-compose ps -q tektra | wc -l)
            local target=$((current - amount))
            if [ "${target}" -lt 1 ]; then
                warn "Cannot scale below 1 replica, setting to 1"
                target=1
            fi
            scale_to "${target}"
            ;;
        kubernetes)
            local namespace="${TEKTRA_NAMESPACE:-tektra}"
            local current=$(kubectl get deployment tektra-app -n "${namespace}" -o jsonpath='{.spec.replicas}')
            local target=$((current - amount))
            if [ "${target}" -lt 1 ]; then
                warn "Cannot scale below 1 replica, setting to 1"
                target=1
            fi
            scale_to "${target}"
            ;;
    esac
}

# Enable auto-scaling
enable_autoscaling() {
    log "🤖 Enabling auto-scaling"
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            warn "Auto-scaling not supported with Docker Compose"
            warn "Consider using Docker Swarm or Kubernetes for auto-scaling"
            ;;
        kubernetes)
            local namespace="${TEKTRA_NAMESPACE:-tektra}"
            
            # Check if HPA exists
            if kubectl get hpa tektra-hpa -n "${namespace}" &>/dev/null; then
                log "HPA already exists, updating..."
                kubectl apply -f k8s/base/hpa.yaml
            else
                log "Creating HPA..."
                kubectl apply -f k8s/base/hpa.yaml
            fi
            
            # Show HPA status
            kubectl get hpa tektra-hpa -n "${namespace}" -o wide
            ;;
    esac
    
    success "✅ Auto-scaling enabled"
}

# Monitor scaling
monitor_scaling() {
    log "👀 Monitoring scaling (Press Ctrl+C to stop)"
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            while true; do
                clear
                echo "=== Tektra Scaling Monitor - Docker Compose ==="
                echo "Time: $(date)"
                echo
                docker-compose ps
                echo
                echo "Container Stats:"
                docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
                sleep 10
            done
            ;;
        kubernetes)
            local namespace="${TEKTRA_NAMESPACE:-tektra}"
            while true; do
                clear
                echo "=== Tektra Scaling Monitor - Kubernetes ==="
                echo "Time: $(date)"
                echo
                echo "Deployment Status:"
                kubectl get deployment tektra-app -n "${namespace}"
                echo
                echo "Pod Status:"
                kubectl get pods -n "${namespace}" -l app.kubernetes.io/name=tektra
                echo
                echo "HPA Status:"
                kubectl get hpa tektra-hpa -n "${namespace}" 2>/dev/null || echo "HPA not found"
                echo
                echo "Resource Usage:"
                kubectl top pods -n "${namespace}" -l app.kubernetes.io/name=tektra 2>/dev/null || echo "Metrics not available"
                sleep 10
            done
            ;;
    esac
}

# Load testing helper
load_test() {
    log "🔥 Starting load test"
    
    # Simple load test using curl
    local endpoint="http://localhost:8000/health"
    local duration=60
    local concurrency=10
    
    log "Load testing ${endpoint} for ${duration}s with ${concurrency} concurrent requests"
    
    # Check if hey is available
    if command -v hey &> /dev/null; then
        hey -z "${duration}s" -c "${concurrency}" "${endpoint}"
    elif command -v ab &> /dev/null; then
        # Apache Bench alternative
        local requests=$((duration * concurrency))
        ab -n "${requests}" -c "${concurrency}" "${endpoint}"
    else
        warn "No load testing tool found (hey or ab)"
        warn "Install hey: go install github.com/rakyll/hey@latest"
        warn "Or install Apache Bench: apt-get install apache2-utils"
    fi
}

# Performance recommendations
performance_recommendations() {
    log "💡 Performance Scaling Recommendations"
    
    case "${DEPLOYMENT_TYPE}" in
        docker-compose)
            echo "Docker Compose Recommendations:"
            echo "  - Monitor CPU and memory usage with 'docker stats'"
            echo "  - Scale horizontally: docker-compose up -d --scale tektra=N"
            echo "  - Consider resource limits in docker-compose.yml"
            echo "  - Use Docker Swarm for automatic scaling"
            ;;
        kubernetes)
            local namespace="${TEKTRA_NAMESPACE:-tektra}"
            
            echo "Kubernetes Recommendations:"
            echo "  - Current resource requests/limits:"
            kubectl get deployment tektra-app -n "${namespace}" -o jsonpath='{.spec.template.spec.containers[0].resources}' | jq '.'
            echo
            echo "  - Enable HPA for automatic scaling"
            echo "  - Monitor metrics with Prometheus/Grafana"
            echo "  - Consider VPA for resource optimization"
            echo "  - Use node affinity for multi-zone deployment"
            ;;
    esac
    
    echo
    echo "General Recommendations:"
    echo "  - Target 70% CPU utilization for optimal performance"
    echo "  - Scale based on agent queue length and response time"
    echo "  - Monitor memory usage patterns for agents"
    echo "  - Use caching to reduce load on AI models"
}

# Main function
main() {
    case "${ACTION}" in
        status)
            show_status
            ;;
        scale)
            if [[ -z "${SCALE_TARGET}" ]]; then
                error "Scale target required for scale action"
                usage
                exit 1
            fi
            scale_to "${SCALE_TARGET}"
            show_status
            ;;
        scale-up)
            scale_up "${SCALE_TARGET}"
            show_status
            ;;
        scale-down)
            scale_down "${SCALE_TARGET}"
            show_status
            ;;
        auto)
            enable_autoscaling
            show_status
            ;;
        monitor)
            monitor_scaling
            ;;
        load-test)
            load_test
            ;;
        recommendations)
            performance_recommendations
            ;;
        *)
            error "Unknown action: ${ACTION}"
            usage
            exit 1
            ;;
    esac
}

# Handle help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"