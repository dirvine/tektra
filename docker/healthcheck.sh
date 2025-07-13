#!/bin/bash
# Tektra AI Assistant - Health Check Script

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

HEALTH_URL="${TEKTRA_HEALTH_URL:-http://localhost:8000/health}"
METRICS_URL="${TEKTRA_METRICS_URL:-http://localhost:8090/metrics}"
TIMEOUT=10

# Check main service health
check_main_service() {
    if curl -f -s --max-time ${TIMEOUT} "${HEALTH_URL}" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check metrics endpoint
check_metrics() {
    if curl -f -s --max-time ${TIMEOUT} "${METRICS_URL}" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check system resources
check_resources() {
    # Check memory usage (should be less than 90%)
    local memory_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
    if (( $(echo "${memory_usage} < 90" | bc -l) )); then
        return 0
    else
        echo "High memory usage: ${memory_usage}%" >&2
        return 1
    fi
}

# Check disk space
check_disk() {
    # Check if data directory has at least 1GB free
    local available=$(df "${TEKTRA_DATA_DIR:-/app/data}" | tail -1 | awk '{print $4}')
    if [ "${available}" -gt 1048576 ]; then  # 1GB in KB
        return 0
    else
        echo "Low disk space: ${available}KB available" >&2
        return 1
    fi
}

# Main health check
main() {
    local status=0
    
    # Check main service
    if ! check_main_service; then
        echo -e "${RED}❌ Main service health check failed${NC}" >&2
        status=1
    fi
    
    # Check metrics (non-critical)
    if ! check_metrics; then
        echo -e "${RED}⚠️ Metrics endpoint unavailable${NC}" >&2
        # Don't fail on metrics - it's not critical
    fi
    
    # Check resources
    if ! check_resources; then
        echo -e "${RED}❌ Resource check failed${NC}" >&2
        status=1
    fi
    
    # Check disk space
    if ! check_disk; then
        echo -e "${RED}❌ Disk space check failed${NC}" >&2
        status=1
    fi
    
    if [ ${status} -eq 0 ]; then
        echo -e "${GREEN}✅ All health checks passed${NC}"
    fi
    
    exit ${status}
}

main "$@"