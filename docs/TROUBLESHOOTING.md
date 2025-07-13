# Tektra AI Assistant - Troubleshooting Guide

## Overview

This troubleshooting guide provides solutions for common issues, debugging techniques, and diagnostic procedures for the Tektra AI Assistant system.

## Quick Diagnostics

### System Health Check

```bash
# Check overall system health
curl http://localhost:8000/health

# Detailed health with metrics
curl http://localhost:8000/health/detailed

# Check specific component
curl http://localhost:8000/health/database
curl http://localhost:8000/health/models
curl http://localhost:8000/health/cache
```

### Service Status

```bash
# Docker deployment
docker-compose ps
docker-compose logs -f tektra

# Kubernetes deployment
kubectl get pods -n tektra
kubectl logs -f deployment/tektra-deployment -n tektra
kubectl describe pod <pod-name> -n tektra
```

### Resource Monitoring

```bash
# System resources
htop
free -h
df -h

# GPU resources (if applicable)
nvidia-smi
nvidia-smi -l 1

# Docker resources
docker stats
```

## Common Issues and Solutions

### 1. Service Won't Start

#### Symptoms
- Service fails to start
- Error messages in logs
- Health check returns 503

#### Diagnostic Commands
```bash
# Check logs for startup errors
docker-compose logs tektra

# Check configuration
cat .env
docker-compose config

# Check dependencies
docker-compose ps
```

#### Common Causes and Solutions

**Database Connection Issues**
```bash
# Check database connectivity
psql $DATABASE_URL -c "SELECT 1;"

# Reset database connection
docker-compose restart postgres
```

**Missing Environment Variables**
```bash
# Verify all required variables are set
env | grep TEKTRA

# Example fix
export DATABASE_URL="postgresql://user:pass@localhost:5432/tektra"
export SECRET_KEY="your_secret_key_here"
```

**Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000
lsof -i :8000

# Fix: Change port in docker-compose.yml
services:
  tektra:
    ports:
      - "8001:8000"  # Use different external port
```

### 2. Model Loading Failures

#### Symptoms
- Model initialization timeouts
- Out of memory errors
- Model download failures

#### Diagnostic Commands
```bash
# Check model directory
ls -la ./models/
du -sh ./models/

# Check available memory
free -h
nvidia-smi  # For GPU memory

# Monitor model loading
docker-compose logs -f tektra | grep "model"
```

#### Solutions

**Insufficient Memory**
```bash
# Reduce model memory usage
export MODEL_QUANTIZATION=int8
export MAX_MEMORY_GB=8

# Or switch to smaller model
export DEFAULT_MODEL="microsoft/DialoGPT-small"
```

**Model Download Issues**
```bash
# Check internet connectivity
curl -I https://huggingface.co

# Pre-download models
python -c "
from transformers import AutoTokenizer, AutoModel
model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
"

# Set custom cache directory
export HF_HOME=/path/to/large/storage/hf_cache
```

**GPU Issues**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Reset GPU state
sudo nvidia-smi --gpu-reset

# Force CPU usage if GPU issues persist
export ENABLE_GPU=false
```

### 3. High Memory Usage

#### Symptoms
- System becomes slow
- Out of memory errors
- Processes being killed (OOM)

#### Diagnostic Commands
```bash
# Monitor memory usage
watch -n 1 'free -h'

# Check process memory usage
ps aux --sort=-%mem | head -10

# Docker memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

#### Solutions

**Model Memory Optimization**
```python
# Optimize model configuration
MODEL_CONFIG = {
    "quantization": "int8",  # or "int4" for even less memory
    "torch_dtype": "float16",
    "device_map": "auto",
    "max_memory": {0: "8GB"},
    "low_cpu_mem_usage": True
}
```

**Cache Management**
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/

# Reduce cache size
export CACHE_SIZE_MB=512

# Enable cache cleanup
export ENABLE_CACHE_CLEANUP=true
export CACHE_CLEANUP_INTERVAL=3600  # 1 hour
```

**System Memory Management**
```bash
# Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Add to /etc/fstab for persistence
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 4. Performance Issues

#### Symptoms
- Slow response times
- High CPU usage
- Request timeouts

#### Diagnostic Commands
```bash
# Monitor performance metrics
curl http://localhost:8000/metrics

# CPU usage
top -p $(pgrep -f tektra)

# Network latency
ping api.tektra.ai
traceroute api.tektra.ai

# Disk I/O
iostat -x 1
```

#### Solutions

**CPU Optimization**
```bash
# Limit CPU usage
docker-compose exec tektra nice -n 10 python app.py

# Use multiple workers
export WORKERS=4
export THREADS_PER_WORKER=2
```

**Database Performance**
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Add indexes for common queries
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
```

**Caching Optimization**
```python
# Tune cache settings
CACHE_CONFIG = {
    "ttl": 3600,  # 1 hour
    "max_size": "2GB",
    "compression": True,
    "levels": {
        "L1": {"size": "512MB", "ttl": 900},   # 15 min
        "L2": {"size": "1GB", "ttl": 3600},    # 1 hour
        "L3": {"size": "2GB", "ttl": 14400}    # 4 hours
    }
}
```

### 5. Security Issues

#### Symptoms
- Authentication failures
- Permission denied errors
- Security violations in logs

#### Diagnostic Commands
```bash
# Check security events
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/v1/security/events

# Check permissions
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/v1/users/me/permissions

# View security logs
docker-compose logs tektra | grep "SECURITY"
```

#### Solutions

**Authentication Issues**
```bash
# Verify API key
export API_KEY="your_api_key"
curl -H "Authorization: Bearer $API_KEY" \
     http://localhost:8000/v1/health

# Regenerate API key
curl -X POST http://localhost:8000/v1/auth/api-keys \
     -H "Content-Type: application/json" \
     -d '{"name": "new-key", "permissions": ["agent.execute"]}'
```

**Permission Problems**
```python
# Check user permissions
from tektra.security import PermissionManager

pm = PermissionManager()
permissions = await pm.get_user_permissions("user_id")
print(permissions)

# Grant missing permission
await pm.grant_permission("user_id", Permission(
    resource_type="agent",
    resource_id="*",
    permission_level=PermissionLevel.EXECUTE
))
```

**Sandbox Failures**
```bash
# Check sandbox configuration
docker run --rm -it \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --read-only \
  --tmpfs /tmp:size=100M \
  tektra-sandbox:latest /bin/sh

# Reset sandbox environment
docker-compose down sandbox
docker-compose up -d sandbox
```

### 6. Database Issues

#### Symptoms
- Connection timeouts
- Slow queries
- Database errors in logs

#### Diagnostic Commands
```bash
# Check database status
docker-compose exec postgres pg_isready

# Monitor connections
docker-compose exec postgres psql -c "
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state;"

# Check database size
docker-compose exec postgres psql -c "
SELECT pg_size_pretty(pg_database_size('tektra'));"
```

#### Solutions

**Connection Pool Issues**
```python
# Tune connection pool
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600
}
```

**Query Performance**
```sql
-- Enable query statistics
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
-- Restart PostgreSQL

-- Analyze slow queries
EXPLAIN ANALYZE SELECT * FROM conversations WHERE user_id = 'user123';

-- Update table statistics
ANALYZE conversations;
ANALYZE messages;
```

**Database Maintenance**
```bash
# Vacuum and analyze
docker-compose exec postgres psql -c "VACUUM ANALYZE;"

# Reindex tables
docker-compose exec postgres psql -c "REINDEX DATABASE tektra;"

# Check for corruption
docker-compose exec postgres psql -c "
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public';"
```

### 7. Network Issues

#### Symptoms
- Connection timeouts
- DNS resolution failures
- Intermittent connectivity

#### Diagnostic Commands
```bash
# Test connectivity
curl -v http://localhost:8000/health

# Check DNS resolution
nslookup api.tektra.ai
dig api.tektra.ai

# Network configuration
ip addr show
route -n
```

#### Solutions

**Docker Network Issues**
```bash
# Recreate Docker networks
docker-compose down
docker network prune
docker-compose up -d

# Check network connectivity between containers
docker-compose exec tektra ping postgres
docker-compose exec tektra nslookup postgres
```

**Firewall Configuration**
```bash
# Check firewall rules
sudo ufw status
sudo iptables -L

# Allow required ports
sudo ufw allow 8000/tcp
sudo ufw allow 5432/tcp  # PostgreSQL
sudo ufw allow 6379/tcp  # Redis
```

**Load Balancer Issues**
```nginx
# Check Nginx configuration
nginx -t

# Reload configuration
nginx -s reload

# Check upstream health
curl http://upstream-server:8000/health
```

## Debugging Techniques

### 1. Log Analysis

#### Comprehensive Logging
```bash
# Set debug logging level
export LOG_LEVEL=DEBUG

# Monitor logs in real-time
tail -f /var/log/tektra/app.log

# Filter logs by component
grep "SecurityManager" /var/log/tektra/app.log

# Search for errors
grep -i "error\|exception\|failed" /var/log/tektra/app.log
```

#### Structured Log Queries
```bash
# Using jq for JSON logs
cat app.log | jq 'select(.level == "ERROR")'
cat app.log | jq 'select(.component == "AgentManager")'
cat app.log | jq 'select(.duration_ms > 1000)'
```

### 2. Performance Profiling

#### Application Profiling
```python
# Enable profiling
import cProfile
import pstats

def profile_endpoint():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute code to profile
    result = agent.execute(message)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

#### Memory Profiling
```python
# Memory usage tracking
import tracemalloc
import psutil

def track_memory():
    tracemalloc.start()
    
    # Execute memory-intensive operation
    model.load()
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()
```

### 3. Database Debugging

#### Query Analysis
```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s

-- Monitor active queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
```

#### Connection Debugging
```python
# Database connection testing
async def test_db_connection():
    try:
        async with database.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            print(f"Database connection successful: {result}")
    except Exception as e:
        print(f"Database connection failed: {e}")
```

### 4. API Debugging

#### Request/Response Logging
```python
# Middleware for request logging
async def debug_middleware(request, call_next):
    start_time = time.time()
    
    # Log request
    print(f"Request: {request.method} {request.url}")
    print(f"Headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    print(f"Response: {response.status_code} ({duration:.3f}s)")
    
    return response
```

#### API Testing
```bash
# Test API endpoints
curl -X GET http://localhost:8000/v1/agents \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -v

# Load testing
ab -n 100 -c 10 http://localhost:8000/health
```

## Monitoring and Alerting

### 1. Health Monitoring

#### Automated Health Checks
```bash
#!/bin/bash
# health_check.sh

HEALTH_URL="http://localhost:8000/health"
SLACK_WEBHOOK="https://hooks.slack.com/services/..."

check_health() {
    response=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)
    
    if [ "$response" != "200" ]; then
        echo "Health check failed: HTTP $response"
        
        # Send alert
        curl -X POST -H 'Content-type: application/json' \
             --data '{"text":"Tektra health check failed"}' \
             $SLACK_WEBHOOK
        
        return 1
    fi
    
    echo "Health check passed"
    return 0
}

# Run health check
check_health
```

### 2. Performance Monitoring

#### Custom Metrics
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
REQUEST_COUNT = Counter('tektra_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('tektra_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('tektra_active_connections', 'Active connections')

# Instrument code
def instrument_request(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        REQUEST_COUNT.labels(method='POST', endpoint='/agents').inc()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    
    return wrapper
```

### 3. Alert Configuration

#### Alertmanager Rules
```yaml
# prometheus/alerts.yml
groups:
- name: tektra.rules
  rules:
  - alert: TektraDown
    expr: up{job="tektra"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Tektra service is down"
      
  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      
  - alert: SlowRequests
    expr: histogram_quantile(0.95, tektra_request_duration_seconds) > 5
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "95% of requests are slower than 5 seconds"
```

## Recovery Procedures

### 1. Service Recovery

#### Automatic Restart
```bash
#!/bin/bash
# auto_restart.sh

SERVICE_NAME="tektra"
MAX_RESTARTS=3
RESTART_COUNT=0

restart_service() {
    echo "Restarting $SERVICE_NAME (attempt $((RESTART_COUNT + 1)))"
    docker-compose restart $SERVICE_NAME
    RESTART_COUNT=$((RESTART_COUNT + 1))
    
    sleep 30  # Wait for service to start
    
    # Check if service is healthy
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "Service restart successful"
        RESTART_COUNT=0
        return 0
    else
        echo "Service restart failed"
        return 1
    fi
}

# Monitor and restart if needed
while true; do
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
            restart_service
        else
            echo "Max restart attempts reached. Manual intervention required."
            # Send critical alert
            break
        fi
    fi
    
    sleep 60  # Check every minute
done
```

### 2. Database Recovery

#### Backup Restoration
```bash
# Restore from backup
docker-compose stop tektra
docker-compose exec postgres dropdb tektra
docker-compose exec postgres createdb tektra
docker-compose exec -T postgres psql tektra < backup.sql
docker-compose start tektra
```

#### Point-in-Time Recovery
```bash
# PostgreSQL PITR
pg_basebackup -D /backup/base -Ft -z -P
# Restore specific point in time
# Configure recovery.conf with target time
```

### 3. Data Migration

#### Model Migration
```python
# Migrate to new model
async def migrate_model(old_model: str, new_model: str):
    # Get all agents using old model
    agents = await agent_manager.get_agents_by_model(old_model)
    
    for agent in agents:
        # Create backup
        await backup_agent_config(agent)
        
        # Update model
        agent.config.model = new_model
        await agent_manager.update_agent(agent)
        
        # Verify functionality
        test_result = await test_agent(agent)
        if not test_result.success:
            # Rollback on failure
            await restore_agent_config(agent)
```

## Prevention Strategies

### 1. Proactive Monitoring

#### Early Warning System
```python
class EarlyWarningSystem:
    """Detect issues before they become critical."""
    
    def __init__(self):
        self.thresholds = {
            "memory_usage": 0.8,  # 80%
            "cpu_usage": 0.7,     # 70%
            "disk_usage": 0.85,   # 85%
            "response_time": 2.0,  # 2 seconds
            "error_rate": 0.05    # 5%
        }
    
    async def check_system_health(self):
        """Check system health against thresholds."""
        metrics = await self.collect_metrics()
        
        warnings = []
        for metric, value in metrics.items():
            if metric in self.thresholds:
                threshold = self.thresholds[metric]
                if value > threshold:
                    warnings.append(f"{metric} is {value:.2%} (threshold: {threshold:.2%})")
        
        if warnings:
            await self.send_warning(warnings)
```

### 2. Capacity Planning

#### Resource Prediction
```python
class CapacityPlanner:
    """Predict future resource needs."""
    
    def analyze_usage_trends(self, days: int = 30):
        """Analyze resource usage trends."""
        # Collect historical data
        data = self.get_historical_metrics(days)
        
        # Calculate trends
        memory_trend = self.calculate_trend(data['memory'])
        cpu_trend = self.calculate_trend(data['cpu'])
        
        # Predict future needs
        predicted_memory = self.predict_usage(memory_trend, days=30)
        predicted_cpu = self.predict_usage(cpu_trend, days=30)
        
        return {
            "memory_prediction": predicted_memory,
            "cpu_prediction": predicted_cpu,
            "scaling_recommendation": self.get_scaling_recommendation(
                predicted_memory, predicted_cpu
            )
        }
```

### 3. Automated Testing

#### Continuous Health Checks
```python
# Automated system tests
class SystemHealthTests:
    """Continuous system health validation."""
    
    async def run_health_tests(self):
        """Run comprehensive health tests."""
        tests = [
            self.test_api_endpoints,
            self.test_database_connectivity,
            self.test_model_loading,
            self.test_agent_creation,
            self.test_security_validation
        ]
        
        results = []
        for test in tests:
            try:
                result = await test()
                results.append({"test": test.__name__, "status": "pass"})
            except Exception as e:
                results.append({"test": test.__name__, "status": "fail", "error": str(e)})
        
        return results
```

Remember: Effective troubleshooting requires understanding the system architecture, having good observability, and following systematic diagnostic procedures. When in doubt, check logs first, verify configuration second, and test connectivity third.