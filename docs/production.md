# Production Deployment Guide

Deploy the Claude Code SDK Wrapper in production environments with enterprise-grade reliability and monitoring.

## Quick Production Setup

### 1. Basic Production Configuration

Create `config/production.json`:
```json
{
  "claude_binary": "claude",
  "timeout": 60.0,
  "max_retries": 5,
  "retry_delay": 2.0,
  "retry_backoff_factor": 2.0,
  "verbose": false,
  "enable_metrics": true,
  "log_level": 20,
  "system_prompt": "You are a professional AI assistant providing accurate information.",
  "environment_vars": {
    "ENVIRONMENT": "production",
    "SERVICE_NAME": "claude-wrapper"
  }
}
```

### 2. Production Service Script

Create `service.py`:
```python
#!/usr/bin/env python3
"""Production Claude Service"""

import os
import json
import logging
from pathlib import Path
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/claude-wrapper/service.log')
    ]
)

class ProductionClaudeService:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            config_data = json.load(f)

        self.config = ClaudeCodeConfig(**config_data)
        self.wrapper = ClaudeCodeWrapper(self.config)
        self.logger = logging.getLogger(__name__)

    def process_request(self, query: str) -> dict:
        try:
            response = self.wrapper.run(query)
            return {
                "success": True,
                "content": response.content,
                "session_id": response.session_id,
                "cost": response.metrics.cost_usd
            }
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

if __name__ == "__main__":
    service = ProductionClaudeService("config/production.json")
    # Your application logic here
```

### 3. Systemd Service (Linux)

Create `/etc/systemd/system/claude-wrapper.service`:
```ini
[Unit]
Description=Claude Code Wrapper Service
After=network.target

[Service]
Type=simple
User=claude-service
Group=claude-service
WorkingDirectory=/opt/claude-wrapper
ExecStart=/opt/claude-wrapper/.venv/bin/python service.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/claude-wrapper
Environment=CLAUDE_CONFIG_PATH=/opt/claude-wrapper/config/production.json

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=claude-wrapper

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/claude-wrapper /var/lib/claude-wrapper

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable claude-wrapper
sudo systemctl start claude-wrapper
sudo systemctl status claude-wrapper
```

## Docker Deployment

### Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash claude

# Set work directory
WORKDIR /app

# Install Claude Code CLI (replace with actual installation)
# This is a placeholder - replace with actual Claude Code installation
RUN echo "#!/bin/bash\necho 'Mock Claude Code - Replace with actual installation'" > /usr/local/bin/claude \
    && chmod +x /usr/local/bin/claude

# Copy application files
COPY claude_code_wrapper.py .
COPY cli_tool.py .
COPY service.py .
COPY config/ ./config/
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs /app/data \
    && chown -R claude:claude /app

# Switch to non-root user
USER claude

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python cli_tool.py health || exit 1

# Default command
CMD ["python", "service.py"]
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  claude-wrapper:
    build: .
    container_name: claude-wrapper
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - CLAUDE_CONFIG_PATH=/app/config/production.json
      - LOG_LEVEL=INFO
    networks:
      - claude-network
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:alpine
    container_name: claude-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - claude-network

  postgres:
    image: postgres:15-alpine
    container_name: claude-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: claude_db
      POSTGRES_USER: claude_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - claude-network

networks:
  claude-network:
    driver: bridge

volumes:
  redis_data:
  postgres_data:
```

### Environment File

Create `.env`:
```bash
POSTGRES_PASSWORD=your_secure_password
CLAUDE_CONFIG_PATH=/app/config/production.json
LOG_LEVEL=INFO
ENVIRONMENT=production
```

Deploy:
```bash
docker-compose up -d
docker-compose logs -f claude-wrapper
```

## Kubernetes Deployment

### ConfigMap

Create `k8s/configmap.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: claude-wrapper-config
  namespace: default
data:
  production.json: |
    {
      "claude_binary": "claude",
      "timeout": 60.0,
      "max_retries": 5,
      "enable_metrics": true,
      "log_level": 20,
      "system_prompt": "You are a professional AI assistant."
    }
```

### Deployment

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-wrapper
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: claude-wrapper
  template:
    metadata:
      labels:
        app: claude-wrapper
    spec:
      containers:
      - name: claude-wrapper
        image: your-registry/claude-wrapper:latest
        ports:
        - containerPort: 8080
        env:
        - name: CLAUDE_CONFIG_PATH
          value: "/app/config/production.json"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
        livenessProbe:
          exec:
            command:
            - python
            - cli_tool.py
            - health
          initialDelaySeconds: 30
          periodSeconds: 60
        readinessProbe:
          exec:
            command:
            - python
            - cli_tool.py
            - health
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: claude-wrapper-config
      - name: logs
        emptyDir: {}
```

### Service

Create `k8s/service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: claude-wrapper-service
  namespace: default
spec:
  selector:
    app: claude-wrapper
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

Deploy:
```bash
kubectl apply -f k8s/
kubectl get pods -l app=claude-wrapper
kubectl logs deployment/claude-wrapper
```

## Monitoring and Observability

### Health Monitoring

Create `monitoring/health_check.py`:
```python
#!/usr/bin/env python3
"""Health monitoring script"""

import requests
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_health():
    """Check service health and log results."""
    try:
        # Using CLI health check
        import subprocess
        result = subprocess.run(
            ["python", "cli_tool.py", "health"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            logger.info("Health check passed")
            return True
        else:
            logger.error(f"Health check failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False

def main():
    """Main monitoring loop."""
    while True:
        timestamp = datetime.now().isoformat()
        is_healthy = check_health()

        # Log to file for external monitoring
        with open("/var/log/claude-wrapper/health.log", "a") as f:
            f.write(f"{timestamp},{is_healthy}\n")

        if not is_healthy:
            # Send alert (implement your alerting logic)
            logger.critical("Service unhealthy - alerting required")

        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
```

### Prometheus Metrics

Create `monitoring/metrics.py`:
```python
"""Prometheus metrics integration"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import threading
from claude_code_wrapper import ClaudeCodeWrapper

# Metrics
request_total = Counter('claude_requests_total', 'Total requests', ['status'])
request_duration = Histogram('claude_request_duration_seconds', 'Request duration')
active_sessions = Gauge('claude_active_sessions', 'Active sessions')
error_rate = Gauge('claude_error_rate', 'Error rate')

class MetricsWrapper:
    """Wrapper with Prometheus metrics."""

    def __init__(self, wrapper: ClaudeCodeWrapper):
        self.wrapper = wrapper
        self.recent_requests = []

    def run(self, query: str, **kwargs):
        """Run with metrics collection."""
        start_time = time.time()

        try:
            with request_duration.time():
                response = self.wrapper.run(query, **kwargs)

            # Record metrics
            status = "error" if response.is_error else "success"
            request_total.labels(status=status).inc()

            # Update recent requests for error rate calculation
            self.recent_requests.append({
                'timestamp': time.time(),
                'error': response.is_error
            })

            # Keep only last 100 requests
            self.recent_requests = self.recent_requests[-100:]

            # Update error rate
            recent_errors = sum(1 for r in self.recent_requests if r['error'])
            error_rate.set(recent_errors / len(self.recent_requests))

            return response

        except Exception as e:
            request_total.labels(status="exception").inc()
            raise

# Start metrics server
start_http_server(8000)
```

### Grafana Dashboard

Create `monitoring/grafana-dashboard.json`:
```json
{
  "dashboard": {
    "title": "Claude Wrapper Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(claude_requests_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(claude_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(claude_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "claude_error_rate"
          }
        ]
      }
    ]
  }
}
```

## Load Balancing and Scaling

### NGINX Load Balancer

Create `nginx/nginx.conf`:
```nginx
upstream claude_wrapper {
    least_conn;
    server claude-wrapper-1:8080 max_fails=3 fail_timeout=30s;
    server claude-wrapper-2:8080 max_fails=3 fail_timeout=30s;
    server claude-wrapper-3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name claude-wrapper.yourdomain.com;

    location / {
        proxy_pass http://claude_wrapper;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        access_log off;
        proxy_pass http://claude_wrapper/health;
    }
}
```

### Horizontal Pod Autoscaler (K8s)

Create `k8s/hpa.yaml`:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: claude-wrapper-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: claude-wrapper
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Security

### Security Configuration

```python
# security_config.py
from claude_code_wrapper import ClaudeCodeConfig
from pathlib import Path

# Security-hardened configuration
SECURE_CONFIG = ClaudeCodeConfig(
    # Restrict tools to safe operations only
    allowed_tools=[
        "Python(import,def,class,print,len,str,int,float,list,dict)",
        "Bash(ls,cat,grep,head,tail,wc,find)",
        "mcp__filesystem__read",
        "mcp__database__query"
    ],

    # Explicitly disallow dangerous operations
    disallowed_tools=[
        "Bash(rm,del,sudo,chmod,chown,mv,cp)",
        "Python(exec,eval,__import__,open,file)",
        "mcp__filesystem__write",
        "mcp__filesystem__delete",
        "mcp__database__write"
    ],

    # Secure execution environment
    working_directory=Path("/app/secure_workspace"),
    environment_vars={
        "SECURITY_MODE": "strict",
        "PYTHONPATH": "/app/secure_libs",
        "PATH": "/usr/local/bin:/usr/bin:/bin"
    },

    # Conservative timeouts
    timeout=30.0,
    max_turns=5,
    max_retries=3
)
```

### Network Security

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: claude-wrapper-netpol
spec:
  podSelector:
    matchLabels:
      app: claude-wrapper
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

## Backup and Recovery

### Configuration Backup

```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/backup/claude-wrapper/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r config/ "$BACKUP_DIR/"

# Backup logs (last 7 days)
find /var/log/claude-wrapper -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;

# Backup metrics data
if [ -d "/var/lib/claude-wrapper" ]; then
    cp -r /var/lib/claude-wrapper "$BACKUP_DIR/"
fi

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

### Disaster Recovery

```python
#!/usr/bin/env python3
"""Disaster recovery script"""

import os
import shutil
import subprocess
from pathlib import Path

class DisasterRecovery:
    def __init__(self, backup_path: str):
        self.backup_path = Path(backup_path)

    def restore_config(self):
        """Restore configuration from backup."""
        config_backup = self.backup_path / "config"
        if config_backup.exists():
            shutil.copytree(config_backup, "config", dirs_exist_ok=True)
            print("Configuration restored")

    def verify_service(self):
        """Verify service is working after recovery."""
        try:
            result = subprocess.run(
                ["python", "cli_tool.py", "health"],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except:
            return False

    def full_recovery(self):
        """Perform full disaster recovery."""
        print("Starting disaster recovery...")

        # Stop service
        subprocess.run(["systemctl", "stop", "claude-wrapper"])

        # Restore configuration
        self.restore_config()

        # Start service
        subprocess.run(["systemctl", "start", "claude-wrapper"])

        # Verify
        if self.verify_service():
            print("Recovery successful")
        else:
            print("Recovery failed - manual intervention required")

if __name__ == "__main__":
    recovery = DisasterRecovery("/backup/claude-wrapper/latest")
    recovery.full_recovery()
```

## Performance Optimization

### Connection Pooling

```python
# connection_pool.py
import queue
import threading
from claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig

class WrapperPool:
    """Connection pool for Claude wrappers."""

    def __init__(self, config: ClaudeCodeConfig, pool_size: int = 5):
        self.pool = queue.Queue(maxsize=pool_size)
        self.config = config

        # Initialize pool
        for _ in range(pool_size):
            wrapper = ClaudeCodeWrapper(config)
            self.pool.put(wrapper)

    def get_wrapper(self):
        """Get wrapper from pool."""
        return self.pool.get()

    def return_wrapper(self, wrapper):
        """Return wrapper to pool."""
        self.pool.put(wrapper)

    def execute(self, query: str, **kwargs):
        """Execute query using pooled wrapper."""
        wrapper = self.get_wrapper()
        try:
            return wrapper.run(query, **kwargs)
        finally:
            self.return_wrapper(wrapper)

# Usage
pool = WrapperPool(ClaudeCodeConfig(), pool_size=10)
response = pool.execute("What is machine learning?")
```

### Caching

```python
# cache.py
import redis
import json
import hashlib
from typing import Optional

class ResponseCache:
    """Redis-based response caching."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour

    def _make_key(self, query: str, config_hash: str) -> str:
        """Create cache key from query and config."""
        content = f"{query}:{config_hash}"
        return f"claude:{hashlib.md5(content.encode()).hexdigest()}"

    def get(self, query: str, config_hash: str) -> Optional[dict]:
        """Get cached response."""
        key = self._make_key(query, config_hash)
        cached = self.redis.get(key)
        return json.loads(cached) if cached else None

    def set(self, query: str, config_hash: str, response: dict):
        """Cache response."""
        key = self._make_key(query, config_hash)
        self.redis.setex(key, self.ttl, json.dumps(response))

# Usage with wrapper
cache = ResponseCache()

def cached_query(query: str):
    config_hash = "default"  # Compute based on actual config

    # Check cache first
    cached = cache.get(query, config_hash)
    if cached:
        return cached

    # Execute query
    wrapper = ClaudeCodeWrapper()
    response = wrapper.run(query)

    # Cache result
    result = {
        "content": response.content,
        "session_id": response.session_id,
        "cost": response.metrics.cost_usd
    }
    cache.set(query, config_hash, result)

    return result
```

This production deployment guide provides comprehensive coverage of deploying the Claude Code SDK Wrapper in enterprise environments with proper monitoring, security, and scalability considerations.
