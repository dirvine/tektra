#!/usr/bin/env python3
"""
Tektra AI Assistant - Production Deployment Tests

Comprehensive testing of production deployment infrastructure including
Docker, Kubernetes, monitoring, and operational readiness.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with pytest,pytest-asyncio,docker,kubernetes,requests python -m pytest test_production_deployment.py -v
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest>=7.4.0",
#     "pytest-asyncio>=0.21.0",
#     "docker>=6.0.0",
#     "kubernetes>=27.0.0",
#     "requests>=2.31.0",
#     "pyyaml>=6.0.0",
#     "loguru>=0.7.0",
# ]
# ///

import asyncio
import json
import os
import subprocess
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest
import requests

from loguru import logger


class TestDockerDeployment:
    """Test Docker-based deployment infrastructure."""
    
    @pytest.fixture(scope="class")
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture(scope="class")
    def docker_compose_file(self, project_root):
        """Get docker-compose file path."""
        return project_root / "docker-compose.yml"
    
    def test_docker_compose_file_exists(self, docker_compose_file):
        """Test that docker-compose.yml exists and is valid."""
        logger.info("🧪 Testing Docker Compose file existence and validity")
        
        assert docker_compose_file.exists(), "docker-compose.yml not found"
        
        # Validate YAML syntax
        with open(docker_compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        assert 'services' in compose_config
        assert 'tektra' in compose_config['services']
        assert 'postgres' in compose_config['services']
        assert 'redis' in compose_config['services']
        
        logger.info("✅ Docker Compose file is valid")
    
    def test_dockerfile_exists_and_valid(self, project_root):
        """Test that Dockerfile exists and has required instructions."""
        logger.info("🧪 Testing Dockerfile existence and validity")
        
        dockerfile = project_root / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"
        
        with open(dockerfile, 'r') as f:
            content = f.read()
        
        # Check for required instructions
        required_instructions = ['FROM', 'WORKDIR', 'COPY', 'RUN', 'EXPOSE', 'CMD']
        for instruction in required_instructions:
            assert instruction in content, f"Missing {instruction} instruction in Dockerfile"
        
        # Check for security best practices
        assert 'USER' in content, "Dockerfile should specify non-root user"
        assert 'HEALTHCHECK' in content, "Dockerfile should include health check"
        
        logger.info("✅ Dockerfile is valid and follows best practices")
    
    def test_environment_configuration(self, project_root):
        """Test environment configuration files."""
        logger.info("🧪 Testing environment configuration")
        
        env_example = project_root / ".env.example"
        assert env_example.exists(), ".env.example not found"
        
        with open(env_example, 'r') as f:
            env_content = f.read()
        
        # Check for required environment variables
        required_vars = [
            'TEKTRA_ENV',
            'POSTGRES_PASSWORD',
            'REDIS_PASSWORD',
            'JWT_SECRET',
            'SECRET_KEY'
        ]
        
        for var in required_vars:
            assert var in env_content, f"Missing required environment variable: {var}"
        
        logger.info("✅ Environment configuration is complete")
    
    def test_docker_scripts_executable(self, project_root):
        """Test that Docker-related scripts are executable."""
        logger.info("🧪 Testing Docker scripts")
        
        scripts = [
            project_root / "docker" / "entrypoint.sh",
            project_root / "docker" / "healthcheck.sh",
            project_root / "scripts" / "deploy.sh",
            project_root / "scripts" / "scale.sh"
        ]
        
        for script in scripts:
            if script.exists():
                assert os.access(script, os.X_OK), f"Script {script} is not executable"
                logger.info(f"✅ Script {script.name} is executable")
    
    @pytest.mark.skipif(not os.environ.get('DOCKER_AVAILABLE'), 
                       reason="Docker not available in test environment")
    def test_docker_build(self, project_root):
        """Test Docker image build process."""
        logger.info("🧪 Testing Docker image build")
        
        # Build the Docker image
        result = subprocess.run([
            'docker', 'build', 
            '-t', 'tektra:test',
            '--build-arg', 'TEKTRA_VERSION=test',
            str(project_root)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"
        logger.info("✅ Docker image built successfully")
        
        # Test image properties
        result = subprocess.run([
            'docker', 'inspect', 'tektra:test'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, "Failed to inspect Docker image"
        
        image_info = json.loads(result.stdout)[0]
        config = image_info['Config']
        
        # Verify exposed ports
        assert '8000/tcp' in config.get('ExposedPorts', {}), "Port 8000 not exposed"
        assert '8090/tcp' in config.get('ExposedPorts', {}), "Port 8090 not exposed"
        
        # Verify non-root user
        assert config.get('User') != '', "Image should run as non-root user"
        
        logger.info("✅ Docker image configuration is correct")


class TestKubernetesDeployment:
    """Test Kubernetes deployment manifests."""
    
    @pytest.fixture(scope="class")
    def k8s_manifests_dir(self):
        """Get Kubernetes manifests directory."""
        return Path(__file__).parent.parent.parent / "k8s"
    
    def test_kubernetes_manifests_exist(self, k8s_manifests_dir):
        """Test that required Kubernetes manifests exist."""
        logger.info("🧪 Testing Kubernetes manifests existence")
        
        required_manifests = [
            "base/namespace.yaml",
            "base/deployment.yaml", 
            "base/service.yaml",
            "base/hpa.yaml"
        ]
        
        for manifest in required_manifests:
            manifest_path = k8s_manifests_dir / manifest
            assert manifest_path.exists(), f"Kubernetes manifest {manifest} not found"
        
        logger.info("✅ All required Kubernetes manifests exist")
    
    def test_kubernetes_manifests_valid_yaml(self, k8s_manifests_dir):
        """Test that Kubernetes manifests are valid YAML."""
        logger.info("🧪 Testing Kubernetes manifests YAML validity")
        
        manifest_files = list((k8s_manifests_dir / "base").glob("*.yaml"))
        
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                try:
                    yaml.safe_load(f)
                    logger.info(f"✅ {manifest_file.name} is valid YAML")
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {manifest_file}: {e}")
    
    def test_deployment_configuration(self, k8s_manifests_dir):
        """Test Kubernetes deployment configuration."""
        logger.info("🧪 Testing Kubernetes deployment configuration")
        
        deployment_file = k8s_manifests_dir / "base" / "deployment.yaml"
        with open(deployment_file, 'r') as f:
            deployment = yaml.safe_load(f)
        
        # Verify deployment structure
        assert deployment['kind'] == 'Deployment'
        assert deployment['metadata']['name'] == 'tektra-app'
        
        spec = deployment['spec']
        assert spec['replicas'] >= 3, "Should have at least 3 replicas for HA"
        
        # Verify container configuration
        container = spec['template']['spec']['containers'][0]
        assert container['name'] == 'tektra'
        
        # Verify resource limits
        resources = container.get('resources', {})
        assert 'requests' in resources, "Should specify resource requests"
        assert 'limits' in resources, "Should specify resource limits"
        
        # Verify health checks
        assert 'livenessProbe' in container, "Should have liveness probe"
        assert 'readinessProbe' in container, "Should have readiness probe"
        
        # Verify security context
        security_context = spec['template']['spec'].get('securityContext', {})
        assert security_context.get('runAsNonRoot') == True, "Should run as non-root"
        
        logger.info("✅ Kubernetes deployment configuration is correct")
    
    def test_hpa_configuration(self, k8s_manifests_dir):
        """Test Horizontal Pod Autoscaler configuration."""
        logger.info("🧪 Testing HPA configuration")
        
        hpa_file = k8s_manifests_dir / "base" / "hpa.yaml"
        with open(hpa_file, 'r') as f:
            hpa = yaml.safe_load(f)
        
        assert hpa['kind'] == 'HorizontalPodAutoscaler'
        
        spec = hpa['spec']
        assert spec['minReplicas'] >= 3, "Minimum replicas should be at least 3"
        assert spec['maxReplicas'] >= 20, "Maximum replicas should allow scaling"
        
        # Verify metrics
        metrics = spec['metrics']
        metric_types = [metric['type'] for metric in metrics]
        assert 'Resource' in metric_types, "Should have resource-based metrics"
        
        logger.info("✅ HPA configuration is correct")
    
    @pytest.mark.skipif(not os.environ.get('KUBECTL_AVAILABLE'), 
                       reason="kubectl not available in test environment")
    def test_kubernetes_manifest_validation(self, k8s_manifests_dir):
        """Test Kubernetes manifest validation with kubectl."""
        logger.info("🧪 Testing Kubernetes manifest validation with kubectl")
        
        manifest_files = list((k8s_manifests_dir / "base").glob("*.yaml"))
        
        for manifest_file in manifest_files:
            result = subprocess.run([
                'kubectl', 'apply', '--dry-run=client', '-f', str(manifest_file)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"kubectl validation failed for {manifest_file}: {result.stderr}"
            logger.info(f"✅ {manifest_file.name} passed kubectl validation")


class TestMonitoringAndObservability:
    """Test monitoring and observability configuration."""
    
    @pytest.fixture(scope="class")
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_prometheus_configuration(self, project_root):
        """Test Prometheus configuration."""
        logger.info("🧪 Testing Prometheus configuration")
        
        prometheus_config = project_root / "docker" / "prometheus" / "prometheus.yml"
        assert prometheus_config.exists(), "Prometheus configuration not found"
        
        with open(prometheus_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify global configuration
        assert 'global' in config
        assert config['global']['scrape_interval'] == '15s'
        
        # Verify scrape configs
        scrape_configs = config['scrape_configs']
        job_names = [job['job_name'] for job in scrape_configs]
        assert 'tektra-app' in job_names, "Should scrape Tektra application"
        assert 'prometheus' in job_names, "Should scrape Prometheus itself"
        
        logger.info("✅ Prometheus configuration is correct")
    
    def test_grafana_provisioning(self, project_root):
        """Test Grafana provisioning configuration."""
        logger.info("🧪 Testing Grafana provisioning")
        
        grafana_dir = project_root / "docker" / "grafana"
        if grafana_dir.exists():
            # Check for provisioning directories
            provisioning_dirs = ['provisioning/datasources', 'provisioning/dashboards']
            for dir_name in provisioning_dirs:
                dir_path = grafana_dir / dir_name
                if dir_path.exists():
                    logger.info(f"✅ Grafana {dir_name} directory exists")
    
    def test_nginx_configuration(self, project_root):
        """Test Nginx reverse proxy configuration."""
        logger.info("🧪 Testing Nginx configuration")
        
        nginx_config = project_root / "docker" / "nginx" / "nginx.conf"
        assert nginx_config.exists(), "Nginx configuration not found"
        
        with open(nginx_config, 'r') as f:
            config_content = f.read()
        
        # Check for essential configurations
        assert 'upstream tektra_app' in config_content, "Should define Tektra upstream"
        assert 'gzip on' in config_content, "Should enable gzip compression"
        assert 'limit_req_zone' in config_content, "Should have rate limiting"
        
        # Check security headers
        security_headers = ['X-Frame-Options', 'X-Content-Type-Options', 'X-XSS-Protection']
        for header in security_headers:
            assert header in config_content, f"Should include {header} security header"
        
        logger.info("✅ Nginx configuration includes security and performance optimizations")


class TestDeploymentScripts:
    """Test deployment and management scripts."""
    
    @pytest.fixture(scope="class")
    def scripts_dir(self):
        """Get scripts directory."""
        return Path(__file__).parent.parent.parent / "scripts"
    
    def test_deploy_script_functionality(self, scripts_dir):
        """Test deployment script functionality."""
        logger.info("🧪 Testing deployment script")
        
        deploy_script = scripts_dir / "deploy.sh"
        assert deploy_script.exists(), "Deploy script not found"
        assert os.access(deploy_script, os.X_OK), "Deploy script is not executable"
        
        # Test help functionality
        result = subprocess.run([
            str(deploy_script), '--help'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, "Deploy script help failed"
        assert 'Usage:' in result.stdout, "Help should show usage information"
        
        logger.info("✅ Deploy script help functionality works")
    
    def test_scale_script_functionality(self, scripts_dir):
        """Test scaling script functionality."""
        logger.info("🧪 Testing scaling script")
        
        scale_script = scripts_dir / "scale.sh"
        assert scale_script.exists(), "Scale script not found"
        assert os.access(scale_script, os.X_OK), "Scale script is not executable"
        
        # Test help functionality
        result = subprocess.run([
            str(scale_script), '--help'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, "Scale script help failed"
        assert 'Usage:' in result.stdout, "Help should show usage information"
        
        logger.info("✅ Scale script help functionality works")
    
    def test_deployment_script_validation(self, scripts_dir):
        """Test deployment script validation logic."""
        logger.info("🧪 Testing deployment script validation")
        
        deploy_script = scripts_dir / "deploy.sh"
        
        # Test with invalid environment
        result = subprocess.run([
            str(deploy_script), 'invalid_env'
        ], capture_output=True, text=True)
        
        assert result.returncode != 0, "Should fail with invalid environment"
        assert 'Invalid environment' in result.stderr, "Should show environment error"
        
        logger.info("✅ Deploy script validation works correctly")


class TestProductionReadiness:
    """Test production readiness aspects."""
    
    def test_security_configuration(self):
        """Test security configuration aspects."""
        logger.info("🧪 Testing security configuration")
        
        # Test that default passwords are not used in production configs
        project_root = Path(__file__).parent.parent.parent
        
        # Check environment example file
        env_example = project_root / ".env.example"
        with open(env_example, 'r') as f:
            env_content = f.read()
        
        # Should not contain weak default passwords
        weak_passwords = ['password', '123456', 'admin', 'root']
        for weak_pwd in weak_passwords:
            assert weak_pwd not in env_content.lower(), f"Should not contain weak password: {weak_pwd}"
        
        # Should contain password generation instructions
        assert 'secrets.token_urlsafe' in env_content, "Should include password generation instructions"
        
        logger.info("✅ Security configuration follows best practices")
    
    def test_resource_limits(self):
        """Test that resource limits are properly configured."""
        logger.info("🧪 Testing resource limits configuration")
        
        project_root = Path(__file__).parent.parent.parent
        
        # Check Docker Compose resource limits
        docker_compose = project_root / "docker-compose.yml"
        with open(docker_compose, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        tektra_service = compose_config['services']['tektra']
        
        # Should have resource configuration through environment variables
        env_vars = tektra_service.get('environment', {})
        memory_vars = [var for var in env_vars if 'MEMORY' in str(var).upper()]
        assert len(memory_vars) > 0, "Should configure memory limits"
        
        logger.info("✅ Resource limits are configured")
    
    def test_health_check_endpoints(self):
        """Test health check endpoint configuration."""
        logger.info("🧪 Testing health check endpoints")
        
        project_root = Path(__file__).parent.parent.parent
        
        # Check that health check script exists
        healthcheck_script = project_root / "docker" / "healthcheck.sh"
        assert healthcheck_script.exists(), "Health check script not found"
        
        with open(healthcheck_script, 'r') as f:
            script_content = f.read()
        
        # Should check multiple endpoints
        assert '/health' in script_content, "Should check main health endpoint"
        assert '/metrics' in script_content, "Should check metrics endpoint"
        assert 'curl' in script_content, "Should use curl for health checks"
        
        logger.info("✅ Health check endpoints are properly configured")
    
    def test_logging_configuration(self):
        """Test logging configuration for production."""
        logger.info("🧪 Testing logging configuration")
        
        project_root = Path(__file__).parent.parent.parent
        
        # Check Docker Compose logging configuration
        docker_compose = project_root / "docker-compose.yml"
        with open(docker_compose, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config['services']
        
        # Check that services have logging configuration
        for service_name, service_config in services.items():
            if 'logging' in service_config:
                logging_config = service_config['logging']
                assert 'driver' in logging_config, f"{service_name} should specify log driver"
                assert 'options' in logging_config, f"{service_name} should specify log options"
                
                options = logging_config['options']
                assert 'max-size' in options, f"{service_name} should limit log file size"
        
        logger.info("✅ Logging configuration is production-ready")


class TestIntegrationWithExternalServices:
    """Test integration with external services and APIs."""
    
    @pytest.mark.skipif(not os.environ.get('INTEGRATION_TESTS_ENABLED'),
                       reason="Integration tests disabled")
    async def test_database_connectivity(self):
        """Test database connectivity and setup."""
        logger.info("🧪 Testing database connectivity")
        
        # This would test actual database connection in a real environment
        # For now, we test the configuration
        
        project_root = Path(__file__).parent.parent.parent
        init_sql = project_root / "docker" / "postgres" / "init.sql"
        
        assert init_sql.exists(), "Database initialization script not found"
        
        with open(init_sql, 'r') as f:
            sql_content = f.read()
        
        # Check for required database setup
        assert 'CREATE EXTENSION' in sql_content, "Should create required extensions"
        assert 'CREATE SCHEMA' in sql_content, "Should create required schemas"
        assert 'CREATE TABLE' in sql_content, "Should create required tables"
        
        logger.info("✅ Database setup script is comprehensive")
    
    def test_monitoring_integration(self):
        """Test monitoring system integration."""
        logger.info("🧪 Testing monitoring integration")
        
        project_root = Path(__file__).parent.parent.parent
        
        # Check Prometheus configuration
        prometheus_config = project_root / "docker" / "prometheus" / "prometheus.yml"
        with open(prometheus_config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Should have Tektra application as target
        scrape_configs = config['scrape_configs']
        tektra_jobs = [job for job in scrape_configs if job['job_name'] == 'tektra-app']
        assert len(tektra_jobs) > 0, "Should monitor Tektra application"
        
        tektra_job = tektra_jobs[0]
        assert 'tektra:8090' in str(tektra_job['static_configs']), "Should scrape metrics port"
        
        logger.info("✅ Monitoring integration is configured correctly")


# Test execution and reporting
def pytest_configure(config):
    """Configure pytest for production deployment tests."""
    logger.info("🧪 Configuring Production Deployment Tests")


def pytest_sessionstart(session):
    """Start of test session."""
    logger.info("🚀 Starting Production Deployment Tests")


def pytest_sessionfinish(session, exitstatus):
    """End of test session."""
    if exitstatus == 0:
        logger.info("✅ All production deployment tests passed!")
    else:
        logger.error(f"❌ Some deployment tests failed with exit status: {exitstatus}")


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    logger.info("🧪 Running Production Deployment Tests")
    
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short"
    ])
    
    sys.exit(result.returncode)