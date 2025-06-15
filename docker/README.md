# Tektra Docker Development Environment

This directory contains Docker configuration for running the Tektra AI Assistant in a containerized environment.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available for containers

### Development Environment

1. **Start all services:**
   ```bash
   cd docker
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   # All services
   docker-compose logs -f
   
   # Specific service
   docker-compose logs -f backend
   docker-compose logs -f frontend
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - PostgreSQL: localhost:5432
   - Redis: localhost:6379

4. **Stop services:**
   ```bash
   docker-compose down
   ```

5. **Reset everything (including data):**
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

## Services

### PostgreSQL Database
- **Image:** postgres:15-alpine
- **Port:** 5432
- **Database:** tektra
- **User:** tektra
- **Password:** tektra_dev_password

### Redis Cache
- **Image:** redis:7-alpine
- **Port:** 6379
- **Use:** Session storage and caching

### Backend API
- **Build:** Custom Python FastAPI application
- **Port:** 8000
- **Dependencies:** PostgreSQL, Redis

### Frontend
- **Build:** Custom Next.js application
- **Port:** 3000
- **Dependencies:** Backend API

## Development Workflow

### Making Changes

**Backend changes:**
```bash
# Rebuild backend only
docker-compose up -d --build backend

# View backend logs
docker-compose logs -f backend
```

**Frontend changes:**
```bash
# Rebuild frontend only
docker-compose up -d --build frontend

# View frontend logs
docker-compose logs -f frontend
```

### Database Management

**Access PostgreSQL:**
```bash
docker-compose exec postgres psql -U tektra -d tektra
```

**Reset database:**
```bash
docker-compose down postgres
docker volume rm docker_postgres_data
docker-compose up -d postgres
```

**Backup database:**
```bash
docker-compose exec postgres pg_dump -U tektra tektra > backup.sql
```

### Debugging

**Execute commands in containers:**
```bash
# Backend shell
docker-compose exec backend bash

# Check backend dependencies
docker-compose exec backend uv pip list

# Frontend shell
docker-compose exec frontend sh

# Check frontend build
docker-compose exec frontend npm run build
```

**Monitor resources:**
```bash
docker-compose ps
docker stats
```

## Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp ../.env.example ../.env
```

Key variables:
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `DEBUG`: Enable debug mode
- `SECRET_KEY`: JWT secret key

## Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Check what's using ports
lsof -i :3000
lsof -i :8000
lsof -i :5432

# Kill processes if needed
kill -9 <PID>
```

**Database connection issues:**
```bash
# Check PostgreSQL is ready
docker-compose exec postgres pg_isready -U tektra

# Check backend can connect
docker-compose exec backend python -c "from app.database import engine; print('DB OK')"
```

**Build issues:**
```bash
# Clean rebuild everything
docker-compose down
docker system prune -f
docker-compose build --no-cache
docker-compose up -d
```

**Volume permissions:**
```bash
# Fix ownership issues (Linux/macOS)
sudo chown -R $USER:$USER ../backend
sudo chown -R $USER:$USER ../frontend
```

## Production Deployment

For production deployment:

1. **Update environment variables:**
   - Set strong passwords
   - Use production database URLs
   - Set `DEBUG=false`

2. **Use production compose file:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Enable SSL/TLS:**
   - Configure reverse proxy (nginx)
   - Add SSL certificates
   - Update CORS settings

## Health Checks

All services include health checks:
- **postgres:** `pg_isready`
- **redis:** `redis-cli ping`
- **backend:** `curl /health`
- **frontend:** `curl /`

Check health status:
```bash
docker-compose ps
```

Healthy services show `Up (healthy)` status.