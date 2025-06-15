-- Initialize Tektra database
-- This script runs when the PostgreSQL container starts for the first time

-- Create the main database (already created by environment variables)
-- CREATE DATABASE tektra;

-- Create additional extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create initial schema (will be handled by migrations later)
-- For now, just ensure the database is ready

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE tektra TO tektra;