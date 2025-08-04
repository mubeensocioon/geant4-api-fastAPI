#!/bin/bash
set -e

echo "🚀 Deploying Geant4 API..."

# Create necessary directories
mkdir -p outputs

# Check if Geant4 installation exists in /root
if [ ! -d "/root/geant4-v11.3.2-install" ]; then
    echo "❌ Geant4 installation not found in /root/"
    echo "Looking for Geant4 installations:"
    ls -la /root/geant4*
    exit 1
fi

echo "✅ Found Geant4 installation at /root/geant4-v11.3.2-install"

# Test Geant4 before building
echo "🔍 Testing Geant4 installation..."
if [ -f "/root/geant4-v11.3.2-install/bin/geant4-config" ]; then
    /root/geant4-v11.3.2-install/bin/geant4-config --version
    echo "✅ Geant4 is working"
else
    echo "❌ geant4-config not found"
    ls -la /root/geant4-v11.3.2-install/bin/
fi

# Check which docker compose command works
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    echo "❌ Neither 'docker-compose' nor 'docker compose' found"
    exit 1
fi

echo "✅ Using: $DOCKER_COMPOSE"

# Stop any existing containers
echo "🛑 Stopping existing containers..."
$DOCKER_COMPOSE down || true

# Build and start services  
echo "🔨 Building Docker containers..."
$DOCKER_COMPOSE build

echo "🚀 Starting services..."
$DOCKER_COMPOSE up -d

echo "⏳ Waiting for services to be ready..."
sleep 20

# Health check
echo "🔍 Checking API health..."
if curl -f http://localhost:8000/health; then
    echo "✅ API is healthy!"
    echo ""
    echo "🔗 Access URLs:"
    echo "  - API: http://localhost:8000"
    echo "  - Health: http://localhost:8000/health"
    echo "  - Docs: http://localhost:8000/docs"
    echo ""
    echo "🎯 For n8n: http://localhost:8000/simulate"
else
    echo "❌ API health check failed"
    echo "📋 Checking container status:"
    $DOCKER_COMPOSE ps
    echo ""
    echo "📋 Container logs:"
    $DOCKER_COMPOSE logs geant4-api
fi

echo "🎉 Deployment script complete!"
