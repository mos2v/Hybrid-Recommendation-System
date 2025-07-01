#!/bin/bash

# Travel Recommendation API Deployment Script
# This script builds and deploys the FastAPI application using Docker

set -e  # Exit on any error

# Configuration
IMAGE_NAME="travel-recommender-api"
CONTAINER_NAME="travel-recommender-api-container"
PORT=8002
DOCKERFILE_PATH="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    print_status "Checking if Docker is running..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to stop and remove existing container
cleanup_existing() {
    print_status "Checking for existing container..."
    
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_warning "Stopping existing container..."
        docker stop $CONTAINER_NAME
        print_success "Container stopped"
    fi
    
    if docker ps -a -q -f name=$CONTAINER_NAME | grep -q .; then
        print_warning "Removing existing container..."
        docker rm $CONTAINER_NAME
        print_success "Container removed"
    fi
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: $IMAGE_NAME"
    docker build -t $IMAGE_NAME $DOCKERFILE_PATH
    print_success "Image built successfully"
}

# Function to run the container
run_container() {
    print_status "Starting container on port $PORT..."
    
    # Create volumes for persistent data
    docker volume create travel-recommender-models 2>/dev/null || true
    docker volume create travel-recommender-logs 2>/dev/null || true
    
    # Run the container with proper volume mounts
    docker run -d \
        --name $CONTAINER_NAME \
        -p $PORT:$PORT \
        -v travel-recommender-models:/app/models \
        -v travel-recommender-logs:/app/logs \
        --restart unless-stopped \
        $IMAGE_NAME
    
    print_success "Container started successfully"
}

# Function to check container health
check_health() {
    print_status "Waiting for API to be ready..."
    
    # Wait for container to start
    sleep 5
    
    # Check if container is running
    if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        print_error "Container is not running. Check logs with: docker logs $CONTAINER_NAME"
        exit 1
    fi
    
    # Check API health (with retries)
    for i in {1..10}; do
        if curl -s -f http://localhost:$PORT/health > /dev/null 2>&1; then
            print_success "API is healthy and responding"
            return 0
        fi
        print_status "Waiting for API to start (attempt $i/10)..."
        sleep 3
    done
    
    print_warning "API health check timed out. Container might still be starting up."
    print_status "You can check the logs with: docker logs $CONTAINER_NAME"
}

# Function to display deployment info
show_info() {
    echo ""
    echo "=========================================="
    echo "  Travel Recommendation API Deployed"
    echo "=========================================="
    echo "API URL: http://localhost:$PORT"
    echo "API Documentation: http://localhost:$PORT/docs"
    echo "Health Check: http://localhost:$PORT/health"
    echo "Container Name: $CONTAINER_NAME"
    echo "Image Name: $IMAGE_NAME"
    echo ""
    echo "Useful Commands:"
    echo "  View logs: docker logs $CONTAINER_NAME"
    echo "  Follow logs: docker logs -f $CONTAINER_NAME"
    echo "  Stop API: docker stop $CONTAINER_NAME"
    echo "  Start API: docker start $CONTAINER_NAME"
    echo "  Remove API: docker rm $CONTAINER_NAME"
    echo "=========================================="
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-only    Only build the image, don't run container"
    echo "  --no-cache      Build image without using cache"
    echo "  --cleanup       Stop and remove existing container and image"
    echo "  --logs          Show container logs"
    echo "  --help          Show this help message"
    echo ""
}

# Parse command line arguments
BUILD_ONLY=false
NO_CACHE=false
CLEANUP=false
SHOW_LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main deployment logic
main() {
    print_status "Starting Travel Recommendation API deployment..."
    
    # Handle special commands
    if [ "$SHOW_LOGS" = true ]; then
        docker logs -f $CONTAINER_NAME
        exit 0
    fi
    
    if [ "$CLEANUP" = true ]; then
        print_status "Cleaning up existing deployment..."
        cleanup_existing
        if docker images -q $IMAGE_NAME | grep -q .; then
            print_status "Removing existing image..."
            docker rmi $IMAGE_NAME
            print_success "Image removed"
        fi
        print_success "Cleanup completed"
        exit 0
    fi
    
    # Check prerequisites
    check_docker
    
    # Clean up existing deployment
    cleanup_existing
    
    # Build image
    if [ "$NO_CACHE" = true ]; then
        print_status "Building image without cache..."
        docker build --no-cache -t $IMAGE_NAME $DOCKERFILE_PATH
    else
        build_image
    fi
    
    # Exit if build-only
    if [ "$BUILD_ONLY" = true ]; then
        print_success "Image built successfully. Use --cleanup to remove or run without --build-only to deploy."
        exit 0
    fi
    
    # Run container
    run_container
    
    # Check health
    check_health
    
    # Show deployment info
    show_info
}

# Run main function
main "$@"
