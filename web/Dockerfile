# ==============================================================================
# RETAIL AI PLATFORM — FRONTEND VITE WEB APP DOCKERFILE
# ==============================================================================
# Stage 1: Build the React single page app
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency definitions
COPY package*.json tsconfig*.json vite.config.ts ./

# Install dependencies
RUN npm ci

# Copy source code
COPY src/ ./src/
COPY index.html ./

# Build production web bundle
RUN npm run build

# Stage 2: Serve compiled web assets with Nginx
FROM nginx:alpine AS runner

# Copy built static assets into Nginx html directory
COPY --from=builder /server/static/app /usr/share/nginx/html
COPY --from=builder /server/static/app /usr/share/nginx/html/static/app

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
