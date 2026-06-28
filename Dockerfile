FROM node:20-slim

# Build tooling required to compile native modules (sqlite3).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ENV NODE_ENV=production

WORKDIR /app

# Install dependencies from the lockfile for reproducible builds. Fall back to
# `npm install` only if the lockfile is missing.
COPY package*.json ./
RUN npm ci --omit=dev || npm install --omit=dev

COPY . .

# Persistent storage for SQLite + WhatsApp session state (Fly volume mounts here).
RUN mkdir -p /app/data

EXPOSE 3000

CMD [ "node", "server.js" ]
