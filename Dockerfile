FROM node:20-slim

# Install dependencies for canvas/sqlite if needed
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./
RUN npm install --production

COPY . .

# Ensure the data directory exists
RUN mkdir -p /app/data

EXPOSE 3000

CMD [ "node", "server.js" ]
