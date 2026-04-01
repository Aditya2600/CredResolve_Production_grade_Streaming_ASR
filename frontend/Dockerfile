FROM node:20-alpine

WORKDIR /app

ARG VITE_DEBUG_LOGS=false
ENV VITE_DEBUG_LOGS=${VITE_DEBUG_LOGS}

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

EXPOSE 5173

CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "5173"]
