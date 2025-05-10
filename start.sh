#!/bin/bash

# Configuration
PORT=5500                      # Port your Python app runs on
APP_COMMAND="python3 app.py"   # Replace with your app's entry point

# Start your Python app
echo "Starting Python app..."
$APP_COMMAND &
APP_PID=$!

# Wait briefly to let app start
sleep 2

# Start ngrok in the background
echo "Starting ngrok on port $PORT..."
ngrok http $PORT > /dev/null &
NGROK_PID=$!

# Give ngrok a moment to initialize
sleep 2

# Fetch the public URL
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o "https://[a-z0-9.-]*\.ngrok[-a-z]*\.app" | head -n 1)

if [[ -n "$PUBLIC_URL" ]]; then
  echo "Public URL: $PUBLIC_URL"
else
  echo "Failed to get ngrok URL"
fi

# Clean up on Ctrl+C
trap "echo Stopping...; kill $NGROK_PID; kill $APP_PID; exit" SIGINT

# Keep script running
wait