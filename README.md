# Card-Stocker-Pro

A high-performance, AI-powered trading card and sports card recognition engine.

Restart plan
- Backup current code: create a `backup-before-restart` branch.
- Create a clean `start-over` branch with this scaffold.
- Implement MVP features first:
  1. Webcam capture & simple local detection
  2. Store detection metadata and thumbnails
  3. Basic web dashboard to view detections

How to get started (local)
1. Install Docker & Docker Compose
2. Copy `.env.example` to `.env` and adjust values
3. Run: `docker compose up --build`
4. Backend: http://localhost:8000
5. Frontend: http://localhost:3000
