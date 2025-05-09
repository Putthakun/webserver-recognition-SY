Real-time Face Recognition System
A distributed system for real-time face detection and recognition, designed for scalability and performance. The project consists of two components: a web server for API and monitoring, and a backend camera for face detection. Built to demonstrate full-stack skills in backend development, API design, and system architecture.
Tech Stack

Backend: FastAPI, Python, OpenCV, YOLO, InsightFace
Database: Redis
DevOps: Nginx, RabbitMQ, Docker, Git, GitHub
Hardware: Raspberry Pi
Tools: Postman, VS Code

Features

Real-time face detection using YOLO and OpenCV on Raspberry Pi
Facial recognition with InsightFace, storing embeddings in Redis
Scalable architecture with Nginx load balancing and RabbitMQ for workload management
Monitoring system with event logging (timestamp, identity, location)
REST API for accessing recognition data and logs

Installation
Follow these steps to set up and run the project locally.
Prerequisites

Python 3.8+
Redis
RabbitMQ
Docker (optional for containerization)
Raspberry Pi (for camera component)

Web Server Setup

Clone the web server repository:
git clone https://github.com/Putthakun/webserver-recognition-SY.git
cd webserver-recognition-SY


Install dependencies:
pip install -r requirements.txt


Set up environment variables:

Create a .env file with:REDIS_URL=redis://localhost:6379
RABBITMQ_URL=amqp://guest:guest@localhost:5672/




Run the web server:
uvicorn main:app --host 0.0.0.0 --port 8000



Backend Camera Setup

Clone the backend camera repository:
git clone https://github.com/Putthakun/backend-camera-SY.git
cd backend-camera-SY


Install dependencies:
pip install -r requirements.txt


Connect Raspberry Pi camera and update config:

Edit config.yaml with camera settings and Redis/RabbitMQ URLs.


Run the camera backend:
python main.py



Usage

Access the web server API at http://localhost:8000/docs to test endpoints (e.g., /recognize, /logs).
The backend camera processes video feed and sends recognized faces to Redis via RabbitMQ.
View event logs (timestamp, identity, location) through the API or monitoring interface.
Example API call using Postman:GET http://localhost:8000/logs



Screenshots

Links

Web Server Repository: github.com/Putthakun/webserver-recognition-SY
Backend Camera Repository: github.com/Putthakun/backend-camera-SY
Demo (if available): face-recognition.putthakun.com

Future Improvements

Add support for cloud deployment (e.g., AWS EC2)
Implement real-time alerts for unrecognized faces
Optimize YOLO model for faster detection on low-power devices
Add unit tests for API endpoints

Contact

Email: putthakun01@gmail.com
GitHub: github.com/Putthakun



# web_server_SY_final


    swagger webserver 1
    http://0.0.0.0:8002/docs#/

    swagger webserver 2
    http://0.0.0.0:8003/docs#/

    swagger webserver 3
    http://0.0.0.0:8003/docs#/


    Rabbit MQ UI
    http://0.0.0.0:15672
    

    grafana
    http://0.0.0.0:3000
        - username: admin
        - password: admin123
