#!/bin/bash
set -e

docker run -d \
    --name student-app \
    --restart unless-stopped \
    -p 80:5000 \
    student-app

docker ps --filter "name=student-app"