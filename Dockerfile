
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*


COPY src/ ./src/


WORKDIR /app/src


RUN pip install --upgrade pip && \
    pip install opencv-python-headless torch torchvision torchaudio \
    numpy norfair


RUN git clone https://github.com/ultralytics/yolov5 && \
    pip install -r yolov5/requirements.txt


ENV DISPLAY=off


CMD ["python", "main.py"]
