FROM "ubuntu:bionic"
FROM python:3.8-slim-buster

RUN mkdir -p /soyabean_yolov7
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

RUN pip install requests
RUN pip install Flask
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install imutils
RUN pip install pandas
RUN pip install -U opencv-python-headless
RUN pip install PyYAML
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install scipy
RUN pip install image-slicer
RUN pip install tqdm
RUN pip install gunicorn

ADD . /3_52_0_0_soyabean_yolov7
WORKDIR /3_52_0_0_soyabean_yolov7
EXPOSE 9002
#CMD ["python", "wsgi_3_52_0_0.py"]
CMD ["gunicorn","--workers","2","--timeout","600","--bind","0.0.0.0:9002","wsgi_3_52_0_0:app"]
