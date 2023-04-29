FROM "ubuntu:bionic"
FROM python:3.8-slim-buster

RUN mkdir -p /quality_master
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

RUN pip install requests
RUN pip install Flask
RUN pip install numpy
RUN pip install mysql-connector-python
RUN pip install gunicorn

ADD . /quality_master
WORKDIR /quality_master
EXPOSE 8081
#CMD ["python", "wsgi_api_quality.py"]
CMD ["gunicorn","--workers","4","--timeout","600","--bind","0.0.0.0:8081","wsgi_quality_master:app"]

