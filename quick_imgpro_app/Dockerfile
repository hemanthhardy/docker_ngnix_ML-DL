FROM "ubuntu:bionic"
FROM python:3.8-slim-buster

#FROM python:3.7
#WORKDIR /quick_analysis
RUN mkdir -p /quick_imgpro

RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

RUN pip install requests
RUN pip install Flask
RUN pip install scikit-image==0.19.3
RUN pip install scikit-learn
RUN pip install imutils
RUN pip install pandas
RUN pip install -U opencv-python-headless
RUN pip install mysql-connector-python
RUN pip install PyYAML
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install scipy
RUN pip install gunicorn

ADD . /quick_imgpro
WORKDIR /quick_imgpro
EXPOSE 8080
#CMD ["python", "wsgi_quick_imgpro.py"]
CMD ["gunicorn","--workers","4","--timeout","600","--bind","0.0.0.0:8080","wsgi_quick_imgpro:app"]
