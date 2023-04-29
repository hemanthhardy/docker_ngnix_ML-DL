FROM nginx:1.15.8

RUN rm /etc/nginx/nginx.conf
RUN apt-get update && apt-get install -y
RUN apt-get install build-essential
RUN apt-get install curl
RUN rm -rf /var/lib/apt/lists/*
COPY nginx.conf /etc/nginx/nginx.conf

CMD ["nginx", "-g", "daemon off;"]
