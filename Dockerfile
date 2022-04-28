# syntax=docker/dockerfile:1
FROM --platform=linux/x86_64 python:3.9
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m pip install --upgrade pip

RUN pip install tensorflow==2.6.2
RUN pip3 install matplotlib
RUN mkdir /app
WORKDIR /app
ADD ./model /app/model
ADD ./static /app/static
ADD ./main_code /app/main_code
ADD ./templates /app/templates
ADD app.py /app/app.py 
ADD log.py /app/log.py
ADD wsgi.py /app/wsgi.py
ADD config.ini /app/config.ini
RUN pip3 install gunicorn==20.1.0
RUN pip3 install Flask==2.1.0
RUN pip3 install Jinja2==3.1.1
RUN pip3 install opencv-python
CMD ["gunicorn", "-w 3", "-b", "0.0.0.0:8004", "wsgi:app","-t 1000"]
