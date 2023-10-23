
FROM python:3.11

WORKDIR /app

COPY * /app/
RUN pip3 install flask
RUN pip3 install tensorflow
RUN pip3 install numpy
RUN pip3 install keras
RUN pip3 install flask-cors
EXPOSE 4000
CMD ["python", "app.py"]
