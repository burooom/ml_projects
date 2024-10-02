FROM python:3.10

COPY requirements.txt /app/requirements.txt
COPY src /app
COPY weights /app/weights

WORKDIR app


RUN pip install -r requirements.txt

CMD ["python", "app.py"]
