FROM python:3.12

COPY requirements.txt /app/requirements.txt
COPY src /app/src

WORKDIR /app

RUN pip install -r requirements.txt
CMD ['python', 'src/app.py']
