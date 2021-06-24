FROM python:3.8.6-buster

COPY api /api
COPY requirements.txt /requirements.txt
COPY le-kingmakers.json /key.json

ENV GOOGLE_APPLICATION_CREDENTIALS=key.json

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/twintproject/twint.git#egg=twint

CMD  uvicorn api.fast:app --host 0.0.0.0 --port $PORT