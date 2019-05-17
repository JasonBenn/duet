# TODO: remove tag when latest updates to 1.1
FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
# Copy requirements and install first to avoid redundant pip installs.
COPY cache /app/cache
COPY requirements.txt requirements.txt
ENV PYTORCH_PRETRAINED_BERT_CACHE /app/cache

RUN pip install waitress
RUN pip install -r requirements.txt


COPY static /app/static
COPY templates /app/templates
COPY transformer_xl /app/transformer_xl
COPY *py /app/
WORKDIR /app
#ENTRYPOINT ["python"]
#CMD ["main.py"]
#EXPOSE 8080
#CMD ["gunicorn"  , "-b", "0.0.0.0:8080", "main:app", "--timeout", "300", "--worker-class", "gevent", "-w", "2", "--log-level", "debug"]
CMD ["waitress-serve", "--port", "8080", "--url-scheme=http", "main:app"]