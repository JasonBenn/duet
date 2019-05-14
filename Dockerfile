FROM pytorch/pytorch
# Copy requirements and install first to avoid redundant pip installs.
COPY cache /app/cache
COPY requirements.txt requirements.txt
ENV PYTORCH_PRETRAINED_BERT_CACHE /app/cache

RUN pip install gunicorn
RUN pip install -r requirements.txt


COPY static /app/static
COPY templates /app/templates
COPY *py /app/
WORKDIR /app
#ENTRYPOINT ["python"]
#CMD ["main.py"]
EXPOSE 8080
CMD ["gunicorn"  , "-b", "0.0.0.0:8080", "main:app", "--timeout", "300"]
