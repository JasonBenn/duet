Run locally with
```sh
pip install -r requirements.txt
python main.py
```

## Build and deploy
Both App Engine and Heroku fail to build, so we build a docker container locally before trying to serve.
For faster startup time, we include the model file in the container image.

```sh
# Copy model cache
cp -r ~/.pytorch_pretrained_bert/ cache/
# Download wiki model
wget -O cache/model-best.pt https://s3.amazonaws.com/yaroslavvb2/runs/transformerxl-wiki/model.pt
# Build docker image
docker build . -t aiduet/web
# Test locally
docker run --rm -it -p 8080:8080 aiduet/web
# One time login
gcloud auth login
# Deploy to AppEngine
yes | gcloud app deploy

## For faster deploy, build container separately:
docker build . -t us.gcr.io/duet-0/appengine
docker push us.gcr.io/duet-0/appengine
gcloud app deploy --image-url us.gcr.io/duet-0/appengine
```
