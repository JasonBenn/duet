`pip install -r requirements.txt`

Run locally with
```sh
python main.py
```

## Build and deploy
Both App Engine and Heroku fail to build, so we build a docker container locally before trying to serve.
For faster startup time, we include the model file in the container image.

```sh
# Copy model cache
cp -r ~/.pytorch_pretrained_bert/ cache/
# Build docker image
docker build . -t aiduet/web
# One time login
heroku container:login
# Push to Heroku
heroku container:push --app aiduet
heroku container:release web
heroku open
```
