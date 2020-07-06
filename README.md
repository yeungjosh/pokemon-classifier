# Deploying [fast.ai](https://www.fast.ai) models on [Docker]

This repo can be used as a starting point to deploy [fast.ai](https://github.com/fastai/fastai) models on Docker.
```
docker build -t fastai-v3 . && docker run --rm -it -p 5000:5000 fastai-v3
```

The guide for production deployment to Render is at https://course.fast.ai/deployment_render.html.

Please use [Render's fast.ai forum thread](https://forums.fast.ai/t/deployment-platform-render/33953) for questions and support.
