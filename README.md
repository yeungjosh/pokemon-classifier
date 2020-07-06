# Deploying [fast.ai](https://www.fast.ai) Pokemon classifier models on [Docker]

Pokemon classifier using transfer learning on Resnet34 deployed as a web application

This repo can be used to deploy [fast.ai](https://github.com/fastai/fastai) models on Docker.
```
docker build -t fastai-v3 . && docker run --rm -it -p 5000:5000 fastai-v3
```

The guide for production deployment to Render is at https://course.fast.ai/deployment_render.html.
