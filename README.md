# Sentiment Analysis

## Dataset
from kaggle: https://www.kaggle.com/crowdflower/twitter-airline-sentiment

## Installation
```
docker build -t ml .
```

## Run
```
docker run --rm -v $(pwd):/app -v $(pwd)/data:/app/data --name mldev -it ml
```