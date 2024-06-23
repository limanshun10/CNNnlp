## Credit
This code is based on the original implementation found [Shawn1993's cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch)

## Introduction
This is the implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.


## Requirement
* python 3
* pytorch > 0.1
* torchtext = 0.6.0
* gensim

## Models Implemented

- **CNN-rand**: A baseline model with randomly initialized word vectors.
- **CNN-static**: A model with pre-trained word vectors that remain static during training.
- **CNN-non-static**: A model with pre-trained word vectors that are fine-tuned during training.
- **CNN-multichannel**: A model using both static and fine-tunable word vectors as separate channels.

## Pre-trained word vectors

- **Word2vec**
- https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300

## Datasets

- MR: Movie Reviews
- SST-1: Stanford Sentiment Treebank - binary
- SST-2: Stanford Sentiment Treebank - fine-grained

## Train
You can choose the models and datasets to train.
For example:
```
python main.py -model_type static -pretrained_embeddings_path ./GoogleNews-vectors-negative300.bin -dataset MR -epochs 10 -batch-size 64 -embed-dim 300
```


## Test
If you has construct you test set, you make testing like:

```
python main.py -test -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt
```
The snapshot option means where your model load from. If you don't assign it, the model will start from scratch.

## Predict
* **Example1**

	```
	python main.py -predict="Hello my dear , I love you so much ." \
	          -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  Hello my dear , I love you so much .
	[Label] positive
	```
* **Example2**

	```
	python main.py -predict="You just make me so sad and I have to leave you ."\
	          -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  You just make me so sad and I have to leave you .
	[Label] negative
	```

Your text must be separated by space, even punctuation.And, your text should longer then the max kernel size.

## Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

