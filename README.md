## Text Summarization by BERT in PyTorch

Transformer-based Summarization by Exploiting Relevant User Comments, as proposed by us.
Our model simulates the nature relationship between relevant user posts and the content of the main documents by sharing information in terms of important words or tokens.

To do that, we empower the model with the equipment of two important aspects: utilizing social information and using the power of transformers, i.e. BERT. More precisely, relevant user posts are used to enrich the information of sentences in the main documents. The enrichment is the combination of hidden features of input sentences and user posts learned from BERT. 

To capture more fine-grained hidden representation, we stack an additional convolution neural network (CNN) on the top of BERT for classification. The final summary is created by selecting top m ranked sentences based on their importance denoted as probabilities.

This simple wrapper based on [Transformers](https://github.com/huggingface/transformers) (for managing BERT model), and [Sentence-Transformer](https://github.com/UKPLab/sentence-transformers) (for managing Sentence-BERT model) and PyTorch achieves 0.284 ROUGE-1 Score on the [USA-CNN](https://github.com/chiennv2000/TextSumarization/blob/master/data/USAToday-CNN.json) and 0.372 ROUGE-1 Score on the [SoLSCSum](https://github.com/chiennv2000/TextSumarization/blob/master/data/SoLSCSum.json) dataset.

## Model architecture

Here we created a custom classification head on top of the BERT backbone. The sequence of a sentence and the title was fed into BERT and the relevant user post was fed into sentenceBERT. ```[CLS-C]``` token represents the final vector of the relevant user post in the final layer of sentenceBERT. We concatenated the 5 hidden representations, and fed it to a convolution neural network (CNN) for classification.

<img src="https://scontent.fhan5-6.fna.fbcdn.net/v/t1.0-9/116874813_1166119750409955_4353412123860951616_o.jpg?_nc_cat=107&_nc_sid=730e14&_nc_ohc=rla-eJTRQBEAX-i9vYG&_nc_ht=scontent.fhan5-6.fna&oh=5b759ad1da6f8ffd665d29dd9bc150a0&oe=5F59C9E8" alt="alt text" width="850">

#### Environment

Python: `3.6`

Torch version: `1.6.0`
Transformers: `2.11.0`

`requirements.txt` exposes the library dependencies

### Training
You need to create directories according to the path to evaluate model by ROUGE-1.5.5:

<img src="https://scontent.fhan5-7.fna.fbcdn.net/v/t1.0-9/117168071_1166154057073191_1999305517093526417_n.jpg?_nc_cat=100&_nc_sid=dbeb18&_nc_ohc=uq0ndgR2EYUAX__9DHV&_nc_ht=scontent.fhan5-7.fna&oh=d28129c8f2b3701015029d872a81965b&oe=5F57622E" alt="alt text" width="300">

To perform training, run the following command:

(You can also change the hyperparameters)
```
python main.py
```

You can also use this [notebook](https://github.com/chiennv2000/TextSumarization/blob/master/Text_Sumarization.ipynb) to train on Google Colaboratory.

## Result

| Datasets  |  ROUGE-1  |  ROUGE-2  |
|-----------|-----------|-----------|
| SoLSCSum  |  0.372    |  0.300    |
| USA-CNN   |  0.284    |  0.093    |
