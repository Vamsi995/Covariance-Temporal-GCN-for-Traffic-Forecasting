# Covariance-Temporal-GCN-for-Traffic-Forecasting

This work presents a novel approach for traffic foreasting, that leverages covariance based temporal embeddings in the data to create a graph filter to prune influential nodes from previous timestep. This is combined with graph convolutions to enhance spatio-temporal learning. 

## Results


These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


## General Usage
PyTorch and TF models are available
​
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentence = "This is something which i cannot understand at all"

text =  "paraphrase: " + sentence + " </s>"

encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    do_sample=True,
    top_k=200,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=5
)

for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(line)

```


## Dockerfile

The repository also contains a minimal reproducible Dockerfile that can be used to spin up a server with the API endpoints to perform text paraphrasing.

_Note_: The Dockerfile uses the built-in Flask development server, hence it's not recommended for production usage. It should be replaced with a production-ready WSGI server.

After cloning the repository, starting the local server it's a two lines script:

```
docker build -t paraphrase .
docker run -p 5000:5000 paraphrase
```

and then the API is available on `localhost:5000`

```
curl -XPOST localhost:5000/run_forward \
-H 'content-type: application/json' \
-d '{"sentence": "What is the best paraphrase of a long sentence that does not say much?", "decoding_params": {"tokenizer": "", "max_len": 512, "strategy": "", "top_k": 168, "top_p": 0.95, "return_sen_num": 3}}'
```

## Built With

* [Streamlit](https://www.streamlit.io/) - Fastest way for building data apps
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - Backend framework
* [Transformers-Huggingface](https://huggingface.co/) - On a mission to solve NLP, one commit at a time. Transformers Library.


## Authors
- [Sai Vamsi Alisetti](https://github.com/Vamsi995)

## Citing

```bibtex
@misc{alisetti2021paraphrase,
  title={Paraphrase generator with t5},
  author={Alisetti, Sai Vamsi},
  year={2021}
}
```
