## Projcect Seshu
SLMs by the public, for the public

### Introduction
chatGPT caught the public imagination and for the first time non-tech people could experience Generative AI. This led to a surge in interest to develop safe applications of LLMs as well as developing domain specific or open source alternatives to chatGPT. Notable among them is LLaMA 2 - an LLM open sourced by Meta. This release catalyzed the development of tasks, tools, and assets ranging from datasets to new models to new applictions. An SLM called Phi 2 released by Microsoft also showed that small models (reletively that is) can also compete with large models, which can be trained and served at substantially lower costs. However, there are some challenges.

Majority, if not all the LLMs, we see today are based on proven Transformer based architectures. And Transfomres have quadratic (in inputs tokens) complexity - therefore slow to train and infer. As a result, new memory and compute efficient attention mechanisms have sprungup, along with Engineering hacks. But, at the end of the day, they are still based on Transformer-based architectures.
Majority, with the exception of some Chinese LLMs, are English-centric and other languages have a token representation (no pun intended).
Often, LLMs have a particulalr tokenizer -- which makes extension to other languages/ domains hard.
Developing SLMs or LLMs is still a compute heavy problem. Therefore, only large consortia with deep pockets, massive talent concentration and GPU farms can afford to build such models.
In this hackathon, we like to address the above challenges.

## Proposal:
### Objectives
- Develop a multilingual Mambaa(S4)-based SLM on multi-lingual indic dataset
- Decentralise the training and development of SLMs/LLMs via simple federated learning framework

## User Stories
### Client (Donor User)

#### Pre-reqs
 - has GPU, registers on hf/mlsquare for write access
 - familair with HuggingFace ecosystem (transfomers, peft, datasets, hub)
 - [optional] can donate time or data or both
#### Actions:
 - Runs client.py which
 - downloads data, ptrained model
 - SFTs via LoRA
 - pushes the adapter to HuggingFace model hub

### Server (who manages the federated learning)

#### Pre-reqs
- has (big) GPU(s)
- is familair with HuggingFace ecosystem (transfomers, peft, datasets, hub), databases, ML Enginneering in general
- [optional] can donate time or data or both

#### Actions:
 - Pretrains a multi-lingual Mamba model, publushed a checkpoint
 - Evaluated the community contributed adapters, and merges them into the PTM
 - Does continous pretrainning, and released checkpoints periodically

### Academic Interests
 - experiment and identity good federating learning policy
 - training configurations to PT, CPT, SFT, FedT SLMs and LLMs
 - develop new task specific adapters
 - contribute your local, vernacular data
 - curate datasets
  
## Roadmap

### Week 0
- Make Mamba compatiable with Transformer class
- Test LoRA adapters (adding, training, merging)
- Pretrain an SLM, SFT on LoRA, Merge, Push
Outcome: A functional end-to-end Pretraining and SFT-ing pipeline

### Week 1
- Develop client-side code
- On multi-lingual indic dataset [samantar](https://huggingface.co/datasets/ai4bharat/samanantar), pretrain a model
- Release a checkpoint

### Week 2
- Drive SFT via community (at least two users)
- Run Federated SFT-ing

### Week 4 and onwards
- Benchmark and eval on test set (against other OSS LLMS)
- Perplexity vs Epochs (and how Seshu is maturing)


### References:
1. [MambaByte ](https://arxiv.org/abs/2401.13660)
2. [Mamba](https://arxiv.org/abs/2312.00752)



### Note:
The views expressed or approach being taken - is of the individuals, and they do not represent any organization explicitly or implicitly.
Likewise, anyone who wants to contribute their time, compute or data must understand that, this is a community experiment to develop LLMs by the community, and may not result in any significant outcome. On the contrary, this may end up in total failure. The contributors must take this risk on their own.
