# GAP

## Introduction

This repository (GAP) is a subproject of [April](https://github.com/Joe-zsc/April) (comming soon), and is also a simplified implementation of the paper [Towards Generalizable Autonomous Penetration Testing via Domain Randomization and Meta-Reinforcement Learning](https://arxiv.org/abs/2412.04078 "paper").  

## Related Projects

[April-AE](https://github.com/Joe-zsc/April-AE)

[April](https://github.com/Joe-zsc/April) (comming soon)

## Getting Started

### Installation

Start by checking out the repository:

```bash
git clone https://github.com/Joe-zsc/GAP.git
cd GAP
pip install -r requirment.txt
```

### Synthetic environment generation

Take vulnerability CVE-2021-3129 for example:

1. Prepare the API key of GLM-4 or other LLMs.

   ```python
   vul="CVE-2021-3129"
   api_key = "" 
   llm = ChatOpenAI(
           temperature=0.96,
           model="glm-4",
           openai_api_key=api_key,
           openai_api_base="https://open.bigmodel.cn/api/paas/v4",
       )
   ```
2. Make sure that the example scenario data exists in `scenarios\auto_probe`. e.g. `scenarios\auto_probe\CVE-2021-3129.json`
3. Make sure that the vulnerability description exists in `GatheredInfo\Action_description.json`
4. run LLM_scenario_generation.py

### Prepare the embedding models

In this project, we directly use sentence-bert to represent the raw state information and action descriptions as vectors.

1. Download pre-trained [Sentence-BERT](https://huggingface.co/models?library=sentence-transformers) models, or train/fine-tune your own embedding models using domain corpus. (reference: [TSDAE](https://github.com/UKPLab/sentence-transformers))
2. Store the embedding models in path  `NLP_Module\Embedding_models`.
3. Modify the config file `config.ini` and write the model names in the corresponding positions.

```ini
[Embedding]
embedding_models = NLP_Module\Embedding_models
sbert_model = MySbertModel ; your sentence-bert model name, e,g., all-MiniLM-L12-v2
```

### Training with GAP

```bash
python April_meta.py --train_env_file meta_scenarios/CVE-2021-3129-train-5.json  --eval_env_file meta_scenarios/CVE-2021-3129-eval.json --meta_algo MAML
```


### Training with single simulated environments

Run the following commands to run a simulation with PPO:

```bash
python April_meta.py --train_env_file auto_probe/CVE-2021-3129.json  --eval_env_file auto_probe/CVE-2021-3129.json
```

The learning curves can be seen via the Tensorboard:

```bash
tensorboard --logdir runs --host localhost --port 6666
```

### Training with real vulnerable host

[April](https://github.com/Joe-zsc/April) (comming soon)

## Citation

**NOTE:** This project is for educational purpose only and the author does not condone any illegal use. Use as your own risk.

Please cite our paper at:

```
@ARTICLE{2024arXiv241204078Z,
       author = {{Zhou}, Shicheng and {Liu}, Jingju and {Lu}, Yuliang and {Yang}, Jiahai and {Zhang}, Yue and {Chen}, Jie},
        title = "{Towards Generalizable Autonomous Penetration Testing via Domain Randomization and Meta-Reinforcement Learning}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Cryptography and Security},
         year = 2024,
        month = dec,
          eid = {arXiv:2412.04078},
        pages = {arXiv:2412.04078},
          doi = {10.48550/arXiv.2412.04078},
archivePrefix = {arXiv},
       eprint = {2412.04078},
 primaryClass = {cs.LG},
}
```
