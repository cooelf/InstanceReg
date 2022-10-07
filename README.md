Code for Paper **Instance Regularization for Discriminative Language Model Pre-training**

##  Installation

We use fastai and transformers toolkits for this work.

```
pip install -r requirements.txt
```

##  Steps

1. `python pretrain_ir.py`
2. set `pretrained_checkcpoint` in `finetune.py` to use the checkpoint you've pretrained and saved in `electra_pytorch/checkpoints/pretrain`. 
3. `python finetune.py` (with `do_finetune` set to `True`)
4. Go to neptune, pick the best run of 10 runs for each task, and set `th_runs` in `finetune.py` according to the numbers in the names of runs you picked.
5. `python finetune.py` (with `do_finetune` set to `False`), this outpus predictions on testset, you can then compress and send `.tsv`s in `test_outputs/<group_name>/*.tsv` to GLUE site to get test score.


