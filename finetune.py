from pathlib import Path
import pandas as pd
import numpy as np
import datasets
from fastai.text.all import *
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
from hugdatafast import *
from _utils.utils import *
from _utils.would_like_to_pr import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--check_name', type=str, default='pretrain_bert_base_ir')
parser.add_argument('--checkpoint', type=str, default='vanilla_11081_100.0%.pth')
parser.add_argument('--size', type=str, default='small')
parser.add_argument('--group_name', type=str, default=None)
args = parser.parse_args()
check_name = args.check_name

c = MyConfig({

  'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # List[int]: use multi gpu (data parallel)
  'start': 0,
  'end': 10,
  'pretrained_checkpoint': args.checkpoint, # None to use pretrained ++ model from HuggingFace
  'seeds': None,
  'weight_decay': 0,
  'adam_bias_correction': False,
  'xavier_reinited_outlayer': True,
  'schedule': 'original_linear',
  'original_lr_layer_decays': True,
  'double_unordered': True,
  # whether to do finetune or test
  'do_finetune': True,  # True -> do finetune ; False -> do test
  # finetuning checkpoint for testing. These will become "ckp_dir/{task}_{group_name}_{th_run}.pth"
  'th_run': {'qqp': 7, 'qnli': 5,
             'mrpc': 7, 'mnli': 2, 'ax': 2,
             'sst2': 3, 'rte': 7, 'wnli': 0,
             'cola': 1, 'stsb': 8,
             },
  'size': args.size,
  'wsc_trick': False,
  'num_workers': 3,
  'my_model': False,  # True only for my personal research
  'logger': 'neptune',
  'group_name': args.group_name,  # the name of represents these runs
})

# only for my personal research purpose
hparam_update = {
  
}

# %%
# Check
if not c.do_finetune: assert c.th_run['mnli'] == c.th_run['ax']
if c.pretrained_checkpoint is None: assert not c.my_model
assert c.schedule in ['original_linear', 'separate_linear', 'one_cycle', 'adjusted_one_cycle']

# Settings of different sizes
if c.size == 'small': c.lr = 3e-4; c.layer_lr_decay = 0.8; c.max_length = 128
elif c.size == 'base': c.lr = 1e-4; c.layer_lr_decay = 0.8; c.max_length = 512
elif c.size == 'large': c.lr = 5e-5; c.layer_lr_decay = 0.9; c.max_length = 512
else: raise ValueError(f"Invalid size {c.size}")
# if c.pretrained_checkpoint is None: c.max_length = 512 # All public models is ++, which use max_length 512

hf_tokenizer = BertTokenizer.from_pretrained(f'bert-{c.size}-uncased')
electra_config = BertConfig.from_pretrained(f'bert-{c.size}-uncased')

if c.wsc_trick:
  from _utils.wsc_trick import * # importing spacy model takes time

# logging
if c.logger == 'neptune':
  import neptune
  from fastai.callback.neptune import NeptuneCallback
  class LightNeptuneCallback(NeptuneCallback):
    def after_batch(self): pass
    def after_epoch(self):
      if self.epoch == (self.n_epoch - 1): super().after_epoch()
  neptune.init(project_qualified_name=f'xxxx/{args.group_name}', api_token='xxxx') # anonymous user account
elif c.logger == 'wandb':
  import wandb
  from fastai.callback.wandb import WandbCallback
  class LightWandbCallback(Callback):
    def __init__(self, run):
      self.run = run
    def after_epoch(self):
      if self.epoch != (self.n_epoch - 1): return
      wandb.log({n:s for n,s in zip(self.recorder.metric_names, self.recorder.log) if n not in ['train_loss', 'epoch', 'time']})
    def after_fit(self):
      wandb.log({}) # ensure sync of last step
      self.run.finish()

# Path
Path('./datasets').mkdir(exist_ok=True)
Path('./checkpoints/glue').mkdir(exist_ok=True, parents=True)
Path('./test_outputs/glue').mkdir(exist_ok=True, parents=True)
c.pretrained_ckp_path = Path(f'./checkpoints/{check_name}/{c.pretrained_checkpoint}')
if c.group_name is None:
  if c.pretrained_checkpoint: c.group_name = c.pretrained_checkpoint[:-4]
  elif c.pretrained_checkpoint is None: c.group_name = f"{c.size}++"

# Print info
print(f"process id: {os.getpid()}")
print(c)


# %%
METRICS = {
  **{ task:[MatthewsCorrCoef()] for task in ['cola']},
  **{ task:[accuracy] for task in ['sst2', 'mnli', 'qnli', 'rte', 'wnli', 'snli','ax']},
  **{ task:[F1Score(), accuracy] for task in ['mrpc', 'qqp']}, 
  **{ task:[PearsonCorrCoef(), SpearmanCorrCoef()] for task in ['stsb']}
}
NUM_CLASS = {
    **{ task:1 for task in ['stsb']},
    **{ task:2 for task in ['cola', 'sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli']},
    **{ task:3 for task in ['mnli','ax']},
}
TEXT_COLS = {
    **{ task:['question', 'sentence'] for task in ['qnli']},
    **{ task:['sentence1', 'sentence2'] for task in ['mrpc','stsb','wnli','rte']},
    **{ task:['question1','question2'] for task in ['qqp']},
    **{ task:['premise','hypothesis'] for task in ['mnli','ax']},
    **{ task:['sentence'] for task in ['cola','sst2']},
}
LOSS_FUNC = {
    **{ task: CrossEntropyLossFlat() for task in ['cola','sst2','mrpc','qqp','mnli','qnli','rte','wnli', 'ax']},
    **{ task: MyMSELossFlat(low=0.0, high=5.0) for task in ['stsb']}
}
if c.wsc_trick: 
  LOSS_FUNC['wnli'] = ELECTRAWSCTrickLoss()
  METRICS['wnli'] = [wsc_trick_accuracy]

def tokenize_sents_max_len(example, cols, max_len, swap=False):
  # Follow BERT and ELECTRA, truncate the examples longer than max length
  tokens_a = hf_tokenizer.tokenize(example[cols[0]])
  tokens_b = hf_tokenizer.tokenize(example[cols[1]]) if len(cols)==2 else []
  _max_length = max_len - 1 - len(cols) # preserved for cls and sep tokens
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= _max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()
  if swap:
    tokens_a, tokens_b = tokens_b, tokens_a
  tokens = [hf_tokenizer.cls_token, *tokens_a, hf_tokenizer.sep_token]
  token_type = [0]*len(tokens)
  if tokens_b: 
    tokens += [*tokens_b, hf_tokenizer.sep_token]
    token_type += [1]*(len(tokens_b)+1)
  example['inp_ids'] = hf_tokenizer.convert_tokens_to_ids(tokens)
  example['attn_mask'] = [1] * len(tokens)
  example['token_type_ids'] = token_type
  return example


# %%
glue_dsets = {}; glue_dls = {}
for task in ['cola', 'sst2', 'mrpc', 'stsb', 'mnli', 'qqp', 'qnli', 'rte', 'wnli', 'ax']:
# for task in ['cola']:

  # Load / download datasets.
  dsets = datasets.load_dataset('glue', task, cache_dir='./datasets')

  # There is two samples broken in QQP training set
  if task=='qqp': dsets['train'] = dsets['train'].filter(lambda e: e['question2']!='',
                        cache_file_name=os.path.join(dsets['train'].cache_directory(), 'fixed_train.arrow'))

  # Load / Make tokenized datasets
  tok_func = partial(tokenize_sents_max_len, cols=TEXT_COLS[task], max_len=c.max_length)
  glue_dsets[task] = dsets.my_map(tok_func, cache_file_names=f"tokenized_{c.max_length}_{{split}}")

  if c.double_unordered and task in ['mrpc', 'stsb']:
    swap_tok_func = partial(tokenize_sents_max_len, cols=TEXT_COLS[task], max_len=c.max_length, swap=True)
    swapped_train = dsets['train'].my_map(swap_tok_func, 
                                          cache_file_name=f"swapped_tokenized_{c.max_length}_train")
    glue_dsets[task]['train'] = datasets.concatenate_datasets([glue_dsets[task]['train'], swapped_train])

  # Load / Make dataloaders
  hf_dsets = HF_Datasets(glue_dsets[task], hf_toker=hf_tokenizer, n_inp=3,
                cols={'inp_ids':TensorText, 'attn_mask':noop, 'token_type_ids':noop, 'label':TensorCategory})
  if c.double_unordered and task in ['mrpc', 'stsb']:
    dl_kwargs = {'train': {'cache_name': f"double_dl_{c.max_length}_train.json"}}
  else: dl_kwargs = None
  glue_dls[task] = hf_dsets.dataloaders(bs=32, shuffle_train=True, num_workers=c.num_workers,
                                        cache_name=f"dl_{c.max_length}_{{split}}.json",
                                        dl_kwargs=dl_kwargs,)


# %%
if c.wsc_trick:
  wsc = datasets.load_dataset('super_glue', 'wsc', cache_dir='./datasets')
  glue_dsets['wnli'] = wsc.my_map(partial(wsc_trick_process, hf_toker=hf_tokenizer),
                                  cache_file_names="tricked_{split}.arrow")
  cols={'prefix':TensorText,'suffix':TensorText,'cands':TensorText,'cand_lens':noop,'label':TensorCategory}
  glue_dls['wnli'] = HF_Datasets(glue_dsets['wnli'], hf_toker=hf_tokenizer, n_inp=4, 
                                 cols=cols).dataloaders(bs=32, cache_name="dl_tricked_{split}.json")

class SentencePredictor(nn.Module):

  def __init__(self, model, hidden_size, num_class):
    super().__init__()
    self.base_model = model
    self.dropout = nn.Dropout(0.1)
    self.classifier = nn.Linear(hidden_size, num_class)
    if c.xavier_reinited_outlayer:
      nn.init.xavier_uniform_(self.classifier.weight.data)
      self.classifier.bias.data.zero_()

  def forward(self, input_ids, attention_mask, token_type_ids):
    x = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
    return self.classifier(self.dropout(x[:,0,:])).squeeze(-1).float() # if regression task, squeeze to (B), else (B,#class)

def list_parameters(model, submod_name):
  return list(eval(f"model.{submod_name}").parameters())

def hf_electra_param_splitter(model, wsc_trick=False):
  base = 'discriminator.electra' if wsc_trick else 'base_model'
  embed_name = 'embedding' if c.my_model else 'embeddings'
  scaler_name = 'dimension_scaler' if c.my_model else 'embeddings_project'
  layers_name = 'layers' if c.my_model else 'layer'
  output_name = 'classifier' if not wsc_trick else f'discriminator.discriminator_predictions'
  
  groups = [ list_parameters(model, f"{base}.{embed_name}") ]
  for i in range(electra_config.num_hidden_layers):
    groups.append( list_parameters(model, f"{base}.encoder.{layers_name}[{i}]") )
  groups.append( list_parameters(model, output_name) )
  # if electra_config.hidden_size != electra_config.embedding_size:
  #   groups[0] += list_parameters(model, f"{base}.{scaler_name}")
  if c.my_model and hparam['pre_norm']:
    groups[-2] += list_parameters(model, f"{base}.encoder.norm")

  assert len(list(model.parameters())) == sum([ len(g) for g in groups])
  for i, (p1, p2) in enumerate(zip(model.parameters(), [ p for g in groups for p in g])):
    assert torch.equal(p1, p2), f"The {i} th tensor"
  return groups

def get_layer_lrs(lr, decay_rate, num_hidden_layers):
  lrs = [ lr * (decay_rate ** depth) for depth in range(num_hidden_layers+2)]
  if c.original_lr_layer_decays:
    for i in range(1, len(lrs)): lrs[i] *= decay_rate
  return list(reversed(lrs))

def load_part_model(file, model, prefix, device=None, strict=True):
  "assume `model` is part of (child attribute at any level) of model whose states save in `file`."
  distrib_barrier()
  if prefix[-1] != '.': prefix += '.'
  # if isinstance(device, int): device = torch.device('cuda', device)
  # elif device is None: device = 'cpu'
  device = torch.device(c.device)
  state = torch.load(file, map_location=device)
  hasopt = set(state)=={'model', 'opt'}
  model_state = state['model'] if hasopt else state
  model_state = {k[len(prefix):] : v for k,v in model_state.items() if k.startswith(prefix)}
  get_model(model).load_state_dict(model_state, strict=strict)

def get_glue_learner(task, run_name=None, inference=False):
  is_wsc_trick = task=='wnli' and c.wsc_trick

  # Num_epochs
  if task in ['rte', 'stsb']: num_epochs = 10
  else: num_epochs = 3
  
  # Dataloaders
  dls = glue_dls[task]
  dls.to(torch.device(c.device))
  discriminator = BertForMaskedLM(electra_config)
  load_part_model(c.pretrained_ckp_path, discriminator, 'generator')

  torch.backends.cudnn.benchmark = True
  if c.seeds:
    dls[0].rng = random.Random(c.seeds[i]) # for fastai dataloader
    random.seed(c.seeds[i])
    np.random.seed(c.seeds[i])
    torch.manual_seed(c.seeds[i])

  # Create finetuning model
  if is_wsc_trick: 
    model = ELECTRAWSCTrickModel(discriminator, hf_tokenizer.pad_token_id)
  else:
    model = SentencePredictor(discriminator.bert, electra_config.hidden_size, num_class=NUM_CLASS[task])

  # Discriminative learning rates
  splitter = partial( hf_electra_param_splitter, wsc_trick=is_wsc_trick)
  layer_lrs = get_layer_lrs(lr=c.lr, 
                            decay_rate=c.layer_lr_decay,
                            num_hidden_layers=electra_config.num_hidden_layers,)
  
  # Optimizer
  if c.adam_bias_correction: opt_func = partial(Adam, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=c.weight_decay)
  else: opt_func = partial(Adam_no_bias_correction, eps=1e-6, mom=0.9, sqr_mom=0.999, wd=c.weight_decay)
  
  # Learner
  learn = Learner(dls, model,
                  loss_func=LOSS_FUNC[task], 
                  opt_func=opt_func,
                  metrics=METRICS[task],
                  splitter=splitter if not inference else trainable_params,
                  lr=layer_lrs if not inference else defaults.lr,
                  path='./checkpoints/glue',
                  model_dir=c.group_name,)

  # Multi gpu
  # if isinstance(c.device, list) or c.device is None:
  #   learn.create_opt()
  #   learn.model = nn.DataParallel(learn.model, device_ids=c.device)



  # Mixed precision
  learn.to_native_fp16(init_scale=2.**14)

  # Gradient clip
  learn.add_cb(GradientClipping(1.0))

  # Logging
  if run_name and not inference:
    if c.logger == 'neptune':
      neptune.create_experiment(name=run_name, params={'task':task, **c, **hparam_update})
      learn.add_cb(LightNeptuneCallback(False))
    elif c.logger == 'wandb':
      wandb_run = wandb.init(name=run_name, project='electra_glue', config={'task':task, **c, **hparam_update}, reinit=True)
      learn.add_cb(LightWandbCallback(wandb_run))

  learn.create_opt()
  learn.model = nn.DataParallel(learn.model)

  # Learning rate schedule
  if c.schedule == 'one_cycle': 
    return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs, lr_max=layer_lrs)
  elif c.schedule == 'adjusted_one_cycle':
    return learn, partial(learn.fit_one_cycle, n_epoch=num_epochs, lr_max=layer_lrs, div=1e5, pct_start=0.1)
  else:
    lr_shed_func = linear_warmup_and_then_decay if c.schedule=='separate_linear' else linear_warmup_and_decay
    lr_shedule = ParamScheduler({'lr': partial(lr_shed_func,
                                               lr_max=np.array(layer_lrs),
                                               warmup_pct=0.1,
                                               total_steps=num_epochs*(len(dls.train)))})

    return learn, partial(learn.fit, n_epoch=num_epochs, cbs=[lr_shedule])

# %% [markdown]
# ## 2.4 Do finetuning

# %%
if c.do_finetune:
  for i in range(c.start, c.end):
    for task in ['cola', 'sst2', 'mrpc', 'stsb', 'rte', 'qnli', 'qqp', 'mnli', 'wnli']:
      if c.group_name: run_name = f"{c.group_name}_{task}_{i}";
      else: run_name = None; print(task)
      learn, fit_fc = get_glue_learner(task, run_name)
      fit_fc()
      if run_name: learn.save(f"{task}_{i}")

# %% [markdown]
# # 3. Testing

# %%
# Haven't found way to validate and log two datasets in the training loop, so validate mnli-mm here as a workaround
if not c.do_finetune:
  learn, _ = get_glue_learner('mnli', inference=True)
  learn.load(f"mnli_{c.th_run['mnli']}")
  with learn.no_mbar():
    print(learn.validate(ds_idx=2))


# %%
def get_identifier(task, split):
  "Turn task name to official task identifier defined."
  map = {'cola': 'CoLA', 'sst2':'SST-2', 'mrpc':'MRPC', 'qqp':'QQP', 'stsb':'STS-B', 'qnli':'QNLI', 'rte':'RTE', 'wnli':'WNLI', 'ax':'AX'}
  if task =='mnli' and split == 'test_matched': return 'MNLI-m'
  elif task == 'mnli' and split == 'test_mismatched': return 'MNLI-mm'
  else: return map[task]

def predict_test(task, checkpoint, dl_idx):
  output_dir = Path(f'./test_outputs/glue/{c.group_name}')
  output_dir.mkdir(exist_ok=True)
  device = torch.device(c.device)

  # Load checkpoint and get predictions
  learn, _ = get_glue_learner(task, inference=True)
  if task == 'wnli' and c.wsc_trick:
    load_model_(learn, checkpoint, merge_out_fc=wsc_trick_merge)
  else:
    load_model_(learn, checkpoint)
  results = learn.get_preds(dl=learn.dls[dl_idx], with_decoded=True)
  preds = results[-1] # preds -> (predictions logits, targets, decoded prediction)

  # Decode target class index to its class name 
  if task in ['mnli','ax']:
    preds = [ ['entailment','neutral','contradiction'][p] for p in preds]
  elif task in ['qnli','rte']: 
    preds = [ ['entailment','not_entailment'][p] for p in preds ]
  elif task == 'wnli' and c.wsc_trick:
    preds = preds.to(dtype=torch.long).tolist()
  else: preds = preds.tolist()
    
  # Form test dataframe and save
  test_df = pd.DataFrame( {'index':range(len(list(glue_dsets[task].values())[dl_idx])), 'prediction': preds} )
  split = list(glue_dsets['mnli'].keys())[dl_idx] if task == 'mnli' else 'test'
  identifier = get_identifier(task, split)
  test_df.to_csv( output_dir/f'{identifier}.tsv', sep='\t' )
  return test_df


# %%
if not c.do_finetune:
  for task, th in c.th_run.items():
    print(task)
    # ax use mnli ckp
    if isinstance(th, int):
      ckp = f"{task}_{th}" if task != 'ax' else f"mnli_{th}"
    else:
      ckp = [f"{task}_{i}" if task != 'ax' else f"mnli_{i}" for i in th]
    # run test for all testset in this task
    dl_idxs = [-1, -2] if task=='mnli' else [-1]
    for dl_idx in dl_idxs:
      df = predict_test(task, ckp, dl_idx)


