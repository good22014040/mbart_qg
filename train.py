import torch
import math
from tqdm.auto import tqdm, trange
import transformers
from accelerate import Accelerator
from transformers import (
    get_scheduler,
    MBartForConditionalGeneration, 
    MBart50TokenizerFast
)

from get_dataset import *
from config import *
from get_optimizer import *

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="zh_CN", tgt_lang="zh_CN")

train_dataset = get_dataset('drcd/train.json', tokenizer)
eval_dataset = get_dataset('drcd/dev.json', tokenizer)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
optimizer = get_optimizer(model)

# Scheduler and math around the number of training steps.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = num_train_epochs * num_update_steps_per_epoch

# scheduler
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=max_train_steps,
)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# Train!
output_dir = 'model/'

total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
completed_steps = 0
best_epoch = {"epoch:": 0, "loss": float('Inf')}

for epoch in trange(num_train_epochs, desc="Epoch"):
  model.train()
  train_loss = 0
  for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    outputs = model(**batch)
    loss = outputs.loss
    train_loss += loss.item()
    loss = loss / gradient_accumulation_steps
    accelerator.backward(loss)
    if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()
      completed_steps += 1

    if step % 100 == 0:
      print({'epoch': epoch, 'step': step, 'loss': loss.item()})

    if completed_steps >= max_train_steps:
      break
  train_loss /= len(train_dataloader)
  print(f"epoch : {epoch}, train_loss: {train_loss}")   

  model.eval()
  eval_loss = 0
  for step, batch in enumerate(tqdm(eval_dataloader, desc="Eval Iteration")):
    outputs = model(**batch)
    eval_loss += outputs.loss.item()

  eval_loss /= len(eval_dataloader)
  print(f"epoch : {epoch}, eval_loss: {eval_loss}")
  if eval_loss < best_epoch['loss']:
    best_epoch['epoch'] = epoch
    best_epoch['loss'] = eval_loss


  if output_dir is not None:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(f"{output_dir}epoch_{str(epoch)}_train_loss={train_loss}_eval_loss={eval_loss}/", save_function=accelerator.save)
