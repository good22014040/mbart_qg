import torch
import math
from tqdm.auto import tqdm, trange
import transformers
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast
)
from rouge import Rouge 

from get_dataset import *
from config import *
from get_optimizer import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="zh_CN", tgt_lang="zh_CN")


model_path = best_model
test_dataset = get_dataset('drcd/DRCD_test.json', tokenizer)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=30)
model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)




model.eval()
with torch.no_grad():
    rouge = Rouge()
    total_question = 0
    rouge_score = {"rouge-1" : 0, "rouge-2" : 0, "rouge-l" : 0}
    for step, batch in enumerate(tqdm(test_dataloader, desc="Eval Iteration")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        outputs = model.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_length=128,
                early_stopping=True,
                num_beams=5,
                no_repeat_ngram_size=5,
                num_return_sequences=1,)
        for output, label in zip(outputs,labels):
            hypothesis = tokenizer.decode(output, skip_special_tokens=True)
            reference = tokenizer.decode(label, skip_special_tokens=True)
            hypothesis = " ".join([word for word in hypothesis])
            reference = " ".join([word for word in reference])

            total_question += 1
            scores = rouge.get_scores(hypothesis, reference)
            rouge_score["rouge-1"] += scores[0]["rouge-1"]["f"]
            rouge_score["rouge-2"] += scores[0]["rouge-2"]["f"]
            rouge_score["rouge-l"] += scores[0]["rouge-l"]["f"]
    rouge_score["rouge-1"] /= total_question
    rouge_score["rouge-2"] /= total_question
    rouge_score["rouge-l"] /= total_question
    print(rouge_score)
    
