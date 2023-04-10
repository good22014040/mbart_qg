from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def predict(context, answer, model, tokenizer):
        input = tokenizer(
                        context, 
                        answer , 
                        truncation=True, 
                        padding=True,
                        return_tensors="pt"
                ).to(device)
        with torch.no_grad():
                outputs = model.generate(
                        **input,
                        max_length=128,
                        early_stopping=True,
                        num_beams=5,
                        no_repeat_ngram_size=5,
                        num_return_sequences=1,
                )
        
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

model_path = "model_size_16/epoch_3_train_loss=0.1130604538561061_eval_loss=0.21756762835629512"
model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")

while(True):
        context = input("context : ")
        answer = input("answer : ")
        if(context == "0"):
                break
        
        predict(context, answer, model, tokenizer)