from datasets import load_from_disk
import datasets 
import sys
from transformers import LayoutLMv2Processor
from data_args import DataTrainingArguments
from model_args import ModelArguments
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser,TrainingArguments,LayoutLMv2ForTokenClassification, AdamW
)
from datasets import load_metric
import torch
from tqdm.notebook import tqdm


metric = load_metric("seqeval")
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

model_args, data_args, training_args = parser.parse_args_into_dataclasses()


train_dataset = load_from_disk('data/train')
eval_dataset = load_from_disk('data/eval')

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset.set_format(type="torch", device=device)
eval_dataset.set_format(type="torch", device=device)

train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_gpu_train_batch_size, shuffle=True)
test_dataloader = DataLoader(eval_dataset, batch_size=training_args.per_gpu_eval_batch_size)

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                          num_labels=25)


model.to(device)
optimizer = AdamW(model.parameters(), lr=training_args.lr)

global_step = 0
num_train_epochs = training_args.num_train_epochs
t_total = len(train_dataloader) * num_train_epochs

#put the model in training mode
model.train()
for epoch in range(num_train_epochs):  
   print("Epoch:", epoch)
   for batch in tqdm(train_dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch) 
        loss = outputs.loss
        
        # print loss every 100 steps
        if global_step % 100 == 0:
          print(f"Loss after {global_step} steps: {loss.item()}")

        loss.backward()
        optimizer.step()
        global_step += 1

        if global_step % training_args.eval_steps == 0:
            # put model in evaluation mode
            model.eval()
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(device)
                    bbox = batch['bbox'].to(device)
                    image = batch['image'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    labels = batch['labels'].to(device)

                    # forward pass
                    outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                                    token_type_ids=token_type_ids, labels=labels)
                    
                    # predictions
                    predictions = outputs.logits.argmax(dim=2)

                    # Remove ignored index (special tokens)
                    true_predictions = [
                        [p.item() for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)
                    ]
                    true_labels = [
                        [l.item() for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)
                    ]

                    metric.add_batch(predictions=true_predictions, references=true_labels)

            final_score = metric.compute()
            print(final_score)