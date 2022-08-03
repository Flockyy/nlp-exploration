import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np

class BERTClassification(nn.Module):
    def __init__ (self):
        super(BERTClassification, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-cased')
        self.bert_drop = nn.Dropout(0.4)
        self.out = nn.Linear(768, 1)
        
    def forward(self, ids, mask, token_type_ids):
        _, pooledOut = self.bert(ids, attention_mask = mask,
                                token_type_ids=token_type_ids)
        bertOut = self.bert_drop(pooledOut)
        output = self.out(bertOut)
        
        return output
    
class DATALoader:
    def __init__(self, data, target, max_length):
        self.data = data
        self.target = target #make sure to convert the target into numerical values
        self.tokeniser = transformers.BertTokenizer.from_pretrained('bert-base-cased')
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        data = str(self.data[item])
        data = " ".join(data.split())
        
        inputs = self.tokeniser.encode_plus(
            data, 
            None,
            add_special_tokens=True,
            max_length = self.max_length,
            pad_to_max_length=True
            
        )
        
        ids = inputs["input_ids"]
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        padding_length = self.max_length - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.long)
        }
    
    
   

def loss_fn(output, targets):
    return nn.BCEWithLogitsLoss()(output, targets.view(-1,1))


def train_func(data_loader, model, optimizer, device, scheduler):
    model.to(device)
    model.train()
    
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        output = model(
            ids=ids,
            mask = mask,
            token_type_ids = token_type_ids
        )
        
        
        loss = loss_fn(output, targets)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        
def eval_func(data_loader, model, device):
    model.eval()
    
    fin_targets = []
    fin_output = []
    
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)


            output = model(
                ids=ids,
                masks = mask,
                token_type_ids = token_type_ids
            )
        
            fin_targets.extend(targets.cpu().detach().numpy().to_list())
            fin_targets.extend(torch.sigmoid(output).cpu().detach().numpy().to_list())
            
        return fin_output, fin_targets
    
def run():
    df = pd.read_json('', lines = True)
    data = pd.DataFrame({
        'text' : df['headline'] + df['short_description'],
        'label' : df['category']
    })


    encoder = LabelEncoder()
    data['label'] = encoder.fit_transform(data['label'])

    df_train, df_valid = train_test_split(data, test_size = 0.1, random_state=23, stratify=data.label.values)

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = DATALoader(
        data=df_train.text.values,
        target=df_train.label.values,
        max_length=512
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=8,
        num_workers=4,
    )

    val_dataset = DATALoader(
        data=df_valid.text.values,
        target=df_valid.label.values,
        max_length=512
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=4,
        num_workers=1,
    )

    device = torch.device("cuda")
    model = BERTClassification()

    param_optimizer = list(model.named_parameters())
    no_decay = [
        "bias", 
        "LayerNorm,bias",
        "LayerNorm.weight",
               ]
    optimizer_parameters = [
        {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
                   'weight_decay':0.001},
        {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
                   'weight_decay':0.0}
    ]

    num_train_steps = int(len(df_train)/ 8*10)

    optimizers = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizers,
        num_warmup_steps=0,
        num_training_steps=num_train_steps

    )

    best_accuracy = 0
    for epoch in range(5):
        train_func(data_loader=train_data_loader, model=model, optimizer=optimizers, device=device, scheduler=scheduler)
        outputs, targets = eval_func(data_loader=train_data_loader, model=model, device=device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score()
        print(f"Accuracy Score: {accuracy}")

        if accuracy>best_accuracy:
            torch.save(model.state_dict(), "model.bin")
            best_accuracy = accuracy
                
                
if __name__ == "__main__":
    run()