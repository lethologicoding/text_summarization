from os.path import join, isfile
from os import listdir
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from rouge_score import rouge_scorer
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler #Dataset,
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration

#######################
MODEL_NAME = 't5-base'
#######################


class T5FineTuner(pl.LightningModule):
    '''
    Documentation-In-Progress
    '''

    def __init__(self, df = pd.DataFrame):
        super().__init__()
        self.save_hyperparameters()
        self.source_len = 512
        self.summ_len = 200
        self.lr = .0001
        self.bs = 8
        self.num_workers = 8
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        self.data = df
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.output = 'temp/'
        self.name = 'test'

    def encode_text(self, context, text):
        ctext = str(context) # context text 
        ctext = ' '.join(ctext.split())
        text = str(text) # summarized text
        text = ' '.join(text.split())
        source = self.tokenizer.batch_encode_plus([ctext], 
                                                max_length= self.source_len, 
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], 
                                                max_length= self.summ_len,
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors='pt')
        y = target['input_ids']
        target_id = y[:, :-1].contiguous()
        target_label = y[:, 1:].clone().detach()
        target_label[y[:, 1:] == self.tokenizer.pad_token_id] = -100 #edge-case  handling when no labels are there
        return source['input_ids'], source['attention_mask'], target_id, target_label
    
    def prepare_data(self):
        source_ids, source_masks, target_ids, target_labels = [], [], [], [] 
        for _, row in self.data.iterrows():
            source_id, source_mask, target_id, target_label = self.encode_text(row.ctext, row.text)
            source_ids.append(source_id)
            source_masks.append(source_mask)
            target_ids.append(target_id)
            target_labels.append(target_label)

        # Transforming lists into tensors
        source_ids = torch.cat(source_ids, dim=0)
        source_masks = torch.cat(source_masks, dim=0)
        target_ids = torch.cat(target_ids, dim=0)
        target_labels = torch.cat(target_labels, dim=0)
        # Splitting data into standard train, val, and test sets 
        data = TensorDataset(source_ids, source_masks, target_ids, target_labels)
        train_size, val_size = int(0.8 * len(data)), int(0.1 * len(data))
        test_size = len(data) - (train_size + val_size)
        self.train_dat, self.val_dat, self.test_dat = \
            random_split(data, [train_size, val_size, test_size])
    
    def forward(self, batch, batch_idx):
        source_ids, source_mask, target_ids, target_labels = batch[:4]
        return self.model(
            input_ids = source_ids, 
            attention_mask = source_mask, 
            decoder_input_ids=target_ids, 
            labels=target_labels
        )
        
    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log('train loss', loss, prog_bar = True, logger = True)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log('valid loss', loss, prog_bar = True, logger = True)
        return {'loss': loss}

    def validation_epoch_end(self, outputs): 
        loss = sum([o['loss'] for o in outputs]) / len(outputs)
        out = {'val_loss': loss}
        return {**out, 'log': out}

    def test_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log('test loss', loss, prog_bar = True, logger = True)
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        loss = sum([o['loss'] for o in outputs]) / len(outputs)
        out = {'test_loss': loss}
        return {**out, 'log': out}
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dat,
            batch_size=self.bs,
            num_workers=self.num_workers, 
            sampler=RandomSampler(self.train_dat)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dat, 
            batch_size=self.bs, 
            num_workers=self.num_workers,
            sampler=SequentialSampler(self.val_dat)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dat, 
            batch_size=self.bs, 
            num_workers=self.num_workers,
            sampler=SequentialSampler(self.test_dat)
        )    

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-4)
        return {'optimizer': optimizer}
    
    def save_core_model(self):
        store_path = join(self.output, self.name, 'core')
        self.model.save_pretrained(store_path)
        self.tokenizer.save_pretrained(store_path)