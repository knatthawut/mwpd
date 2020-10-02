import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer,AutoModelForSequenceClassification,AdamW
from sklearn.metrics import classification_report
from config import Config
from utils import ProductCorpus
import time
import datetime
from tqdm import tqdm
import argparse

# Allocate gpu0
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train(config):
    # Build tokenizer and bert model
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    model = AutoModelForSequenceClassification.from_pretrained(config.model)
    model.cuda()   # Load this model on the GPU.
    model.train()  # set model to train mode

    # Skip training if the model exist
    if config.exist_model != '':
        print("Loading model from : ",config.exist_model)
        checkpoint = torch.load(config.exist_model)
        model.load_state_dict(checkpoint['model'])
        return model

    #Build Dataloader
    pc = ProductCorpus(config.train_data, tokenizer) # Load dataset
    dataloader = DataLoader(pc, sampler=RandomSampler(pc),  batch_size=config.batch_size)

    # Build optimizer and scheduler
    optimizer = AdamW(model.parameters(),lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0, num_training_steps=len(dataloader) * config.epochs)

    # ========================================
    #               Training
    # ========================================
    for epoch_i in range(0, config.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config.epochs))
        print('Training...')

        total_train_loss = 0  # Reset loss for the epoch
        t0 = time.time()  # set timer

        # For each batch of training data...
        for step, batch in tqdm(enumerate(dataloader),total=len(dataloader)):
            if step % config.log_interval == 0 and step > 0: #print out every log interval
                elapsed = str(datetime.timedelta(seconds=time.time() - t0))
                cur_loss = total_train_loss / (step * config.batch_size)
                print('Batch {:>5,}  of  {:>5,}.    Loss: {:.5f}.    Elapsed: {:}.'.format(step, len(dataloader), cur_loss, elapsed))

            #unpack data from batch and load to device (CPU or GPU)
            b_input_ids = torch.squeeze(batch['input'],1).to(config.device)
            b_input_attn = torch.squeeze(batch['attn'],1).to(config.device)
            b_token_type_ids = torch.squeeze(batch['token'],1).to(config.device)
            b_labels = batch['label'].to(config.device)

            # Perform a forward pass
            loss, logits = model(b_input_ids,
                                 token_type_ids=b_token_type_ids,
                                 attention_mask=b_input_attn,
                                 labels=b_labels)
            # Accumulate the training loss over all of the batches
            total_train_loss += loss.item()

            loss.backward()     # Perform a backward pass to calculate the gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # Clip the norm of the gradients to 1.0.
            optimizer.step()    # Update parameters and take a step using the computed gradient.
            scheduler.step()    # Update the learning rate.
            model.zero_grad()   # Clear gradients before performing a backward

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(dataloader)
        training_time = str(datetime.timedelta(seconds=time.time() - t0))
        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Training epcoh took: {:}".format(training_time))

    #evaluate model
    print("\nRunning Train...")
    eval(config, model, pc)

    #get state parameter and save
    checkpoint = {'model': model.state_dict()}
    torch.save(checkpoint, config.model_path + config.model + '.pt')

    return model

def test(config,model):
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    test = ProductCorpus(config.test_data, tokenizer)  # Load dataset
    pair_ids = test.getColumn('pair_id').to_list()  # get pair ids

    print("\nRunning Test...")
    predicts = eval(config, model, test)    # predict label

    #Save result
    result = pd.DataFrame(list(zip(pair_ids, predicts)), columns=['ID', 'predict_val'])
    result.to_csv(config.result_path + config.model + '.csv', index = False)

def eval(config, model, dataset):

    # set model to test mode
    model.eval()
    # Build Testdata
    tester = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=config.batch_size)

    # Collect label
    predict_label = []
    gold_label = []
    t0 = time.time()  # set timer

    # Evaluate data for one epoch
    for idx, batch in tqdm(enumerate(tester), total=len(tester)):
        # unpack data from batch and load to device (CPU or GPU)
        b_input_ids = torch.squeeze(batch['input'], 1).to(config.device)
        b_input_attn = torch.squeeze(batch['attn'], 1).to(config.device)
        b_token_type_ids = torch.squeeze(batch['token'], 1).to(config.device)
        b_labels = batch['label'].to(config.device)

        # Tell pytorch not to construct the compute graph
        with torch.no_grad():
            output = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_attn)
        # Move logits and labels to CPU
        logits = output[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predict_label += np.argmax(logits, axis=1).tolist()
        gold_label += label_ids.tolist()

    # Measure how long the test run.
    print("Evaluation took: {:}".format(str(datetime.timedelta(seconds=time.time() - t0))))
    # Classification Report Score
    print(classification_report(gold_label, predict_label))

    return predict_label

# Aggrate the results from different models
def aggegrate(config):
    df = pd.read_json(config.test_data, lines=True)
    for target in config.target_list:
        # Remove Duplicate Col when merge
        if 'ID' in df:
            del df['ID']
        tmp = pd.read_csv(config.result_path + target + '.csv')
        tmp = tmp.rename(columns={'predict_val': target})
        df = pd.merge(df, tmp, how='inner', left_on='pair_id', right_on='ID')

    df['sum'] = df[config.target_list].sum(axis='columns') # Sum from classifier results
    df['final_predict'] = (df['sum'] > 0).astype(int)
    df.loc[:,['pair_id','final_predict']].to_csv(config.result_path + 'final.csv', index=False)

if __name__ == "__main__":
    config = Config()   #load all config

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="load the existing model")
    args = parser.parse_args()
    if args.model:
        config.exist_model = args.model

    model = train(config)
    test(config, model) # Test on the test data
    aggegrate(config)   # produce final output
