import argparse
import pathlib

import pandas as pd

from sklearn import model_selection

import torch
from transformers import BertModel

from jeopardy_pytorch_dataset import JeopardyDataset, tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
eval_every = 1
batch_size = 12


class ValueRegressor(torch.nn.Module):
	def __init__(self):
		super(ValueRegressor, self).__init__()

		self.bert = BertModel.from_pretrained('bert-base-uncased')
		#self.bert.resize_token_embeddings(len(tokenizer))
		self.dropout = torch.nn.Dropout(0.5)
		self.linear = torch.nn.Linear(768+2,1)

	def forward(self, ids, attn_mask, air_date, rounds):
		_, logits = self.bert(ids, attention_mask=attn_mask, token_type_ids=None, return_dict=False)
		logits = self.dropout(logits)

		## Concat Inputs
		feats = torch.cat((logits, air_date.unsqueeze(1), rounds.unsqueeze(1)), 1)
		value = self.linear(feats)
		return value

	def inference(self, ids, attn_mask, air_date, rounds):
		_, logits = self.bert(ids, attention_mask=attn_mask, token_type_ids=None, return_dict=False)
		logits = self.dropout(logits)

		## Concat Inputs
		feats = torch.cat((logits, air_date.unsqueeze(1), rounds.unsqueeze(1)), 1)
		value = self.linear(feats)
		return value


def train(model, df, criterion, optimizer, epoch):
	model.train()
	training_set = JeopardyDataset(df, max_seq_length=256, device=device)
	training_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)

	total_loss = 0
	for text_ids, attn_mask, air_date, rounds, values in training_dataloader:
		values = values.float()

		optimizer.zero_grad()
		y_preds = model(text_ids, attn_mask, air_date, rounds)
		loss = criterion(y_preds.squeeze(), values)
		loss = torch.sqrt(loss)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	print(f"Total training loss for epoch: {epoch} is: {total_loss/len(training_set)}")

def eval(model, df, criterion, optimizer, epoch):
	with torch.no_grad():
		model.eval()
		eval_set = JeopardyDataset(df, max_seq_length=256, device=device)
		eval_dataloader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size)

		total_loss = 0
		for text_ids, attn_mask, air_date, rounds, values in eval_dataloader:
			y_preds = model.inference(text_ids, attn_mask, air_date, rounds)
			
			loss = criterion(y_preds.squeeze(), values)
			loss = torch.sqrt(loss)

			total_loss += loss.item()
		print(f"Total eval loss for epoch: {epoch} is: {total_loss/len(eval_set)}")


def main(args):
	df = pd.read_csv(args.input_filename).sample(frac = args.data_frac)
	df.dropna(inplace=True)

	## Generate_text_feats
	df.loc[:, "text_feats"] = df.loc[:, " Question"] +" "+df.loc[:, " Answer"] +" "+ df.loc[:, " Category"]

	## Generate train, validation pair
	df_train, df_eval = model_selection.train_test_split(df, train_size=0.7)

	model = ValueRegressor()
	model.to(device)

	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(
	[
		{"params":model.bert.parameters(),"lr": 1e-4},
		{"params":model.linear.parameters(), "lr": 1e-3},
	   
   ])

	for epoch in range(args.epochs):
		train(model, df_train, criterion, optimizer, epoch)
		if epoch%eval_every == 0:
			eval(model, df_eval, criterion, optimizer, epoch)

	


if __name__=="__main__":
	parser = argparse.ArgumentParser(description = "Use this script to finetune bert. Example: python3 finetune_bert.py <input filepath> --epochs 5 --data_frac 0.1")
	parser.add_argument("input_filename", metavar="fi", type=pathlib.Path, help="Path to input csv file.")
	parser.add_argument("--epochs", metavar="num_e", type=int, default= 5, help="Number of epochs to train on, default 5")
	parser.add_argument("--data_frac", metavar="num_e", type=float, default= 1, help="\% of data to train on. Default=1")
	args = parser.parse_args()
	assert (args.data_frac<=1 and args.data_frac>0), "Data frac should strictly be in range 0-1"

	print ("Finetuning bert")
	print (f"Using: {device} for training")
	main(args)