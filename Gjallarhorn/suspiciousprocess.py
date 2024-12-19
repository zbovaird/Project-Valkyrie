import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import string

# Model parameters
all_letters = string.ascii_lowercase
n_letters = len(all_letters)
n_hidden = 32
n_categories = 2
learning_rate = 5e-4

class ProcessnameClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProcessnameClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def evaluate(model, line_tensor):
    hidden = model.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    return output

def preprocess_txt(input_line):
    input_line = input_line.split('/')[-1]
    input_line = input_line.lower()
    input_line = input_line.replace('.exe','')
    input_line = ''.join(filter(str.islower, input_line))
    return input_line

def load_model(model_path):
    model = ProcessnameClassifier(n_letters, n_hidden, n_categories)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_process(model, process_name):
    input_line = preprocess_txt(process_name)
    if len(input_line) == 0:
        return 1.0
        
    with torch.no_grad():
        output = evaluate(model, lineToTensor(input_line))
    output = torch.exp(output)
    return round(output[0][1].item(), 4)

def predict_batch(model, df):
    is_malicious_prob_lst = []
    for idx, row in df.iterrows():
        is_mal_prob = predict_process(model, row['text'])
        is_malicious_prob_lst.append(is_mal_prob)
    
    output = pd.DataFrame()
    output['is_malicious_prob'] = is_malicious_prob_lst
    return output