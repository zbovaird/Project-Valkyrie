import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import string
from uhg import UHG, UHGLayer
import torch.nn.functional as F

# Model parameters
all_letters = string.ascii_lowercase
n_letters = len(all_letters)
n_hidden = 32
n_categories = 2
learning_rate = 5e-4
uhg_dim = 4  # Dimension of hyperbolic space
curvature = 1.0  # Curvature of hyperbolic space

class UHGProcessClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, uhg_dim, curvature):
        super(UHGProcessClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.uhg_dim = uhg_dim
        self.curvature = curvature
        
        # UHG layers for processing in hyperbolic space
        self.uhg = UHG(curvature=curvature)
        self.embed = nn.Linear(input_size, uhg_dim)
        self.uhg_layer = UHGLayer(uhg_dim, hidden_size)
        
        # Output layers
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        # Project input into hyperbolic space
        batch_size = input_tensor.size(0)
        embedded = self.embed(input_tensor)
        
        # Map to hyperbolic space using UHG
        hyperbolic_repr = self.uhg.expmap0(embedded)
        
        # Process in hyperbolic space
        hidden = self.uhg_layer(hyperbolic_repr)
        
        # Map back to Euclidean space for classification
        output = self.uhg.logmap0(hidden)
        output = self.output(output)
        output = self.softmax(output)
        
        return output

def letterToIndex(letter):
    return all_letters.find(letter)

def processToTensor(process_name):
    # Convert process name to character frequency vector
    freq = torch.zeros(n_letters)
    for letter in process_name:
        if letter in all_letters:
            freq[letterToIndex(letter)] += 1
    # Normalize
    if freq.sum() > 0:
        freq = freq / freq.sum()
    return freq

def preprocess_txt(input_line):
    input_line = input_line.split('/')[-1]
    input_line = input_line.lower()
    input_line = input_line.replace('.exe','')
    input_line = ''.join(filter(str.islower, input_line))
    return input_line

def load_model(model_path):
    model = UHGProcessClassifier(n_letters, n_hidden, n_categories, uhg_dim, curvature)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_process(model, process_name):
    input_line = preprocess_txt(process_name)
    if len(input_line) == 0:
        return 1.0
    
    with torch.no_grad():
        input_tensor = processToTensor(input_line).unsqueeze(0)
        output = model(input_tensor)
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