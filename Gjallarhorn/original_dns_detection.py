"""
Original DNS Exfiltration Detection
Simple deep learning approach with fixed dimensions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import string
import math
import random
from collections import Counter
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Constants
PRINTABLE_CHARS = list(string.printable.strip())
BATCH_SIZE = 256
THRESHOLD = 0.5

def generate_normal_domain() -> str:
    """Generate extremely sophisticated legitimate domains for a credit card processing company."""
    tlds = ['.com', '.net', '.org', '.io', '.cloud', '.internal', '.payments', '.secure', '.pci', '.bank']
    payment_words = ['api', 'gateway', 'payment', 'process', 'auth', 'verify', 'secure', 'token', 
                    'vault', 'merchant', 'billing', 'checkout', 'pci', 'card', 'crypto', 'settle',
                    'clearing', 'batch', 'realtime', 'fraud', 'risk', 'compliance', 'kyc', 'aml',
                    'dispute', 'chargeback', 'reconcile', 'settlement', 'escrow', '3ds', 'ach']
    partners = ['visa', 'mastercard', 'amex', 'discover', 'stripe', 'square', 'paypal', 'adyen',
               'worldpay', 'chase', 'firstdata', 'fiserv', 'cybersource', 'braintree', 'klarna',
               'affirm', 'shopify', 'checkout', 'globalpay', 'paysafe', 'authnet', 'spreedly',
               'recurly', 'chargebee', 'mollie', 'wise', 'plaid']
    environments = ['prod', 'staging', 'dev', 'qa', 'uat', 'sandbox', 'canary', 'blue', 'green',
                   'perf', 'stress', 'security', 'audit', 'backup', 'dr', 'failover']
    regions = ['us', 'eu', 'ap', 'na', 'global', 'latam', 'emea', 'apac', 'cn', 'in', 'jp', 
              'kr', 'sg', 'au', 'br', 'mx', 'uk', 'de', 'fr', 'it']
    
    domain_types = [
        # Complex microservices with detailed versioning and health checks
        lambda: f"svc-{random.choice(payment_words)}-v{random.randint(1,5)}.{random.randint(0,9)}-" + \
               f"{random.choice(environments)}-{random.choice(regions)}-" + \
               f"health-{random.choice(['ok', 'degraded', 'critical'])}.{random.choice(partners)}{random.choice(tlds)}",
        
        # Sophisticated API gateway routing
        lambda: f"api-gw-{random.choice(regions)}-{random.choice(['primary', 'secondary', 'fallback'])}-" + \
               f"route{random.randint(1,9)}-shard-{random.randint(1,5)}.{random.choice(partners)}{random.choice(tlds)}",
        
        # Complex feature flags with A/B testing
        lambda: f"exp-{random.choice(payment_words)}-cohort-{random.randint(1,5)}-" + \
               f"{random.choice(['control', 'variant-a', 'variant-b', 'shadow'])}-" + \
               f"{random.choice(environments)}.{random.choice(partners)}{random.choice(tlds)}",
        
        # Detailed monitoring and metrics endpoints
        lambda: f"monitor-{random.choice(['latency', 'errors', 'throughput', 'saturation'])}-" + \
               f"{random.choice(regions)}-{random.choice(environments)}-" + \
               f"aggregation-{random.randint(1,15)}min.metrics{random.choice(tlds)}",
        
        # Complex compliance and audit logging
        lambda: f"{random.choice(['audit', 'compliance', 'pci'])}-{random.choice(['log', 'trace', 'event'])}-" + \
               f"{random.choice(regions)}-{random.choice(environments)}-" + \
               f"retention-{random.randint(30,365)}d.{random.choice(partners)}{random.choice(tlds)}",
        
        # Sophisticated payment processing pipelines
        lambda: f"payment-pipeline-{random.choice(['ingress', 'process', 'enrich', 'validate', 'settle'])}-" + \
               f"step-{random.randint(1,5)}-{random.choice(regions)}.{random.choice(partners)}{random.choice(tlds)}",
        
        # Complex service mesh topology
        lambda: f"mesh-{random.choice(payment_words)}-{random.choice(environments)}-" + \
               f"dc-{random.choice(regions)}-zone-{random.choice(['a', 'b', 'c'])}-" + \
               f"replica-{random.randint(1,5)}.svc{random.choice(tlds)}",
        
        # Detailed transaction processing stages
        lambda: f"txn-{random.choice(['auth', 'capture', 'settle', 'refund', 'void'])}-" + \
               f"{random.choice(['sync', 'async', 'batch'])}-" + \
               f"priority-{random.choice(['high', 'medium', 'low'])}-" + \
               f"v{random.randint(1,3)}.{random.choice(partners)}{random.choice(tlds)}",
        
        # Complex load balancer configurations
        lambda: f"lb-{random.choice(regions)}-{random.choice(['internal', 'external'])}-" + \
               f"tier-{random.randint(1,3)}-{random.choice(['ssl', 'tcp', 'http'])}-" + \
               f"backend-{random.randint(1,5)}.{random.choice(partners)}{random.choice(tlds)}"
    ]
    
    domain = random.choice(domain_types)()
    
    # Add sophisticated complexity layers
    if random.random() < 0.20:  # Transaction and trace IDs
        parts = domain.split('.')
        trace_id = ''.join(random.choices(string.hexdigits.lower(), k=16))
        span_id = ''.join(random.choices(string.hexdigits.lower(), k=8))
        parts[0] = parts[0] + f"-trace-{trace_id}-span-{span_id}"
        domain = '.'.join(parts)
    
    if random.random() < 0.15:  # Add detailed timestamp and version info
        parts = domain.split('.')
        timestamp = int(time.time())
        parts[0] = parts[0] + f"-ts-{timestamp}-v{random.randint(1,9)}.{random.randint(0,99)}"
        domain = '.'.join(parts)
    
    if random.random() < 0.12:  # Add deployment and infrastructure metadata
        parts = domain.split('.')
        parts[0] = parts[0] + f"-{random.choice(['canary', 'blue', 'green'])}-" + \
                  f"deploy-{random.choice(['k8s', 'ecs', 'vm'])}-" + \
                  f"{random.choice(['spot', 'ondemand'])}-" + \
                  ''.join(random.choices(string.hexdigits.lower(), k=8))
        domain = '.'.join(parts)
    
    return domain

def generate_exfil_domain() -> str:
    """Generate extremely sophisticated exfiltration domains that perfectly mimic payment processing patterns."""
    tlds = ['.com', '.net', '.org', '.io', '.cloud', '.internal', '.payments', '.secure', '.pci', '.bank']
    payment_words = ['api', 'gateway', 'payment', 'process', 'auth', 'verify', 'secure', 'token']
    partners = ['visa', 'mastercard', 'amex', 'discover', 'stripe', 'square', 'paypal']
    
    exfil_types = [
        # Exfiltration disguised as detailed API calls
        lambda: f"{random.choice(payment_words)}-{random.choice(['v1', 'v2', 'v3'])}.{random.randint(0,9)}-" + \
               f"{''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(35, 45)))}-" + \
               f"shard-{random.randint(1,5)}.{random.choice(partners)}{random.choice(tlds)}",
        
        # Exfiltration mimicking transaction processing
        lambda: f"txn-{random.choice(['auth', 'capture', 'settle'])}-" + \
               f"{random.choice(['sync', 'async', 'batch'])}-" + \
               f"{''.join(random.choices(string.hexdigits.lower(), k=random.randint(40, 50)))}" + \
               f".process{random.choice(tlds)}",
        
        # Exfiltration disguised as compliance logging
        lambda: '.'.join([
            f"audit-{random.choice(['pci', 'kyc', 'aml'])}-{random.choice(['log', 'trace'])}",
            f"{''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(30, 40)))}",
            f"{int(time.time())}-{random.randint(1000,9999)}",
            f"secure{random.choice(tlds)}"
        ]),
        
        # Exfiltration in service mesh traffic
        lambda: f"mesh-{random.choice(payment_words)}-{random.choice(['primary', 'secondary'])}-" + \
               f"{''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(35, 45)))}" + \
               f"-zone-{random.choice(['a', 'b', 'c'])}.svc{random.choice(tlds)}",
        
        # Exfiltration in monitoring data
        lambda: f"monitor-{random.choice(['metrics', 'traces', 'logs'])}-" + \
               f"{''.join(random.choices(string.hexdigits.lower(), k=random.randint(35, 45)))}" + \
               f"-aggregation-{random.randint(1,15)}min.{random.choice(partners)}{random.choice(tlds)}"
    ]
    
    domain = random.choice(exfil_types)()
    
    # Add legitimate-looking elements to make detection harder
    if random.random() < 0.18:  # Add realistic trace IDs
        parts = domain.split('.')
        trace_id = ''.join(random.choices(string.hexdigits.lower(), k=16))
        parts[0] = parts[0] + f"-trace-{trace_id}"
        domain = '.'.join(parts)
    
    if random.random() < 0.15:  # Add legitimate-looking metadata
        parts = domain.split('.')
        parts[0] = parts[0] + f"-{random.choice(['prod', 'staging', 'dev'])}-" + \
                  f"{random.choice(['us', 'eu', 'ap'])}-" + \
                  f"v{random.randint(1,3)}"
        domain = '.'.join(parts)
    
    return domain

def generate_test_dataset(size: int = 1000) -> pd.DataFrame:
    """Generate extremely challenging test dataset with sophisticated patterns."""
    print(f"Generating {size} test queries...")
    
    # More extreme class imbalance
    normal_size = int(size * 0.90)  # 90% normal queries
    exfil_size = size - normal_size
    
    normal_queries = [generate_normal_domain() for _ in range(normal_size)]
    exfil_queries = [generate_exfil_domain() for _ in range(exfil_size)]
    
    # Add sophisticated noise to normal queries (10% chance)
    noisy_normal = int(normal_size * 0.10)
    for i in range(noisy_normal):
        idx = random.randint(0, normal_size-1)
        parts = normal_queries[idx].split('.')
        parts[0] = parts[0] + '-' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(12, 18)))
        normal_queries[idx] = '.'.join(parts)
    
    # Make some exfiltration queries look extremely legitimate (15% chance)
    clean_exfil = int(exfil_size * 0.15)
    for i in range(clean_exfil):
        idx = random.randint(0, exfil_size-1)
        parts = exfil_queries[idx].split('.')
        parts[0] = f"{random.choice(['api', 'gateway', 'auth'])}-" + \
                  f"{random.choice(['prod', 'staging'])}-" + \
                  f"{random.choice(['us', 'eu', 'ap'])}-" + \
                  f"v{random.randint(1,3)}.{random.randint(0,9)}"
        exfil_queries[idx] = '.'.join(parts)
    
    df = pd.DataFrame({
        'query': normal_queries + exfil_queries,
        'is_exfiltration': [0] * normal_size + [1] * exfil_size
    })
    
    # Add realistic internal network IPs for different segments with more detail
    subnets = {
        'payment_prod': ['10.0.0', '10.0.1'],  # Production payment processing
        'payment_staging': ['10.0.2', '10.0.3'],  # Staging payment processing
        'compliance_prod': ['192.168.10', '192.168.11'],  # Production PCI compliance
        'compliance_staging': ['192.168.12', '192.168.13'],  # Staging PCI compliance
        'gateway_prod': ['172.16.0', '172.16.1'],  # Production payment gateway
        'gateway_staging': ['172.16.2', '172.16.3'],  # Staging payment gateway
        'internal_tools': ['10.1.0', '10.1.1'],  # Internal tooling
        'monitoring': ['192.168.20', '192.168.21']  # Monitoring infrastructure
    }
    
    # Assign IPs based on domain type and environment
    def assign_ip(domain):
        if 'prod' in domain:
            if 'gateway' in domain:
                subnet = random.choice(subnets['gateway_prod'])
            elif 'compliance' in domain or 'pci' in domain:
                subnet = random.choice(subnets['compliance_prod'])
            else:
                subnet = random.choice(subnets['payment_prod'])
        elif 'monitor' in domain:
            subnet = random.choice(subnets['monitoring'])
        elif 'internal' in domain:
            subnet = random.choice(subnets['internal_tools'])
        else:
            if 'gateway' in domain:
                subnet = random.choice(subnets['gateway_staging'])
            elif 'compliance' in domain or 'pci' in domain:
                subnet = random.choice(subnets['compliance_staging'])
            else:
                subnet = random.choice(subnets['payment_staging'])
        return f"{subnet}.{random.randint(2, 254)}"
    
    df['src'] = [assign_ip(query) for query in df['query']]
    df['rank'] = 1
    
    return df.sample(frac=1).reset_index(drop=True)

class DNSFeatureProcessor:
    def __init__(self):
        self.text_rows = []
        self.size_avg = []
        self.entropy_avg = []
        self.scaler = StandardScaler()
        self.feature_size = len(PRINTABLE_CHARS)
    
    def clear_state(self):
        self.text_rows.clear()
        self.size_avg.clear()
        self.entropy_avg.clear()
    
    def compute_entropy(self, domain: str) -> float:
        char_counts = Counter(domain)
        domain_len = float(len(domain))
        return -sum(count / domain_len * math.log(count / domain_len, 2) 
                   for count in char_counts.values())
    
    def index_chars(self, query: str) -> None:
        char_freq = {}
        for char in query:
            if char in PRINTABLE_CHARS:
                idx = PRINTABLE_CHARS.index(char)
                char_freq[idx] = char_freq.get(idx, 0) + 1
        self.text_rows.append(char_freq)
    
    def compute_aggregated_features(self, row: pd.Series, df: pd.DataFrame) -> None:
        prev_events = df[(df['src'] == row['src']) & (df['tld'] == row['tld'])]
        self.size_avg.append(prev_events['len'].mean() if len(prev_events) > 0 else 0)
        self.entropy_avg.append(prev_events['entropy'].mean() if len(prev_events) > 0 else 0)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.clear_state()
        df['query'].apply(self.index_chars)
        char_freq_df = pd.DataFrame(self.text_rows, columns=range(len(PRINTABLE_CHARS)))
        char_freq_df.fillna(0, inplace=True)
        
        df = pd.concat([char_freq_df, df.reset_index(drop=True)], axis=1)
        
        df['subdomain'] = df['query'].apply(lambda x: str(x).rsplit('.', 2)[0])
        df['tld'] = df['query'].apply(lambda x: '.'.join(str(x).rsplit('.', 2)[1:]))
        
        df['len'] = df['subdomain'].apply(len)
        df['entropy'] = df['subdomain'].apply(self.compute_entropy)
        
        recent_df = df[df['rank'] == 1].copy()
        recent_df.apply(lambda x: self.compute_aggregated_features(x, df), axis=1)
        recent_df['size_avg'] = self.size_avg
        recent_df['entropy_avg'] = self.entropy_avg
        
        # Return only the numeric features
        feature_cols = list(range(len(PRINTABLE_CHARS))) + ['len', 'entropy', 'size_avg', 'entropy_avg']
        return recent_df[feature_cols]

class DNSExfiltrationDetector(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = len(PRINTABLE_CHARS) + 4  # printable chars + len, entropy, size_avg, entropy_avg
        self.layer_1 = nn.Linear(input_size, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_out = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        print(f"Model initialized with input size: {input_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        return x

# Generate dataset
data = generate_test_dataset(size=1000)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Initialize model and processor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DNSExfiltrationDetector().to(device)
processor = DNSFeatureProcessor()

# Process features
processed_train = processor.prepare_features(train_data)
processed_test = processor.prepare_features(test_data)

print(f"Feature shape: {processed_train.shape}")

# Prepare training data
X_train = torch.FloatTensor(processed_train.values).to(device)
y_train = torch.FloatTensor(train_data['is_exfiltration'].values).to(device)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# Training loop
print("Training model...")
history = []
for epoch in range(100):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    history.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {avg_loss:.4f}')

# Testing
print("\nEvaluating model...")
model.eval()
test_features = processed_test
X_test = torch.FloatTensor(test_features.values).to(device)
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = test_outputs.cpu().numpy()

# Calculate metrics
y_true = test_data['is_exfiltration'].values
y_pred = predictions >= THRESHOLD
metrics = {
    'Accuracy': accuracy_score(y_true, y_pred),
    'Precision': precision_score(y_true, y_pred),
    'Recall': recall_score(y_true, y_pred),
    'F1': f1_score(y_true, y_pred),
    'AUC-ROC': roc_auc_score(y_true, predictions)
}

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.bar(metrics.keys(), metrics.values())
plt.title('Model Performance')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Print metrics and example predictions
print("\nPerformance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nExample Predictions:")
example_df = pd.DataFrame({
    'Query': test_data['query'].values[:5],
    'True Label': test_data['is_exfiltration'].values[:5],
    'Predicted': y_pred[:5].flatten(),
    'Probability': predictions[:5].flatten()
})
print(example_df) 