# All imports first
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import string
import random
from tqdm import tqdm
import time
import re
import matplotlib.pyplot as plt
from uhg.projective import ProjectiveUHG

# Feature Extraction Class
class DNSFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, domain):
        """Extract enhanced numerical features from a domain name."""
        features = []
        
        # Basic length features
        features.append(len(domain))
        features.append(len(domain.split('.')))
        
        # Character type ratios with positional weighting
        total_len = len(domain) + 1
        segments = domain.split('.')
        main_segment = segments[0] if segments else ""
        
        # Weighted character distributions
        for i, char in enumerate(main_segment):
            position_weight = 1.0 - (i / len(main_segment)) if main_segment else 0
            if char.isdigit():
                features.append(position_weight)
            elif char.isalpha():
                features.append(position_weight * 0.8)
            else:
                features.append(position_weight * 0.5)
        
        # Pad or truncate position features to fixed length
        position_features = features[3:]  # Skip the first 3 basic features
        target_length = 50  # Fixed length for position features
        if len(position_features) < target_length:
            position_features.extend([0] * (target_length - len(position_features)))
        else:
            position_features = position_features[:target_length]
        features = features[:3] + position_features
        
        # Advanced pattern detection
        features.extend([
            sum(c.isdigit() for c in domain) / total_len,
            sum(c.isalpha() for c in domain) / total_len,
            sum(c in string.punctuation for c in domain) / total_len,
            sum(c.isupper() for c in domain) / total_len,
            sum(c.islower() for c in domain) / total_len
        ])
        
        # Entropy calculations for different parts
        def calculate_entropy(text):
            if not text:
                return 0
            char_freq = {}
            for c in text:
                char_freq[c] = char_freq.get(c, 0) + 1
            entropy = 0
            for freq in char_freq.values():
                prob = freq / len(text)
                entropy -= prob * np.log2(prob) if prob > 0 else 0
            return entropy
        
        features.extend([
            calculate_entropy(domain),
            calculate_entropy(main_segment),
            calculate_entropy(''.join(segments[1:]) if len(segments) > 1 else '')
        ])
        
        # Segment analysis
        features.extend([
            max(len(seg) for seg in segments),
            min(len(seg) for seg in segments),
            np.mean([len(seg) for seg in segments]),
            np.std([len(seg) for seg in segments])
        ])
        
        # Payment-specific pattern detection
        payment_keywords = ['payment', 'gateway', 'auth', 'token', 'process', 'txn', 'card',
                          'vault', 'merchant', 'billing', 'checkout', 'pci', 'api', 'secure']
        features.extend([
            sum(1 for kw in payment_keywords if kw in domain.lower()),
            sum(1 for kw in payment_keywords if kw in main_segment.lower()),
            max(domain.lower().count(kw) for kw in payment_keywords)
        ])
        
        # Environment and region detection
        env_keywords = ['prod', 'staging', 'dev', 'qa', 'uat', 'sandbox']
        region_keywords = ['us', 'eu', 'ap', 'na', 'global', 'latam']
        features.extend([
            sum(1 for env in env_keywords if env in domain.lower()),
            sum(1 for region in region_keywords if region in domain.lower())
        ])
        
        # Special pattern detection
        features.extend([
            int(any(seg.startswith('xn--') for seg in segments)),  # Punycode
            sum(c == '-' for c in domain),
            sum(c1.isdigit() and c2.isdigit() for c1, c2 in zip(domain, domain[1:])),
            sum(c1.isalpha() and c2.isdigit() for c1, c2 in zip(domain, domain[1:])),
            sum(c in string.hexdigits for c in domain) / total_len,
            int(bool(re.search(r'\d{8,}', domain))),  # Long number sequences
            int(bool(re.search(r'[a-f0-9]{8,}', domain.lower())))  # Hex-like sequences
        ])
        
        return [float(f) for f in features]

    def build_graph(self, df, k=5):
        """Build a graph from the dataset using k-nearest neighbors."""
        # Extract features for each domain
        features_list = []
        for domain in df['query']:
            features_list.append(self.extract_features(domain))
        
        # Scale features
        node_features = self.scaler.fit_transform(np.array(features_list))
        
        # Build adjacency matrix using k-nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(node_features)
        distances, indices = nbrs.kneighbors(node_features)
        
        # Create edge index for PyTorch Geometric
        edge_list = []
        for i in range(len(indices)):
            for j in indices[i][1:]:  # Skip first neighbor (self)
                edge_list.append([i, j])
                edge_list.append([j, i])  # Add reverse edge for undirected graph
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        return torch.FloatTensor(node_features), edge_index

# UHG Model Class
class UHGModel(torch.nn.Module):
    def __init__(self, num_features):
        super(UHGModel, self).__init__()
        
        # Initialize UHG geometric operations
        self.uhg = ProjectiveUHG()
        
        # Neural network layers
        self.input_transform = torch.nn.Linear(num_features, 128)
        self.hidden1 = torch.nn.Linear(128, 128)
        self.hidden2 = torch.nn.Linear(128, 64)
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1)
        )
    
    def forward(self, x, edge_index):
        # Transform features to projective space
        x = self.input_transform(x)
        x = F.relu(x)
        x = self.uhg.normalize(x)  # Use UHG's normalize function
        
        # Message passing using UHG operations
        row, col = edge_index
        
        # First layer
        messages = []
        for i in range(len(row)):
            source = x[row[i]]
            target = x[col[i]]
            
            # Use UHG's distance function
            dist = self.uhg.distance(source.unsqueeze(0), target.unsqueeze(0))
            message = source * torch.exp(-dist)
            messages.append(message)
        
        # Aggregate messages
        messages = torch.stack(messages)
        agg_messages = torch.zeros_like(x)
        for i in range(len(row)):
            agg_messages[col[i]] += messages[i]
        
        # Update node features
        x = self.hidden1(agg_messages)
        x = F.relu(x)
        x = self.uhg.normalize(x)
        
        # Second layer with UHG operations
        messages = []
        for i in range(len(row)):
            source = x[row[i]]
            target = x[col[i]]
            
            # Use UHG's distance function
            dist = self.uhg.distance(source.unsqueeze(0), target.unsqueeze(0))
            message = source * torch.exp(-dist)
            messages.append(message)
        
        messages = torch.stack(messages)
        agg_messages = torch.zeros_like(x)
        for i in range(len(row)):
            agg_messages[col[i]] += messages[i]
        
        # Final feature transformation
        x = self.hidden2(agg_messages)
        x = F.relu(x)
        x = self.uhg.normalize(x)
        
        # Classification
        out = self.classifier(x)
        return torch.sigmoid(out)

# Domain Generation Functions
def generate_normal_domain():
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

def generate_exfil_domain():
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

def generate_test_dataset(size: int = 1000):
    """Generate extremely challenging test dataset with sophisticated patterns."""
    print(f"Generating {size} test queries...")
    
    # Realistic class imbalance for enterprise payment processor
    normal_size = int(size * 0.997)  # 99.7% normal queries
    exfil_size = size - normal_size  # 0.3% malicious queries
    
    normal_queries = [generate_normal_domain() for _ in range(normal_size)]
    exfil_queries = [generate_exfil_domain() for _ in range(exfil_size)]
    
    # Add sophisticated noise to normal queries (0.5% chance)
    noisy_normal = int(normal_size * 0.005)
    for i in range(noisy_normal):
        idx = random.randint(0, normal_size-1)
        parts = normal_queries[idx].split('.')
        parts[0] = parts[0] + '-' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(12, 18)))
        normal_queries[idx] = '.'.join(parts)
    
    # Make some exfiltration queries look extremely legitimate (25% chance)
    clean_exfil = int(exfil_size * 0.25)
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

# Main execution block
if __name__ == "__main__":
    # Generate dataset
    print("Generating dataset...")
    data = generate_test_dataset(size=1000)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Initialize model and process features
    print("\nInitializing model and processing features...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    feature_extractor = DNSFeatureExtractor()
    train_features, train_edge_index = feature_extractor.build_graph(train_data)
    train_features = train_features.to(device)
    train_edge_index = train_edge_index.to(device)
    train_labels = torch.FloatTensor(train_data['is_exfiltration'].values).to(device)

    print(f"Feature shape: {train_features.shape}")

    # Initialize model
    input_dim = train_features.shape[1]
    model = UHGModel(num_features=input_dim).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Training loop with progress bar
    print("\nTraining model...")
    history = []
    progress_bar = tqdm(range(100), desc='Training')
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_features, train_edge_index)
        loss = criterion(outputs.squeeze(), train_labels)
        
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # Testing
    print("\nEvaluating model...")
    model.eval()
    test_features, test_edge_index = feature_extractor.build_graph(test_data)
    test_features = test_features.to(device)
    test_edge_index = test_edge_index.to(device)

    with torch.no_grad():
        test_outputs = model(test_features, test_edge_index)
        predictions = test_outputs.cpu().numpy()

    # Calculate metrics
    y_true = test_data['is_exfiltration'].values
    y_pred = predictions >= 0.5
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