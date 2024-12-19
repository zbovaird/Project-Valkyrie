# Credit Card Fraud Detection: GNN and Geometric Space Analysis

## Problem Context
The financial industry, particularly credit card issuers, heavily prioritizes AI for fraud detection (81% enthusiasm rate, 62% expected implementation). This creates a dual challenge:
1. Technical effectiveness in pattern detection
2. Regulatory compliance and explainability

 Issuers see combating fraud as the most important application for AI. Among the 27 European issuers we surveyed, 22 of them (81%) were very or somewhat enthusiastic about AI’s impact on their business in the future, matching closely with other regions. Among all issuers, 62% said they expect AI to be used for fraud detection, more than any other area. (Customer service came next, at 54%, followed by sales forecasting, at 48%).

Our conversations with issuers reveal that AI generates its own concerns. In theory, AI and machine learning have the potential to analyze large amounts of data to detect anomalies and identify patterns. But in practice, issuers need to be wary of how they construct their anti-fraud models. For example, the Financial Conduct Authority requires firms to treat customers fairly and communicate with them clearly, so issuers should be transparent in how they apply AI. The rules they construct should be documented and rationalized in order to meet regulatory requirements, issuers note. If an issuer says a transaction is fraudulent, they should be able to say why.
*Source: https://www.tsys.com/insights/2024/02/20/amidst-a-payment-revolution-issuers-cite-security-as-the-highest-priority

## Build a GNN for Fraud Detection using GraphSAGE, supported by Splunk SOAR, optimized with uhg library

## GNN Requirements Analysis

### Do We Need a GNN?
- a GNN is highly recommended because:
    - Transaction patterns form natural graphs
    - Relationships between entities are crucial
    - Pattern detection requires contextual understanding
    - Historical transaction paths matter
    - Entity relationships evolve over time

### Optimal GNN Architecture

#### GraphSAGE Advantages
- Better suited for fraud detection because:
    - Handles dynamic graphs (new merchants/customers)
    - Scales well with large transaction volumes
    - Effective at inductive learning
    - Good performance on unseen nodes

#### Why Not GCN?
- Limited by:
    - Static graph requirement
    - Full graph training needed
    - Poor scalability
    - Struggles with new nodes

## Geometric Space Considerations

### Hyperbolic Space Not Required
Standard Euclidean space is sufficient because:
1. Transaction patterns are relatively flat
2. Relationships don't form deep hierarchies
3. Distance metrics are straightforward
4. Pattern complexity is manageable in Euclidean space

### Exception Cases
Consider hyperbolic space only for:
- Multi-level merchant hierarchies
- Complex authorization chains
- Nested payment processor relationships

## Explainability Requirements

### Model Design Considerations
1. Feature Attribution
   - Transaction characteristics
   - Historical patterns
   - Entity relationships
   - Temporal aspects

2. Decision Path Tracking
   - Clear reasoning chain
   - Documented rule application
   - Pattern identification evidence

### Regulatory Compliance
Must provide:
- Clear decision rationale
- Documented rule base
- Fair treatment evidence
- Transparent methodology

## Implementation Recommendations

1. Use GraphSAGE with attention mechanisms
2. Implement in Euclidean space
3. Add explainability layers:
   - SHAP values (SHapley Additive exPlanations)
     * Game theory approach to feature importance
     * Shows how each feature contributes to decisions
     * Example: Transaction amount contributed 40% to fraud score
     * Helps explain why a transaction was flagged
   - Decision paths
   - Pattern highlights
   - Rule activation tracking

4. Documentation system for:
   - Model decisions
   - Rule applications
   - Pattern detections
   - Fairness metrics 

## GraphSAGE Model Planning

### Model Architecture Overview
1. Input Layer
   - Transaction nodes
   - Entity nodes (customers, merchants, processors)
   - Edge features (transaction amounts, timestamps, types)
   - Node features (entity attributes, historical metrics)

2. Sampling Layers
   - Two-hop neighborhood sampling
     * 1st hop: Transaction -> Connected Entities (Merchant, Card Holder, Processor)
     * 2nd hop: Connected Entities -> Their Other Transactions
       Example: 
       - A suspicious transaction at Merchant A
       - 1st hop: Look at Merchant A's profile
       - 2nd hop: Look at other recent transactions at Merchant A
   - Adaptive sampling rates based on node importance
   - Temporal sampling considerations
   - Attention-based neighbor selection

3. Aggregation Layers
   - Mean aggregator for stable patterns
   - LSTM aggregator for sequential patterns
   - Attention mechanism for important neighbors
   - Skip connections for multi-scale features

4. Neural Network Components
   - Feature transformation layers
   - Pattern recognition layers
   - Temporal encoding layers
   - Output classification layer

### Required Data Elements

1. Node Features
   - Transaction Data
     * Amount
     * Time
     * Location
     * MCC code
     * Currency
   
   - Entity Data
     * Customer profiles
     * Merchant categories
     * Processor relationships
     * Historical metrics

2. Edge Features
   - Transaction metadata
   - Relationship types
   - Historical patterns
   - Risk scores

3. Temporal Aspects
   - Transaction sequences
   - Pattern evolution
   - Seasonal factors
   - Time-based risk factors

### Data Sources to Consider

1. Internal Sources
   - Transaction processing systems
   - Customer databases
   - Merchant management systems
   - Authorization systems
   - Risk management platforms

2. External Sources
   - Credit bureaus
   - Fraud databases
   - Industry blacklists
   - Geographic risk data
   - Economic indicators

3. Derived Features
   - Velocity metrics
   - Pattern indicators
   - Risk scores
   - Anomaly indicators

### Model Training Strategy

1. Training Phases
   - Initial pattern learning
   - Fine-tuning on recent data
   - Continuous learning updates
   - Periodic retraining
   - Percentage Training
    - 1% for testing code
    - 2-3% for memory errors
    - 5-10% for initial model training
    - 10-20% for fine-tuning
    - 20-30% for continuous learning
    - 30-40% for periodic retraining

2. Data Requirements and Splits
   - Minimum Dataset Size:
     * At least 1 million transactions
       - Ensures 1,000-3,000 fraud cases (at 0.1-0.3% fraud rate)
       - Provides sufficient graph density for pattern detection
       - Matches typical business processing volumes
       - Minimum needed for reliable GNN training
     * Collected over 3-6 months (not longer)
     * Reason: Fraud patterns evolve rapidly
   
   - Time Window Considerations:
     * Recent data (last 6 months) weighted more heavily
     * Older data gradually discounted
     * Monthly pattern validation
   
   - Split Ratios:
     * Training: 80% (most recent data)
     * Validation: 10% (chronologically after training)
     * Testing: 10% (most recent data)
     * Rationale: 
       - Larger training set needed for graph patterns
       - Recent data crucial for current fraud patterns
       - Validation/Test must be chronologically after training

   - Minimum Fraud Examples:
     * At least 1000 confirmed fraud cases
     * Distributed across different fraud types
     * If insufficient, use synthetic data generation
     * Maintain realistic fraud/legitimate ratio (typically 0.1-0.3%)

3. Validation Approach
   - Time-based splitting
     * No random splits (maintains temporal integrity)
     * Rolling window validation
     * Monthly performance evaluation
   
   - Entity-based considerations
     * Ensure merchant representation
     * Balance transaction volumes
     * Account for seasonal patterns
   
   - Cross-validation strategy
     * Time-series cross validation
     * Forward-chaining validation
     * No standard k-fold (violates time sequence)

4. Performance Monitoring
   - False positive tracking
   - Detection rate monitoring
   - Pattern effectiveness
   - Model drift detection
   - Monthly retraining evaluation

### Integration Points

1. Input Processing
   - Real-time transaction feeds
   - Batch updates
   - Entity updates
   - Pattern updates

2. Output Handling
   - Risk score generation
   - Alert triggering
   - Pattern reporting
   - Investigation queuing

3. Feedback Loops
   - Investigation results
   - False positive corrections
   - Pattern updates
   - Model adjustments

## Code Implementation

### 1. Basic Model Structure
```python
import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

class FraudGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(FraudGraphSAGE, self).__init__()
        
        # Initial feature transformation
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList([
            geom_nn.SAGEConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                normalize=True,
                aggr='mean'
            ) for _ in range(num_layers)
        ])
        
        # Final classification
        self.classifier = nn.Linear(hidden_dim, output_dim)
```

**Why This Structure?**
- Handles dynamic transaction graphs (new merchants/cards appear constantly)
- Aggregates neighborhood information effectively
- Scales well with large transaction volumes
- Can process unseen nodes (new merchants/customers)

### 2. Attention Mechanism
```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=0)
        return attention_weights * x
```

**Why Add Attention?**
- Not all transactions are equally important
- Helps focus on suspicious patterns
- Improves model interpretability
- Aids in explaining decisions to regulators

### 3. Temporal Encoding
```python
class TemporalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalEncoding, self).__init__()
        self.time_embed = nn.Linear(1, hidden_dim)
        
    def forward(self, x, timestamps):
        # Normalize timestamps to [0,1]
        norm_time = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-6)
        time_features = self.time_embed(norm_time.unsqueeze(-1))
        return x + time_features
```

**Why Temporal Encoding?**
- Fraud patterns evolve over time
- Captures time-based relationships
- Important for sequence-dependent fraud
- Helps detect velocity-based attacks

### 3.1 Time-Based Weighting
```python
class TimeWeighting:
    def __init__(self, decay_rate=0.1, cutoff_months=6):
        self.decay_rate = decay_rate
        self.cutoff_months = cutoff_months
        
    def calculate_weights(self, timestamps, reference_time=None):
        """
        Calculate weights based on time difference.
        More recent = higher weight
        > 6 months = exponentially decaying weight
        """
        if reference_time is None:
            reference_time = timestamps.max()
            
        # Convert to months difference
        time_diff = (reference_time - timestamps) / (30 * 24 * 60 * 60)  # seconds to months
        
        # Recent data (within cutoff) gets full weight
        weights = torch.ones_like(timestamps, dtype=torch.float32)
        
        # Older data gets exponentially decaying weight
        old_data_mask = time_diff > self.cutoff_months
        weights[old_data_mask] = torch.exp(
            -self.decay_rate * (time_diff[old_data_mask] - self.cutoff_months)
        )
        
        return weights

class WeightedLoss(nn.Module):
    def __init__(self, base_criterion=nn.BCELoss()):
        super(WeightedLoss, self).__init__()
        self.base_criterion = base_criterion
        self.time_weighting = TimeWeighting()
        
    def forward(self, predictions, targets, timestamps):
        # Calculate time-based weights
        weights = self.time_weighting.calculate_weights(timestamps)
        
        # Apply weights to base loss
        base_loss = self.base_criterion(predictions, targets)
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean()

# Usage in training loop:
def train_step(model, data, optimizer, criterion):
    """
    Training step with time-weighted loss
    """
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(
        data.x,
        data.edge_index,
        data.timestamps
    )
    
    # Calculate weighted loss
    loss = criterion(
        predictions,
        data.y,
        data.timestamps
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

**Why This Time Weighting?**
- Full weight for data within last 6 months
- Exponential decay for older data
- Maintains historical patterns while prioritizing recent trends
- Adjustable decay rate for different fraud patterns

### 4. Main Forward Pass
```python
def forward(self, x, edge_index, timestamps):
    # Initial feature transformation
    x = self.input_transform(x)
    x = self.temporal_encoding(x, timestamps)
    
    # GraphSAGE layers with attention
    for conv in self.convs:
        # Sample neighborhood
        # Note: In Splunk SOAR implementation, this will be 
        # optimized for streaming data
        x_neigh = conv(x, edge_index)
        
        # Apply attention
        x_attended = self.attention(x_neigh)
        
        # Residual connection
        x = x + x_attended
        x = torch.relu(x)
    
    # Final classification
    return torch.sigmoid(self.classifier(x))
```

**Why This Forward Pass Structure?**
- Efficiently processes transaction neighborhoods
- Maintains temporal awareness
- Balances local and global patterns
- Provides clear decision pathways for explainability

### 5. Splunk SOAR Integration Notes
```python
class SplunkSOARInterface:
    def process_event(self, event):
        """
        Process transaction event in Splunk format
        
        Args:
            event: Splunk event dictionary containing:
                - _time: Timestamp
                - source: Source of transaction
                - sourcetype: Transaction type
                - host: Processing system
                - _raw: Raw transaction data
        """
        # Convert Splunk event to graph format
        graph_data = self.build_graph(
            event_data=event.get('_raw'),
            _time=event.get('_time'),
            source=event.get('source')
        )
        
        # Get model prediction
        with torch.no_grad():
            severity = self.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data._time
            )
        
        # Format for SOAR
        return {
            'severity': float(severity),  # Splunk severity field
            'weight': self.get_weights(),  # Splunk weight field
            'neighbor_count': len(graph_data.edge_index[0]),  # Number of related events
            'temporal_pattern': self.get_temporal_analysis(),
            '_time': event.get('_time'),
            'source': event.get('source'),
            'sourcetype': 'fraud_detection',
            'event_type': 'transaction_analysis',
            'status': 'new'  # Splunk SOAR status field
        }

    def get_weights(self):
        """Get attention weights in Splunk format"""
        weights = self.model.attention.get_weights()
        return {
            'transaction_weight': float(weights['transaction']),
            'merchant_weight': float(weights['merchant']),
            'pattern_weight': float(weights['pattern'])
        }

    def get_temporal_analysis(self):
        """Get temporal analysis in Splunk format"""
        return {
            'earliest_time': self.earliest_time,
            'latest_time': self.latest_time,
            'span': self.time_span,
            'velocity': self.transaction_velocity
        }
```

**Splunk SOAR Integration Notes:**
- Uses standard Splunk field names (_time, source, etc.)
- Matches Splunk severity levels (0-100)
- Follows Splunk event format
- Compatible with Splunk SOAR playbooks
- Maintains Splunk search compatibility

**Why This Belongs in SOAR:**
- Fraud is a security threat that involves:
    Phishing, Account Takeover, other cyber attacks
- Shared IOCs: IP addresses, email addresses, behavioral anomalies, geolocation anomalies (don't use Splunk _iplocation)
- Automated response:
    - Block transaction
    - Flag accounts
    - Notify security team
- Investigation
    - creates event
    - Artifact investigation
    - Data enrichment
    - trigger playbooks
    - 
