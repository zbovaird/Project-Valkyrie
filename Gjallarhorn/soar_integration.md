# Gjallarhorn UHG Model - SOAR Integration

## Overview
This document outlines how the Gjallarhorn UHG Model integrates with Splunk SOAR for automated response to network anomalies. Functions and components are specifically designed for SOAR playbook automation.

## Key Functions for SOAR Analysts

### enhanced_artifact_investigation()
```python
def enhanced_artifact_investigation(artifact, baseline):
    """Investigates network artifacts against baseline profile"""
```
- **Purpose**: Primary investigation function for SOAR playbooks
- **Input**: Network artifact (IP, protocol, packet length)
- **Output**: Risk assessment and evidence
- **Use Case**: Initial triage of suspicious network activity
- **Playbook Trigger**: New notable event or suspicious IP

### Risk Assessment Components
```python
risk_assessment = {
    'risk_level': str,          # Critical, High, Medium, Low, Info
    'risk_factors': list,       # List of identified risk factors
    'evidence': list,           # Detailed evidence for each factor
    'confidence': float,        # 0.0 to 1.0 confidence score
    'risk_score': float,        # Raw risk score
    'statistical_data': dict    # Statistical evidence
}
```

## SOAR Playbook Integration

### 1. Artifact Analysis Playbook
```yaml
Trigger: New Network Artifact
Steps:
1. Extract artifact details
2. Call enhanced_artifact_investigation()
3. Branch based on risk_level:
   - Critical: Immediate block
   - High: Enrich and investigate
   - Medium: Add to watchlist
   - Low/Info: Log and monitor
```

### 2. Risk Response Actions
Based on risk_factors:
- **New Source IP**:
  - Add to watchlist
  - Trigger IP enrichment
  - Check against threat feeds

- **Highly Anomalous Packet Length**:
  - Capture packet samples
  - Create network filter
  - Alert SOC team

- **Unknown/Rare Protocol**:
  - Protocol analysis
  - Port scanning detection
  - Update firewall rules

### 3. Confidence-Based Actions
```python
if risk_assessment['confidence'] >= 0.8:
    # Automated response
    - Block IP
    - Create case
    - Alert team
elif risk_assessment['confidence'] >= 0.5:
    # Semi-automated
    - Create case
    - Await approval
else:
    # Manual review
    - Add to investigation queue
```

## SOAR Observable Types

### 1. Network Observables
- source_ip
- destination_ip
- protocol
- packet_length

### 2. Analysis Observables
- risk_level
- risk_score
- statistical_deviations
- behavior_cluster

### 3. Context Observables
- baseline_metrics
- historical_patterns
- correlation_scores

## Automation Examples

### 1. High-Risk IP Workflow
```python
if 'New Source IP' in risk_factors and risk_level == 'Critical':
    actions = [
        'enrich_ip_reputation',
        'check_threat_feeds',
        'create_block_rule',
        'notify_team'
    ]
```

### 2. Protocol Analysis Workflow
```python
if 'Unknown Protocol' in risk_factors:
    actions = [
        'capture_traffic_sample',
        'analyze_port_patterns',
        'check_known_exploits',
        'update_ids_rules'
    ]
```

### 3. Statistical Analysis Workflow
```python
if std_deviations > 4:
    actions = [
        'create_baseline_deviation_case',
        'collect_similar_patterns',
        'update_correlation_rules'
    ]
```

## Best Practices for SOAR Integration

1. **Artifact Handling**
   - Always check artifact completeness
   - Validate input data
   - Handle missing fields gracefully

2. **Response Automation**
   - Start with low-risk actions
   - Implement approval workflows for critical actions
   - Log all automated responses

3. **Baseline Management**
   - Regular baseline updates
   - Version control for baselines
   - Backup before updates

4. **Alert Throttling**
   - Implement rate limiting
   - Group similar alerts
   - Prevent alert storms 