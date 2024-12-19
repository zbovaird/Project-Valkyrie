# Gjallarhorn UHG Model - Splunk Integration

## Overview
The Gjallarhorn UHG Model provides anomaly detection capabilities that can be integrated with Splunk for network traffic analysis. This document outlines the key components and functions useful for Splunk integration.

## Key Components

### Baseline Profile Generation
```python
baseline_profile = {
    'normal_patterns': {
        'packet_length_stats': {
            'mean', 'std', 'q1', 'q3', 'iqr'
        },
        'known_sources': set(unique_sources),
        'known_destinations': set(unique_dests),
        'protocol_distribution': protocol_frequencies
    }
}
```
- **Purpose**: Creates statistical baseline for normal network behavior
- **Use in Splunk**: Create correlation searches and alerts based on deviations from baseline

### Statistical Metrics
The model tracks several key metrics useful for Splunk dashboards:
- Packet length distributions
- Protocol frequency analysis
- Source/Destination IP patterns
- Behavior clusters

## Splunk Integration Points

### 1. Risk Level Distribution
```python
risk_level_stats = {
    'Info': count,
    'Low': count,
    'Medium': count,
    'High': count,
    'Critical': count
}
```
- **Dashboard Use**: Track risk level trends over time
- **Alert Thresholds**: Monitor changes in risk distribution

### 2. Network Profile Analysis
```python
network_stats = {
    'unique_sources': len(unique_sources),
    'unique_destinations': len(unique_dests),
    'behavior_clusters': num_clusters
}
```
- **Visualization**: Network behavior patterns
- **Trend Analysis**: Track network growth and changes

### 3. Feature Correlations
```python
correlations = {
    'protocol_anomaly': correlation_value,
    'packet_length_anomaly': correlation_value
}
```
- **Analysis**: Identify strong indicators of anomalies
- **Search Terms**: Focus on highly correlated features

## Splunk Search Examples

### 1. High-Risk Events
```spl
index=network sourcetype=uhg_analysis risk_level IN ("High", "Critical")
| stats count by source_ip, risk_factors
```

### 2. Anomalous Packet Lengths
```spl
index=network sourcetype=uhg_analysis 
| where std_deviations > 3
| table _time source_ip packet_length std_deviations
```

### 3. Protocol Analysis
```spl
index=network sourcetype=uhg_analysis 
| stats count by protocol
| where count < baseline_frequency
```

## Dashboard Components

1. **Risk Overview**
   - Risk level distribution
   - Trend analysis
   - Top risky IPs

2. **Network Behavior**
   - Protocol distribution
   - Packet length patterns
   - Behavior cluster analysis

3. **Anomaly Metrics**
   - Statistical deviations
   - Correlation patterns
   - Baseline comparisons

## Alert Configuration

### 1. Critical Risk Alert
- Trigger: risk_level="Critical"
- Throttle: 5 minutes
- Action: Create notable event

### 2. Statistical Deviation Alert
- Trigger: std_deviations > 4
- Throttle: 15 minutes
- Action: Add to investigation

### 3. New Behavior Pattern Alert
- Trigger: New behavior cluster detected
- Throttle: 1 hour
- Action: Update baseline 