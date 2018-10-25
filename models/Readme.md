#### Models

- retain_ex.py
  - RetainEX model with additional weight embeddings, bidirectionality and time decay
  - **Model used in paper**
- retain_time.py
  - Time-decaying LSTM cell
  - sub-par performance
- retain_bidirectional.py
  - bidirectional RNN settings added to RETAIN (NIPS'16) paper
- gru_bidirectional.py
  - bidirectional GRU setting for baseline
- retain_copy.py
  - version incorporating a copying mechanism (Guo et al., ACL'16)
  - not mentioned in paper for performance
- retain_dc.py
  - variant of retain_copy
  - not mentioned in paper
