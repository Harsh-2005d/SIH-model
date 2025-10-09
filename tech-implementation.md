
# Fusion Model — Satellite-Meteorology Integration
---
## Problem Understanding
Air pollution in rapidly urbanizing megacities such as Delhi poses a persistent and escalating threat to public health. Among the most critical gaseous pollutants are Nitrogen Dioxide (NO₂) and Ozone (O₃), both of which frequently exceed globally accepted air quality thresholds. These pollutants, particularly at ground level, contribute to respiratory illnesses, cardiovascular stress, and long-term climate implications.

Accurate short-term forecasting of these pollutants is essential for enabling proactive mitigation, public advisories, and policy formulation. However, current air quality monitoring systems suffer from limited spatial coverage and temporal inconsistency, resulting in poor generalization across different regions and seasons.

---

## Data Collection and Preprocessing
- Sentinel-5P data- the data is at a daily frequency of NO2,O3,SO2,Coud,Methane,UV Aeorosol Index,HCHO,CO
- ERA5 DATA - This is at an hourly frequency collected from the GEE of the relevant meterological variables 
- The data from sentinel is at 0.01x0.01 res and era5 at 0.25x0.25
- The missing data in sentinel is temporally imputed from the prev days , whereas the ERA5 data is imputed through KNN imputation setting the hyperparam to 3
- The data is also Standard Scaled.
```python
def impute_missing_pixels(arr: np.ndarray) -> np.ndarray:
    """
    Robustly fills NaN pixels in a 3D [time, H, W] array.
    Uses temporal neighbors' mean; falls back gracefully if neighbors are empty.
    """
    arr = arr.copy()
    t, h, w = arr.shape

    # --- Handle first day ---
    if np.isnan(arr[0]).all():
        # find first non-empty day
        for d in range(1, t):
            if not np.isnan(arr[d]).all():
                arr[0] = arr[d]
                break
            else:
                arr[0] = np.zeros((h, w))
    else:
        mean_val = np.nanmean(arr[0])
        arr[0][np.isnan(arr[0])] = mean_val

    # --- Handle rest of the days ---
    for day in range(1, t):
        elem = arr[day]
        missing = np.isnan(elem)

        prev = arr[day - 1]
        after = arr[day + 1] if day + 1 < t else None

        # if both prev and after exist
        if after is not None:
            stack = np.stack([prev, after])
            mean = np.nanmean(stack, axis=0)
        else:
            mean = np.nan_to_num(prev, nan=np.nanmean(prev))

        # fallback: if mean itself has NaNs (prev & after both NaN there)
        if np.isnan(mean).any():
            mean[np.isnan(mean)] = np.nanmean(arr[max(day-3, 0):day+1])

        # still might happen if entire slice is NaN
        if np.isnan(elem).all():
            elem = mean
        else:
            elem[missing] = mean[missing]

        arr[day] = elem

    # Final safety pass — replace any remaining NaN with global mean
    global_mean = np.nanmean(arr)
    arr[np.isnan(arr)] = global_mean

    return arr
```
---

## 3️⃣ Model Architecture

The proposed architecture is a hybrid deep learning model that fuses satellite imagery and station-level meteorological data to forecast short-term concentrations of NO₂ and O₃.
It combines spatial encoders, temporal sequence modeling, and an attention mechanism to capture both spatial and temporal dependencies effectively.

### CNN Encoder — Spatial Feature Extraction

The CNNEncoder is designed to process 9-channel satellite grids, extracting high-level spatial representations that summarize pollutant patterns, cloud dynamics, and land-surface influences.

3 convolutional blocks with Batch Normalization, ReLU, and Max-Pooling progressively downsample and learn spatial hierarchies.

A global pooling layer compresses spatial information into a compact embedding vector (output_size = 64).

A fully connected projection layer refines the spatial features for fusion.

This module ensures that each hourly satellite snapshot is transformed into a robust spatial descriptor.

```python
class CNNEncoder(nn.Module):
    """Refined CNN encoder for 9-channel satellite grids."""
    def __init__(self, in_channels=9, output_size=64, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # → [B, 64, 1, 1]
            nn.Flatten()               # → [B, 64]
        )

        self.fc = nn.Sequential(
            nn.Linear(64, output_size),
            nn.LayerNorm(output_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x
```
---
### Bahdanau Attention — Focused Temporal Weighting

To enhance interpretability and focus, a Bahdanau Attention mechanism is applied over the LSTM outputs.

Learns attention weights over timesteps, highlighting hours most influential for prediction.

Produces a context vector representing the weighted temporal summary.

Separate attention modules are used for O₃ and NO₂, allowing pollutant-specific temporal sensitivity.
```python
class BahdanauAttention(nn.Module):
    
    def __init__(self, hidden_size, attn_dim=64):
        super().__init__()
        # Projects hidden state into attention space
        self.W = nn.Linear(hidden_size, attn_dim)
        # Scoring vector u0
        self.v = nn.Linear(attn_dim, 1, bias=False)
        # Optional bias term d_j is already absorbed in nn.Linear

    def forward(self, lstm_out):
        """
        lstm_out: [batch, seq_len, hidden_size]
        """
        # 1. Project hidden states with tanh nonlinearity
        energy = torch.tanh(self.W(lstm_out))           # [batch, seq_len, attn_dim]

        # 2. Compute scores for each timestep
        scores = self.v(energy).squeeze(-1)             # [batch, seq_len]

        # 3. Normalize to get attention weights
        weights = torch.softmax(scores, dim=1)          # [batch, seq_len]

        # 4. Weighted sum of hidden states
        context = torch.bmm(weights.unsqueeze(1), lstm_out)  # [batch, 1, hidden_size]

        return context.squeeze(1), weights 
```
---
### The Fusion Module 
  It integrates multi-source atmospheric and meteorological datasets into a unified spatiotemporal framework. Satellite-derived gaseous concentrations (NO₂, O₃, CO, HCHO) are co-registered with reanalysis meteorological variables (temperature, wind vectors, humidity, boundary-layer height) through spatial resampling and temporal synchronization. Advanced geostatistical interpolation and dynamic lag alignment preserve diurnal and seasonal variability. Missing data are reconstructed via hybrid imputation combining temporal interpolation and model-based estimation. The fused dataset encapsulates synergistic dependencies between trace gases and meteorological drivers, serving as a robust input foundation for the downstream CNN–BiLSTM–Attention architecture.
```python
class FusionModel(nn.Module):
    """Fuses CNN-encoded satellite data with station features, then forecasts pollutants."""
    def __init__(
        self,
        sat_channels=2,
        station_features=18,
        cnn_out=64,
        out_dim=32,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.15,
        hidden_size=128,
    ):
        super().__init__()

        self.cnn_o3 = CNNEncoder(in_channels=9,output_size=cnn_out)

        lstm_input_size = out_dim+cnn_out
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.attention=BahdanauAttention(2*lstm_hidden)

        self.station_proj=MLPProj(station_features,hidden=hidden_size,out_dim=out_dim)
        # Heads for O3 and NO2
        self.head_O3 = nn.Linear(2*lstm_hidden, 24)
        self.head_NO2 = nn.Linear(2*lstm_hidden, 24)

    def forward(self, sat_x, station_x):
        """
        sat_x: [batch, seq_len, channels, H, W]
        station_x: [batch, seq_len, station_features]
        """
        batch_size, seq_len, _, _, _ = sat_x.shape

        # Encode each time step
        cnn_embeds = []
        for t in range(seq_len):
            grid = sat_x[:, t, ...]   # [batch, 1, 9, 10]
        
            sat_feat = self.cnn_o3(grid)     # [batch, 32

            cnn_embeds.append(sat_feat)

    
        cnn_embeds = torch.stack(cnn_embeds, dim=1)
        
        station_x=self.station_proj(station_x)

        lstm_in = torch.cat((cnn_embeds, station_x), dim=-1)

        lstm_out, _ = self.lstm(lstm_in)  # [B, seq_len, 2*lstm_hidden]
        last_out = lstm_out[:, -1]        # [B, 2*lstm_hidden]

        context_O3,_=self.attention(lstm_out)
        context_No2,_=self.attention(lstm_out)
        pred_o3 = self.head_O3(context_O3)  # [B, 24]
        pred_no2 = self.head_NO2(context_No2) 

        return torch.cat((pred_no2,pred_o3), dim=-1) # [batch, 2]

```
---
## Results
![bias v horizon](./model2.0/bias.png)
![rmse](rmse.png)



---
