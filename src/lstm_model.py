import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        num_layers: int = 2,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Stacked, (optionally) bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Attention: score each timestep, then take weighted sum
        self.use_attention = use_attention
        if use_attention:
            self.attn_fc = nn.Linear(hidden_dim * self.num_directions, 1)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_directions, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, embedding_dim)
        """
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        lstm_out, (h_n, _) = self.lstm(x)

        if self.use_attention:
            # Compute unnormalized attention scores and normalize
            # scores: (batch, seq_len, 1)
            scores = self.attn_fc(lstm_out)
            weights = torch.softmax(scores, dim=1)
            # context: (batch, hidden_dim * num_directions)
            context = torch.sum(weights * lstm_out, dim=1)
        else:
            # If bidirectional, last layer has 2 states: [-2]=forward, [-1]=backward
            if self.bidirectional:
                # h_n: (num_layers*2, batch, hidden_dim)
                forward_last  = h_n[-2]
                backward_last = h_n[-1]
                context = torch.cat([forward_last, backward_last], dim=1)
            else:
                # h_n[-1]: (batch, hidden_dim)
                context = h_n[-1]

        logits = self.classifier(context)
        return logits