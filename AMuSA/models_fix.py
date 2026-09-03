import torch
import torch.nn as nn
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SignatureClassifier(nn.Module):
    def __init__(self, num_signatures, encoding_dim=128, dropout_rate=0.2, threshold=0.4):
        super(SignatureClassifier, self).__init__()
        self.num_signatures = num_signatures
        
        if isinstance(threshold, (list, np.ndarray)):
            self.thresholds = torch.tensor(threshold, dtype=torch.float).to(device)
        else:
            self.thresholds = torch.ones(num_signatures, dtype=torch.float) * threshold
            self.thresholds = self.thresholds.to(device)

        self.input_proj = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.attention = AttentionLayer(256)
        self.resblock1 = ResidualBlock(256, dropout_rate)
        self.resblock2 = ResidualBlock(256, dropout_rate)

        self.down_proj = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Linear(128, num_signatures)
        self._init_parameters()
        
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update_thresholds(self, new_thresholds):
        if isinstance(new_thresholds, (list, np.ndarray)):
            self.thresholds = torch.tensor(new_thresholds, dtype=torch.float).to(device)
        else:
            self.thresholds = torch.ones(self.num_signatures, dtype=torch.float) * new_thresholds
            self.thresholds = self.thresholds.to(device)

    def forward(self, encoded_x):
        x = self.input_proj(encoded_x)
        x = self.attention(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.down_proj(x)
        
        logits = self.output_layer(x)
        probs = torch.sigmoid(logits)

        active_preds = (probs >= self.thresholds).float()
        return probs, active_preds, logits

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.net(x)

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=128, dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, encoding_dim),
            nn.SiLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
