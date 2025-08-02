from scipy.stats import skew
import torch
import torch.nn.functional as F
import numpy as np

class EmbeddingGenerator:
    def __init__(self, feature_dim=10, embedding_dim=512):
        """Initialize the parameter update embedding generator."""
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Simple MLP for feature projection
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, embedding_dim)
        )
        
        # Initialize projection weights
        with torch.no_grad():
            for layer in self.projection:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)
    
    def generate_embedding(self, parameter_update):
        """Generate an embedding vector from parameter updates."""
        # Extract statistical features
        features = self._extract_features(parameter_update)
        
        # Convert to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Handle NaN or infinite values
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize features
        features = F.normalize(features.unsqueeze(0), p=2, dim=1).squeeze(0)
        
        # Project to embedding space
        with torch.no_grad():
            embedding = self.projection(features)
            
        return embedding
    
    def _extract_features(self, parameter_update):
        """Extract comprehensive statistical features from parameter updates."""
        # Convert to numpy for processing
        if isinstance(parameter_update, torch.Tensor):
            update_values = parameter_update.detach().cpu().numpy().flatten()
        elif isinstance(parameter_update, dict):
            # Handle state dict
            if all(isinstance(v, torch.Tensor) for v in parameter_update.values()):
                update_values = torch.cat([p.flatten() for p in parameter_update.values()]).detach().cpu().numpy()
            else:
                update_values = np.concatenate([np.array(p).flatten() for p in parameter_update.values()])
        else:
            update_values = np.array(parameter_update).flatten()
        
        # Handle edge cases
        if len(update_values) == 0:
            return np.zeros(self.feature_dim)
        
        # Remove NaN and infinite values
        update_values = update_values[np.isfinite(update_values)]
        if len(update_values) == 0:
            return np.zeros(self.feature_dim)
        
        # Calculate robust statistical features
        try:
            mean_val = np.mean(update_values)
            std_val = np.std(update_values)
            l2_norm = np.linalg.norm(update_values, 2)
            l1_norm = np.linalg.norm(update_values, 1)
            min_val = np.min(update_values)
            max_val = np.max(update_values)
            median_val = np.median(update_values)
            
            # Handle skewness calculation
            if std_val > 1e-8:
                skewness = skew(update_values)
            else:
                skewness = 0.0
            
            # Additional robust features
            q75, q25 = np.percentile(update_values, [75, 25])
            iqr = q75 - q25
            
            features = np.array([
                mean_val, std_val, l2_norm, l1_norm,
                min_val, max_val, median_val, skewness,
                iqr, np.mean(np.abs(update_values))  # Mean absolute value
            ])
            
        except Exception as e:
            # Fallback to zeros if feature extraction fails
            features = np.zeros(self.feature_dim)
        
        return features