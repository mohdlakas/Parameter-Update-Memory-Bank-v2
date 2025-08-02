import numpy as np
import torch
import logging


class QualityMetric:
    # Q_loss: Normalized loss improvement.
    # Q_consistency: Consistency of model updates.
    # Q_data: Normalized data size.
    # quality = α·Q_loss + β·Q_consistency + γ·Q_data
    # where α + β + γ = 1.0
    # Default weights are α=0.6, β=0.3, γ= 0.1
    
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        """Quality weights as specified in paper Section IV.B"""
        self.alpha = alpha  # Loss improvement weight
        self.beta = beta    # Consistency weight  
        self.gamma = gamma  # Data size weight
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1"
        self.logger = logging.getLogger('PUMB_Quality')

    def calculate_quality(self, loss_before, loss_after, data_sizes, param_update, 
                         round_num, client_id, all_loss_improvements=None):
        """
        THEORY-ALIGNED: Implement q_i^t = α·Q_loss + β·Q_consistency + γ·Q_data
        """
        # Q_loss: Normalized loss improvement
        loss_improvement = max(0, loss_before - loss_after)  # ΔL_i^t
        
        if all_loss_improvements is not None and len(all_loss_improvements) > 0:
            max_improvement = max(all_loss_improvements) + 1e-8
            Q_loss = loss_improvement / max_improvement
        else:
            Q_loss = 1.0 if loss_improvement > 0 else 0.1
        
        # Q_consistency: As per paper equation
        param_values = torch.cat([p.flatten() for p in param_update.values()])
        param_np = param_values.detach().cpu().numpy()
        
        if len(param_np) > 0:
            mean_val = np.mean(param_np)
            var_val = np.var(param_np)
            # Q_consistency = exp(-Var[Δθ]/Mean[Δθ]^2 + ε)
            consistency_ratio = var_val / (mean_val**2 + 1e-8)
            Q_consistency = np.exp(-consistency_ratio)
        else:
            Q_consistency = 0.5
        
        # Q_data: Normalized data size
        max_data_size = max(data_sizes.values()) if data_sizes else 1
        Q_data = data_sizes.get(client_id, 1) / max_data_size
        
        # Combine according to paper formula
        quality = self.alpha * Q_loss + self.beta * Q_consistency + self.gamma * Q_data
        
        self.logger.info(f"Client {client_id}: Q_loss={Q_loss:.4f}, Q_consistency={Q_consistency:.4f}, Q_data={Q_data:.4f}, final_score={quality:.4f}")
        return max(0.01, min(1.0, quality))  # Clamp to reasonable range
# class QualityMetric:
#   # quality = (loss_improvement * actual_data_size * (1 + alignment_score)) / (update_norm + epsilon)
#     def __init__(self, epsilon=1e-8, momentum=0.9):
#         """Initialize the quality metric calculator with global direction estimation."""
#         self.epsilon = epsilon
#         self.momentum = momentum
#         self.global_direction = None
#         self.logger = logging.getLogger('PUMB_Quality')


#     def update_global_direction(self, current_updates):
#         """Update global direction using momentum-based estimation."""
#         if not current_updates:
#             return
#         # Stack and mean updates
#         if isinstance(current_updates[0], torch.Tensor):
#             current_mean = torch.stack(current_updates).mean(dim=0)
#         else:
#             current_mean = np.mean(current_updates, axis=0)
#         # Momentum update
#         if self.global_direction is None:
#             self.global_direction = current_mean
#         else:
#             if isinstance(current_mean, torch.Tensor):
#                 self.global_direction = (
#                     self.momentum * self.global_direction +
#                     (1 - self.momentum) * current_mean
#                 )
#             else:
#                 self.global_direction = (
#                     self.momentum * self.global_direction +
#                     (1 - self.momentum) * current_mean
#                 )

#     def calculate_quality(self, loss_before, loss_after, data_size, client_update, update_norm=None, round_num=0, client_id=None):
#         """Calculate efficiency-based quality metric using global direction."""
#         loss_improvement = max(0, loss_before - loss_after)
        
#         # Handle data_size parameter - extract the actual size if it's a dict
#         if isinstance(data_size, dict):
#             if client_id is not None and client_id in data_size:
#                 actual_data_size = data_size[client_id]
#             else:
#                 # Fallback: use the first value or sum of all values
#                 actual_data_size = list(data_size.values())[0] if data_size else 1
#         else:
#             actual_data_size = data_size
        
#         # Flatten client update
#         if isinstance(client_update, dict):
#             client_flat = torch.cat([v.flatten() for v in client_update.values()])
#         elif isinstance(client_update, torch.Tensor):
#             client_flat = client_update.flatten()
#         else:
#             client_flat = torch.tensor(client_update).flatten()
        
#         # Update norm
#         if update_norm is None:
#             update_norm = torch.norm(client_flat, p=2).item()
        
#         # Initialize alignment score
#         alignment_score = 0.0
        
#         # Alignment with global direction
#         if self.global_direction is not None and round_num > 2:
#             if isinstance(self.global_direction, dict):
#                 # If global_direction is a dict, flatten it
#                 global_flat = torch.cat([v.flatten() for v in self.global_direction.values()])
#             elif isinstance(self.global_direction, torch.Tensor):
#                 global_flat = self.global_direction.flatten()
#             else:
#                 global_flat = torch.tensor(self.global_direction).flatten()
            
#             min_size = min(len(client_flat), len(global_flat))
#             client_flat = client_flat[:min_size]
#             global_flat = global_flat[:min_size]
            
#             if torch.norm(global_flat) > self.epsilon:
#                 alignment_value = torch.cosine_similarity(
#                     client_flat.unsqueeze(0),
#                     global_flat.unsqueeze(0)
#                 ).item()
#                 alignment_score = max(0, alignment_value)
        
#         # Quality score using the actual data size
#         quality = (loss_improvement * actual_data_size * (1 + alignment_score)) / (update_norm + self.epsilon)

#         # Use info level to see this information by default
#         self.logger.info(f"Client {client_id}: loss_imp={loss_improvement:.4f}, final_score={quality:.4f}")
#         return quality


    def _flatten_params(self, model_params):
        """Flatten PyTorch parameters into a single vector."""
        if isinstance(model_params, dict):
            # If it's a state dict
            return torch.cat([p.flatten() for p in model_params.values()])
        elif isinstance(model_params, list):
            # If it's a list of tensors
            return torch.cat([p.flatten() for p in model_params])
        else:
            # Assume it's already a tensor
            return model_params.flatten()
    
    def _flatten_params_numpy(self, model_params):
        """Flatten numpy parameters into a single vector."""
        if isinstance(model_params, dict):
            # If it's a dict
            return np.concatenate([p.flatten() for p in model_params.values()])
        elif isinstance(model_params, list):
            # If it's a list of arrays
            return np.concatenate([p.flatten() for p in model_params])
        else:
            # Assume it's already an array
            return model_params.flatten()