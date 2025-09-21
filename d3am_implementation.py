"""
Dynamic Activation-Aware Alpha Modulation (DA3M) Implementation
PyTorch implementation of the DA3M method for model diffing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ActivationStats:
    """Container for activation statistics"""
    mean: torch.Tensor
    variance: torch.Tensor
    std: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

class ActivationHook:
    """Hook to capture activations from model layers"""
    
    def __init__(self):
        self.activations = []
        self.hooks = []
    
    def hook_fn(self, module, input, output):
        """Hook function to capture activations"""
        if isinstance(output, torch.Tensor):
            self.activations.append(output.detach())
        elif isinstance(output, tuple):
            # Handle tuple outputs (e.g., from transformer layers)
            self.activations.append(output[0].detach())
    
    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """Register hooks on specified layers"""
        for name, module in model.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
                logger.info(f"Registered hook on layer: {name}")
    
    def clear_activations(self):
        """Clear stored activations"""
        self.activations = []
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def compute_activation_stats(activations: List[torch.Tensor]) -> ActivationStats:
    """Compute statistics from a list of activation tensors"""
    if not activations:
        raise ValueError("No activations provided")
    
    # Concatenate all activations along the batch dimension
    all_activations = torch.cat([act.view(act.size(0), -1) for act in activations], dim=1)
    
    # Compute statistics
    mean = torch.mean(all_activations, dim=1, keepdim=True)
    variance = torch.var(all_activations, dim=1, keepdim=True)
    std = torch.std(all_activations, dim=1, keepdim=True)
    min_val = torch.min(all_activations, dim=1, keepdim=True)[0]
    max_val = torch.max(all_activations, dim=1, keepdim=True)[0]
    
    return ActivationStats(
        mean=mean,
        variance=variance,
        std=std,
        min_val=min_val,
        max_val=max_val
    )

class AlphaController(ABC):
    """Abstract base class for Alpha Controllers"""
    
    @abstractmethod
    def compute_alpha(self, base_stats: ActivationStats, tuned_stats: ActivationStats) -> torch.Tensor:
        """Compute dynamic alpha value based on activation statistics"""
        pass

class HeuristicAlphaController(AlphaController):
    """Heuristic-based Alpha Controller"""
    
    def __init__(self, k: float = 10.0, eps: float = 1e-8):
        self.k = k
        self.eps = eps
    
    def compute_alpha(self, base_stats: ActivationStats, tuned_stats: ActivationStats) -> torch.Tensor:
        """Compute alpha using heuristic formula"""
        # Normalize differences
        mean_diff = torch.abs(tuned_stats.mean - base_stats.mean) / (torch.abs(base_stats.mean) + self.eps)
        var_diff = torch.abs(tuned_stats.variance - base_stats.variance) / (torch.abs(base_stats.variance) + self.eps)
        
        # Heuristic formula: alpha = k * (1 + |mean_diff|) * (1 + |var_diff|)
        alpha = self.k * (1 + mean_diff) * (1 + var_diff)
        
        return alpha.squeeze()

class LinearAlphaController(AlphaController, nn.Module):
    """Learned Linear Alpha Controller"""
    
    def __init__(self, input_dim: int = 4, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.scale_factor = 50.0  # Scale sigmoid output to reasonable alpha range
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def compute_alpha(self, base_stats: ActivationStats, tuned_stats: ActivationStats) -> torch.Tensor:
        """Compute alpha using learned linear mapping"""
        # Create input vector [mean_base, var_base, mean_tuned, var_tuned]
        # Ensure all tensors are at least 1D
        base_mean = base_stats.mean.flatten()
        base_var = base_stats.variance.flatten()
        tuned_mean = tuned_stats.mean.flatten()
        tuned_var = tuned_stats.variance.flatten()
        
        # Take mean across all dimensions to get scalar features
        input_vector = torch.cat([
            base_mean.mean().unsqueeze(0),
            base_var.mean().unsqueeze(0),
            tuned_mean.mean().unsqueeze(0),
            tuned_var.mean().unsqueeze(0)
        ], dim=0)
        
        # Normalize input to prevent numerical issues
        input_vector = F.normalize(input_vector, dim=-1)
        
        # Forward pass
        alpha = self.linear(input_vector)
        alpha = self.sigmoid(alpha) * self.scale_factor
        
        return alpha.squeeze()

class MLPAlphaController(AlphaController, nn.Module):
    """MLP-based Alpha Controller"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.scale_factor = 50.0
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def compute_alpha(self, base_stats: ActivationStats, tuned_stats: ActivationStats) -> torch.Tensor:
        """Compute alpha using MLP"""
        # Create input vector
        # Ensure all tensors are at least 1D
        base_mean = base_stats.mean.flatten()
        base_var = base_stats.variance.flatten()
        tuned_mean = tuned_stats.mean.flatten()
        tuned_var = tuned_stats.variance.flatten()
        
        # Take mean across all dimensions to get scalar features
        input_vector = torch.cat([
            base_mean.mean().unsqueeze(0),
            base_var.mean().unsqueeze(0),
            tuned_mean.mean().unsqueeze(0),
            tuned_var.mean().unsqueeze(0)
        ], dim=0)
        
        # Normalize input
        input_vector = F.normalize(input_vector, dim=-1)
        
        # Forward pass
        alpha = self.network(input_vector) * self.scale_factor
        
        return alpha.squeeze()

class DA3MModelDiffer:
    """Main DA3M Model Diffing class"""
    
    def __init__(
        self,
        base_model: nn.Module,
        tuned_model: nn.Module,
        alpha_controller: AlphaController,
        hook_layers: List[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.base_model = base_model.to(device)
        self.tuned_model = tuned_model.to(device)
        self.alpha_controller = alpha_controller
        self.device = device
        
        # Default hook layers for transformer models
        if hook_layers is None:
            hook_layers = ["layers", "blocks", "transformer"]
        self.hook_layers = hook_layers
        
        # Initialize hooks
        self.base_hook = ActivationHook()
        self.tuned_hook = ActivationHook()
        
        # Set models to evaluation mode
        self.base_model.eval()
        self.tuned_model.eval()
    
    def setup_hooks(self):
        """Setup activation hooks on both models"""
        self.base_hook.register_hooks(self.base_model, self.hook_layers)
        self.tuned_hook.register_hooks(self.tuned_model, self.hook_layers)
    
    def cleanup_hooks(self):
        """Remove all hooks"""
        self.base_hook.remove_hooks()
        self.tuned_hook.remove_hooks()
    
    def forward_with_diffing(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_static_alpha: bool = False,
        static_alpha: float = 20.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform forward pass with model diffing
        
        Returns:
            base_logits: Logits from base model
            tuned_logits: Logits from tuned model  
            amplified_logits: Amplified logits using DA3M or static alpha
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Clear previous activations
        self.base_hook.clear_activations()
        self.tuned_hook.clear_activations()
        
        with torch.no_grad():
            # Forward pass through base model
            if attention_mask is not None:
                base_outputs = self.base_model(input_ids, attention_mask=attention_mask)
            else:
                base_outputs = self.base_model(input_ids)
            
            base_logits = base_outputs.logits if hasattr(base_outputs, 'logits') else base_outputs
            
            # Forward pass through tuned model
            if attention_mask is not None:
                tuned_outputs = self.tuned_model(input_ids, attention_mask=attention_mask)
            else:
                tuned_outputs = self.tuned_model(input_ids)
            
            tuned_logits = tuned_outputs.logits if hasattr(tuned_outputs, 'logits') else tuned_outputs
        
        # Compute alpha
        if use_static_alpha:
            alpha = torch.tensor(static_alpha, device=self.device)
        else:
            # Compute activation statistics
            base_stats = compute_activation_stats(self.base_hook.activations)
            tuned_stats = compute_activation_stats(self.tuned_hook.activations)
            
            # Get dynamic alpha from controller
            alpha = self.alpha_controller.compute_alpha(base_stats, tuned_stats)
            
            # Ensure alpha is scalar or broadcastable
            if alpha.dim() > 0 and alpha.size(0) == 1:
                alpha = alpha.item()
        
        # Apply model diffing formula
        # amplified_logits = tuned_logits + alpha * (tuned_logits - base_logits)
        # Ensure alpha can be broadcast with logits
        if isinstance(alpha, torch.Tensor) and alpha.dim() > 0:
            # Reshape alpha to be broadcastable with logits
            while alpha.dim() < tuned_logits.dim():
                alpha = alpha.unsqueeze(-1)
        
        amplified_logits = tuned_logits + alpha * (tuned_logits - base_logits)
        
        return base_logits, tuned_logits, amplified_logits
    
    def generate_with_diffing(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        use_static_alpha: bool = False,
        static_alpha: float = 20.0,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text using amplified logits"""
        
        current_input_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone() if attention_mask is not None else None
        
        for _ in range(max_new_tokens):
            # Get amplified logits for next token
            _, _, amplified_logits = self.forward_with_diffing(
                current_input_ids,
                current_attention_mask,
                use_static_alpha=use_static_alpha,
                static_alpha=static_alpha
            )
            
            # Get logits for the last token
            next_token_logits = amplified_logits[:, -1, :] / temperature
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            
            # Update attention mask if provided
            if current_attention_mask is not None:
                new_attention = torch.ones((current_attention_mask.size(0), 1), 
                                         device=current_attention_mask.device)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
            
            # Check for EOS token (assuming token_id 2 is EOS)
            if next_token.item() == 2:
                break
        
        return current_input_ids

def train_alpha_controller(
    controller: Union[LinearAlphaController, MLPAlphaController],
    training_data: List[Tuple[ActivationStats, ActivationStats, float]],
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """Train the alpha controller on labeled data"""
    
    if not isinstance(controller, nn.Module):
        raise ValueError("Controller must be a PyTorch module for training")
    
    controller = controller.to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    controller.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for base_stats, tuned_stats, target_alpha in training_data:
            # Move stats to device
            base_stats.mean = base_stats.mean.to(device)
            base_stats.variance = base_stats.variance.to(device)
            tuned_stats.mean = tuned_stats.mean.to(device)
            tuned_stats.variance = tuned_stats.variance.to(device)
            
            target_alpha = torch.tensor(target_alpha, device=device)
            
            # Forward pass
            predicted_alpha = controller.compute_alpha(base_stats, tuned_stats)
            
            # Compute loss
            loss = criterion(predicted_alpha, target_alpha)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / len(training_data)
            logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    controller.eval()
    logger.info("Alpha controller training completed")

# Example usage and testing functions
def create_mock_models(vocab_size: int = 1000, hidden_size: int = 768) -> Tuple[nn.Module, nn.Module]:
    """Create mock transformer models for testing"""
    
    class MockTransformer(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True)
                for _ in range(6)
            ])
            self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            
            for layer in self.layers:
                x = layer(x, src_key_padding_mask=attention_mask)
            
            logits = self.lm_head(x)
            return type('Output', (), {'logits': logits})()
    
    base_model = MockTransformer(vocab_size, hidden_size)
    tuned_model = MockTransformer(vocab_size, hidden_size)
    
    # Make tuned model slightly different
    with torch.no_grad():
        for param in tuned_model.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    return base_model, tuned_model

if __name__ == "__main__":
    # Example usage
    logger.info("Creating mock models...")
    base_model, tuned_model = create_mock_models()
    
    # Test different alpha controllers
    controllers = {
        "heuristic": HeuristicAlphaController(k=10.0),
        "linear": LinearAlphaController(),
        "mlp": MLPAlphaController()
    }
    
    for name, controller in controllers.items():
        logger.info(f"Testing {name} controller...")
        
        # Create DA3M differ
        differ = DA3MModelDiffer(base_model, tuned_model, controller)
        differ.setup_hooks()
        
        # Test input
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Test forward pass
        base_logits, tuned_logits, amplified_logits = differ.forward_with_diffing(input_ids)
        
        logger.info(f"Base logits shape: {base_logits.shape}")
        logger.info(f"Tuned logits shape: {tuned_logits.shape}")
        logger.info(f"Amplified logits shape: {amplified_logits.shape}")
        
        # Test generation
        generated = differ.generate_with_diffing(input_ids[:1], max_new_tokens=5)
        logger.info(f"Generated sequence shape: {generated.shape}")
        
        differ.cleanup_hooks()
        logger.info(f"{name} controller test completed\n")
