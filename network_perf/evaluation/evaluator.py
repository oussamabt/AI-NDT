"""
Model evaluation utilities for network performance prediction models.

This module provides classes for evaluating GNN models on network performance data.
"""

import torch
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class ProperModelEvaluator:
    """Evaluator for GNN models with proper train/val/test splitting and metrics reporting."""
    
    def __init__(self, data, test_size=0.2, val_size=0.2, random_state=42):
        """
        Initialize evaluator with data and split ratios.
        
        Args:
            data: PyTorch Geometric Data object
            test_size: Fraction of data to use for test set
            val_size: Fraction of remaining data to use for validation set
            random_state: Random seed for reproducibility
        """
        self.data = data
        self.device = data.x.device
        self.test_size = test_size
        self.val_size = val_size
        
        # Split the data into train/val/test
        self._split_data(random_state)
    
    def _split_data(self, random_state):
        """Split data into train/val/test sets."""
        num_nodes = self.data.x.shape[0]
        indices = np.arange(num_nodes)
        
        # First split off the test set
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, random_state=random_state
        )
        
        # Then split the remaining data into train/val
        val_ratio = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio, random_state=random_state
        )
        
        # Save indices
        self.train_idx = torch.tensor(train_idx, dtype=torch.long).to(self.device)
        self.val_idx = torch.tensor(val_idx, dtype=torch.long).to(self.device) 
        self.test_idx = torch.tensor(test_idx, dtype=torch.long).to(self.device)
        
        print(f"Data split: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test nodes")
    
    def train_with_validation(self, model, config, num_epochs=200, patience=20):
        """
        Train model with early stopping based on validation performance.
        
        Args:
            model: The GNN model to train
            config: Configuration dict with optimizer settings
            num_epochs: Maximum number of epochs to train
            patience: Number of epochs to wait for improvement before stopping
            
        Returns:
            dict with training results
        """
        # Use the evaluator's device instead of trying to access model.device
        device = self.device
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate) #lr=config.get('learning_rate', 0.01))
        loss_fn = torch.nn.MSELoss()
        
        # For early stopping
        best_val_loss = float('inf')
        best_val_r2 = float('-inf')
        best_model_state = None
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        val_r2_scores = []
        
        # Get target values
        y = self.data.y.to(device)
        edge_index = self.data.edge_index.to(device)
        x = self.data.x.to(device)
        
        for epoch in range(num_epochs):
            # Training step
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(x, edge_index)
            
            # Loss on training nodes
            train_loss = loss_fn(out[self.train_idx], y[self.train_idx])
            train_losses.append(train_loss.item())
            
            # Backpropagation
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                val_loss = loss_fn(out[self.val_idx], y[self.val_idx])
                val_losses.append(val_loss.item())
                
                # Calculate R² score
                val_pred = out[self.val_idx].cpu().numpy()
                val_true = y[self.val_idx].cpu().numpy()
                val_r2 = r2_score(val_true, val_pred, multioutput='variance_weighted')
                val_r2_scores.append(val_r2)
            
            # Check for early stopping (prioritize R² over loss)
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_val_loss = val_loss.item()
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss.item():.4f}, "
                      f"Val Loss = {val_loss.item():.4f}, Val R² = {val_r2:.4f}")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)
            
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'best_val_loss': best_val_loss,
            'best_val_r2': best_val_r2,
            'epochs_trained': len(train_losses)
        }
    
    def evaluate_final_performance(self, model):
        """
        Evaluate model performance on test set.
        
        Args:
            model: Trained GNN model
            
        Returns:
            dict with test metrics
        """
        # Use the evaluator's device instead of trying to access model.device
        device = self.device
        model = model.to(device)
        model.eval()
        
        x = self.data.x.to(device)
        edge_index = self.data.edge_index.to(device)
        y = self.data.y.to(device)
        
        with torch.no_grad():
            out = model(x, edge_index)
            
            # Calculate predictions and true values
            test_pred = out[self.test_idx].cpu().numpy()
            test_true = y[self.test_idx].cpu().numpy()
            
            # Calculate overall R² (multioutput weighted by target variance)
            r2_overall = r2_score(test_true, test_pred, multioutput='variance_weighted')
            
            # Calculate R² for each target dimension (assuming multiple targets)
            r2_per_target = []
            if test_true.ndim > 1 and test_true.shape[1] > 1:
                for i in range(test_true.shape[1]):
                    r2 = r2_score(test_true[:, i], test_pred[:, i])
                    r2_per_target.append(r2)
            else:
                r2_per_target = [r2_overall]
            
            # Calculate MSE
            mse = np.mean((test_pred - test_true) ** 2)
            
            # Calculate MAE
            mae = np.mean(np.abs(test_pred - test_true))
            
        results = {
            'test_r2_overall': r2_overall,
            'test_mse': mse,
            'test_mae': mae,
        }
        
        # Add individual target R² scores if appropriate
        if len(r2_per_target) == 2:  # Assuming 2 targets: rtt and retransmissions
            results['test_r2_rtt'] = r2_per_target[0]
            results['test_r2_retrans'] = r2_per_target[1]
        
        return results
    
    def detect_overfitting(self, training_results):
        """
        Analyze training history to detect overfitting.
        
        Args:
            training_results: Results dict from train_with_validation
            
        Returns:
            dict with overfitting metrics
        """
        train_losses = training_results['train_losses']
        val_losses = training_results['val_losses']
        
        # Check final performance gap
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        
        # Calculate loss ratio (val/train) in the final epoch
        # Higher values indicate more overfitting
        loss_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')
        
        # Calculate slope of validation loss in last third of training
        last_third = len(val_losses) // 3
        if last_third > 0:
            recent_val_losses = val_losses[-last_third:]
            epochs = np.arange(len(recent_val_losses))
            
            if len(recent_val_losses) > 1:
                # Simple linear regression to get slope
                slope = np.polyfit(epochs, recent_val_losses, 1)[0]
            else:
                slope = 0
        else:
            slope = 0
            
        # Determine overfitting risk level
        if loss_ratio > 2.0 and slope > 0:
            risk = "high"
        elif loss_ratio > 1.5 or slope > 0:
            risk = "moderate" 
        else:
            risk = "low"
            
        return {
            'loss_ratio': loss_ratio,
            'val_loss_trend': slope,
            'overall_risk': risk
        }