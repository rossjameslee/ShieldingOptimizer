"""
Machine learning models for shielding optimization.

This module contains the neural network models and Gekko integration
classes used for predicting radiation dose through shielding materials.
"""

import numpy as np
from typing import List, Optional, Union
from gekko import GEKKO
import joblib
import logging

logger = logging.getLogger(__name__)


class GekkoDense:
    """
    A dense neural network layer that integrates with Gekko optimization.
    
    This class implements a fully connected layer with various activation
    functions that can be used within Gekko optimization problems.
    """
    
    def __init__(self, layer: List, m: Optional[GEKKO] = None):
        """
        Initialize the dense layer.
        
        Args:
            layer: List containing [n_in, n_out, weights, bias, activation]
            m: Gekko model instance for optimization integration
        """
        n_in, n_out, W, b, activation = layer
        self.weights = W
        self.bias = b
        self.af = activation
        
        if m is not None:
            self.hook_gekko(m)
    
    def hook_gekko(self, m: GEKKO) -> None:
        """Hook this layer to a Gekko model for optimization."""
        self.m = m
    
    def activation(self, x, skip_act: bool = False):
        """
        Apply activation function to input.
        
        Args:
            x: Input tensor
            skip_act: Whether to skip activation (for output layer)
            
        Returns:
            Activated output
        """
        if skip_act:
            return x
            
        af = self.af
        
        if af == 'relu':
            return self.m.max3(0, x)
        elif af == 'sigmoid':
            return 1 / (1 + self.m.exp(-x))
        elif af == 'tanh':
            return self.m.tanh(x)
        elif af == 'softsign':
            return x / (self.m.abs2(x) + 1)
        elif af == 'exponential':
            return self.m.exp(x)
        elif af == 'softplus':
            return self.m.log(self.m.exp(x) + 1)
        elif af == 'elu':
            alpha = 1.0
            return self.m.if3(x, alpha * (self.m.exp(x) - 1), x)
        elif af == 'selu':
            alpha = 1.67326324
            scale = 1.05070098
            return self.m.if3(x, scale * alpha * (self.m.exp(x) - 1), scale * x)
        else:
            return x
    
    def forward(self, x, skip_act: bool = False) -> List:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor
            skip_act: Whether to skip activation
            
        Returns:
            Output tensor
        """
        n = self.weights.shape[1]
        return [
            self.m.Intermediate(
                self.activation(
                    self.m.sum(self.weights[:, i] * x) + self.bias[i], 
                    skip_act
                )
            ) for i in range(n)
        ]
    
    def __call__(self, x, skip_act: bool = False) -> List:
        """Call forward method."""
        return self.forward(x, skip_act)


class GekkoSklearnModel:
    """
    Wrapper for scikit-learn models to integrate with Gekko optimization.
    
    This class converts trained scikit-learn neural network models
    into a format that can be used within Gekko optimization problems.
    """
    
    def __init__(self, model, m: GEKKO):
        """
        Initialize the Gekko-sklearn model wrapper.
        
        Args:
            model: Trained scikit-learn MLPRegressor
            m: Gekko model instance
        """
        self.m = m
        self.W = model.coefs_
        self.b = model.intercepts_
        self.hidden_layer_sizes = model.hidden_layer_sizes
        self.n_in = model.n_features_in_
        self.n_out = model.n_outputs_
        self.activation = model.activation
        
        self.layers = self._build_layers()
    
    def _build_layers(self) -> List[GekkoDense]:
        """Build the neural network layers."""
        layers = []
        
        if len(self.hidden_layer_sizes) == 0:
            # Single layer (input to output)
            layer = [self.n_in, self.n_out, self.W[0], self.b[0], self.activation]
            layers.append(GekkoDense(layer, self.m))
        else:
            # Input layer
            layer = [self.n_in, self.hidden_layer_sizes[0], self.W[0], self.b[0], self.activation]
            layers.append(GekkoDense(layer, self.m))
            
            # Hidden layers
            for i in range(1, len(self.hidden_layer_sizes)):
                layer = [
                    self.hidden_layer_sizes[i-1], 
                    self.hidden_layer_sizes[i], 
                    self.W[i], 
                    self.b[i], 
                    self.activation
                ]
                layers.append(GekkoDense(layer, self.m))
            
            # Output layer
            layer = [self.hidden_layer_sizes[-1], self.n_out, self.W[-1], self.b[-1], self.activation]
            layers.append(GekkoDense(layer, self.m))
        
        return layers
    
    def forward(self, x) -> List:
        """
        Forward pass through the entire network.
        
        Args:
            x: Input features
            
        Returns:
            Network output
        """
        l = x
        for i, layer in enumerate(self.layers):
            skip_act = (i == len(self.layers) - 1)  # Skip activation for output layer
            l = layer(l, skip_act)
        return l
    
    def __call__(self, x) -> List:
        """Call forward method."""
        return self.forward(x)
    
    def predict(self, x, return_std: bool = False) -> Union[float, tuple]:
        """
        Make prediction using the model.
        
        Args:
            x: Input features
            return_std: Whether to return standard deviation (not implemented)
            
        Returns:
            Predicted value
        """
        return self.forward(x)[0]


class MLModel:
    """
    High-level machine learning model interface.
    
    This class provides a clean interface for loading and using
    trained machine learning models for dose prediction.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the ML model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model from file."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            features: Input features array
            
        Returns:
            Predicted dose values
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            predictions = self.model.predict(features)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def create_gekko_model(self, m: GEKKO) -> GekkoSklearnModel:
        """
        Create a Gekko-compatible version of the model.
        
        Args:
            m: Gekko model instance
            
        Returns:
            GekkoSklearnModel wrapper
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        return GekkoSklearnModel(self.model, m) 