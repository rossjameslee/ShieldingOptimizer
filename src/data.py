"""
Data handling and material database management.

This module provides classes and functions for managing shielding material
properties, feature processing, and data transformations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from .config import Config, MaterialProperties
import logging

logger = logging.getLogger(__name__)


class MaterialDatabase:
    """
    Database of shielding materials and their properties.
    
    This class manages the material properties database and provides
    methods for accessing and manipulating material data.
    """
    
    def __init__(self):
        """Initialize the material database."""
        self.materials = {mat.id: mat for mat in Config.MATERIALS}
        self.material_names = Config.MATERIAL_NAMES
    
    def get_material(self, material_id: int) -> Optional[MaterialProperties]:
        """
        Get material properties by ID.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Material properties or None if not found
        """
        return self.materials.get(material_id)
    
    def get_material_name(self, material_id: int) -> str:
        """
        Get material name by ID.
        
        Args:
            material_id: Material identifier
            
        Returns:
            Material name
        """
        return self.material_names.get(material_id, f"Unknown Material {material_id}")
    
    def list_materials(self) -> List[Tuple[int, str]]:
        """
        List all available materials.
        
        Returns:
            List of (id, name) tuples
        """
        return [(mat_id, self.get_material_name(mat_id)) 
                for mat_id in self.materials.keys()]
    
    def get_material_array(self, material_id: int) -> List[float]:
        """
        Get material properties as a flat array for optimization.
        
        Args:
            material_id: Material identifier
            
        Returns:
            List of material properties
        """
        mat = self.get_material(material_id)
        if mat is None:
            raise ValueError(f"Material {material_id} not found")
        
        return [
            mat.id, mat.density, mat.true_radius, mat.log_xs_gam_abe6,
            mat.log_xs_gam_abe7, mat.log_xs_gam_pro_e2, mat.log_xs_gam_pro_e1,
            mat.log_xs_gam_pro_e4, mat.log_xs_gam_pro_e5, mat.log_xs_gam_pro_e6,
            mat.log_xs_gam_pro_e7, mat.log_xs_alpha_pro_e2, mat.log_xs_alpha_pro_e1,
            mat.log_xs_alpha_pro_e4, mat.log_xs_alpha_pro_e5, mat.log_xs_alpha_pro_e6,
            mat.log_xs_alpha_pro_e7, mat.log_xs_elastic_e2, mat.log_xs_elastic_e1,
            mat.log_xs_elastic_e4, mat.log_xs_elastic_e5, mat.log_xs_elastic_e6,
            mat.log_xs_elastic_e7, mat.log_peak_elastic, mat.log_peak_alpha,
            mat.log_peak_ga, mat.log_peak_ge, mat.line_rate, mat.line_cost,
            mat.total_cost, mat.total_mass
        ]


class FeatureProcessor:
    """
    Handles feature processing and data transformations.
    
    This class provides methods for preparing input data for the ML model
    and handling feature extraction and normalization.
    """
    
    def __init__(self):
        """Initialize the feature processor."""
        self.feature_columns = Config.FEATURE_COLUMNS
        self.columns_to_drop = Config.COLUMNS_TO_DROP
        self.max_values = Config.MAX_VALUES
    
    def extract_features(self, data_array: List[float]) -> List[float]:
        """
        Extract features from raw data array.
        
        Args:
            data_array: Raw data array with all columns
            
        Returns:
            Feature array for ML model
        """
        # Indices to drop for feature extraction
        indices_to_drop = [0, 2, 5, 6, 7] + list(range(9, 33)) + [39] + list(range(57, 61))  # Drop index 60 to get 27 features
        
        # Filter out non-feature columns
        features = [item for index, item in enumerate(data_array) 
                   if index not in indices_to_drop]
        
        return features
    
    def normalize_features(self, features: List[float]) -> List[float]:
        """
        Normalize features using maximum values.
        
        Args:
            features: Raw feature values
            
        Returns:
            Normalized feature values
        """
        if len(features) != len(self.max_values):
            raise ValueError(f"Feature count {len(features)} doesn't match max values {len(self.max_values)}")
        
        return [feature / max_val for feature, max_val in zip(features, self.max_values)]
    
    def denormalize_features(self, normalized_features: List[float]) -> List[float]:
        """
        Denormalize features back to original scale.
        
        Args:
            normalized_features: Normalized feature values
            
        Returns:
            Denormalized feature values
        """
        if len(normalized_features) != len(self.max_values):
            raise ValueError(f"Feature count {len(normalized_features)} doesn't match max values {len(self.max_values)}")
        
        return [feature * max_val for feature, max_val in zip(normalized_features, self.max_values)]


class DataBuilder:
    """
    Builds data arrays for optimization from material properties.
    
    This class constructs the data arrays needed for the optimization
    process from material properties and layer configurations.
    """
    
    def __init__(self):
        """Initialize the data builder."""
        self.material_db = MaterialDatabase()
        self.feature_processor = FeatureProcessor()
    
    def build_layer_data(self, magic_num: float, material_array: List[float], 
                        layer_position: int) -> List[List[float]]:
        """
        Build data array for a single layer.
        
        Args:
            magic_num: Magic number for calculations
            material_array: Material properties array
            layer_position: Position of the layer
            
        Returns:
            Data array for the layer
        """
        # Initialize test array with zeros
        test_array = [[0.0] * 61]  # 61 columns as per original code
        
        # Unpack material array
        (material_id, density, true_radius, log_xs_gam_abe6, log_xs_gam_abe7,
         log_xs_gam_pro_e2, log_xs_gam_pro_e1, log_xs_gam_pro_e4, log_xs_gam_pro_e5,
         log_xs_gam_pro_e6, log_xs_gam_pro_e7, log_xs_alpha_pro_e2, log_xs_alpha_pro_e1,
         log_xs_alpha_pro_e4, log_xs_alpha_pro_e5, log_xs_alpha_pro_e6, log_xs_alpha_pro_e7,
         log_xs_elastic_e2, log_xs_elastic_e1, log_xs_elastic_e4, log_xs_elastic_e5,
         log_xs_elastic_e6, log_xs_elastic_e7, log_peak_elastic, log_peak_alpha,
         log_peak_ga, log_peak_ge, line_rate, line_cost, total_cost, total_mass) = material_array
        
        # Fill basic properties
        test_array[0][0] = material_id
        test_array[0][1] = density
        test_array[0][3] = true_radius
        test_array[0][4] = true_radius + magic_num * layer_position
        
        # Calculate derived properties
        radius = test_array[0][4]
        cross_sectional_area = np.pi * radius**2
        volume = cross_sectional_area * magic_num
        mass = density * volume
        
        test_array[0][5] = cross_sectional_area
        test_array[0][6] = volume
        test_array[0][7] = volume
        test_array[0][8] = mass
        
        # Fill cross-section data
        test_array[0][9] = log_xs_gam_abe6
        test_array[0][10] = log_xs_gam_abe7
        test_array[0][11] = log_xs_gam_pro_e2
        test_array[0][12] = log_xs_gam_pro_e1
        test_array[0][13] = log_xs_gam_pro_e4
        test_array[0][14] = log_xs_gam_pro_e5
        test_array[0][15] = log_xs_gam_pro_e6
        test_array[0][16] = log_xs_gam_pro_e7
        test_array[0][17] = log_xs_alpha_pro_e2
        test_array[0][18] = log_xs_alpha_pro_e1
        test_array[0][19] = log_xs_alpha_pro_e4
        test_array[0][20] = log_xs_alpha_pro_e5
        test_array[0][21] = log_xs_alpha_pro_e6
        test_array[0][22] = log_xs_alpha_pro_e7
        test_array[0][23] = log_xs_elastic_e2
        test_array[0][24] = log_xs_elastic_e1
        test_array[0][25] = log_xs_elastic_e4
        test_array[0][26] = log_xs_elastic_e5
        test_array[0][27] = log_xs_elastic_e6
        test_array[0][28] = log_xs_elastic_e7
        test_array[0][29] = log_peak_elastic
        test_array[0][30] = log_peak_alpha
        test_array[0][31] = log_peak_ga
        test_array[0][32] = log_peak_ge
        
        # Calculate and fill Nave features (averaged values)
        test_array[0][2] = test_array[0][1]  # Density Nave
        test_array[0][33] = test_array[0][9]  # LogXSgamABe6Nave
        test_array[0][34] = test_array[0][10]  # LogXSgamABe7Nave
        test_array[0][35] = test_array[0][11]  # LogXSgamPROe-2Nave
        test_array[0][36] = test_array[0][12]  # LogXSgamPROe-1Nave
        test_array[0][37] = test_array[0][13]  # LogXSgamPROe4Nave
        test_array[0][38] = test_array[0][14]  # LogXSgamPROe5Nave
        test_array[0][39] = test_array[0][15]  # LogXSgamPROe6Nave
        test_array[0][40] = test_array[0][16]  # LogXSgamPROe7Nave
        test_array[0][41] = test_array[0][17]  # LogXSalphaPROe-2Nave
        test_array[0][42] = test_array[0][18]  # LogXSalphaPROe-1Nave
        test_array[0][43] = test_array[0][19]  # LogXSalphaPROe4Nave
        test_array[0][44] = test_array[0][20]  # LogXSalphaPROe5Nave
        test_array[0][45] = test_array[0][21]  # LogXSalphaPROe6Nave
        test_array[0][46] = test_array[0][22]  # LogXSalphaPROe7Nave
        test_array[0][47] = test_array[0][23]  # LogXSelasticE-2Nave
        test_array[0][48] = test_array[0][24]  # LogXSelasticE-1Nave
        test_array[0][49] = test_array[0][25]  # LogXSelasticE4Nave
        test_array[0][50] = test_array[0][26]  # LogXSelasticE5Nave
        test_array[0][51] = test_array[0][27]  # LogXSelasticE6Nave
        test_array[0][52] = test_array[0][28]  # LogXSelasticE7Nave
        test_array[0][53] = test_array[0][29]  # LogPeakElasticNave
        test_array[0][54] = test_array[0][30]  # LogPeakAlphaNave
        test_array[0][55] = test_array[0][31]  # LogPeakGANave
        test_array[0][56] = test_array[0][32]  # LogPeakGENave
        
        # Assign line cost
        test_array[0][57] = line_cost
        
        return test_array 