"""
Tests for the data handling module.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data import MaterialDatabase, FeatureProcessor, DataBuilder
from src.config import Config


class TestMaterialDatabase(unittest.TestCase):
    """Test cases for the MaterialDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.material_db = MaterialDatabase()
    
    def test_material_database_initialization(self):
        """Test that material database is properly initialized."""
        self.assertIsNotNone(self.material_db.materials)
        self.assertIsNotNone(self.material_db.material_names)
        self.assertEqual(len(self.material_db.materials), 6)
    
    def test_get_material(self):
        """Test getting material by ID."""
        material = self.material_db.get_material(0)
        self.assertIsNotNone(material)
        self.assertEqual(material.id, 0)
        self.assertEqual(material.density, 11300)
        
        # Test non-existent material
        material = self.material_db.get_material(999)
        self.assertIsNone(material)
    
    def test_get_material_name(self):
        """Test getting material name by ID."""
        name = self.material_db.get_material_name(0)
        self.assertEqual(name, "Lead")
        
        name = self.material_db.get_material_name(1)
        self.assertEqual(name, "B4C")
        
        # Test unknown material
        name = self.material_db.get_material_name(999)
        self.assertEqual(name, "Unknown Material 999")
    
    def test_list_materials(self):
        """Test listing all materials."""
        materials = self.material_db.list_materials()
        self.assertEqual(len(materials), 6)
        
        # Check that all materials have valid IDs and names
        for material_id, material_name in materials:
            self.assertIsInstance(material_id, int)
            self.assertIsInstance(material_name, str)
            self.assertGreater(len(material_name), 0)
    
    def test_get_material_array(self):
        """Test getting material properties as array."""
        material_array = self.material_db.get_material_array(0)
        self.assertIsInstance(material_array, list)
        self.assertEqual(len(material_array), 31)  # 31 properties per material (including id)
        
        # Test with non-existent material
        with self.assertRaises(ValueError):
            self.material_db.get_material_array(999)


class TestFeatureProcessor(unittest.TestCase):
    """Test cases for the FeatureProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_processor = FeatureProcessor()
    
    def test_feature_processor_initialization(self):
        """Test that feature processor is properly initialized."""
        self.assertIsNotNone(self.feature_processor.feature_columns)
        self.assertIsNotNone(self.feature_processor.columns_to_drop)
        self.assertIsNotNone(self.feature_processor.max_values)
    
    def test_extract_features(self):
        """Test feature extraction from data array."""
        # Create a dummy data array with 61 elements
        data_array = [float(i) for i in range(61)]
        
        features = self.feature_processor.extract_features(data_array)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 27)  # 27 features
        
        # Check that dropped indices are not in features
        indices_to_drop = [0, 2, 5, 6, 7] + list(range(9, 33)) + [39] + list(range(57, 61))  # Drop index 60 to get 27 features
        for idx in indices_to_drop:
            self.assertNotIn(data_array[idx], features)
    
    def test_normalize_features(self):
        """Test feature normalization."""
        features = [1.0, 2.0, 3.0, 4.0, 5.0] * 5 + [1.0, 2.0]  # 27 features
        
        # Use the actual max values from config
        max_values = self.feature_processor.max_values
        self.assertEqual(len(max_values), 27)

        normalized = self.feature_processor.normalize_features(features)
        self.assertEqual(len(normalized), 27)

        # Check that normalization worked correctly
        for i, norm_val in enumerate(normalized):
            expected = features[i] / max_values[i]
            self.assertAlmostEqual(norm_val, expected)
    
    def test_normalize_features_mismatch(self):
        """Test that normalization fails with mismatched feature counts."""
        features = [1.0, 2.0, 3.0]  # Wrong number of features
        max_values = [10.0] * 27
        
        with self.assertRaises(ValueError):
            self.feature_processor.normalize_features(features)
    
    def test_denormalize_features(self):
        """Test feature denormalization."""
        normalized_features = [0.1, 0.2, 0.3, 0.4, 0.5] * 5 + [0.1, 0.2]  # 27 features
        
        # Use the actual max values from config
        max_values = self.feature_processor.max_values
        self.assertEqual(len(max_values), 27)

        denormalized = self.feature_processor.denormalize_features(normalized_features)
        self.assertEqual(len(denormalized), 27)

        # Check that denormalization worked correctly
        for i, denorm_val in enumerate(denormalized):
            expected = normalized_features[i] * max_values[i]
            self.assertAlmostEqual(denorm_val, expected)


class TestDataBuilder(unittest.TestCase):
    """Test cases for the DataBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_builder = DataBuilder()
    
    def test_data_builder_initialization(self):
        """Test that data builder is properly initialized."""
        self.assertIsNotNone(self.data_builder.material_db)
        self.assertIsNotNone(self.data_builder.feature_processor)
    
    def test_build_layer_data(self):
        """Test building layer data array."""
        magic_num = 0.1
        material_array = self.data_builder.material_db.get_material_array(0)
        layer_position = 0
        
        data_array = self.data_builder.build_layer_data(magic_num, material_array, layer_position)
        
        self.assertIsInstance(data_array, list)
        self.assertEqual(len(data_array), 1)  # Single layer
        self.assertEqual(len(data_array[0]), 61)  # 61 columns
        
        # Check that basic properties are set
        self.assertEqual(data_array[0][0], 0)  # Material ID
        self.assertEqual(data_array[0][1], material_array[1])  # Density
        self.assertEqual(data_array[0][3], material_array[2])  # True radius
    
    def test_build_layer_data_with_position(self):
        """Test building layer data with non-zero position."""
        magic_num = 0.1
        material_array = self.data_builder.material_db.get_material_array(0)
        layer_position = 2
        
        data_array = self.data_builder.build_layer_data(magic_num, material_array, layer_position)
        
        # Check that radius is adjusted for position
        expected_radius = material_array[2] + magic_num * layer_position
        self.assertEqual(data_array[0][4], expected_radius)


if __name__ == '__main__':
    unittest.main() 