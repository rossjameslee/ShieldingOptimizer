"""
Tests for the configuration module.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config, MaterialProperties


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
    
    def test_materials_exist(self):
        """Test that materials are properly defined."""
        self.assertGreater(len(self.config.MATERIALS), 0)
        self.assertEqual(len(self.config.MATERIALS), 6)
    
    def test_material_properties(self):
        """Test that material properties are valid."""
        for material in self.config.MATERIALS:
            self.assertIsInstance(material, MaterialProperties)
            self.assertGreater(material.density, 0)
            self.assertGreater(material.true_radius, 0)
    
    def test_feature_columns(self):
        """Test that feature columns are properly defined."""
        self.assertGreater(len(self.config.FEATURE_COLUMNS), 0)
        self.assertEqual(len(self.config.FEATURE_COLUMNS), 27)
    
    def test_material_names(self):
        """Test that material names are properly mapped."""
        expected_names = {
            0: "Lead",
            1: "B4C",
            2: "HDPE",
            3: "BC",
            4: "Tungsten",
            5: "Depleted Uranium"
        }
        self.assertEqual(self.config.MATERIAL_NAMES, expected_names)
    
    def test_optimization_parameters(self):
        """Test that optimization parameters are reasonable."""
        self.assertGreater(self.config.MAX_LAYERS, 0)
        self.assertGreater(self.config.MASS_LIMIT, 0)
        self.assertGreater(self.config.COST_LIMIT, 0)
        self.assertLess(self.config.DOSE_LIMIT, 0)  # Should be negative
    
    def test_solver_options(self):
        """Test that solver options are properly defined."""
        self.assertGreater(len(self.config.SOLVER_OPTIONS), 0)
        for option in self.config.SOLVER_OPTIONS:
            self.assertIsInstance(option, str)
            self.assertGreater(len(option), 0)


class TestMaterialProperties(unittest.TestCase):
    """Test cases for the MaterialProperties dataclass."""
    
    def test_material_properties_creation(self):
        """Test creating a MaterialProperties instance."""
        material = MaterialProperties(
            id=0, density=11300, true_radius=2.2,
            log_xs_gam_abe6=1.00E-07, log_xs_gam_abe7=3.00E-05,
            log_xs_gam_pro_e2=5.00E-04, log_xs_gam_pro_e1=1.00E-04,
            log_xs_gam_pro_e4=2.00E-06, log_xs_gam_pro_e5=1.00E+00,
            log_xs_gam_pro_e6=2.50E-04, log_xs_gam_pro_e7=2.00E-04,
            log_xs_alpha_pro_e2=1.00E+08, log_xs_alpha_pro_e1=1.00E+08,
            log_xs_alpha_pro_e4=1.00E+08, log_xs_alpha_pro_e5=1.00E+08,
            log_xs_alpha_pro_e6=1.00E-05, log_xs_alpha_pro_e7=1.10E+01,
            log_xs_elastic_e2=1.10E+01, log_xs_elastic_e1=1.10E+01,
            log_xs_elastic_e4=8.00E+01, log_xs_elastic_e5=1.10E+01,
            log_xs_elastic_e6=2.60E+00, log_xs_elastic_e7=45,
            log_peak_elastic=8.00E-08, log_peak_alpha=3.00E-03,
            log_peak_ga=1, log_peak_ge=1,
            line_rate=1, line_cost=1, total_cost=1, total_mass=1
        )
        
        self.assertEqual(material.id, 0)
        self.assertEqual(material.density, 11300)
        self.assertEqual(material.true_radius, 2.2)


if __name__ == '__main__':
    unittest.main() 