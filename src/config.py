"""
Configuration settings for the ShieldingOptimizer package.

This module contains all configuration constants, material properties,
and optimization parameters used throughout the application.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class MaterialProperties:
    """Properties of shielding materials."""
    id: int
    density: float
    true_radius: float
    log_xs_gam_abe6: float
    log_xs_gam_abe7: float
    log_xs_gam_pro_e2: float
    log_xs_gam_pro_e1: float
    log_xs_gam_pro_e4: float
    log_xs_gam_pro_e5: float
    log_xs_gam_pro_e6: float
    log_xs_gam_pro_e7: float
    log_xs_alpha_pro_e2: float
    log_xs_alpha_pro_e1: float
    log_xs_alpha_pro_e4: float
    log_xs_alpha_pro_e5: float
    log_xs_alpha_pro_e6: float
    log_xs_alpha_pro_e7: float
    log_xs_elastic_e2: float
    log_xs_elastic_e1: float
    log_xs_elastic_e4: float
    log_xs_elastic_e5: float
    log_xs_elastic_e6: float
    log_xs_elastic_e7: float
    log_peak_elastic: float
    log_peak_alpha: float
    log_peak_ga: float
    log_peak_ge: float
    line_rate: float
    line_cost: float
    total_cost: float
    total_mass: float


class Config:
    """Main configuration class containing all application settings."""
    
    # Material database
    MATERIALS = [
        # Lead
        MaterialProperties(
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
        ),
        # B4C
        MaterialProperties(
            id=1, density=2500, true_radius=17.967,
            log_xs_gam_abe6=4.00E-09, log_xs_gam_abe7=8.02E-07,
            log_xs_gam_pro_e2=4.01E-01, log_xs_gam_pro_e1=1.60E-01,
            log_xs_gam_pro_e4=4.02E-04, log_xs_gam_pro_e5=1.63E-04,
            log_xs_gam_pro_e6=5.80E-05, log_xs_gam_pro_e7=3.40E-05,
            log_xs_alpha_pro_e2=4.80E+03, log_xs_alpha_pro_e1=1.60E+03,
            log_xs_alpha_pro_e4=3.20E+00, log_xs_alpha_pro_e5=8.00E-01,
            log_xs_alpha_pro_e6=1.60E-01, log_xs_alpha_pro_e7=4.80E-02,
            log_xs_elastic_e2=3.00E+00, log_xs_elastic_e1=2.66E+00,
            log_xs_elastic_e4=2.74E+00, log_xs_elastic_e5=3.30E+00,
            log_xs_elastic_e6=2.14E+00, log_xs_elastic_e7=7.70E-01,
            log_peak_elastic=4.72E+01, log_peak_alpha=1.60E+05,
            log_peak_ga=4.00E-06, log_peak_ge=1.35E+01,
            line_rate=1, line_cost=1, total_cost=1, total_mass=1
        ),
        # HDPE
        MaterialProperties(
            id=2, density=998.88, true_radius=2.28,
            log_xs_gam_abe6=2.03E-09, log_xs_gam_abe7=1.30376E-06,
            log_xs_gam_pro_e2=0.012736752, log_xs_gam_pro_e1=0.005094701,
            log_xs_gam_pro_e4=1.43664E-05, log_xs_gam_pro_e5=9.98363E-06,
            log_xs_gam_pro_e6=1.76293E-05, log_xs_gam_pro_e7=2.97779E-05,
            log_xs_alpha_pro_e2=1.33E+02, log_xs_alpha_pro_e1=4.44E+01,
            log_xs_alpha_pro_e4=8.89E-02, log_xs_alpha_pro_e5=2.22E-02,
            log_xs_alpha_pro_e6=4.44E-03, log_xs_alpha_pro_e7=3.98E-02,
            log_xs_elastic_e2=1.69E+00, log_xs_elastic_e1=1.64E+00,
            log_xs_elastic_e4=1.65E+00, log_xs_elastic_e5=1.53E+00,
            log_xs_elastic_e6=9.24E-01, log_xs_elastic_e7=2.30E-01,
            log_peak_elastic=1.25E+03, log_peak_alpha=4.44E+03,
            log_peak_ga=1.96E-03, log_peak_ge=1.32E+01,
            line_rate=1, line_cost=1, total_cost=1, total_mass=1
        ),
        # BC
        MaterialProperties(
            id=3, density=2019.5, true_radius=2.165,
            log_xs_gam_abe6=4.75E-09, log_xs_gam_abe7=2.00E-06,
            log_xs_gam_pro_e2=2.53E-01, log_xs_gam_pro_e1=1.01E-01,
            log_xs_gam_pro_e4=2.55E-04, log_xs_gam_pro_e5=1.09E-04,
            log_xs_gam_pro_e6=5.50E-05, log_xs_gam_pro_e7=5.50E-05,
            log_xs_alpha_pro_e2=3.00E+03, log_xs_alpha_pro_e1=1.00E+03,
            log_xs_alpha_pro_e4=2.00000001E+00, log_xs_alpha_pro_e5=5.00E-01,
            log_xs_alpha_pro_e6=1.00E-01, log_xs_alpha_pro_e7=7.50E-02,
            log_xs_elastic_e2=3.75E+00, log_xs_elastic_e1=3.50E+00,
            log_xs_elastic_e4=3.55E+00, log_xs_elastic_e5=3.75E+00,
            log_xs_elastic_e6=2.35E+00, log_xs_elastic_e7=7.25E-01,
            log_peak_elastic=5.95E+01, log_peak_alpha=1.00E+05,
            log_peak_ga=4.00E-06, log_peak_ge=1.01E+01,
            line_rate=1, line_cost=1, total_cost=1, total_mass=1
        ),
        # Tungsten
        MaterialProperties(
            id=4, density=19300, true_radius=30.3,
            log_xs_gam_abe6=2.25E-05, log_xs_gam_abe7=2.83E-03,
            log_xs_gam_pro_e2=2.603073, log_xs_gam_pro_e1=0.8227445,
            log_xs_gam_pro_e4=1.93E-01, log_xs_gam_pro_e5=1.70E-01,
            log_xs_gam_pro_e6=8.07E-02, log_xs_gam_pro_e7=7.19E-04,
            log_xs_alpha_pro_e2=0.00000001E+00, log_xs_alpha_pro_e1=0.00000001E+00,
            log_xs_alpha_pro_e4=0.00000001E+00, log_xs_alpha_pro_e5=0.00000001E+00,
            log_xs_alpha_pro_e6=0.00000001E+00, log_xs_alpha_pro_e7=1.98E-05,
            log_xs_elastic_e2=7.38606, log_xs_elastic_e1=7.33006,
            log_xs_elastic_e4=11.3226, log_xs_elastic_e5=8.55177,
            log_xs_elastic_e6=4.17167, log_xs_elastic_e7=2.48124,
            log_peak_elastic=1.00E+04, log_peak_alpha=7.50E-03,
            log_peak_ga=2.50E-02, log_peak_ge=1.00E+03,
            line_rate=1, line_cost=1, total_cost=1, total_mass=1
        ),
        # Depleted Uranium
        MaterialProperties(
            id=5, density=19000, true_radius=117.04,
            log_xs_gam_abe6=2.96E-05, log_xs_gam_abe7=0.000182087,
            log_xs_gam_pro_e2=4.261824, log_xs_gam_pro_e1=1.3647543,
            log_xs_gam_pro_e4=1.58E+00, log_xs_gam_pro_e5=1.79E-01,
            log_xs_gam_pro_e6=1.28E-01, log_xs_gam_pro_e7=9.17E-04,
            log_xs_alpha_pro_e2=0.00000001E+00, log_xs_alpha_pro_e1=0.00000001E+00,
            log_xs_alpha_pro_e4=0.00000001E+00, log_xs_alpha_pro_e5=0.00000001E+00,
            log_xs_alpha_pro_e6=0.00000001E+00, log_xs_alpha_pro_e7=1.00E-06,
            log_xs_elastic_e2=9.33244, log_xs_elastic_e1=9.283234,
            log_xs_elastic_e4=12.2224, log_xs_elastic_e5=11.107,
            log_xs_elastic_e6=4.25407, log_xs_elastic_e7=2.72168,
            log_peak_elastic=1.00E+04, log_peak_alpha=5.00E-02,
            log_peak_ga=7.00E-03, log_peak_ge=1.00E+04,
            line_rate=1, line_cost=1, total_cost=1, total_mass=1
        )
    ]
    
    # Feature columns for ML model
    FEATURE_COLUMNS = [
        'Density', 'True Radius (m)', 'Radius (m)', 'Mass (kg)',
        'LogXSgamABe6Nave', 'LogXSgamABe7Nave', 'LogXSgamPROe-2Nave',
        'LogXSgamPROe-1Nave', 'LogXSgamPROe4Nave', 'LogXSgamPROe5Nave',
        'LogXSgamPROe7Nave', 'LogXSalphaPROe-2Nave', 'LogXSalphaPROe-1Nave',
        'LogXSalphaPROe4Nave', 'LogXSalphaPROe5Nave', 'LogXSalphaPROe6Nave',
        'LogXSalphaPROe7Nave', 'LogXSelasticE-2Nave', 'LogXSelasticE-1Nave',
        'LogXSelasticE4Nave', 'LogXSelasticE5Nave', 'LogXSelasticE6Nave',
        'LogXSelasticE7Nave', 'LogPeakElasticNave', 'LogPeakAlphaNave',
        'LogPeakGANave', 'LogPeakGENave'
    ]
    
    # Columns to drop for feature extraction
    COLUMNS_TO_DROP = [
        'MaterialIDs', 'CellCount', 'Cross-Sectional Area (m^2)',
        'LogXSgamABe6', 'LogXSgamABe7', 'LogXSgamPROe-2', 'LogXSgamPROe-1',
        'LogXSgamPROe4', 'LogXSgamPROe5', 'LogXSgamPROe6', 'LogXSgamPROe7',
        'LogXSalphaPROe-2', 'LogXSalphaPROe-1', 'LogXSalphaPROe4',
        'LogXSalphaPROe5', 'LogXSalphaPROe6', 'LogXSalphaPROe7',
        'LogXSelasticE-2', 'LogXSelasticE-1', 'LogXSelasticE4',
        'LogXSelasticE5', 'LogXSelasticE6', 'LogXSelasticE7',
        'LogPeakElastic', 'LogPeakAlpha', 'LogPeakGA', 'LogPeakGE',
        'LineRate', 'LineCost', 'TotalCost', 'TotalMass'
    ]
    
    # Optimization parameters
    MAX_LAYERS = 10
    DOSE_LIMIT = -0.90
    MASS_LIMIT = 200000.0  # kg (matching original)
    COST_LIMIT = 3000000.0  # USD (matching original)
    
    # Gekko solver options
    SOLVER_OPTIONS = [
        'minlp_gap_tol 1.0e-2',
        'minlp_maximum_iterations 10000',
        'minlp_max_iter_with_int_sol 500',
        'minlp_branch_method 1',
        'nlp_maximum_iterations 20'
    ]
    
    # Maximum values for normalization
    MAX_VALUES = [
        19300, -8.691793984, -6.096042291, -3.301029996, -4,
        -5.698970004, -5.000711535, -4.526106548, -1000,
        -1000, -1000, -1000, -5.99910194, 1.810113597,
        1.576092407, 1.496538635, 1.903089987, 1.041392685,
        -0.638980905, 4, -7.096910013, -5.397940009, 4, 4, 4, 4, 4
    ]
    
    # Material names for better readability
    MATERIAL_NAMES = {
        0: "Lead",
        1: "B4C", 
        2: "HDPE",
        3: "BC",
        4: "Tungsten",
        5: "Depleted Uranium"
    }
    
    # Initial material guesses (matching original)
    INITIAL_MATERIAL_GUESSES = [1, 2, 1, 1, 3, 1, 2, 1, 3, 1] 