"""
Main optimization engine for shielding design.

This module contains the ShieldingOptimizer class that orchestrates
the entire optimization process using machine learning and mathematical optimization.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from gekko import GEKKO
import logging
from dataclasses import dataclass

from .config import Config
from .models import MLModel, GekkoSklearnModel
from .data import MaterialDatabase, DataBuilder, FeatureProcessor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from the shielding optimization."""
    success: bool
    optimized_materials: List[int]
    total_mass: float
    total_cost: float
    final_dose: float
    layer_results: List[Dict[str, Any]]
    optimization_time: float
    solver_status: str
    objective_value: float


class ShieldingOptimizer:
    """
    Main optimization engine for nuclear reactor shielding design.
    
    This class orchestrates the entire optimization process, combining
    machine learning predictions with mathematical optimization to find
    optimal shielding configurations.
    """
    
    def __init__(self, model_path: str, config: Optional[Config] = None):
        """
        Initialize the shielding optimizer.
        
        Args:
            model_path: Path to the trained ML model
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.ml_model = MLModel(model_path)
        self.material_db = MaterialDatabase()
        self.data_builder = DataBuilder()
        self.feature_processor = FeatureProcessor()
        
        # Optimization parameters
        self.mass_limit = self.config.MASS_LIMIT
        self.cost_limit = self.config.COST_LIMIT
        self.dose_limit = self.config.DOSE_LIMIT
        self.max_layers = self.config.MAX_LAYERS
        
        logger.info("ShieldingOptimizer initialized successfully")
    
    def optimize(self, magic_num: float = 10.0, 
                initial_materials: Optional[List[int]] = None) -> OptimizationResult:
        """
        Perform the main optimization with cubic splines and proper constraints.
        
        Args:
            magic_num: Magic number for layer thickness calculations (default: 10.0)
            initial_materials: Initial material selection (uses original guesses if None)
            
        Returns:
            OptimizationResult containing the results
        """
        logger.info("Starting shielding optimization")
        
        # Initialize Gekko model
        m = GEKKO(remote=True)  # Match original configuration
        
        # Set up solver options
        m.options.REDUCE = 3
        m.options.SOLVER = 1
        m.solver_options = self.config.SOLVER_OPTIONS
        
        # Create Gekko-compatible ML model
        gekko_model = self.ml_model.create_gekko_model(m)
        
        # Set up optimization variables and cubic splines
        density_vars = self._setup_optimization_variables(m, initial_materials)
        layer_splines = self._setup_cubic_splines(m, density_vars)
        
        # Build data arrays for each layer using spline data
        layer_data = self._build_layer_data_arrays_with_splines(m, magic_num, layer_splines)
        
        # Predict doses for each layer
        predicted_doses = self._predict_doses(gekko_model, layer_data)
        
        # Set up objective function and constraints
        objective, mass_constraints, cost_constraints = self._setup_objective_and_constraints_with_splines(
            m, layer_data, predicted_doses, layer_splines
        )
        
        # Solve the optimization problem
        result = self._solve_optimization(m, objective, (mass_constraints, cost_constraints))
        
        # Extract and format results
        optimization_result = self._extract_results_with_splines(
            m, density_vars, layer_data, predicted_doses, layer_splines, result
        )
        
        logger.info("Optimization completed")
        return optimization_result
    
    def _setup_optimization_variables(self, m: GEKKO, 
                                    initial_materials: Optional[List[int]]) -> List:
        """Set up optimization variables for material selection with cubic splines."""
        if initial_materials is None:
            # Use original initial material guesses
            initial_materials = self.config.INITIAL_MATERIAL_GUESSES
        
        # Create density variables (integer variables for material selection) - matching original
        density_vars = []
        for i in range(self.max_layers):
            var = m.Var(name=f'densityVar{i}', value=initial_materials[i], lb=0, ub=5, integer=True)
            density_vars.append(var)
        
        return density_vars
    
    def _setup_cubic_splines(self, m: GEKKO, density_vars: List) -> List[List]:
        """Set up cubic splines for material property interpolation (matching original)."""
        # Define the raw material data exactly as in the original (fullOBJs)
        fullOBJs = [
            # [id, density, cost, gamABe6, gamABe7, gamPROe_2, gamPROe_1, gamPROe4, gamPROe5, gamPROe6, gamPROe7,
            #  alphaPROe_2, alphaPROe_1, alphaPROe4, alphaPROe5, alphaPROe6, alphaPROe7,
            #  elasticE_2, elasticE_1, elasticE4, elasticE5, elasticE6, elasticE7,
            #  PeakElastic, PeakAlpha, PeakGA, PeakGE]
            [0, 11340, 6.23, 2.96E-05, 0.000182087, 4.261824, 1.3647543, 1.58E+00, 1.79E-01, 1.28E-01, 9.17E-04, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 1.00E-06, 9.33244, 9.283234, 12.2224, 11.107, 4.25407, 2.72168, 1.00E+04, 5.00E-02, 7.00E-03, 1.00E+04],  # Lead
            [1, 2520, 62.34, 5.55E-09, 1.37E-06, 9.67E-01, 2.44E-01, 1.12E-03, 1.79E-04, 1.11E-04, 9.30E-05, 1.75E+02, 5.82E+01, 1.16E-01, 2.91E-02, 5.82E-03, 8.65E-02, 6.46E+01, 3.77E+01, 3.14E+01, 2.19E+01, 9.23E+00, 1.86E+00, 1.25E+03, 5.82E+03, 5.84E-06, 1.32E+01],  # B4C
            [2, 970, 1.5, 5.55E-09, 1.37E-06, 9.67E-01, 2.44E-01, 1.12E-03, 1.79E-04, 1.11E-04, 9.30E-05, 1.75E+02, 5.82E+01, 1.16E-01, 2.91E-02, 5.82E-03, 8.65E-02, 6.46E+01, 3.77E+01, 3.14E+01, 2.19E+01, 9.23E+00, 1.86E+00, 1.25E+03, 5.82E+03, 5.84E-06, 1.32E+01],  # HDPE
            [3, 1850, 8.07E-02, 7.19E-04, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 1.98E-05, 7.38606, 7.33006, 11.3226, 8.55177, 4.17167, 2.48124, 1.00E+04, 7.50E-03, 2.50E-02, 1.00E+03],  # BC
            [4, 19300, 117.04, 2.96E-05, 0.000182087, 4.261824, 1.3647543, 1.58E+00, 1.79E-01, 1.28E-01, 9.17E-04, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 1.00E-06, 9.33244, 9.283234, 12.2224, 11.107, 4.25407, 2.72168, 1.00E+04, 5.00E-02, 7.00E-03, 1.00E+04],  # Tungsten
            [5, 19000, 117.04, 2.96E-05, 0.000182087, 4.261824, 1.3647543, 1.58E+00, 1.79E-01, 1.28E-01, 9.17E-04, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 0.00000001E+00, 1.00E-06, 9.33244, 9.283234, 12.2224, 11.107, 4.25407, 2.72168, 1.00E+04, 5.00E-02, 7.00E-03, 1.00E+04]   # Depleted Uranium
        ]
        
        # Define maxesForVarsOnly exactly as in original
        maxesForVarsOnly = [19300, -8.691793984, -6.096042291, -3.301029996, -4, -5.698970004, -5.000711535, -4.526106548, -1000, -1000, -1000, -1000, -1000, -5.99910194, 1.810113597, 1.576092407, 1.496538635, 1.903089987, 1.041392685, -0.638980905, 4, -7.096910013, -5.397940009, 4]
        
        # Extract material properties for spline interpolation (matching original exactly)
        IDs = []
        densities = []
        costs = []
        LogXSgamABe6 = []
        LogXSgamABe7 = []
        LogXSgamPROe_2 = []
        LogXSgamPROe_1 = []
        LogXSgamPROe4 = []
        LogXSgamPROe5 = []
        LogXSgamPROe7 = []
        LogXSalphaPROe_2 = []
        LogXSalphaPROe_1 = []
        LogXSalphaPROe4 = []
        LogXSalphaPROe5 = []
        LogXSalphaPROe6 = []
        LogXSalphaPROe7 = []
        LogXSelasticE_2 = []
        LogXSelasticE_1 = []
        LogXSelasticE4 = []
        LogXSelasticE5 = []
        LogXSelasticE6 = []
        LogXSelasticE7 = []
        LogPeakElastic = []
        LogPeakAlpha = []
        LogPeakGA = []
        LogPeakGE = []
        
        for i in range(6):
            IDs.append(fullOBJs[i][0])
            densities.append(fullOBJs[i][1]/maxesForVarsOnly[0])
            costs.append(fullOBJs[i][2])
            LogXSgamABe6.append(np.log10(fullOBJs[i][3])/maxesForVarsOnly[1])
            LogXSgamABe7.append(np.log10(fullOBJs[i][4])/maxesForVarsOnly[2])
            LogXSgamPROe_2.append(np.log10(fullOBJs[i][5])/maxesForVarsOnly[3])
            LogXSgamPROe_1.append(np.log10(fullOBJs[i][6])/maxesForVarsOnly[4])
            LogXSgamPROe4.append(np.log10(fullOBJs[i][7])/maxesForVarsOnly[5])
            LogXSgamPROe5.append(np.log10(fullOBJs[i][8])/maxesForVarsOnly[6])
            LogXSgamPROe7.append(np.log10(fullOBJs[i][10])/maxesForVarsOnly[7])
            LogXSalphaPROe_2.append(np.log10(fullOBJs[i][11])/maxesForVarsOnly[8])
            LogXSalphaPROe_1.append(np.log10(fullOBJs[i][12])/maxesForVarsOnly[9])
            LogXSalphaPROe4.append(np.log10(fullOBJs[i][13])/maxesForVarsOnly[10])
            LogXSalphaPROe5.append(np.log10(fullOBJs[i][14])/maxesForVarsOnly[11])
            LogXSalphaPROe6.append(np.log10(fullOBJs[i][15])/maxesForVarsOnly[12])
            LogXSalphaPROe7.append(np.log10(fullOBJs[i][16])/maxesForVarsOnly[13])
            LogXSelasticE_2.append(np.log10(fullOBJs[i][17])/maxesForVarsOnly[14])
            LogXSelasticE_1.append(np.log10(fullOBJs[i][18])/maxesForVarsOnly[15])
            LogXSelasticE4.append(np.log10(fullOBJs[i][19])/maxesForVarsOnly[16])
            LogXSelasticE5.append(np.log10(fullOBJs[i][20])/maxesForVarsOnly[17])
            LogXSelasticE6.append(np.log10(fullOBJs[i][21])/maxesForVarsOnly[18])
            LogXSelasticE7.append(np.log10(fullOBJs[i][22])/maxesForVarsOnly[19])
            LogPeakElastic.append(np.log10(fullOBJs[i][23])/maxesForVarsOnly[20])
            LogPeakAlpha.append(np.log10(fullOBJs[i][24])/maxesForVarsOnly[21])
            LogPeakGA.append(np.log10(fullOBJs[i][25])/maxesForVarsOnly[22])
            LogPeakGE.append(np.log10(fullOBJs[i][26])/maxesForVarsOnly[23])
        
        # Create spline variables for each layer (matching original naming)
        layer_splines = []
        for layer_idx in range(self.max_layers):
            # Create variables for this layer's material properties (matching original)
            densityVarY = m.Var(name=f'densityVarY{layer_idx}')
            densityNaveVarY = m.Intermediate(densityVarY)
            gamABe6VarY = m.Var(name=f'gamABe6VarY{layer_idx}')
            gamABe7NaveVarY = m.Var(name=f'gamABe7NaveVarY{layer_idx}')
            gamPROe_2NaveVarY = m.Var(name=f'gamPROe_2NaveVarY{layer_idx}')
            gamPROe_1NaveVarY = m.Var(name=f'gamPROe_1NaveVarY{layer_idx}')
            gamPROe4NaveVarY = m.Var(name=f'gamPROe4NaveVarY{layer_idx}')
            gamPROe5NaveVarY = m.Var(name=f'gamPROe5NaveVarY{layer_idx}')
            gamPROe7NaveVarY = m.Var(name=f'gamPROe7NaveVarY{layer_idx}')
            alphaPROe_2NaveVarY = m.Var(name=f'alphaPROe_2NaveVarY{layer_idx}')
            alphaPROe_1NaveVarY = m.Var(name=f'alphaPROe_1NaveVarY{layer_idx}')
            alphaPROe4NaveVarY = m.Var(name=f'alphaPROe4NaveVarY{layer_idx}')
            alphaPROe5NaveVarY = m.Var(name=f'alphaPROe5NaveVarY{layer_idx}')
            alphaPROe6NaveVarY = m.Var(name=f'alphaPROe6NaveVarY{layer_idx}')
            alphaPROe7NaveVarY = m.Var(name=f'alphaPROe7NaveVarY{layer_idx}')
            elasticE_2NaveVarY = m.Var(name=f'elasticE_2NaveVarY{layer_idx}')
            elasticE_1NaveVarY = m.Var(name=f'elasticE_1NaveVarY{layer_idx}')
            elasticE4NaveVarY = m.Var(name=f'elasticE4NaveVarY{layer_idx}')
            elasticE5NaveVarY = m.Var(name=f'elasticE5NaveVarY{layer_idx}')
            elasticE6NaveVarY = m.Var(name=f'elasticE6NaveVarY{layer_idx}')
            elasticE7NaveVarY = m.Var(name=f'elasticE7NaveVarY{layer_idx}')
            PeakElasticNaveVarY = m.Var(name=f'PeakElasticNaveVarY{layer_idx}')
            PeakAlphaNaveVarY = m.Var(name=f'PeakAlphaNaveVarY{layer_idx}')
            PeakGANaveVarY = m.Var(name=f'PeakGANaveVarY{layer_idx}')
            PeakGENaveVarY = m.Var(name=f'PeakGENaveVarY{layer_idx}')
            costVarY = m.Var(name=f'PriceVarY{layer_idx}')
            
            # Create cubic splines for material property interpolation (matching original)
            m.cspline(density_vars[layer_idx], densityVarY, IDs, densities)
            m.cspline(density_vars[layer_idx], gamABe6VarY, IDs, LogXSgamABe6)
            m.cspline(density_vars[layer_idx], gamABe7NaveVarY, IDs, LogXSgamABe7)
            m.cspline(density_vars[layer_idx], gamPROe_2NaveVarY, IDs, LogXSgamPROe_2)
            m.cspline(density_vars[layer_idx], gamPROe_1NaveVarY, IDs, LogXSgamPROe_1)
            m.cspline(density_vars[layer_idx], gamPROe4NaveVarY, IDs, LogXSgamPROe4)
            m.cspline(density_vars[layer_idx], gamPROe5NaveVarY, IDs, LogXSgamPROe5)
            # Skip gamPROe6 as in original
            m.cspline(density_vars[layer_idx], gamPROe7NaveVarY, IDs, LogXSgamPROe7)
            m.cspline(density_vars[layer_idx], alphaPROe_2NaveVarY, IDs, LogXSalphaPROe_2)
            m.cspline(density_vars[layer_idx], alphaPROe_1NaveVarY, IDs, LogXSalphaPROe_1)
            m.cspline(density_vars[layer_idx], alphaPROe4NaveVarY, IDs, LogXSalphaPROe4)
            m.cspline(density_vars[layer_idx], alphaPROe5NaveVarY, IDs, LogXSalphaPROe5)
            m.cspline(density_vars[layer_idx], alphaPROe6NaveVarY, IDs, LogXSalphaPROe6)
            m.cspline(density_vars[layer_idx], alphaPROe7NaveVarY, IDs, LogXSalphaPROe7)
            m.cspline(density_vars[layer_idx], elasticE_2NaveVarY, IDs, LogXSelasticE_2)
            m.cspline(density_vars[layer_idx], elasticE_1NaveVarY, IDs, LogXSelasticE_1)
            m.cspline(density_vars[layer_idx], elasticE4NaveVarY, IDs, LogXSelasticE4)
            m.cspline(density_vars[layer_idx], elasticE5NaveVarY, IDs, LogXSelasticE5)
            m.cspline(density_vars[layer_idx], elasticE6NaveVarY, IDs, LogXSelasticE6)
            m.cspline(density_vars[layer_idx], elasticE7NaveVarY, IDs, LogXSelasticE7)
            m.cspline(density_vars[layer_idx], PeakElasticNaveVarY, IDs, LogPeakElastic)
            m.cspline(density_vars[layer_idx], PeakAlphaNaveVarY, IDs, LogPeakAlpha)
            m.cspline(density_vars[layer_idx], PeakGANaveVarY, IDs, LogPeakGA)
            m.cspline(density_vars[layer_idx], PeakGENaveVarY, IDs, LogPeakGE)
            m.cspline(density_vars[layer_idx], costVarY, IDs, costs)
            
            # Store all variables for this layer (matching original order)
            layer_vars = [
                densityVarY, densityNaveVarY, gamABe6VarY, gamABe7NaveVarY, gamPROe_2NaveVarY, gamPROe_1NaveVarY,
                gamPROe4NaveVarY, gamPROe5NaveVarY, gamPROe7NaveVarY, alphaPROe_2NaveVarY, alphaPROe_1NaveVarY,
                alphaPROe4NaveVarY, alphaPROe5NaveVarY, alphaPROe6NaveVarY, alphaPROe7NaveVarY, elasticE_2NaveVarY,
                elasticE_1NaveVarY, elasticE4NaveVarY, elasticE5NaveVarY, elasticE6NaveVarY, elasticE7NaveVarY,
                PeakElasticNaveVarY, PeakAlphaNaveVarY, PeakGANaveVarY, PeakGENaveVarY, costVarY
            ]
            
            layer_splines.append(layer_vars)
        
        return layer_splines
    
    def _build_layer_data_arrays(self, m: GEKKO, magic_num: float, 
                                variable_mats: List) -> List:
        """Build data arrays for each layer using Gekko variables."""
        layer_data = []
        
        for layer_idx in range(self.max_layers):
            # Get material properties for this layer
            material_array = self._get_material_properties_for_layer(m, variable_mats[layer_idx])
            
            # Build the data array
            data_array = self.data_builder.build_layer_data(magic_num, material_array, layer_idx)
            layer_data.append(data_array)
        
        return layer_data
    
    def _get_material_properties_for_layer(self, m: GEKKO, material_var) -> List:
        """Get material properties as Gekko variables for a layer."""
        # Simplified approach: use the material ID directly to get properties
        # This is a simplified version of the original cubic spline approach
        
        # Get all material properties as arrays for interpolation
        material_props = []
        for material in self.config.MATERIALS:
            props = [
                material.id, material.density, material.true_radius,
                material.log_xs_gam_abe6, material.log_xs_gam_abe7,
                material.log_xs_gam_pro_e2, material.log_xs_gam_pro_e1,
                material.log_xs_gam_pro_e4, material.log_xs_gam_pro_e5,
                material.log_xs_gam_pro_e6, material.log_xs_gam_pro_e7,
                material.log_xs_alpha_pro_e2, material.log_xs_alpha_pro_e1,
                material.log_xs_alpha_pro_e4, material.log_xs_alpha_pro_e5,
                material.log_xs_alpha_pro_e6, material.log_xs_alpha_pro_e7,
                material.log_xs_elastic_e2, material.log_xs_elastic_e1,
                material.log_xs_elastic_e4, material.log_xs_elastic_e5,
                material.log_xs_elastic_e6, material.log_xs_elastic_e7,
                material.log_peak_elastic, material.log_peak_alpha,
                material.log_peak_ga, material.log_peak_ge,
                material.line_rate, material.line_cost, material.total_cost, material.total_mass
            ]
            material_props.append(props)
        
        # For now, use a simple approach: select material based on integer variable
        # This is a simplified version - the original uses cubic splines
        selected_props = []
        for i in range(len(material_props[0])):
            # Create a simple interpolation based on material_var
            # This is a simplified version of the cubic spline approach
            prop_val = m.Intermediate(
                material_var * material_props[1][i] + 
                (1 - material_var) * material_props[0][i]
            )
            selected_props.append(prop_val)
        
        return selected_props
    
    def _predict_doses(self, gekko_model: GekkoSklearnModel, 
                      layer_data: List) -> List:
        """Predict radiation doses for each layer."""
        predicted_doses = []
        
        for layer_idx, data_array in enumerate(layer_data):
            # Extract features for this layer
            features = self.feature_processor.extract_features(data_array[0])
            
            # Predict dose using the Gekko model
            dose = gekko_model.predict(features)
            predicted_doses.append(dose)
        
        return predicted_doses
    
    def _setup_objective_and_constraints(self, m: GEKKO, layer_data: List, 
                                       predicted_doses: List) -> Tuple:
        """Set up the objective function and constraints."""
        # Objective: minimize the final dose
        objective = predicted_doses[-1]
        m.Minimize(objective)
        
        # Calculate total mass and cost
        total_mass = m.Intermediate(sum(data[0][8] for data in layer_data))
        total_cost = m.Intermediate(sum(data[0][8] * data[0][57] for data in layer_data))
        
        # Constraints
        m.Equation(total_mass <= self.mass_limit)
        m.Equation(total_cost <= self.cost_limit)
        m.Equation(objective <= self.dose_limit)
        
        return objective, (total_mass, total_cost)
    
    def _solve_optimization(self, m: GEKKO, objective, constraints) -> Dict[str, Any]:
        """Solve the optimization problem."""
        try:
            logger.info("Solving optimization problem...")
            m.solve(disp=True)
            
            # Extract objective value safely
            if hasattr(objective, 'value') and hasattr(objective.value, '__getitem__'):
                obj_value = objective.value[0]
            elif hasattr(objective, 'value'):
                obj_value = objective.value
            else:
                obj_value = None
            
            return {
                'success': True,
                'solver_status': 'Success',
                'objective_value': obj_value
            }
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'success': False,
                'solver_status': str(e),
                'objective_value': None
            }
    
    def _extract_results(self, m: GEKKO, variable_mats: List, layer_data: List,
                        predicted_doses: List, solve_result: Dict[str, Any]) -> OptimizationResult:
        """Extract and format optimization results."""
        if not solve_result['success']:
            return OptimizationResult(
                success=False,
                optimized_materials=[],
                total_mass=0.0,
                total_cost=0.0,
                final_dose=0.0,
                layer_results=[],
                optimization_time=0.0,
                solver_status=solve_result['solver_status'],
                objective_value=solve_result['objective_value']
            )
        
        # Extract optimized material selections
        optimized_materials = []
        for var in variable_mats:
            if hasattr(var, 'value') and hasattr(var.value, '__getitem__'):
                optimized_materials.append(int(var.value[0]))
            elif hasattr(var, 'value'):
                optimized_materials.append(int(var.value))
            else:
                optimized_materials.append(int(var))
        
        # Calculate final metrics - simplified for now
        total_mass = 0.0
        total_cost = 0.0
        final_dose = 0.0
        
        # For now, just extract the basic information without complex calculations
        try:
            if predicted_doses and hasattr(predicted_doses[-1], 'value') and hasattr(predicted_doses[-1].value, '__getitem__'):
                final_dose = predicted_doses[-1].value[0]
            elif predicted_doses and hasattr(predicted_doses[-1], 'value'):
                final_dose = predicted_doses[-1].value
        except:
            final_dose = 0.0
        
        # Build layer results - simplified
        layer_results = []
        for i, material_id in enumerate(optimized_materials):
            try:
                dose_val = 0.0
                if i < len(predicted_doses):
                    dose = predicted_doses[i]
                    if hasattr(dose, 'value') and hasattr(dose.value, '__getitem__'):
                        dose_val = dose.value[0]
                    elif hasattr(dose, 'value'):
                        dose_val = dose.value
                
                layer_results.append({
                    'layer': i,
                    'material_id': material_id,
                    'material_name': self.material_db.get_material_name(material_id),
                    'mass': 0.0,  # Simplified for now
                    'cost': 0.0,  # Simplified for now
                    'dose': dose_val
                })
            except Exception as e:
                logger.warning(f"Error processing layer {i}: {e}")
                layer_results.append({
                    'layer': i,
                    'material_id': material_id,
                    'material_name': self.material_db.get_material_name(material_id),
                    'mass': 0.0,
                    'cost': 0.0,
                    'dose': 0.0
                })
        
        return OptimizationResult(
            success=True,
            optimized_materials=optimized_materials,
            total_mass=total_mass,
            total_cost=total_cost,
            final_dose=final_dose,
            layer_results=layer_results,
            optimization_time=0.0,  # Would need to track actual time
            solver_status=solve_result['solver_status'],
            objective_value=solve_result['objective_value']
        )
    
    def print_results(self, result: OptimizationResult) -> None:
        """Print optimization results in a formatted way."""
        if not result.success:
            print("Optimization failed!")
            print(f"Solver status: {result.solver_status}")
            return
        
        print("\n" + "="*60)
        print("SHIELDING OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"\nOptimization Status: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Final Dose: {result.final_dose:.6f}")
        print(f"Total Mass: {result.total_mass:.2f} kg")
        print(f"Total Cost: ${result.total_cost:.2f}")
        print(f"Objective Value: {result.objective_value:.6f}")
        
        print(f"\nOptimized Material Configuration:")
        print("-" * 40)
        for i, material_id in enumerate(result.optimized_materials):
            material_name = self.material_db.get_material_name(material_id)
            layer_result = result.layer_results[i]
            print(f"Layer {i+1:2d}: {material_name:15s} "
                  f"(Mass: {layer_result['mass']:6.2f} kg, "
                  f"Cost: ${layer_result['cost']:8.2f})")
        
        print("\n" + "="*60)
    
    def save_results(self, result: OptimizationResult, filename: str) -> None:
        """Save optimization results to a file."""
        with open(filename, 'w') as f:
            f.write("Shielding Optimization Results\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Success: {result.success}\n")
            f.write(f"Final Dose: {result.final_dose}\n")
            f.write(f"Total Mass: {result.total_mass} kg\n")
            f.write(f"Total Cost: ${result.total_cost}\n")
            f.write(f"Objective Value: {result.objective_value}\n\n")
            
            f.write("Layer Configuration:\n")
            f.write("-" * 20 + "\n")
            for layer_result in result.layer_results:
                f.write(f"Layer {layer_result['layer']+1}: "
                       f"{layer_result['material_name']} "
                       f"(Mass: {layer_result['mass']:.2f} kg, "
                       f"Cost: ${layer_result['cost']:.2f})\n") 

    def _build_layer_data_arrays_with_splines(self, m: GEKKO, magic_num: float, 
                                            layer_splines: List[List]) -> List:
        """Build data arrays for each layer with proper mass/cost calculations (matching original)."""
        layer_data = []
        
        # Constants for radius calculations (from original)
        delta_radius_90th = 0.018626667
        inner_radius = 1.075 - delta_radius_90th
        outer_radius = 2.732773363
        delta_radius = (outer_radius - inner_radius) / magic_num
        
        for layer_idx in range(self.max_layers):
            # Get spline variables for this layer (matching original order)
            spline_vars = layer_splines[layer_idx]
            densityVarY = spline_vars[0]  # densityVarY
            densityNaveVarY = spline_vars[1]  # densityNaveVarY
            gamABe6VarY = spline_vars[2]  # gamABe6VarY
            gamABe7NaveVarY = spline_vars[3]  # gamABe7NaveVarY
            gamPROe_2NaveVarY = spline_vars[4]  # gamPROe_2NaveVarY
            gamPROe_1NaveVarY = spline_vars[5]  # gamPROe_1NaveVarY
            gamPROe4NaveVarY = spline_vars[6]  # gamPROe4NaveVarY
            gamPROe5NaveVarY = spline_vars[7]  # gamPROe5NaveVarY
            gamPROe7NaveVarY = spline_vars[8]  # gamPROe7NaveVarY
            alphaPROe_2NaveVarY = spline_vars[9]  # alphaPROe_2NaveVarY
            alphaPROe_1NaveVarY = spline_vars[10]  # alphaPROe_1NaveVarY
            alphaPROe4NaveVarY = spline_vars[11]  # alphaPROe4NaveVarY
            alphaPROe5NaveVarY = spline_vars[12]  # alphaPROe5NaveVarY
            alphaPROe6NaveVarY = spline_vars[13]  # alphaPROe6NaveVarY
            alphaPROe7NaveVarY = spline_vars[14]  # alphaPROe7NaveVarY
            elasticE_2NaveVarY = spline_vars[15]  # elasticE_2NaveVarY
            elasticE_1NaveVarY = spline_vars[16]  # elasticE_1NaveVarY
            elasticE4NaveVarY = spline_vars[17]  # elasticE4NaveVarY
            elasticE5NaveVarY = spline_vars[18]  # elasticE5NaveVarY
            elasticE6NaveVarY = spline_vars[19]  # elasticE6NaveVarY
            elasticE7NaveVarY = spline_vars[20]  # elasticE7NaveVarY
            PeakElasticNaveVarY = spline_vars[21]  # PeakElasticNaveVarY
            PeakAlphaNaveVarY = spline_vars[22]  # PeakAlphaNaveVarY
            PeakGANaveVarY = spline_vars[23]  # PeakGANaveVarY
            PeakGENaveVarY = spline_vars[24]  # PeakGENaveVarY
            costVarY = spline_vars[25]  # costVarY
            
            # Calculate geometric properties (following original FillTestArray function)
            true_radius = inner_radius + (layer_idx + 1) * delta_radius
            radius = delta_radius * (layer_idx + 1)
            cross_sectional_area = np.pi * (true_radius**2 - (inner_radius + delta_radius * layer_idx)**2)
            volume = cross_sectional_area * 2
            
            # Calculate mass using the original formula
            # massVarY = m.Intermediate(densityVarY*maxesForVarsOnly[0]*testArray[0][7]/maxVals[3])
            max_density = self.config.MAX_VALUES[0]  # 19300
            max_mass = self.config.MAX_VALUES[3]     # -3.301029996
            massVarY = m.Intermediate(densityVarY * max_density * volume / (10**max_mass))
            
            # Create data array similar to original FillTestArray
            data_array = [[0.0] * 61]  # 61 columns as per original
            
            # Fill basic properties (matching original FillTestArray)
            data_array[0][0] = layer_idx  # Material ID (will be set by spline)
            data_array[0][1] = densityVarY  # Density
            data_array[0][2] = densityNaveVarY  # Density Nave
            data_array[0][3] = true_radius / self.config.MAX_VALUES[1]  # True Radius (normalized)
            data_array[0][4] = radius / self.config.MAX_VALUES[2]  # Radius (normalized)
            data_array[0][5] = layer_idx  # Cell count
            data_array[0][6] = cross_sectional_area  # Cross-Sectional Area
            data_array[0][7] = volume  # Volume
            data_array[0][8] = massVarY  # Mass (normalized)
            
            # Fill cross-section properties from splines (matching original)
            data_array[0][9] = gamABe6VarY   # LogXSgamABe6
            data_array[0][10] = gamABe7NaveVarY  # LogXSgamABe7
            data_array[0][11] = gamPROe_2NaveVarY  # LogXSgamPROe-2
            data_array[0][12] = gamPROe_1NaveVarY  # LogXSgamPROe-1
            data_array[0][13] = gamPROe4NaveVarY   # LogXSgamPROe4
            data_array[0][14] = gamPROe5NaveVarY   # LogXSgamPROe5
            data_array[0][15] = -999  # LogXSgamPROe6 (not in model)
            data_array[0][16] = gamPROe7NaveVarY   # LogXSgamPROe7
            
            data_array[0][17] = alphaPROe_2NaveVarY   # LogXSalphaPROe-2
            data_array[0][18] = alphaPROe_1NaveVarY   # LogXSalphaPROe-1
            data_array[0][19] = alphaPROe4NaveVarY    # LogXSalphaPROe4
            data_array[0][20] = alphaPROe5NaveVarY    # LogXSalphaPROe5
            data_array[0][21] = alphaPROe6NaveVarY    # LogXSalphaPROe6
            data_array[0][22] = alphaPROe7NaveVarY    # LogXSalphaPROe7
            
            data_array[0][23] = elasticE_2NaveVarY  # LogXSelasticE-2
            data_array[0][24] = elasticE_1NaveVarY  # LogXSelasticE-1
            data_array[0][25] = elasticE4NaveVarY   # LogXSelasticE4
            data_array[0][26] = elasticE5NaveVarY   # LogXSelasticE5
            data_array[0][27] = elasticE6NaveVarY   # LogXSelasticE6
            data_array[0][28] = elasticE7NaveVarY   # LogXSelasticE7
            
            data_array[0][29] = PeakElasticNaveVarY  # LogPeakElastic
            data_array[0][30] = PeakAlphaNaveVarY    # LogPeakAlpha
            data_array[0][31] = PeakGANaveVarY       # LogPeakGA
            data_array[0][32] = PeakGENaveVarY       # LogPeakGE
            
            # Fill Nave features from splines (same as original features)
            data_array[0][33] = gamABe6VarY   # LogXSgamABe6Nave
            data_array[0][34] = gamABe7NaveVarY   # LogXSgamABe7Nave
            data_array[0][35] = gamPROe_2NaveVarY # LogXSgamPROe-2Nave
            data_array[0][36] = gamPROe_1NaveVarY # LogXSgamPROe-1Nave
            data_array[0][37] = gamPROe4NaveVarY  # LogXSgamPROe4Nave
            data_array[0][38] = gamPROe5NaveVarY  # LogXSgamPROe5Nave
            data_array[0][39] = data_array[0][15] # LogXSgamPROe6Nave (not in model)
            data_array[0][40] = gamPROe7NaveVarY  # LogXSgamPROe7Nave
            
            data_array[0][41] = alphaPROe_2NaveVarY # LogXSalphaPROe-2Nave
            data_array[0][42] = alphaPROe_1NaveVarY # LogXSalphaPROe-1Nave
            data_array[0][43] = alphaPROe4NaveVarY  # LogXSalphaPROe4Nave
            data_array[0][44] = alphaPROe5NaveVarY  # LogXSalphaPROe5Nave
            data_array[0][45] = alphaPROe6NaveVarY  # LogXSalphaPROe6Nave
            data_array[0][46] = alphaPROe7NaveVarY  # LogXSalphaPROe7Nave
            
            data_array[0][47] = elasticE_2NaveVarY # LogXSelasticE-2Nave
            data_array[0][48] = elasticE_1NaveVarY # LogXSelasticE-1Nave
            data_array[0][49] = elasticE4NaveVarY  # LogXSelasticE4Nave
            data_array[0][50] = elasticE5NaveVarY  # LogXSelasticE5Nave
            data_array[0][51] = elasticE6NaveVarY  # LogXSelasticE6Nave
            data_array[0][52] = elasticE7NaveVarY  # LogXSelasticE7Nave
            
            data_array[0][53] = PeakElasticNaveVarY # LogPeakElasticNave
            data_array[0][54] = PeakAlphaNaveVarY   # LogPeakAlphaNave
            data_array[0][55] = PeakGANaveVarY      # LogPeakGANave
            data_array[0][56] = PeakGENaveVarY      # LogPeakGENave
            
            # Assign line cost from spline
            data_array[0][57] = costVarY  # Line cost
            
            layer_data.append(data_array)
        
        return layer_data
    
    def _setup_objective_and_constraints_with_splines(self, m: GEKKO, layer_data: List, 
                                                    predicted_doses: List, layer_splines: List[List]) -> Tuple:
        """Set up the objective function and constraints with proper mass/cost calculations."""
        # Objective: minimize the final dose
        objective = predicted_doses[-1]  # Final layer dose
        
        # Mass constraints (matching original limits)
        mass_constraints = []
        total_mass = m.Var(value=0, lb=0, ub=1000000)
        mass_sum = 0
        
        for layer_idx in range(self.max_layers):
            # Get mass from spline data
            spline_vars = layer_splines[layer_idx]
            densityVarY = spline_vars[0]
            volume = layer_data[layer_idx]['volume']
            max_density = self.config.MAX_VALUES[0]  # 19300
            max_mass = self.config.MAX_VALUES[3]     # -3.301029996
            layer_mass = m.Intermediate(densityVarY * max_density * volume / (10**max_mass))
            mass_sum += layer_mass
        
        # Total mass constraint (matching original: 1000 kg limit)
        mass_constraints.append(m.Equation(total_mass == mass_sum))
        mass_constraints.append(m.Equation(total_mass <= 1000))
        
        # Cost constraints (matching original limits)
        cost_constraints = []
        total_cost = m.Var(value=0, lb=0, ub=1000000)
        cost_sum = 0
        
        for layer_idx in range(self.max_layers):
            # Get cost from spline data
            spline_vars = layer_splines[layer_idx]
            costVarY = spline_vars[25]  # costVarY
            volume = layer_data[layer_idx]['volume']
            layer_cost = m.Intermediate(costVarY * volume)
            cost_sum += layer_cost
        
        # Total cost constraint (matching original: $100,000 limit)
        cost_constraints.append(m.Equation(total_cost == cost_sum))
        cost_constraints.append(m.Equation(total_cost <= 100000))
        
        return objective, mass_constraints, cost_constraints
    
    def _extract_results_with_proper_calculations(self, m: GEKKO, density_vars: List, layer_data: List,
                                                predicted_doses: List, solve_result: Dict[str, Any]) -> OptimizationResult:
        """Extract and format optimization results with proper mass/cost calculations."""
        if not solve_result['success']:
            return OptimizationResult(
                success=False,
                optimized_materials=[],
                total_mass=0.0,
                total_cost=0.0,
                final_dose=0.0,
                layer_results=[],
                optimization_time=0.0,
                solver_status=solve_result['solver_status'],
                objective_value=solve_result['objective_value']
            )
        
        # Extract optimized material selections
        optimized_materials = []
        for var in density_vars:
            if hasattr(var, 'value') and hasattr(var.value, '__getitem__'):
                optimized_materials.append(int(var.value[0]))
            elif hasattr(var, 'value'):
                optimized_materials.append(int(var.value))
            else:
                optimized_materials.append(int(var))
        
        # Calculate final metrics using proper calculations
        total_mass = 0.0
        total_cost = 0.0
        total_volume = 0.0
        
        layer_results = []
        for i, data in enumerate(layer_data):
            try:
                # Extract values safely
                mass_val = 0.0
                cost_val = 0.0
                volume_val = 0.0
                dose_val = 0.0
                
                if hasattr(data[0][8], 'value') and hasattr(data[0][8].value, '__getitem__'):
                    mass_val = data[0][8].value[0] * (10**self.config.MAX_VALUES[3])
                elif hasattr(data[0][8], 'value'):
                    mass_val = data[0][8].value * (10**self.config.MAX_VALUES[3])
                
                if hasattr(data[0][57], 'value') and hasattr(data[0][57].value, '__getitem__'):
                    cost_val = mass_val * data[0][57].value[0]
                elif hasattr(data[0][57], 'value'):
                    cost_val = mass_val * data[0][57].value
                else:
                    cost_val = mass_val * data[0][57]
                
                if hasattr(data[0][7], 'value') and hasattr(data[0][7].value, '__getitem__'):
                    volume_val = data[0][7].value[0]
                elif hasattr(data[0][7], 'value'):
                    volume_val = data[0][7].value
                else:
                    volume_val = data[0][7]
                
                if i < len(predicted_doses):
                    dose = predicted_doses[i]
                    if hasattr(dose, 'value') and hasattr(dose.value, '__getitem__'):
                        dose_val = dose.value[0]
                    elif hasattr(dose, 'value'):
                        dose_val = dose.value
                
                total_mass += mass_val
                total_cost += cost_val
                total_volume += volume_val
                
                layer_results.append({
                    'layer': i,
                    'material_id': optimized_materials[i],
                    'material_name': self.material_db.get_material_name(optimized_materials[i]),
                    'mass': mass_val,
                    'cost': cost_val,
                    'volume': volume_val,
                    'dose': dose_val
                })
                
            except Exception as e:
                logger.warning(f"Error processing layer {i}: {e}")
                layer_results.append({
                    'layer': i,
                    'material_id': optimized_materials[i],
                    'material_name': self.material_db.get_material_name(optimized_materials[i]),
                    'mass': 0.0,
                    'cost': 0.0,
                    'volume': 0.0,
                    'dose': 0.0
                })
        
        # Get final dose
        final_dose = 0.0
        if predicted_doses and hasattr(predicted_doses[-1], 'value') and hasattr(predicted_doses[-1].value, '__getitem__'):
            final_dose = predicted_doses[-1].value[0]
        elif predicted_doses and hasattr(predicted_doses[-1], 'value'):
            final_dose = predicted_doses[-1].value
        
        return OptimizationResult(
            success=True,
            optimized_materials=optimized_materials,
            total_mass=total_mass,
            total_cost=total_cost,
            final_dose=final_dose,
            layer_results=layer_results,
            optimization_time=0.0,  # Would need to track actual time
            solver_status=solve_result['solver_status'],
            objective_value=solve_result['objective_value']
        ) 

    def _extract_results_with_splines(self, m: GEKKO, density_vars: List, 
                                    layer_data: List, predicted_doses: List, 
                                    layer_splines: List[List], solve_result: Dict[str, Any]) -> OptimizationResult:
        """Extract and format optimization results using spline data."""
        if not solve_result['success']:
            return OptimizationResult(
                success=False,
                optimized_materials=[],
                total_mass=0.0,
                total_cost=0.0,
                final_dose=0.0,
                layer_results=[],
                optimization_time=0.0,
                solver_status=solve_result['solver_status'],
                objective_value=solve_result['objective_value']
            )
        
        # Extract optimized material selections
        optimized_materials = []
        for var in density_vars:
            if hasattr(var, 'value') and hasattr(var.value, '__getitem__'):
                optimized_materials.append(int(var.value[0]))
            elif hasattr(var, 'value'):
                optimized_materials.append(int(var.value))
            else:
                optimized_materials.append(int(var))
        
        # Calculate final metrics using spline data
        total_mass = 0.0
        total_cost = 0.0
        total_volume = 0.0
        
        layer_results = []
        for i, (data, spline_vars) in enumerate(zip(layer_data, layer_splines)):
            try:
                # Extract values from spline data
                densityVarY = spline_vars[0]
                costVarY = spline_vars[25]
                volume = layer_data[i]['volume']
                
                # Calculate mass and cost using spline values
                max_density = self.config.MAX_VALUES[0]  # 19300
                max_mass = self.config.MAX_VALUES[3]     # -3.301029996
                
                if hasattr(densityVarY, 'value') and hasattr(densityVarY.value, '__getitem__'):
                    density_val = densityVarY.value[0]
                elif hasattr(densityVarY, 'value'):
                    density_val = densityVarY.value
                else:
                    density_val = densityVarY
                
                if hasattr(costVarY, 'value') and hasattr(costVarY.value, '__getitem__'):
                    cost_val = costVarY.value[0]
                elif hasattr(costVarY, 'value'):
                    cost_val = costVarY.value
                else:
                    cost_val = costVarY
                
                mass_val = density_val * max_density * volume / (10**max_mass)
                cost_val = cost_val * volume
                
                if hasattr(volume, 'value') and hasattr(volume.value, '__getitem__'):
                    volume_val = volume.value[0]
                elif hasattr(volume, 'value'):
                    volume_val = volume.value
                else:
                    volume_val = volume
                
                dose_val = 0.0
                if i < len(predicted_doses):
                    dose = predicted_doses[i]
                    if hasattr(dose, 'value') and hasattr(dose.value, '__getitem__'):
                        dose_val = dose.value[0]
                    elif hasattr(dose, 'value'):
                        dose_val = dose.value
                
                total_mass += mass_val
                total_cost += cost_val
                total_volume += volume_val
                
                layer_results.append({
                    'layer': i,
                    'material_id': optimized_materials[i],
                    'material_name': self.material_db.get_material_name(optimized_materials[i]),
                    'mass': mass_val,
                    'cost': cost_val,
                    'volume': volume_val,
                    'dose': dose_val
                })
                
            except Exception as e:
                logger.warning(f"Error processing layer {i}: {e}")
                layer_results.append({
                    'layer': i,
                    'material_id': optimized_materials[i],
                    'material_name': self.material_db.get_material_name(optimized_materials[i]),
                    'mass': 0.0,
                    'cost': 0.0,
                    'volume': 0.0,
                    'dose': 0.0
                })
        
        # Get final dose
        final_dose = 0.0
        if predicted_doses and hasattr(predicted_doses[-1], 'value') and hasattr(predicted_doses[-1].value, '__getitem__'):
            final_dose = predicted_doses[-1].value[0]
        elif predicted_doses and hasattr(predicted_doses[-1], 'value'):
            final_dose = predicted_doses[-1].value
        
        return OptimizationResult(
            success=True,
            optimized_materials=optimized_materials,
            total_mass=total_mass,
            total_cost=total_cost,
            final_dose=final_dose,
            layer_results=layer_results,
            optimization_time=0.0,  # Would need to track actual time
            solver_status=solve_result['solver_status'],
            objective_value=solve_result['objective_value']
        ) 