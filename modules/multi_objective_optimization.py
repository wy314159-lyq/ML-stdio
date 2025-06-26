"""
Module for Single & Multi-Objective Optimization using one or more separate, pre-trained models.
This module allows users to find a set of Pareto-optimal solutions
that represent the trade-offs between two different objectives predicted by two models.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any, Optional, Tuple
import traceback
import warnings
import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
                           QLabel, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
                           QAbstractItemView, QHeaderView, QSpinBox, QComboBox, QTextEdit,
                           QTabWidget, QMessageBox, QProgressBar, QDoubleSpinBox,
                           QCheckBox, QFormLayout, QFrame, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap

# Matplotlib for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import seaborn as sns

# Set matplotlib style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Pymoo for optimization - Robust import with version detection
PYMOO_AVAILABLE = False
PYMOO_VERSION = 'not_installed'
NSGA2 = None
NSGA3 = None
SPEA2 = None
MOEAD = None
CMA_ES = None
DE = None
Problem = None
minimize = None
get_termination = None
Callback = None
Sampling = None
Repair = None
FloatRandomSampling = None
Termination = None
SBX = None
PM = None
DifferentialEvolution = None
UniformCrossover = None

try:
    # First, let's detect what's actually available in pymoo
    import pymoo
    print(f"Found pymoo package at: {pymoo.__file__}")
    
    # Try to get version info
    try:
        PYMOO_VERSION = pymoo.__version__
    except:
        PYMOO_VERSION = '0.5.0'  # Assume 0.5.0 if no version info
    
    print(f"Detected pymoo version: {PYMOO_VERSION}")
    
    # Import core functions that should be available in all versions
    from pymoo.optimize import minimize
    from pymoo.factory import get_termination
    
    # Try different import strategies based on what's available
    # Strategy 1: Try modern pymoo (0.6+) structure
    try:
        print("Trying modern pymoo structure (0.6+)...")
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.core.callback import Callback
        from pymoo.core.sampling import Sampling
        from pymoo.core.repair import Repair
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.core.termination import Termination
        
        try:
            from pymoo.algorithms.moo.nsga3 import NSGA3
        except ImportError:
            NSGA3 = None
            
        try:
            from pymoo.algorithms.moo.spea2 import SPEA2
        except ImportError:
            SPEA2 = None
            
        # Try to import additional algorithms
        try:
            from pymoo.algorithms.moo.moead import MOEAD
        except ImportError:
            MOEAD = None
            
        try:
            from pymoo.algorithms.soo.nonconvex.cmaes import CMA_ES
        except ImportError:
            CMA_ES = None
            
        try:
            from pymoo.algorithms.soo.nonconvex.de import DE
        except ImportError:
            DE = None
            
        # Try to import operators
        try:
            from pymoo.operators.crossover.sbx import SBX
        except ImportError:
            SBX = None
        try:
            from pymoo.operators.mutation.pm import PM
        except ImportError:
            PM = None
        try:
            from pymoo.operators.crossover.dex import DifferentialEvolution
        except ImportError:
            DifferentialEvolution = None
        try:
            from pymoo.operators.crossover.ux import UniformCrossover
        except ImportError:
            UniformCrossover = None
        
        print("✅ Successfully imported modern pymoo structure")
        PYMOO_VERSION = f"{PYMOO_VERSION} (modern)"
        
    except ImportError:
        print("Modern structure failed, trying classic pymoo structure (0.5.0)...")
        # Strategy 2: Try classic pymoo (0.5.0) structure
        try:
            from pymoo.model.problem import Problem
            from pymoo.model.callback import Callback
            from pymoo.model.sampling import Sampling
            from pymoo.model.repair import Repair
            from pymoo.model.termination import Termination
            
            # For NSGA2, try several possible locations
            try:
                from pymoo.algorithms.moo.nsga2 import NSGA2
            except ImportError:
                try:
                    from pymoo.algorithms.nsga2 import NSGA2
                except ImportError:
                    # Last resort - check if there's a generic GA we can use
                    try:
                        from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
                        NSGA2 = GeneticAlgorithm
                        print("⚠️ Using GeneticAlgorithm as NSGA2 fallback")
                    except ImportError:
                        raise ImportError("Could not find NSGA2 or compatible algorithm")
            
            # For FloatRandomSampling, try different locations
            try:
                from pymoo.operators.sampling.random_sampling import FloatRandomSampling
            except ImportError:
                try:
                    from pymoo.operators.sampling.random import Random
                    FloatRandomSampling = Random
                except ImportError:
                    # Create a simple random sampling fallback
                    class SimpleRandomSampling(Sampling):
                        def _do(self, problem, n_samples, **kwargs):
                            return np.random.uniform(problem.xl, problem.xu, (n_samples, problem.n_var))
                    FloatRandomSampling = SimpleRandomSampling
                    print("⚠️ Using simple random sampling fallback")
            
            # Try to get NSGA3
            try:
                from pymoo.algorithms.moo.nsga3 import NSGA3
            except ImportError:
                NSGA3 = None
                print("⚠️ NSGA3 not available in this version")
            
            # Try to get SPEA2
            try:
                from pymoo.algorithms.moo.spea2 import SPEA2
            except ImportError:
                SPEA2 = None
                print("⚠️ SPEA2 not available in this version")
            
            # Try to get additional algorithms
            try:
                from pymoo.algorithms.moo.moead import MOEAD
            except ImportError:
                MOEAD = None
                print("⚠️ MOEA/D not available in this version")
            
            try:
                from pymoo.algorithms.soo.nonconvex.cmaes import CMA_ES
            except ImportError:
                CMA_ES = None
                print("⚠️ CMA-ES not available in this version")
                
            try:
                from pymoo.algorithms.soo.nonconvex.de import DE
            except ImportError:
                DE = None
                print("⚠️ DE not available in this version")
            
            # Try to import operators for classic structure
            try:
                from pymoo.operators.crossover.simulated_binary_crossover import SBX
            except ImportError:
                SBX = None
            try:
                from pymoo.operators.mutation.polynomial_mutation import PM
            except ImportError:
                PM = None
            try:
                from pymoo.operators.crossover.dex import DifferentialEvolution
            except ImportError:
                DifferentialEvolution = None
            try:
                from pymoo.operators.crossover.ux import UniformCrossover
            except ImportError:
                UniformCrossover = None
            
            print("✅ Successfully imported classic pymoo structure")
            PYMOO_VERSION = f"{PYMOO_VERSION} (classic)"
            
        except ImportError as e:
            raise ImportError(f"Failed to import from classic pymoo structure: {e}")
    
    # If we get here, we have successfully imported the essentials
    PYMOO_AVAILABLE = True
    print(f"✅ pymoo successfully loaded: {PYMOO_VERSION}")
    
    # Disable compile warnings if available
    try:
        from pymoo.config import Config
        Config.show_compile_hint = False
    except:
        pass  # Not available in this version
    
except ImportError as e:
    print(f"❌ Failed to import pymoo: {e}")
    print("Please install pymoo using: pip install pymoo")
except Exception as e:
    print(f"❌ Unexpected error while importing pymoo: {e}")
    print("Please check your pymoo installation")


class PipelineAnalyzer:
    """Smart analyzer for extracting information from sklearn pipelines"""
    
    @staticmethod
    def extract_feature_bounds(pipeline, feature_names, log_callback=None):
        """
        Extract original feature ranges from various types of sklearn preprocessing steps
        
        Parameters:
        - pipeline: sklearn Pipeline object
        - feature_names: list of feature names
        - log_callback: function to log messages
        
        Returns:
        - List of (min, max) tuples for each feature, or None if extraction fails
        """
        def log(message):
            if log_callback:
                log_callback(message)
            else:
                print(message)
        
        try:
            bounds = []
            
            # Strategy 1: Check for ColumnTransformer (most common in modern pipelines)
            preprocessor = PipelineAnalyzer._find_preprocessor(pipeline)
            
            if preprocessor and hasattr(preprocessor, 'transformers_'):
                log("🔍 Analyzing ColumnTransformer...")
                bounds = PipelineAnalyzer._extract_from_column_transformer(
                    preprocessor, feature_names, log
                )
            
            # Strategy 2: Direct scaler in pipeline
            elif preprocessor:
                log("🔍 Analyzing direct preprocessor...")
                bounds = PipelineAnalyzer._extract_from_direct_scaler(
                    preprocessor, feature_names, log
                )
            
            # Strategy 3: Check pipeline steps individually
            else:
                log("🔍 Analyzing pipeline steps...")
                bounds = PipelineAnalyzer._extract_from_pipeline_steps(
                    pipeline, feature_names, log
                )
            
            # Validate results
            if bounds and len(bounds) == len(feature_names):
                log(f"✅ Successfully extracted bounds for all {len(bounds)} features")
                return bounds
            elif bounds:
                log(f"⚠️  Extracted bounds for {len(bounds)} of {len(feature_names)} features")
                return None  # Partial extraction not supported for now
            else:
                log("❌ Could not extract feature bounds from pipeline")
                return None
                
        except Exception as e:
            log(f"❌ Error during pipeline analysis: {str(e)}")
            return None
    
    @staticmethod
    def _find_preprocessor(pipeline):
        """Find the main preprocessor in the pipeline"""
        if not hasattr(pipeline, 'named_steps'):
            return None
        
        # Common preprocessor step names
        preprocessor_names = [
            'preprocessor', 'scaler', 'normalizer', 'standardscaler', 
            'minmaxscaler', 'robustscaler', 'columntransformer'
        ]
        
        for name in preprocessor_names:
            if name in pipeline.named_steps:
                return pipeline.named_steps[name]
        
        # Check all steps for known preprocessor types
        for step_name, step in pipeline.named_steps.items():
            if PipelineAnalyzer._is_known_preprocessor(step):
                return step
        
        return None
    
    @staticmethod
    def _is_known_preprocessor(obj):
        """Check if object is a known preprocessor type"""
        class_name = obj.__class__.__name__
        return any(name in class_name.lower() for name in [
            'scaler', 'normalizer', 'transformer', 'preprocessor'
        ])
    
    @staticmethod
    def _extract_from_column_transformer(transformer, feature_names, log):
        """Extract bounds from ColumnTransformer"""
        bounds = []
        processed_features = 0
        
        for name, step_transformer, columns in transformer.transformers_:
            if name == 'remainder':  # Skip remainder transformer
                continue
                
            log(f"   Processing transformer '{name}' for {len(columns)} columns")
            
            # Handle different column specifications
            if isinstance(columns, (list, np.ndarray)):
                n_cols = len(columns)
            else:
                n_cols = 1
            
            # Extract bounds from this transformer
            step_bounds = PipelineAnalyzer._extract_bounds_from_transformer(
                step_transformer, n_cols, log
            )
            
            if step_bounds:
                bounds.extend(step_bounds)
                processed_features += n_cols
            else:
                log(f"   ⚠️  Could not extract bounds from transformer '{name}'")
                return None
        
        return bounds if processed_features == len(feature_names) else None
    
    @staticmethod
    def _extract_from_direct_scaler(scaler, feature_names, log):
        """Extract bounds from direct scaler (no ColumnTransformer)"""
        return PipelineAnalyzer._extract_bounds_from_transformer(
            scaler, len(feature_names), log
        )
    
    @staticmethod
    def _extract_from_pipeline_steps(pipeline, feature_names, log):
        """Extract bounds by checking individual pipeline steps"""
        if not hasattr(pipeline, 'named_steps'):
            return None
        
        for step_name, step in pipeline.named_steps.items():
            if PipelineAnalyzer._is_known_preprocessor(step):
                log(f"   Found preprocessor step: {step_name}")
                bounds = PipelineAnalyzer._extract_bounds_from_transformer(
                    step, len(feature_names), log
                )
                if bounds:
                    return bounds
        
        return None
    
    @staticmethod
    def _extract_bounds_from_transformer(transformer, n_features, log):
        """Extract bounds from a specific transformer"""
        transformer_name = transformer.__class__.__name__
        log(f"     Analyzing {transformer_name}...")
        
        # MinMaxScaler - has exact min/max values
        if hasattr(transformer, 'data_min_') and hasattr(transformer, 'data_max_'):
            log(f"     ✅ Found MinMaxScaler with exact bounds")
            bounds = []
            for i in range(min(len(transformer.data_min_), n_features)):
                min_val = float(transformer.data_min_[i])
                max_val = float(transformer.data_max_[i])
                bounds.append((min_val, max_val))
            return bounds
        
        # StandardScaler - estimate bounds using mean ± k*std
        elif hasattr(transformer, 'mean_') and hasattr(transformer, 'scale_'):
            log(f"     ✅ Found StandardScaler, estimating bounds using mean ± 3σ")
            bounds = []
            for i in range(min(len(transformer.mean_), n_features)):
                mean = float(transformer.mean_[i])
                std = float(transformer.scale_[i])
                # Use mean ± 3*std as reasonable bounds (covers ~99.7% of data)
                min_val = mean - 3 * std
                max_val = mean + 3 * std
                bounds.append((min_val, max_val))
            return bounds
        
        # RobustScaler - use median and scale
        elif hasattr(transformer, 'center_') and hasattr(transformer, 'scale_'):
            log(f"     ✅ Found RobustScaler, estimating bounds")
            bounds = []
            for i in range(min(len(transformer.center_), n_features)):
                center = float(transformer.center_[i])
                scale = float(transformer.scale_[i])
                # Use center ± 3*scale as reasonable bounds
                min_val = center - 3 * scale
                max_val = center + 3 * scale
                bounds.append((min_val, max_val))
            return bounds
        
        # PowerTransformer, QuantileTransformer, etc.
        elif hasattr(transformer, 'lambdas_'):  # PowerTransformer
            log(f"     ⚠️  {transformer_name} detected but bounds extraction not implemented")
            return None
        
        # Unknown transformer
        else:
            log(f"     ❌ Unknown transformer type: {transformer_name}")
            return None


class MixedVariableSampling(Sampling):
    """
    Custom sampling that handles mixed variable types (continuous, binary, categorical)
    """
    
    def __init__(self, feature_types, categorical_ranges=None):
        """
        Args:
            feature_types: List of feature types ('continuous', 'binary', 'categorical')
            categorical_ranges: Dict mapping feature index to list of valid values
        """
        super().__init__()
        self.feature_types = feature_types
        self.categorical_ranges = categorical_ranges or {}
        
    def _do(self, problem, n_samples, **kwargs):
        """Generate initial population with proper variable types"""
        X = np.zeros((n_samples, problem.n_var))
        
        for j in range(problem.n_var):
            feature_type = self.feature_types[j] if j < len(self.feature_types) else 'continuous'
            
            if feature_type == 'binary':
                # Binary features: only 0 or 1
                X[:, j] = np.random.choice([0, 1], size=n_samples)
            elif feature_type == 'categorical':
                # Categorical features: integer values within range
                if j in self.categorical_ranges:
                    valid_values = self.categorical_ranges[j]
                    X[:, j] = np.random.choice(valid_values, size=n_samples)
                else:
                    # Default: integer sampling between bounds
                    low = int(problem.xl[j])
                    high = int(problem.xu[j]) + 1
                    X[:, j] = np.random.randint(low, high, size=n_samples)
            else:
                # Continuous features: uniform sampling
                X[:, j] = np.random.uniform(problem.xl[j], problem.xu[j], size=n_samples)
        
        return X


class EnhancedBoundaryRepair(Repair):
    """
    Enhanced repair operator that ensures all constraints are satisfied
    """
    
    def __init__(self, feature_types, categorical_ranges=None, fixed_features=None):
        """
        Args:
            feature_types: List of feature types ('continuous', 'binary', 'categorical')
            categorical_ranges: Dict mapping feature index to list of valid values
            fixed_features: Dict mapping feature index to fixed value
        """
        super().__init__()
        self.feature_types = feature_types
        self.categorical_ranges = categorical_ranges or {}
        self.fixed_features = fixed_features or {}
        
    def _do(self, problem, X, **kwargs):
        """Repair solutions to satisfy all constraints"""
        X_repaired = X.copy()
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Apply fixed features first
                if j in self.fixed_features:
                    X_repaired[i, j] = self.fixed_features[j]
                    continue
                
                # Get feature type
                feature_type = self.feature_types[j] if j < len(self.feature_types) else 'continuous'
                
                # Apply bounds constraint
                X_repaired[i, j] = np.clip(X_repaired[i, j], problem.xl[j], problem.xu[j])
                
                # Apply type-specific constraints
                if feature_type == 'binary':
                    # Binary: round to nearest 0 or 1
                    X_repaired[i, j] = round(X_repaired[i, j])
                    X_repaired[i, j] = np.clip(X_repaired[i, j], 0, 1)
                elif feature_type == 'categorical':
                    # Categorical: round to nearest integer and ensure in valid range
                    if j in self.categorical_ranges:
                        valid_values = np.array(self.categorical_ranges[j])
                        # Find closest valid value
                        closest_idx = np.argmin(np.abs(valid_values - X_repaired[i, j]))
                        X_repaired[i, j] = valid_values[closest_idx]
                    else:
                        # Round to integer and clip to bounds
                        X_repaired[i, j] = round(X_repaired[i, j])
                        X_repaired[i, j] = np.clip(X_repaired[i, j], problem.xl[j], problem.xu[j])
                # Continuous features don't need additional processing beyond clipping
        
        return X_repaired


def batch_predict_models(models, indices_list, X_pop, feature_names):
    """
    Vectorized prediction for all models to improve performance.
    
    Args:
        models: List of trained model pipelines
        indices_list: List of feature indices for each model
        X_pop: Population matrix (n_individuals x n_features)
        feature_names: List of all feature names
    
    Returns:
        Predictions matrix (n_individuals x n_models)
    """
    try:
        n_individuals = X_pop.shape[0]
        n_models = len(models)
        predictions = np.zeros((n_individuals, n_models))
        
        for model_idx, (model, indices) in enumerate(zip(models, indices_list)):
            # Extract features for current model
            X_subset = X_pop[:, indices]
            
            # Create DataFrame with correct feature names
            model_feature_names = [feature_names[idx] for idx in indices]
            X_df = pd.DataFrame(X_subset, columns=model_feature_names)
            
            # Batch prediction for all individuals at once
            model_predictions = model.predict(X_df)
            predictions[:, model_idx] = model_predictions
        
        return predictions
    except Exception as e:
        print(f"Batch prediction error: {e}")
        # Fallback to individual predictions
        predictions = np.zeros((X_pop.shape[0], len(models)))
        for i in range(X_pop.shape[0]):
            for j, (model, indices) in enumerate(zip(models, indices_list)):
                try:
                    X_subset = X_pop[i, indices]
                    model_feature_names = [feature_names[idx] for idx in indices]
                    X_df = pd.DataFrame([X_subset], columns=model_feature_names)
                    predictions[i, j] = model.predict(X_df)[0]
                except Exception:
                    predictions[i, j] = float('inf')
        return predictions


class HypervolumeEarlyStop(Termination):
    """
    Early stopping termination based on hypervolume convergence.
    Stops optimization when hypervolume improvement stagnates.
    """
    
    def __init__(self, min_delta=1e-6, patience=30, n_last=10):
        """
        Args:
            min_delta: Minimum hypervolume improvement required
            patience: Number of generations to wait without improvement
            n_last: Number of last generations to consider for convergence
        """
        super().__init__()
        self.min_delta = min_delta
        self.patience = patience
        self.n_last = n_last
        self.hypervolume_history = []
        self.best_hv = -np.inf
        self.counter = 0
        self.min_generations = max(10, patience // 3)  # 至少运行几代才开始早停检查
        
    def _update(self, algorithm):
        """Check if termination criteria is met"""
        try:
            # Get current generation number
            current_gen = algorithm.n_gen if hasattr(algorithm, 'n_gen') else len(self.hypervolume_history)
            
            # Get current population objectives
            pop = algorithm.pop
            if pop is None or len(pop) == 0:
                return False
                
            F = pop.get("F")
            if F is None or len(F) == 0:
                return False
            
            # Calculate hypervolume (simplified approximation)
            current_hv = self._calculate_hypervolume(F)
            self.hypervolume_history.append(current_hv)
            
            # Don't trigger early stopping until minimum generations have passed
            if current_gen < self.min_generations:
                return False
            
            # Use relative improvement instead of absolute
            if len(self.hypervolume_history) > 1:
                # Calculate relative improvement
                prev_hv = self.hypervolume_history[-2]
                if prev_hv > 0:
                    relative_improvement = (current_hv - prev_hv) / abs(prev_hv)
                else:
                    relative_improvement = current_hv - prev_hv
                
                # Check for significant improvement
                if relative_improvement > self.min_delta:
                    self.best_hv = current_hv
                    self.counter = 0
                else:
                    self.counter += 1
            else:
                # First generation
                self.best_hv = current_hv
                self.counter = 0
            
            # Also check trend in last n generations (only if we have enough history)
            if len(self.hypervolume_history) >= self.n_last:
                recent_hvs = self.hypervolume_history[-self.n_last:]
                if len(recent_hvs) > 2:  # Need at least 3 points for trend
                    try:
                        trend = np.polyfit(range(len(recent_hvs)), recent_hvs, 1)[0]
                        # Use relative trend (normalized by average hypervolume)
                        avg_hv = np.mean(recent_hvs)
                        if avg_hv > 0:
                            relative_trend = abs(trend) / avg_hv
                            if relative_trend < self.min_delta / 10:  # Very conservative threshold
                                self.counter = max(self.counter, self.patience // 3)
                    except np.RankWarning:
                        pass  # Skip trend analysis if there's an issue
            
            # Only terminate if we've been stagnant for patience generations AND past minimum
            should_terminate = (self.counter >= self.patience and 
                               current_gen >= self.min_generations and 
                               len(self.hypervolume_history) >= self.n_last)
            
            return should_terminate
            
        except Exception as e:
            print(f"Early stopping check error: {e}")
            return False
    
    def _calculate_hypervolume(self, objectives):
        """Calculate hypervolume approximation"""
        try:
            if len(objectives) == 0:
                return 0.0
            
            # Use nadir point as reference
            ref_point = np.max(objectives, axis=0) + 0.1 * np.abs(np.max(objectives, axis=0))
            
            if objectives.shape[1] == 1:
                # For single objective, use simple range
                return ref_point[0] - np.min(objectives[:, 0])
            elif objectives.shape[1] == 2:
                # For 2D, calculate exact hypervolume
                sorted_idx = np.argsort(objectives[:, 0])
                sorted_objs = objectives[sorted_idx]
                
                hv = 0.0
                prev_x = ref_point[0]
                for obj in sorted_objs:
                    if obj[0] < ref_point[0] and obj[1] < ref_point[1]:
                        width = prev_x - obj[0]
                        height = ref_point[1] - obj[1]
                        hv += width * height
                        prev_x = obj[0]
                return max(0.0, hv)
            else:
                # For higher dimensions, use product approximation
                dominated_volume = 1.0
                for i in range(objectives.shape[1]):
                    min_obj = np.min(objectives[:, i])
                    if min_obj < ref_point[i]:
                        dominated_volume *= (ref_point[i] - min_obj)
                    else:
                        dominated_volume = 0.0
                        break
                return dominated_volume
        except Exception:
            return 0.0


class OptimizationCheckpoint:
    """
    Checkpoint system for saving and resuming optimization progress.
    """
    
    def __init__(self, save_interval=25, max_checkpoints=5):
        """
        Args:
            save_interval: Save checkpoint every N generations
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = Path("optimization_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def should_save(self, generation):
        """Check if checkpoint should be saved at this generation"""
        return generation % self.save_interval == 0 and generation > 0
    
    def save_checkpoint(self, generation, algorithm, config, additional_data=None):
        """Save optimization checkpoint"""
        try:
            checkpoint_data = {
                'generation': generation,
                'population_X': algorithm.pop.get("X") if algorithm.pop else None,
                'population_F': algorithm.pop.get("F") if algorithm.pop else None,
                'config': config,
                'timestamp': datetime.datetime.now().isoformat(),
                'additional_data': additional_data or {}
            }
            
            checkpoint_file = self.checkpoint_dir / f"checkpoint_gen_{generation:04d}.joblib"
            joblib.dump(checkpoint_data, checkpoint_file)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_file)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_gen_*.joblib"))
            if len(checkpoint_files) > self.max_checkpoints:
                # Sort by creation time and remove oldest
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in checkpoint_files[:-self.max_checkpoints]:
                    old_file.unlink()
        except Exception as e:
            print(f"Failed to cleanup checkpoints: {e}")
    
    def list_checkpoints(self):
        """List available checkpoints"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_gen_*.joblib"))
            checkpoints = []
            
            for file in checkpoint_files:
                try:
                    data = joblib.load(file)
                    checkpoints.append({
                        'file': file,
                        'generation': data.get('generation', 0),
                        'timestamp': data.get('timestamp', 'unknown'),
                        'size_mb': file.stat().st_size / (1024 * 1024)
                    })
                except Exception:
                    continue
            
            return sorted(checkpoints, key=lambda x: x['generation'], reverse=True)
        except Exception as e:
            print(f"Failed to list checkpoints: {e}")
            return []
    
    def load_checkpoint(self, checkpoint_file):
        """Load checkpoint data"""
        try:
            return joblib.load(checkpoint_file)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None


class ConstraintViolationMonitor:
    """
    Monitor and report constraint violations during optimization
    """
    
    def __init__(self):
        self.violation_history = []
        self.total_evaluations = 0
        self.total_violations = 0
        
    def check_violations(self, X, problem, feature_types, fixed_features=None):
        """Check for constraint violations in population"""
        violations = []
        n_violations = 0
        
        for i in range(X.shape[0]):
            individual_violations = []
            
            for j in range(X.shape[1]):
                # Check bounds violation
                if X[i, j] < problem.xl[j] or X[i, j] > problem.xu[j]:
                    individual_violations.append(f"Feature {j}: bounds violation ({X[i, j]:.4f} not in [{problem.xl[j]:.4f}, {problem.xu[j]:.4f}])")
                    n_violations += 1
                
                # Check type violations
                feature_type = feature_types[j] if j < len(feature_types) else 'continuous'
                if feature_type == 'binary' and X[i, j] not in [0, 1]:
                    individual_violations.append(f"Feature {j}: binary violation (value={X[i, j]:.4f})")
                    n_violations += 1
                elif feature_type == 'categorical' and X[i, j] != round(X[i, j]):
                    individual_violations.append(f"Feature {j}: categorical violation (non-integer value={X[i, j]:.4f})")
                    n_violations += 1
                
                # Check fixed feature violations
                if fixed_features and j in fixed_features and abs(X[i, j] - fixed_features[j]) > 1e-6:
                    individual_violations.append(f"Feature {j}: fixed value violation (expected={fixed_features[j]}, got={X[i, j]:.4f})")
                    n_violations += 1
            
            if individual_violations:
                violations.append(f"Individual {i}: {'; '.join(individual_violations)}")
        
        self.total_evaluations += X.shape[0]
        self.total_violations += n_violations
        
        return violations, n_violations
    
    def get_violation_rate(self):
        """Get overall violation rate"""
        if self.total_evaluations == 0:
            return 0.0
        return self.total_violations / self.total_evaluations


class ExplicitConstraintHandler:
    """
    Handle explicit constraints (<=, >=, =) with adaptive penalty coefficients
    """
    def __init__(self, constraint_specs=None):
        """
        Parameters:
        - constraint_specs: List of dicts with constraint definitions
          Format: {'type': 'leq'/'geq'/'eq', 'target_idx': int, 'bound': float, 'feature_indices': [int]}
        """
        self.constraint_specs = constraint_specs or []
        self.penalty_coeffs = {}
        self.violation_history = {}
        self.adaptive_penalty = True
        
        # Initialize penalty coefficients
        for i, spec in enumerate(self.constraint_specs):
            self.penalty_coeffs[i] = 1.0
            self.violation_history[i] = []
    
    def evaluate_constraints(self, x, objectives):
        """
        Evaluate constraint violations and return constraint vector
        
        Parameters:
        - x: Decision variables
        - objectives: Objective values
        
        Returns:
        - cv: Constraint violation vector (positive = violation)
        """
        cv = []
        
        for i, spec in enumerate(self.constraint_specs):
            if spec['type'] == 'leq':  # g(x) <= bound
                if 'target_idx' in spec:
                    # Constraint on objective value
                    violation = objectives[spec['target_idx']] - spec['bound']
                else:
                    # Constraint on feature values
                    value = sum(x[j] for j in spec['feature_indices'])
                    violation = value - spec['bound']
                    
            elif spec['type'] == 'geq':  # g(x) >= bound
                if 'target_idx' in spec:
                    violation = spec['bound'] - objectives[spec['target_idx']]
                else:
                    value = sum(x[j] for j in spec['feature_indices'])
                    violation = spec['bound'] - value
                    
            elif spec['type'] == 'eq':  # g(x) = bound
                if 'target_idx' in spec:
                    violation = abs(objectives[spec['target_idx']] - spec['bound'])
                else:
                    value = sum(x[j] for j in spec['feature_indices'])
                    violation = abs(value - spec['bound'])
            else:
                violation = 0.0
            
            cv.append(max(0.0, violation))
            
            # Track violation history for adaptive penalty
            self.violation_history[i].append(violation > 0)
            
        return np.array(cv)
    
    def update_penalty_coefficients(self, generation):
        """Update penalty coefficients based on violation rates"""
        if not self.adaptive_penalty or generation < 10:
            return
            
        for i, spec in enumerate(self.constraint_specs):
            recent_violations = self.violation_history[i][-10:]  # Last 10 generations
            violation_rate = sum(recent_violations) / len(recent_violations)
            
            if violation_rate > 0.8:  # High violation rate
                self.penalty_coeffs[i] *= 1.2
            elif violation_rate < 0.1:  # Low violation rate
                self.penalty_coeffs[i] *= 0.95
                
            # Keep penalty coefficient in reasonable range
            self.penalty_coeffs[i] = max(0.1, min(100.0, self.penalty_coeffs[i]))


class RobustOptimizationHandler:
    """
    Handle robust optimization with uncertainty propagation
    """
    def __init__(self, uncertainty_config=None):
        """
        Parameters:
        - uncertainty_config: Dict with uncertainty specifications
          Format: {'method': 'bootstrap/monte_carlo', 'n_samples': int, 'noise_level': float}
        """
        self.config = uncertainty_config or {
            'method': 'bootstrap',
            'n_samples': 10,
            'noise_level': 0.01
        }
        self.uncertainty_cache = {}
    
    def evaluate_robust_objectives(self, models, model_indices, x, feature_names, directions):
       
        method = self.config['method']
        n_samples = self.config['n_samples']
        noise_level = self.config['noise_level']
        
        if method == 'bootstrap':
            return self._bootstrap_evaluation(models, model_indices, x, feature_names, directions, n_samples)
        elif method == 'monte_carlo':
            return self._monte_carlo_evaluation(models, model_indices, x, feature_names, directions, n_samples, noise_level)
        else:
            # Fallback to deterministic evaluation
            objectives = []
            for j, (model, model_idx) in enumerate(zip(models, model_indices)):
                try:
                    x_df = pd.DataFrame([x], columns=feature_names)
                    prediction = model.predict(x_df)[0]
                    objectives.append(directions[j] * prediction)
                except:
                    objectives.append(float('inf'))
            
            return np.array(objectives), {'uncertainty': 0.0, 'method': 'deterministic'}
    
    def _bootstrap_evaluation(self, models, model_indices, x, feature_names, directions, n_samples):
        """Bootstrap-based uncertainty evaluation"""
        all_predictions = []
        
        for j, (model, model_idx) in enumerate(zip(models, model_indices)):
            predictions = []
            x_df = pd.DataFrame([x], columns=feature_names)
            
            for _ in range(n_samples):
                try:
                    # Add small noise to input for bootstrap-like sampling
                    x_noisy = x + np.random.normal(0, 0.001, size=x.shape)
                    x_noisy_df = pd.DataFrame([x_noisy], columns=feature_names)
                    
                    prediction = model.predict(x_noisy_df)[0]
                    predictions.append(directions[j] * prediction)
                except:
                    predictions.append(float('inf'))
            
            all_predictions.append(predictions)
        
        # Calculate robust statistics
        means = [np.mean(pred) for pred in all_predictions]
        stds = [np.std(pred) for pred in all_predictions]
        
        # Robust objective: mean + risk_factor * std (minimize worst-case)
        risk_factor = 0.5  # Adjustable risk aversion parameter
        robust_objectives = [mean + risk_factor * std for mean, std in zip(means, stds)]
        
        uncertainty_metrics = {
            'means': means,
            'stds': stds,
            'uncertainty': np.mean(stds),
            'method': 'bootstrap'
        }
        
        return np.array(robust_objectives), uncertainty_metrics
    
    def _monte_carlo_evaluation(self, models, model_indices, x, feature_names, directions, n_samples, noise_level):
        """Monte Carlo-based uncertainty evaluation"""
        all_predictions = []
        
        for j, (model, model_idx) in enumerate(zip(models, model_indices)):
            predictions = []
            
            for _ in range(n_samples):
                try:
                    # Add Gaussian noise to input variables
                    x_noisy = x + np.random.normal(0, noise_level * np.abs(x), size=x.shape)
                    x_noisy_df = pd.DataFrame([x_noisy], columns=feature_names)
                    
                    prediction = model.predict(x_noisy_df)[0]
                    predictions.append(directions[j] * prediction)
                except:
                    predictions.append(float('inf'))
            
            all_predictions.append(predictions)
        
        # Calculate statistics
        means = [np.mean(pred) for pred in all_predictions]
        stds = [np.std(pred) for pred in all_predictions]
        
        # CVaR-based robust objective (Conditional Value at Risk)
        alpha = 0.1  # Risk level (10% worst cases)
        robust_objectives = []
        
        for predictions in all_predictions:
            sorted_pred = np.sort(predictions)
            cvar_idx = int(len(sorted_pred) * (1 - alpha))
            cvar = np.mean(sorted_pred[cvar_idx:])  # Mean of worst α% cases
            robust_objectives.append(cvar)
        
        uncertainty_metrics = {
            'means': means,
            'stds': stds,
            'cvar': robust_objectives,
            'uncertainty': np.mean(stds),
            'method': 'monte_carlo'
        }
        
        return np.array(robust_objectives), uncertainty_metrics


class ScenarioBasedOptimization:
    """
    Multi-scenario robust optimization handler
    """
    def __init__(self, scenario_config=None):
        """
        Parameters:
        - scenario_config: Dict with scenario specifications
          Format: {'scenarios': [{'name': str, 'perturbations': dict}], 'aggregation': str}
        """
        self.config = scenario_config or {
            'scenarios': [
                {'name': 'nominal', 'perturbations': {}},
                {'name': 'worst_case', 'perturbations': {'noise_scale': 0.05}}
            ],
            'aggregation': 'worst_case'  # 'worst_case', 'average', 'weighted'
        }
    
    def evaluate_multi_scenario(self, models, model_indices, x, feature_names, directions):
        """
        Evaluate objectives across multiple scenarios
        
        Returns:
        - aggregated_objectives: Scenario-aggregated objective values
        - scenario_details: Dict with per-scenario results
        """
        scenario_results = {}
        
        for scenario in self.config['scenarios']:
            name = scenario['name']
            perturbations = scenario.get('perturbations', {})
            
            # Apply scenario perturbations
            x_scenario = self._apply_perturbations(x, perturbations)
            
            # Evaluate in this scenario
            objectives = []
            for j, (model, model_idx) in enumerate(zip(models, model_indices)):
                try:
                    x_df = pd.DataFrame([x_scenario], columns=feature_names)
                    prediction = model.predict(x_df)[0]
                    objectives.append(directions[j] * prediction)
                except:
                    objectives.append(float('inf'))
            
            scenario_results[name] = np.array(objectives)
        
        # Aggregate across scenarios
        aggregated = self._aggregate_scenarios(scenario_results)
        
        return aggregated, scenario_results
    
    def _apply_perturbations(self, x, perturbations):
        """Apply scenario-specific perturbations to input"""
        x_perturbed = x.copy()
        
        if 'noise_scale' in perturbations:
            noise = np.random.normal(0, perturbations['noise_scale'], size=x.shape)
            x_perturbed += noise
        
        if 'bias' in perturbations:
            x_perturbed += perturbations['bias']
        
        return x_perturbed
    
    def _aggregate_scenarios(self, scenario_results):
        """Aggregate objectives across scenarios"""
        method = self.config['aggregation']
        
        if method == 'worst_case':
            # For minimization: take maximum (worst) value across scenarios
            all_values = np.array(list(scenario_results.values()))
            return np.max(all_values, axis=0)
            
        elif method == 'average':
            all_values = np.array(list(scenario_results.values()))
            return np.mean(all_values, axis=0)
            
        elif method == 'weighted':
            # Use equal weights for now (could be configurable)
            weights = np.ones(len(scenario_results)) / len(scenario_results)
            all_values = np.array(list(scenario_results.values()))
            return np.average(all_values, axis=0, weights=weights)
        
        else:
            # Default to average
            all_values = np.array(list(scenario_results.values()))
            return np.mean(all_values, axis=0)


class OptimizationCallback(Callback):
    """Enhanced callback for real-time progress updates and convergence tracking"""
    
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.generation_history = []
        self.hypervolume_history = []
        self.best_objectives_history = []
        
    def notify(self, algorithm):
        """Called after each generation"""
        try:
            # Get current generation data
            gen = algorithm.n_gen
            pop = algorithm.pop
            
            if pop is not None and len(pop) > 0:
                # Extract objective values
                F = pop.get("F")
                X = pop.get("X")
                
                if F is not None and X is not None:
                    # Calculate convergence metrics
                    self._calculate_metrics(gen, F)
                    
                    # Emit progress signal with enhanced data
                    progress_data = {
                        'generation': gen,
                        'objectives': F,
                        'variables': X,
                        'n_solutions': len(F),
                        'hypervolume_history': self.hypervolume_history.copy(),
                        'best_objectives_history': self.best_objectives_history.copy(),
                        'convergence_metrics': self._get_convergence_metrics(F)
                    }
                    self.worker.progress_updated.emit(progress_data)
                    
                    # Handle checkpointing
                    if hasattr(self.worker, 'checkpoint_system'):
                        if self.worker.checkpoint_system.should_save(gen):
                            try:
                                checkpoint_file = self.worker.checkpoint_system.save_checkpoint(
                                    gen, algorithm, self.worker.config,
                                    additional_data={
                                        'hypervolume_history': self.hypervolume_history.copy(),
                                        'best_objectives_history': self.best_objectives_history.copy()
                                    }
                                )
                                if checkpoint_file:
                                    self.worker.status_updated.emit(f"💾 Checkpoint saved: {checkpoint_file}")
                            except Exception as e:
                                print(f"Checkpoint save error: {e}")
                                
        except Exception as e:
            print(f"Callback error: {e}")
    
    def _calculate_metrics(self, generation, objectives):
        """Calculate convergence metrics for current generation"""
        try:
            # Store generation
            self.generation_history.append(generation)
            
            # Calculate hypervolume (simplified approximation)
            if len(objectives) > 0:
                # Use maximum values as reference point
                ref_point = np.max(objectives, axis=0) + 0.1 * np.abs(np.max(objectives, axis=0))
                # Simple hypervolume approximation using dominated volume
                hv = self._approximate_hypervolume(objectives, ref_point)
                self.hypervolume_history.append(hv)
            else:
                self.hypervolume_history.append(0.0)
            
            # Track best objectives (minimum for each objective)
            if len(objectives) > 0:
                best_objs = np.min(objectives, axis=0)
                self.best_objectives_history.append(best_objs.tolist())
            else:
                self.best_objectives_history.append([float('inf')] * objectives.shape[1] if len(objectives.shape) > 1 else [float('inf')])
                
        except Exception as e:
            print(f"Metrics calculation error: {e}")
    
    def _approximate_hypervolume(self, objectives, ref_point):
        """Simple hypervolume approximation"""
        try:
            if len(objectives) == 0:
                return 0.0
            
            # For 2D case, use exact calculation
            if objectives.shape[1] == 2:
                # Sort by first objective
                sorted_idx = np.argsort(objectives[:, 0])
                sorted_objs = objectives[sorted_idx]
                
                hv = 0.0
                prev_x = ref_point[0]
                for obj in sorted_objs:
                    if obj[0] < ref_point[0] and obj[1] < ref_point[1]:
                        width = prev_x - obj[0]
                        height = ref_point[1] - obj[1]
                        hv += width * height
                        prev_x = obj[0]
                return max(0.0, hv)
            else:
                # For higher dimensions, use simple approximation
                dominated_volume = 1.0
                for i in range(objectives.shape[1]):
                    min_obj = np.min(objectives[:, i])
                    if min_obj < ref_point[i]:
                        dominated_volume *= (ref_point[i] - min_obj)
                    else:
                        dominated_volume = 0.0
                        break
                return dominated_volume
        except Exception:
            return 0.0
    
    def _get_convergence_metrics(self, objectives):
        """Get current convergence metrics"""
        if len(objectives) == 0:
            return {}
        
        return {
            'n_solutions': len(objectives),
            'min_objectives': np.min(objectives, axis=0).tolist(),
            'max_objectives': np.max(objectives, axis=0).tolist(),
            'mean_objectives': np.mean(objectives, axis=0).tolist(),
            'std_objectives': np.std(objectives, axis=0).tolist()
        }


class MLProblem(Problem):
  
    
    def __init__(self, models, model_indices, directions, n_var, xl, xu, feature_names, 
                 fixed_features=None, feature_types=None, categorical_ranges=None, 
                 diversity_noise_scale=1e-10, robust_config=None, constraint_config=None):
        """
        Initialize the multi-objective problem
        
        Args:
            models: List of trained model pipelines
            model_indices: List of feature indices for each model
            directions: List of optimization directions (-1 for max, 1 for min)
            n_var: Number of variables (combined features)
            xl: Lower bounds array
            xu: Upper bounds array
            feature_names: Names of all features
            fixed_features: Dictionary of feature_index -> fixed_value
            feature_types: List of feature types ('continuous', 'binary', 'categorical')
            categorical_ranges: Dict mapping feature index to list of valid values
            diversity_noise_scale: Scale factor for diversity noise (relative to objective magnitude)
        """
        n_obj = len(models)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        
        self.models = models
        self.model_indices = [np.array(indices) for indices in model_indices]
        self.directions = directions
        self.feature_names = feature_names
        self.n_objectives = n_obj
        self.fixed_features = fixed_features or {}
        self.feature_types = feature_types or ['continuous'] * n_var
        self.categorical_ranges = categorical_ranges or {}
        self.diversity_noise_scale = diversity_noise_scale
        
        # Initialize enhanced handlers
        self.robust_handler = None
        self.constraint_handler = None
        self.scenario_handler = None
        
        # Setup robust optimization if enabled
        if robust_config and robust_config.get('enable_robust_optimization', False):
            uncertainty_config = {
                'method': robust_config.get('robust_method', 'bootstrap'),
                'n_samples': robust_config.get('robust_samples', 10),
                'noise_level': robust_config.get('robust_noise_level', 0.01)
            }
            self.robust_handler = RobustOptimizationHandler(uncertainty_config)
            print(f"✅ Robust optimization enabled: {uncertainty_config['method']}")
            
            if uncertainty_config['method'] == 'scenario_based':
                self.scenario_handler = ScenarioBasedOptimization()
        
        # Setup constraint handling if enabled
        if constraint_config and constraint_config.get('enable_explicit_constraints', False):
            self.constraint_handler = ExplicitConstraintHandler()
            print(f"✅ Explicit constraint handling enabled")
        
        # Validate inputs
        if len(models) != len(model_indices) or len(models) != len(directions):
            raise ValueError("Number of models, indices, and directions must match")
        
        for i, indices in enumerate(self.model_indices):
            if len(indices) == 0:
                raise ValueError(f"Model {i+1} indices cannot be empty")
        
        # Validate feature types
        if len(self.feature_types) != n_var:
            print(f"Warning: feature_types length ({len(self.feature_types)}) != n_var ({n_var}), padding with 'continuous'")
            self.feature_types.extend(['continuous'] * (n_var - len(self.feature_types)))
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Enhanced evaluation with batch prediction and controlled diversity preservation
        
        Args:
            x: Input solutions (n_solutions x n_variables)
            out: Output dictionary to store results
        """
        try:
            n_solutions = x.shape[0]
            
            # Apply bounds clipping and fixed features to all solutions
            x_processed = np.clip(x.copy(), self.xl, self.xu)
            
            # Apply fixed features if any
            for feature_idx, fixed_value in self.fixed_features.items():
                x_processed[:, feature_idx] = fixed_value
            
            # Use batch prediction or robust evaluation
            try:
                if self.robust_handler:
                    # Use robust optimization for uncertainty handling
                    objectives = np.zeros((n_solutions, self.n_objectives))
                    
                    for i in range(n_solutions):
                        solution = x_processed[i]
                        
                        if self.scenario_handler:
                            # Scenario-based evaluation
                            robust_objs, _ = self.scenario_handler.evaluate_multi_scenario(
                                self.models, self.model_indices, solution, self.feature_names, self.directions
                            )
                        else:
                            # Bootstrap or Monte Carlo evaluation
                            robust_objs, _ = self.robust_handler.evaluate_robust_objectives(
                                self.models, self.model_indices, solution, self.feature_names, self.directions
                            )
                        
                        objectives[i] = robust_objs
                        
                        # Add diversity term if configured
                        if self.diversity_noise_scale > 0:
                            for j in range(self.n_objectives):
                                if np.isfinite(objectives[i, j]):
                                    diversity_term = np.random.normal(0, abs(objectives[i, j]) * self.diversity_noise_scale)
                                    objectives[i, j] += diversity_term
                else:
                    # Standard batch prediction
                    predictions = batch_predict_models(
                        self.models, self.model_indices, x_processed, self.feature_names
                    )
                    
                    # Apply optimization directions
                    objectives = np.zeros((n_solutions, self.n_objectives))
                    for j in range(self.n_objectives):
                        direction = self.directions[j]
                        valid_mask = np.isfinite(predictions[:, j])
                        
                        # Apply direction to valid predictions
                        objectives[valid_mask, j] = direction * predictions[valid_mask, j]
                        
                        # Set invalid predictions to infinity
                        objectives[~valid_mask, j] = float('inf')
                        
                        # Add configurable diversity term to prevent identical objective values
                        if self.diversity_noise_scale > 0 and np.any(valid_mask):
                            valid_objs = objectives[valid_mask, j]
                            diversity_terms = np.random.normal(
                                0, np.abs(valid_objs) * self.diversity_noise_scale
                            )
                            objectives[valid_mask, j] += diversity_terms
                        
            except Exception as e:
                print(f"Batch prediction failed, falling back to individual evaluation: {e}")
                # Fallback to original individual evaluation
                objectives = self._evaluate_individual(x_processed)
            
            # Additional diversity check for multi-objective cases
            if (n_solutions > 1 and self.n_objectives >= 2 and 
                0 < self.diversity_noise_scale < 1e-8):
                for j in range(self.n_objectives):
                    obj_values = objectives[:, j]
                    valid_values = obj_values[np.isfinite(obj_values)]
                    if len(valid_values) > 1:
                        # Check if all values are too similar
                        value_range = np.max(valid_values) - np.min(valid_values)
                        mean_value = np.mean(valid_values)
                        if value_range < abs(mean_value) * 1e-10:
                            # Add minimal artificial diversity
                            for idx, val in enumerate(obj_values):
                                if np.isfinite(val):
                                    objectives[idx, j] += np.random.normal(0, abs(mean_value) * 1e-12)
            
            out["F"] = objectives
            
        except Exception as e:
            print(f"Critical evaluation error: {e}")
            import traceback
            traceback.print_exc()
            # Return invalid objectives
            out["F"] = np.full((x.shape[0], self.n_objectives), float('inf'))
    
    def _evaluate_individual(self, x_processed):
        """Fallback individual evaluation method (original logic)"""
        n_solutions = x_processed.shape[0]
        objectives = np.zeros((n_solutions, self.n_objectives))
        
        for i in range(n_solutions):
            solution = x_processed[i]
            
            # Evaluate each objective/model
            for j in range(self.n_objectives):
                try:
                    model = self.models[j]
                    indices = self.model_indices[j]
                    direction = self.directions[j]
                    
                    # Extract features for current model
                    x_subset = solution[indices]
                    
                    # Validate input dimensions
                    if len(x_subset) != len(indices):
                        print(f"Warning: Dimension mismatch for model {j+1}")
                        objectives[i, j] = float('inf')
                        continue
                    
                    # Create DataFrame with correct feature names
                    model_feature_names = [self.feature_names[idx] for idx in indices]
                    x_df = pd.DataFrame([x_subset], columns=model_feature_names)
                    
                    # Predict and apply optimization direction
                    prediction = model.predict(x_df)[0]
                    
                    # Enhanced prediction validation
                    if not np.isfinite(prediction):
                        print(f"Warning: Invalid prediction from model {j+1}")
                        objectives[i, j] = float('inf')
                    else:
                        # Apply direction
                        base_objective = direction * prediction
                        
                        # Add diversity term if configured
                        if self.diversity_noise_scale > 0:
                            diversity_term = np.random.normal(0, abs(base_objective) * self.diversity_noise_scale)
                            objectives[i, j] = base_objective + diversity_term
                        else:
                            objectives[i, j] = base_objective
                    
                except Exception as e:
                    print(f"Model {j+1} prediction error: {str(e)}")
                    objectives[i, j] = float('inf')
        
        return objectives


class MultiObjectiveOptimizationWorker(QThread):
    """
    Worker thread to run the NSGA-II optimization without freezing the UI.
    """
    
    # Signals
    progress_updated = pyqtSignal(dict)
    optimization_completed = pyqtSignal(dict)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.should_stop = False
        self.checkpoint_system = OptimizationCheckpoint(
            save_interval=config.get('checkpoint_interval', 25),
            max_checkpoints=config.get('max_checkpoints', 5)
        )
        
    def run(self):
        """Main optimization loop"""
        try:
            # Extract configuration
            models = self.config['models']
            model_indices = self.config['model_indices']
            directions = self.config['directions']
            feature_bounds = self.config['feature_bounds']
            fixed_features = self.config['fixed_features']
            population_size = self.config['population_size']
            n_generations = self.config['n_generations']
            feature_names = self.config['feature_names']
            
            # Determine optimization type based on number of models
            is_single_objective = len(models) == 1
            if is_single_objective:
                self.status_updated.emit("Starting single-objective optimization...")
            else:
                self.status_updated.emit("Starting multi-objective optimization...")
            
            # Prepare bounds
            n_var = len(feature_names)
            xl = np.array([bounds[0] for bounds in feature_bounds])
            xu = np.array([bounds[1] for bounds in feature_bounds])
            
            self.status_updated.emit(f"Initializing problem with {n_var} variables...")
            
            # Extract feature type information
            feature_types = self.config.get('feature_types', ['continuous'] * n_var)
            categorical_ranges = self.config.get('categorical_ranges', {})
            
            # Debug: log fixed features info
            if fixed_features:
                self.status_updated.emit(f"Fixed features configured: {len(fixed_features)} features")
                for feature_idx, value in fixed_features.items():
                    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
                    self.status_updated.emit(f"  • {feature_name} = {value}")
            
            # Extract diversity noise scale from config (advanced parameter)
            diversity_noise_scale = self.config.get('diversity_noise_scale', 1e-10)
            
            # Prepare robust and constraint configurations
            robust_config = {
                'enable_robust_optimization': self.config.get('enable_robust_optimization', False),
                'robust_method': self.config.get('robust_method', 'bootstrap'),
                'robust_samples': self.config.get('robust_samples', 10),
                'robust_noise_level': self.config.get('robust_noise_level', 0.01)
            }
            
            constraint_config = {
                'enable_explicit_constraints': self.config.get('enable_explicit_constraints', False),
                'adaptive_penalty_coefficients': self.config.get('adaptive_penalty_coefficients', True)
            }
            
            # Create enhanced problem instance with robust optimization and constraint handling
            problem = MLProblem(
                models=models,
                model_indices=model_indices,
                directions=directions,
                n_var=n_var,
                xl=xl,
                xu=xu,
                feature_names=feature_names,
                fixed_features=fixed_features,
                feature_types=feature_types,
                categorical_ranges=categorical_ranges,
                diversity_noise_scale=diversity_noise_scale,
                robust_config=robust_config,
                constraint_config=constraint_config
            )
            
            # Extract algorithm parameters
            crossover_prob = self.config.get('crossover_prob', 0.9)
            crossover_eta = self.config.get('crossover_eta', 15.0)
            mutation_prob = self.config.get('mutation_prob', None)
            mutation_eta = self.config.get('mutation_eta', 20.0)
            eliminate_duplicates = self.config.get('eliminate_duplicates', True)
            random_seed = self.config.get('random_seed', None)
            verbose = self.config.get('verbose', False)
            
            # Configure optimization strategy based on objective count
            if is_single_objective:
                # For single-objective optimization, use standard settings
                eliminate_duplicates_adjusted = eliminate_duplicates
                min_pop_size = max(30, len(feature_names) * 2)  # Smaller population for single objective
                self.status_updated.emit("Configuring single-objective optimization strategy...")  
            else:
                # CRITICAL FIX: 调整重复解消除策略，避免帕累托前沿收缩为单点
                # 对于多目标优化，过度的重复解消除会严重限制解的多样性
                if eliminate_duplicates and len(models) >= 2:
                    # 对于多目标情况，使用更宽松的重复检测，保持解的多样性
                    eliminate_duplicates_adjusted = False
                    self.status_updated.emit("⚠️  Adjusting duplicate elimination strategy to maintain Pareto front diversity")
                else:
                    eliminate_duplicates_adjusted = eliminate_duplicates
                
                # 自动调整种群大小，确保有足够的多样性
                min_pop_size = max(50, len(models) * 20)  # 每个目标至少20个个体
                self.status_updated.emit("Configuring multi-objective optimization strategy...")
            
            # Adjust population size
            if population_size < min_pop_size:
                population_size_adjusted = min_pop_size
                self.status_updated.emit(f"⚠️  Population size adjusted to {min_pop_size} to ensure convergence")
            else:
                population_size_adjusted = population_size
            
            # Set random seed if specified
            if random_seed is not None:
                np.random.seed(random_seed)
                self.status_updated.emit(f"Random seed set to: {random_seed}")
            
            # Set up algorithm with proper mixed-variable support
            if is_single_objective:
                self.status_updated.emit("Configuring GA algorithm for single-objective optimization...")
            else:
                self.status_updated.emit("Configuring NSGA-II algorithm for multi-objective optimization...")
            
            # Check if we have mixed variables (discrete + continuous)
            has_mixed_variables = any(ftype in ['binary', 'categorical'] for ftype in feature_types)
            
            # 使用改进的配置，专门针对混合变量优化
            try:
                
                if has_mixed_variables:
                    # For mixed variables, use custom sampling and repair
                    self.status_updated.emit("🔧 Detected mixed variables, using specialized constraint handling strategy...") 
                    
                    # Create mixed-variable sampling
                    sampling = MixedVariableSampling(
                        feature_types=feature_types,
                        categorical_ranges=categorical_ranges
                    )
                    
                    # Create enhanced repair operator for constraint handling
                    repair = EnhancedBoundaryRepair(
                        feature_types=feature_types,
                        categorical_ranges=categorical_ranges,
                        fixed_features=fixed_features
                    )
                    
                    self.status_updated.emit(f"   📋 Feature type distribution:")
                    type_counts = {}
                    for ftype in feature_types:
                        type_counts[ftype] = type_counts.get(ftype, 0) + 1
                    for ftype, count in type_counts.items():
                        self.status_updated.emit(f"      {ftype}: {count} features")
                else:
                    # For continuous variables only, use standard operators
                    try:
                        from pymoo.operators.sampling.rnd import FloatRandomSampling
                    except ImportError:
                        # pymoo 0.5.0 path
                        from pymoo.operators.sampling.random_sampling import FloatRandomSampling
                    sampling = FloatRandomSampling()
                    repair = None
                    self.status_updated.emit("📈 All features are continuous, using standard optimization strategy")
                
                # Configure crossover operator
                crossover = SBX(prob=crossover_prob, eta=crossover_eta) if SBX else None
                
                # Configure mutation operator
                if mutation_prob is None:
                    mutation_prob_adjusted = min(0.3, max(0.1, 1.0 / len(feature_names)))
                else:
                    mutation_prob_adjusted = mutation_prob
                
                mutation = PM(prob=mutation_prob_adjusted, eta=mutation_eta) if PM else None
                
                # Create algorithm based on selection and objective count
                selected_algorithm = self.config.get('algorithm', 'NSGA-II')
                
                if selected_algorithm == "GA (Single-Objective)" or is_single_objective:
                    # Use Genetic Algorithm for single objective
                    GA = NSGA2  # Use NSGA2 as GA for single objective optimization
                    self.status_updated.emit("Using NSGA-II for single-objective optimization")
                    algorithm = GA(
                        pop_size=population_size_adjusted,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"✅ Created GA algorithm for single-objective optimization")
                elif selected_algorithm == "NSGA-III" and NSGA3 is not None:
                    # Use NSGA-III for many-objective problems
                    try:
                        from pymoo.util.ref_dirs import get_reference_directions
                    except ImportError:
                        # Try pymoo 0.5.0 path
                        try:
                            from pymoo.factory import get_reference_directions
                        except ImportError:
                            # If reference directions not available, create simple uniform directions
                            def get_reference_directions(method, n_obj, n_partitions=12):
                                if n_obj == 2:
                                    return np.array([[1, 0], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0, 1]])
                                elif n_obj == 3:
                                    return np.array([[1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])
                                else:
                                    # Simple uniform grid for higher dimensions
                                    refs = []
                                    for i in range(n_obj):
                                        ref = np.zeros(n_obj)
                                        ref[i] = 1.0
                                        refs.append(ref)
                                    return np.array(refs)
                            self.status_updated.emit("⚠️ Using simplified reference directions")
                    # Create reference directions for NSGA-III
                    if len(models) <= 3:
                        ref_dirs = get_reference_directions("das-dennis", len(models), n_partitions=12)
                    else:
                        ref_dirs = get_reference_directions("das-dennis", len(models), n_partitions=6)
                    
                    algorithm = NSGA3(
                        pop_size=population_size_adjusted,
                        ref_dirs=ref_dirs,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"✅ Created NSGA-III algorithm with {len(ref_dirs)} reference directions")
                elif selected_algorithm == "NSGA-III" and NSGA3 is None:
                    # Fallback to NSGA-II if NSGA3 not available
                    algorithm = NSGA2(
                        pop_size=population_size_adjusted,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"⚠️ NSGA-III not available in this pymoo version, using NSGA-II instead")
                elif selected_algorithm == "SPEA2" and SPEA2 is not None:
                    # Use SPEA2 algorithm
                    algorithm = SPEA2(
                        pop_size=population_size_adjusted,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"✅ Created SPEA2 algorithm for multi-objective optimization")
                elif selected_algorithm == "SPEA2" and SPEA2 is None:
                    # Fallback to NSGA-II if SPEA2 not available
                    algorithm = NSGA2(
                        pop_size=population_size_adjusted,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"⚠️ SPEA2 not available in this pymoo version, using NSGA-II instead")
                elif selected_algorithm == "MOEA/D" and MOEAD is not None:
                    # Use MOEA/D algorithm
                    try:
                        # MOEA/D requires reference directions for decomposition
                        try:
                            from pymoo.util.ref_dirs import get_reference_directions
                        except ImportError:
                            from pymoo.factory import get_reference_directions
                        
                        if len(models) <= 3:
                            ref_dirs = get_reference_directions("das-dennis", len(models), n_partitions=12)
                        else:
                            ref_dirs = get_reference_directions("das-dennis", len(models), n_partitions=6)
                        
                        algorithm = MOEAD(
                            ref_dirs=ref_dirs,
                            sampling=sampling,
                            crossover=crossover,
                            mutation=mutation,
                            repair=repair
                        )
                        self.status_updated.emit(f"✅ Created MOEA/D algorithm with {len(ref_dirs)} decomposition vectors")
                    except Exception as e:
                        # Fallback to NSGA-II if MOEA/D setup fails
                        algorithm = NSGA2(
                            pop_size=population_size_adjusted,
                            sampling=sampling,
                            crossover=crossover,
                            mutation=mutation,
                            repair=repair,
                            eliminate_duplicates=eliminate_duplicates_adjusted
                        )
                        self.status_updated.emit(f"⚠️ MOEA/D setup failed ({str(e)}), using NSGA-II instead")
                elif selected_algorithm == "MOEA/D" and MOEAD is None:
                    # Fallback to NSGA-II if MOEA/D not available
                    algorithm = NSGA2(
                        pop_size=population_size_adjusted,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"⚠️ MOEA/D not available in this pymoo version, using NSGA-II instead")
                elif selected_algorithm == "CMA-ES" and CMA_ES is not None:
                    # Use CMA-ES for continuous optimization
                    try:
                        algorithm = CMA_ES(
                            x0=np.mean([xl, xu], axis=0),  # Initial center point
                            sigma=0.1,  # Initial standard deviation
                            restarts=0  # No restarts for now
                        )
                        self.status_updated.emit(f"✅ Created CMA-ES algorithm for continuous optimization")
                    except Exception as e:
                        # Fallback to NSGA-II if CMA-ES setup fails
                        algorithm = NSGA2(
                            pop_size=population_size_adjusted,
                            sampling=sampling,
                            crossover=crossover,
                            mutation=mutation,
                            repair=repair,
                            eliminate_duplicates=eliminate_duplicates_adjusted
                        )
                        self.status_updated.emit(f"⚠️ CMA-ES setup failed ({str(e)}), using NSGA-II instead")
                elif selected_algorithm == "CMA-ES" and CMA_ES is None:
                    # Fallback to NSGA-II if CMA-ES not available
                    algorithm = NSGA2(
                        pop_size=population_size_adjusted,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"⚠️ CMA-ES not available in this pymoo version, using NSGA-II instead")
                elif selected_algorithm == "Differential Evolution" and DE is not None:
                    # Use Differential Evolution
                    try:
                        # Configure DE-specific operators if available
                        if DifferentialEvolution:
                            crossover_de = DifferentialEvolution(weight=mutation_eta, prob=crossover_prob)
                        else:
                            crossover_de = crossover
                        
                        algorithm = DE(
                            pop_size=population_size_adjusted,
                            sampling=sampling,
                            variant="DE/rand/1/bin",
                            CR=crossover_prob,
                            F=mutation_eta,
                            repair=repair
                        )
                        self.status_updated.emit(f"✅ Created DE algorithm with CR={crossover_prob:.2f}, F={mutation_eta:.2f}")
                    except Exception as e:
                        # Fallback to NSGA-II if DE setup fails
                        algorithm = NSGA2(
                            pop_size=population_size_adjusted,
                            sampling=sampling,
                            crossover=crossover,
                            mutation=mutation,
                            repair=repair,
                            eliminate_duplicates=eliminate_duplicates_adjusted
                        )
                        self.status_updated.emit(f"⚠️ DE setup failed ({str(e)}), using NSGA-II instead")
                elif selected_algorithm == "Differential Evolution" and DE is None:
                    # Fallback to NSGA-II if DE not available
                    algorithm = NSGA2(
                        pop_size=population_size_adjusted,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"⚠️ DE not available in this pymoo version, using NSGA-II instead")
                else:
                    # Default to NSGA-II for multi-objective
                    algorithm = NSGA2(
                        pop_size=population_size_adjusted,
                        sampling=sampling,
                        crossover=crossover,
                        mutation=mutation,
                        repair=repair,  # CRITICAL: This ensures constraints are enforced during optimization
                        eliminate_duplicates=eliminate_duplicates_adjusted
                    )
                    self.status_updated.emit(f"✅ Created NSGA-II algorithm for multi-objective optimization")
                
                if has_mixed_variables:
                    self.status_updated.emit(f"✅ Mixed variable algorithm configuration completed: mutation rate={mutation_prob_adjusted:.3f}")
                else:
                    self.status_updated.emit(f"✅ Continuous variable algorithm configuration completed: mutation rate={mutation_prob_adjusted:.3f}")
                
            except ImportError:
                # 如果无法导入高级算子，使用基础配置
                if has_mixed_variables:
                    # Try to create basic repair at least
                    try:
                        repair = EnhancedBoundaryRepair(
                            feature_types=feature_types,
                            categorical_ranges=categorical_ranges,
                            fixed_features=fixed_features
                        )
                        if is_single_objective:
                            try:
                                from pymoo.algorithms.soo.nonconvex.ga import GA
                            except ImportError:
                                GA = NSGA2  # Use NSGA2 as fallback
                            algorithm = GA(
                                pop_size=population_size_adjusted,
                                repair=repair,
                                eliminate_duplicates=False
                            )
                        else:
                            algorithm = NSGA2(
                                pop_size=population_size_adjusted,
                                repair=repair,
                                eliminate_duplicates=False
                            )
                        self.status_updated.emit("⚠️  Using basic operator configuration + constraint repair")
                    except Exception:
                        if is_single_objective:
                            from pymoo.algorithms.soo.nonconvex.ga import GA
                            algorithm = GA(
                                pop_size=population_size_adjusted,
                                eliminate_duplicates=False
                            )
                        else:
                            algorithm = NSGA2(
                                pop_size=population_size_adjusted,
                                eliminate_duplicates=False
                            )
                        self.status_updated.emit("⚠️  Using basic configuration (constraints may not be fully satisfied)")
                else:
                    if is_single_objective:
                        from pymoo.algorithms.soo.nonconvex.ga import GA
                        algorithm = GA(
                            pop_size=population_size_adjusted,
                            eliminate_duplicates=False
                        )
                    else:
                        algorithm = NSGA2(
                            pop_size=population_size_adjusted,
                            eliminate_duplicates=False
                        )
                    self.status_updated.emit("⚠️  Using basic operator configuration")
            
            if verbose:
                self.status_updated.emit(f"Algorithm configured (enhanced diversity mode):")
                self.status_updated.emit(f"- Population size: {population_size_adjusted} (Original: {population_size})")
                self.status_updated.emit(f"- Generations: {n_generations}")
                self.status_updated.emit(f"- Objectives: {len(models)}")
                self.status_updated.emit(f"- Variables: {len(feature_names)}")
                self.status_updated.emit(f"- Categorical features: {len([t for t in feature_types if t in ['binary', 'categorical']])} features")
                self.status_updated.emit(f"- Eliminate duplicates: {eliminate_duplicates_adjusted} (Original: {eliminate_duplicates})")
                if fixed_features:
                    self.status_updated.emit(f"- Fixed features: {len(fixed_features)} / {len(feature_names)}")
                    free_features = len(feature_names) - len(fixed_features)
                    self.status_updated.emit(f"- Free features: {free_features}")
                    if free_features < 2:
                        self.status_updated.emit("⚠️  Few free features, which may affect Pareto front diversity")
                if random_seed is not None:
                    self.status_updated.emit(f"- Random seed: {random_seed}")
                
                # 计算搜索空间大小估计
                search_space_size = 1.0
                for i, bounds in enumerate(feature_bounds):
                    if i not in fixed_features:
                        search_space_size *= (bounds[1] - bounds[0])
                self.status_updated.emit(f"- Estimated search space size: {search_space_size:.2e}")
                
                if search_space_size < 1e-6:
                    self.status_updated.emit("⚠️  Search space may be too small, please check feature boundary settings")
            
            # Set termination criteria with optional early stopping
            use_early_stopping = self.config.get('enable_early_stopping', True)
            if use_early_stopping and len(models) > 1:
                # For now, disable early stopping to ensure full generations run
                # TODO: Fix early stopping mechanism in future version
                termination = get_termination("n_gen", n_generations)
                early_stop = None
                self.status_updated.emit(f"🔄 Running standard {n_generations} generations (early stopping temporarily disabled to ensure full run)")
            else:
                # Standard generation-based termination
                termination = get_termination("n_gen", n_generations)
                early_stop = None
            
            # Create enhanced callback for progress updates and checkpointing
            callback = OptimizationCallback(self)
            
            self.status_updated.emit("Running NSGA-II optimization...")
            
            # Use pymoo 0.5.0 API - simple and reliable approach
            from pymoo.optimize import minimize
            
            res = minimize(
                problem,
                algorithm,
                termination,
                callback=callback,
                verbose=verbose,
                seed=random_seed
            )
            
            if not self.should_stop:
                # Process results
                self.status_updated.emit("Processing optimization results...")
                
                # Get final results from pymoo 0.5.0 result (different structure for single vs multi-objective)
                if is_single_objective:
                    # For single-objective, res.F is 1D array, res.X is 2D array [pop_size, n_vars]
                    # Get the best solution and entire population for diversity
                    if hasattr(res, 'X') and res.X is not None:
                        if res.X.ndim == 1:
                            # Single best solution
                            pareto_solutions = res.X.reshape(1, -1)  # Shape: [1, n_vars]
                        else:
                            # Population solutions - take top 10 or all if fewer
                            n_solutions_to_keep = min(10, len(res.X))
                            if hasattr(res, 'F') and res.F is not None:
                                # Sort by objective value and take best solutions
                                sorted_indices = np.argsort(res.F)[:n_solutions_to_keep]
                                pareto_solutions = res.X[sorted_indices]
                            else:
                                pareto_solutions = res.X[:n_solutions_to_keep]
                    else:
                        raise ValueError("Single-objective optimization failed to find solutions")
                    
                    # Get corresponding objective values
                    if hasattr(res, 'F') and res.F is not None:
                        if res.F.ndim == 0:
                            # Single objective value
                            pareto_front = np.array([[res.F]])  # Shape: [1, 1]
                        elif res.F.ndim == 1:
                            if len(pareto_solutions) == 1:
                                # Single best solution
                                pareto_front = np.array([[res.F[0]]])  # Take best objective
                            else:
                                # Multiple solutions
                                n_solutions = len(pareto_solutions)
                                pareto_front = res.F[:n_solutions].reshape(-1, 1)  # Shape: [n_solutions, 1]
                        else:
                            pareto_front = res.F[:len(pareto_solutions)]
                    else:
                        raise ValueError("Single-objective optimization failed to get objective values")
                else:
                    # For multi-objective, standard Pareto front processing
                    pareto_front = res.F
                    pareto_solutions = res.X
                
                # Validate results
                if pareto_front is None or len(pareto_front) == 0:
                    raise ValueError("Optimization failed to find any solutions")
                
                # Apply fixed features to solutions (they're not automatically included in res.X)
                if fixed_features:
                    self.status_updated.emit("Applying fixed feature values to solutions...")
                    for i in range(len(pareto_solutions)):
                        for feature_idx, fixed_value in fixed_features.items():
                            pareto_solutions[i, feature_idx] = fixed_value
                
                # Check if additional repair is needed (only if repair wasn't used during optimization)
                has_mixed_variables = any(ftype in ['binary', 'categorical'] for ftype in feature_types)
                if has_mixed_variables:
                    self.status_updated.emit("Validating categorical constraints in final solutions...")
                    
                    # Quick validation check
                    violations_found = False
                    if len(pareto_solutions) > 0:
                        sample_violations = 0
                        for j, feature_type in enumerate(feature_types):
                            if j >= pareto_solutions.shape[1]:
                                break
                            if feature_type == 'binary' and pareto_solutions[0, j] not in [0.0, 1.0]:
                                sample_violations += 1
                                violations_found = True
                            elif feature_type == 'categorical' and j in categorical_ranges:
                                if pareto_solutions[0, j] not in categorical_ranges[j]:
                                    sample_violations += 1
                                    violations_found = True
                    
                    if violations_found:
                        self.status_updated.emit("⚠️  Constraint violations detected, applying final repair...")
                        
                        # Apply final repair only if needed
                        try:
                            pareto_solutions_repaired = pareto_solutions.copy()
                            
                            for i in range(len(pareto_solutions_repaired)):
                                for j in range(len(feature_types)):
                                    if j >= pareto_solutions_repaired.shape[1]:
                                        break
                                        
                                    # Apply fixed features first
                                    if j in fixed_features:
                                        pareto_solutions_repaired[i, j] = fixed_features[j]
                                        continue
                                    
                                    # Get feature type
                                    feature_type = feature_types[j] if j < len(feature_types) else 'continuous'
                                    
                                    # Apply type-specific constraints
                                    if feature_type == 'binary':
                                        # Binary: round to nearest 0 or 1
                                        value = pareto_solutions_repaired[i, j]
                                        pareto_solutions_repaired[i, j] = 1.0 if value >= 0.5 else 0.0
                                    elif feature_type == 'categorical' and j in categorical_ranges:
                                        # Categorical: find closest valid value
                                        value = pareto_solutions_repaired[i, j]
                                        valid_values = categorical_ranges[j]
                                        distances = [abs(value - v) for v in valid_values]
                                        closest_idx = np.argmin(distances)
                                        pareto_solutions_repaired[i, j] = valid_values[closest_idx]
                            
                            pareto_solutions = pareto_solutions_repaired
                            self.status_updated.emit("✅ 最终约束修复完成")
                            
                        except Exception as repair_error:
                            self.status_updated.emit(f"Warning: Final repair failed: {repair_error}")
                    else:
                        self.status_updated.emit("✅ 所有分类约束已满足（优化过程中正确处理）")
                
                # Convert back to original objective values (multiply by direction to restore original scale)
                original_objectives = pareto_front.copy()
                for i, direction in enumerate(directions):
                    original_objectives[:, i] *= direction  # Convert back from minimization
                
                # 详细分析优化结果质量
                n_solutions = len(pareto_front)
                if is_single_objective:
                    self.status_updated.emit(f"分析单目标优化结果质量...")
                else:
                    self.status_updated.emit(f"分析帕累托前沿质量...")
                
                # 检查目标值的多样性
                diversity_info = []
                for i in range(len(models)):
                    obj_values = original_objectives[:, i]
                    obj_range = np.max(obj_values) - np.min(obj_values)
                    obj_std = np.std(obj_values)
                    obj_mean = np.mean(obj_values)
                    
                    diversity_info.append({
                        'range': obj_range,
                        'std': obj_std,
                        'mean': obj_mean,
                        'cv': obj_std / abs(obj_mean) if obj_mean != 0 else 0
                    })
                    
                    if is_single_objective:
                        self.status_updated.emit(f"Objective values: Best={np.min(obj_values):.6f}, Mean={obj_mean:.6f}, Std={obj_std:.6f}")
                    else:
                        self.status_updated.emit(f"Objective {i+1}: Range={obj_range:.6f}, Std={obj_std:.6f}")
                
                # 检查解的多样性
                solution_diversity = []
                for i in range(len(feature_names)):
                    if i not in fixed_features:
                        feature_values = pareto_solutions[:, i]
                        feature_range = np.max(feature_values) - np.min(feature_values)
                        solution_diversity.append(feature_range)
                
                avg_solution_diversity = np.mean(solution_diversity) if solution_diversity else 0
                self.status_updated.emit(f"Average solution diversity: {avg_solution_diversity:.6f}")
                
                # 优化结果质量评估
                if is_single_objective:
                    if n_solutions == 1:
                        self.status_updated.emit("✅ Single-objective optimization found the optimal solution")
                    else:
                        self.status_updated.emit(f"✅ Single-objective optimization found {n_solutions} high-quality solutions")
                else:
                    # 多目标优化的质量评估
                    if n_solutions == 1:
                        self.status_updated.emit("⚠️  Pareto front only has 1 point - this may indicate:")
                        self.status_updated.emit("   1. Search space too restricted")
                        self.status_updated.emit("   2. Too many fixed features")
                        self.status_updated.emit("   3. Target function returns same value")
                        self.status_updated.emit("   4. Duplicate elimination too strict")
                    elif n_solutions < 10:
                        self.status_updated.emit(f"⚠️  Pareto front has few solutions ({n_solutions})")
                    else:
                        self.status_updated.emit(f"✅ Pareto front contains {n_solutions} diverse solutions")
                
                results = {
                    'pareto_front': original_objectives,
                    'pareto_solutions': pareto_solutions,
                    'feature_names': feature_names,
                    'n_solutions': n_solutions,
                    'n_objectives': len(models),
                    'fixed_features': fixed_features,
                    'convergence_history': getattr(res, 'history', None),
                    'model_names': self.config.get('model_names', [f'Model {i+1}' for i in range(len(models))]),
                    'objective_names': self.config.get('objective_names', [f'Objective {i+1}' for i in range(len(models))]),
                    'diversity_info': diversity_info,
                    'solution_diversity': avg_solution_diversity,
                    'is_single_objective': is_single_objective
                }
                
                # Report optimization completion with quality assessment
                if is_single_objective:
                    if n_solutions == 1:
                        completion_msg = f"✅ Single-objective optimization completed successfully! Found the optimal solution."
                    else:
                        completion_msg = f"✅ Single-objective optimization completed successfully! Found {n_solutions} high-quality solutions."
                else:
                    if n_solutions >= 10:
                        completion_msg = f"✅ Multi-objective optimization completed successfully! Found {n_solutions} high-quality Pareto optimal solutions."
                    elif n_solutions > 1:
                        completion_msg = f"⚠️  Multi-objective optimization completed, found {n_solutions} Pareto optimal solutions (consider checking parameter settings)."
                    else:
                        completion_msg = f"⚠️  Multi-objective optimization completed, but only found {n_solutions} solutions (consider adjusting algorithm parameters)."
                
                self.status_updated.emit(completion_msg)
                self.optimization_completed.emit(results)
            else:
                self.status_updated.emit("Optimization stopped by user.")
                
        except Exception as e:
            error_msg = f"Optimization error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
    
    def stop(self):
        """Gracefully stop the optimization"""
        self.should_stop = True
        self.status_updated.emit("Stopping optimization...")


class MultiObjectiveOptimizationModule(QWidget):
    """
    UI for the Single & Multi-Objective Optimization module.
    Supports both single-objective optimization (1 model) and multi-objective optimization (2+ models).
    """
    
    def __init__(self):
        super().__init__()
        
        # Check dependencies
        self.check_dependencies()
        
        if not PYMOO_AVAILABLE:
            self.show_dependency_error()
            return
        
        # Initialize data storage
        self.models_data = []  # List of loaded model data
        self.combined_features = []
        self.model_indices = []  # List of feature indices for each model
        self.worker = None
        
        # Initialize UI
        self.init_ui()
        
        # Initialize algorithm settings display
        self.update_algorithm_settings_display()
    
    def check_dependencies(self):
        """Check and report dependency status"""
        dependency_status = []
        
        # Check PyQt5
        try:
            from PyQt5 import QtCore
            pyqt_version = QtCore.QT_VERSION_STR
            dependency_status.append(f"✅ PyQt5: {pyqt_version}")
        except ImportError:
            dependency_status.append("❌ PyQt5: Not available")
        
        # Check pymoo with version-specific features
        if PYMOO_AVAILABLE:
            dependency_status.append(f"✅ pymoo: {PYMOO_VERSION}")
            
            # Check version-specific features
            if PYMOO_VERSION == '0.5.0':
                dependency_status.append("   ℹ️  pymoo 0.5.0 detected - some features may use fallback implementations")
                if SPEA2 is None:
                    dependency_status.append("   ⚠️  SPEA2 algorithm not available in this pymoo version")
            elif PYMOO_VERSION.startswith('0.6'):
                dependency_status.append("   ℹ️  pymoo 0.6+ detected - full feature support")
        else:
            dependency_status.append("❌ pymoo: Not available - install with 'pip install pymoo'")
        
        # Check matplotlib
        try:
            import matplotlib
            dependency_status.append(f"✅ matplotlib: {matplotlib.__version__}")
        except ImportError:
            dependency_status.append("❌ matplotlib: Not available")
        
        # Check pandas
        try:
            dependency_status.append(f"✅ pandas: {pd.__version__}")
        except:
            dependency_status.append("❌ pandas: Not available")
        
        # Check numpy
        try:
            dependency_status.append(f"✅ numpy: {np.__version__}")
        except:
            dependency_status.append("❌ numpy: Not available")
        
        # Check joblib
        try:
            import joblib as jl
            dependency_status.append(f"✅ joblib: {jl.__version__}")
        except:
            dependency_status.append("❌ joblib: Not available")
        
        print("📦 Dependency Check Results:")
        for status in dependency_status:
            print(f"   {status}")
        
        return dependency_status
        
    def show_dependency_error(self):
        """Show error message for missing pymoo dependency"""
        layout = QVBoxLayout(self)
        error_label = QLabel("Error: pymoo library is not installed.\n\n"
                           "Please install it using:\npip install pymoo")
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("color: red; font-size: 14px;")
        layout.addWidget(error_label)
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Create panels
        control_panel = self.create_control_panel()
        results_panel = self.create_results_panel()
        
        main_splitter.addWidget(control_panel)
        main_splitter.addWidget(results_panel)
        main_splitter.setSizes([400, 800])
        
    def create_control_panel(self) -> QWidget:
        """Create the left-side control panel"""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        
        # Model Loading Section
        model_group = QGroupBox("Model Loading & Objectives")
        model_layout = QVBoxLayout(model_group)
        
        # Add model button
        add_model_btn = QPushButton("+ Add Model/Objective")
        add_model_btn.clicked.connect(self.add_model_row)
        model_layout.addWidget(add_model_btn)
        
        # Scrollable area for models
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.models_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        model_layout.addWidget(scroll_area)
        
        # Store model widgets for dynamic management
        self.model_widgets = []
        
        # Add initial model row (single model minimum)
        self.add_model_row()
        
        layout.addWidget(model_group)
        
        # Feature Bounds Section
        bounds_group = QGroupBox("Feature Bounds Configuration")
        bounds_layout = QVBoxLayout(bounds_group)
        
        # Range detection controls
        range_controls = QHBoxLayout()
        self.use_original_ranges_check = QCheckBox("Use original feature ranges (auto-detected)")
        self.use_original_ranges_check.setChecked(True)
        self.use_original_ranges_check.stateChanged.connect(self.on_range_mode_changed)
        range_controls.addWidget(self.use_original_ranges_check)
        
        # Range detection status
        self.range_status_label = QLabel("⏳ Waiting for models...")
        self.range_status_label.setStyleSheet("color: gray; font-style: italic;")
        range_controls.addWidget(self.range_status_label)
        
        # Refresh bounds button
        refresh_bounds_btn = QPushButton("🔄 Refresh Bounds")
        refresh_bounds_btn.clicked.connect(self.refresh_feature_bounds)
        refresh_bounds_btn.setMaximumWidth(120)
        range_controls.addWidget(refresh_bounds_btn)
        
        range_controls.addStretch()
        bounds_layout.addLayout(range_controls)
        
        # Create feature bounds table
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(5)
        self.feature_table.setHorizontalHeaderLabels(["Feature Name", "Min Value", "Max Value", "Fixed", "Fixed Value"])
        self.feature_table.horizontalHeader().setStretchLastSection(True)
        self.feature_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        bounds_layout.addWidget(self.feature_table)
        
        layout.addWidget(bounds_group)
        
        # Algorithm Status Indicator
        self.algo_status_label = QLabel("🎯 Current Algorithm: GA (Genetic Algorithm) - Single-Objective Mode")
        self.algo_status_label.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                border: 2px solid #2196f3;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
                color: #1976d2;
            }
        """)
        self.algo_status_label.setWordWrap(True)
        layout.addWidget(self.algo_status_label)
        
        # Algorithm Settings Section (will be updated dynamically)
        self.algo_group = QGroupBox("Genetic Algorithm Settings")
        algo_layout = QFormLayout(self.algo_group)
        
        # Basic Parameters
        self.population_spin = QSpinBox()
        self.population_spin.setMinimum(10)
        self.population_spin.setMaximum(500)
        self.population_spin.setValue(50)
        self.population_spin.setToolTip("Number of individuals in each generation (larger = more diverse but slower)")
        self.population_spin.valueChanged.connect(lambda: self._mark_param_modified('population_size'))
        algo_layout.addRow("Population Size:", self.population_spin)
        
        self.generations_spin = QSpinBox()
        self.generations_spin.setMinimum(10)
        self.generations_spin.setMaximum(2000)
        self.generations_spin.setValue(100)
        self.generations_spin.setToolTip("Number of generations to evolve (more = better convergence but longer time)")
        algo_layout.addRow("Number of Generations:", self.generations_spin)
        
        # Crossover Parameters
        self.crossover_prob_spin = QDoubleSpinBox()
        self.crossover_prob_spin.setMinimum(0.1)
        self.crossover_prob_spin.setMaximum(1.0)
        self.crossover_prob_spin.setValue(0.9)
        self.crossover_prob_spin.setSingleStep(0.1)
        self.crossover_prob_spin.setDecimals(2)
        self.crossover_prob_spin.setToolTip("Probability of crossover between parents (0.8-0.95 recommended)")
        self.crossover_prob_spin.valueChanged.connect(lambda: self._mark_param_modified('crossover_prob'))
        algo_layout.addRow("Crossover Probability:", self.crossover_prob_spin)
        
        self.crossover_eta_spin = QDoubleSpinBox()
        self.crossover_eta_spin.setMinimum(1.0)
        self.crossover_eta_spin.setMaximum(50.0)
        self.crossover_eta_spin.setValue(15.0)
        self.crossover_eta_spin.setSingleStep(1.0)
        self.crossover_eta_spin.setDecimals(1)
        self.crossover_eta_spin.setToolTip("Crossover distribution index (lower = more exploration, higher = more exploitation)")
        self.crossover_eta_spin.valueChanged.connect(lambda: self._mark_param_modified('crossover_eta'))
        algo_layout.addRow("Crossover Eta (η_c):", self.crossover_eta_spin)
        
        # Mutation Parameters
        self.mutation_prob_spin = QDoubleSpinBox()
        self.mutation_prob_spin.setMinimum(0.01)
        self.mutation_prob_spin.setMaximum(0.5)
        self.mutation_prob_spin.setValue(0.1)
        self.mutation_prob_spin.setSingleStep(0.01)
        self.mutation_prob_spin.setDecimals(3)
        self.mutation_prob_spin.setToolTip("Probability of mutation (typically 1/n_vars, auto-calculated if 0)")
        self.mutation_prob_spin.valueChanged.connect(lambda: self._mark_param_modified('mutation_prob'))
        algo_layout.addRow("Mutation Probability:", self.mutation_prob_spin)
        
        self.mutation_eta_spin = QDoubleSpinBox()
        self.mutation_eta_spin.setMinimum(1.0)
        self.mutation_eta_spin.setMaximum(100.0)
        self.mutation_eta_spin.setValue(20.0)
        self.mutation_eta_spin.setSingleStep(1.0)
        self.mutation_eta_spin.setDecimals(1)
        self.mutation_eta_spin.setToolTip("Mutation distribution index (lower = more perturbation, higher = fine-tuning)")
        self.mutation_eta_spin.valueChanged.connect(lambda: self._mark_param_modified('mutation_eta'))
        algo_layout.addRow("Mutation Eta (η_m):", self.mutation_eta_spin)
        
        # Algorithm Selection
        self.algorithm_combo = QComboBox()
        # Add algorithms based on availability
        algorithms = ["NSGA-II"]
        if NSGA3:
            algorithms.append("NSGA-III")
        if SPEA2:
            algorithms.append("SPEA2")
        if MOEAD:
            algorithms.append("MOEA/D")
        if CMA_ES:
            algorithms.append("CMA-ES")
        if DE:
            algorithms.append("Differential Evolution")
        algorithms.append("GA (Single-Objective)")
        
        self.algorithm_combo.addItems(algorithms)
        self.algorithm_combo.setCurrentText("NSGA-II")
        self.algorithm_combo.setToolTip("Optimization algorithm to use")
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        algo_layout.addRow("Algorithm:", self.algorithm_combo)
        
        # Selection and Survival
        self.selection_combo = QComboBox()
        self.selection_combo.addItems(["Tournament Selection", "Random Selection"])
        self.selection_combo.setCurrentText("Tournament Selection")
        self.selection_combo.setToolTip("Selection method for choosing parents")
        algo_layout.addRow("Selection Method:", self.selection_combo)
        
        self.tournament_size_spin = QSpinBox()
        self.tournament_size_spin.setMinimum(2)
        self.tournament_size_spin.setMaximum(10)
        self.tournament_size_spin.setValue(2)
        self.tournament_size_spin.setToolTip("Tournament size for tournament selection (2 = binary tournament)")
        algo_layout.addRow("Tournament Size:", self.tournament_size_spin)
        
        # Convergence and Termination
        self.eliminate_duplicates_check = QCheckBox()
        self.eliminate_duplicates_check.setChecked(False)  # 默认关闭以保持帕累托前沿多样性
        self.eliminate_duplicates_check.setToolTip("Remove duplicate solutions (注意：启用可能会减少帕累托前沿的解数量)")
        self.eliminate_duplicates_check.stateChanged.connect(lambda: self._mark_param_modified('eliminate_duplicates'))
        algo_layout.addRow("Eliminate Duplicates:", self.eliminate_duplicates_check)
        
        # Advanced Parameters
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout(advanced_group)
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setMinimum(-1)
        self.seed_spin.setMaximum(999999)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("Random")
        self.seed_spin.setToolTip("Random seed for reproducibility (-1 for random)")
        advanced_layout.addRow("Random Seed:", self.seed_spin)
        
        self.verbose_check = QCheckBox()
        self.verbose_check.setChecked(False)
        self.verbose_check.setToolTip("Enable detailed progress logging")
        advanced_layout.addRow("Verbose Output:", self.verbose_check)
        
        # Auto-calculate mutation probability checkbox
        self.auto_mutation_check = QCheckBox()
        self.auto_mutation_check.setChecked(True)
        self.auto_mutation_check.setToolTip("Automatically set mutation probability to 1/n_variables")
        self.auto_mutation_check.stateChanged.connect(self.on_auto_mutation_changed)
        advanced_layout.addRow("Auto Mutation Prob:", self.auto_mutation_check)
        
        # Early Stopping Parameters
        self.enable_early_stopping_check = QCheckBox()
        self.enable_early_stopping_check.setChecked(True)
        self.enable_early_stopping_check.setToolTip("Enable early stopping based on hypervolume convergence (for multi-objective)")
        advanced_layout.addRow("Enable Early Stopping:", self.enable_early_stopping_check)
        
        self.early_stop_patience_spin = QSpinBox()
        self.early_stop_patience_spin.setMinimum(10)
        self.early_stop_patience_spin.setMaximum(200)
        self.early_stop_patience_spin.setValue(30)
        self.early_stop_patience_spin.setToolTip("Number of generations to wait without improvement before stopping")
        advanced_layout.addRow("Early Stop Patience:", self.early_stop_patience_spin)
        
        # Checkpoint Parameters
        self.enable_checkpoints_check = QCheckBox()
        self.enable_checkpoints_check.setChecked(True)
        self.enable_checkpoints_check.setToolTip("Enable automatic checkpointing during optimization")
        advanced_layout.addRow("Enable Checkpoints:", self.enable_checkpoints_check)
        
        self.checkpoint_interval_spin = QSpinBox()
        self.checkpoint_interval_spin.setMinimum(5)
        self.checkpoint_interval_spin.setMaximum(100)
        self.checkpoint_interval_spin.setValue(25)
        self.checkpoint_interval_spin.setToolTip("Save checkpoint every N generations")
        advanced_layout.addRow("Checkpoint Interval:", self.checkpoint_interval_spin)
        
        # Diversity noise scale parameter
        self.diversity_noise_spin = QDoubleSpinBox()
        self.diversity_noise_spin.setMinimum(0.0)
        self.diversity_noise_spin.setMaximum(1e-6)
        self.diversity_noise_spin.setValue(1e-10)
        self.diversity_noise_spin.setDecimals(12)
        self.diversity_noise_spin.setSingleStep(1e-11)
        self.diversity_noise_spin.setSpecialValueText("Disabled")
        self.diversity_noise_spin.setToolTip("Scale factor for diversity preservation noise (relative to objective magnitude). Set to 0 to disable. Recommended: 1e-10 to 1e-8")
        advanced_layout.addRow("Diversity Noise Scale:", self.diversity_noise_spin)
        
        # Robust Optimization Settings
        robust_group = QGroupBox("Robust Optimization")
        robust_layout = QFormLayout(robust_group)
        
        self.enable_robust_check = QCheckBox()
        self.enable_robust_check.setChecked(False)
        self.enable_robust_check.setToolTip("Enable robust optimization to handle uncertainty in predictions")
        robust_layout.addRow("Enable Robust Optimization:", self.enable_robust_check)
        
        self.robust_method_combo = QComboBox()
        self.robust_method_combo.addItems(["Bootstrap", "Monte Carlo", "Scenario-Based"])
        self.robust_method_combo.setCurrentText("Bootstrap")
        self.robust_method_combo.setToolTip("Method for uncertainty evaluation")
        robust_layout.addRow("Uncertainty Method:", self.robust_method_combo)
        
        self.robust_samples_spin = QSpinBox()
        self.robust_samples_spin.setMinimum(5)
        self.robust_samples_spin.setMaximum(100)
        self.robust_samples_spin.setValue(10)
        self.robust_samples_spin.setToolTip("Number of samples for uncertainty evaluation")
        robust_layout.addRow("Uncertainty Samples:", self.robust_samples_spin)
        
        self.robust_noise_spin = QDoubleSpinBox()
        self.robust_noise_spin.setMinimum(0.001)
        self.robust_noise_spin.setMaximum(0.1)
        self.robust_noise_spin.setValue(0.01)
        self.robust_noise_spin.setDecimals(3)
        self.robust_noise_spin.setToolTip("Noise level for Monte Carlo uncertainty (fraction of input magnitude)")
        robust_layout.addRow("Noise Level:", self.robust_noise_spin)
        
        # Constraint Handling Settings
        constraint_group = QGroupBox("Constraint Handling")
        constraint_layout = QFormLayout(constraint_group)
        
        self.enable_constraints_check = QCheckBox()
        self.enable_constraints_check.setChecked(False)
        self.enable_constraints_check.setToolTip("Enable explicit constraint handling (experimental)")
        constraint_layout.addRow("Enable Constraints:", self.enable_constraints_check)
        
        self.adaptive_penalty_check = QCheckBox()
        self.adaptive_penalty_check.setChecked(True)
        self.adaptive_penalty_check.setToolTip("Automatically adjust penalty coefficients based on violation rates")
        constraint_layout.addRow("Adaptive Penalties:", self.adaptive_penalty_check)
        
        layout.addWidget(self.algo_group)
        layout.addWidget(advanced_group)
        layout.addWidget(robust_group)
        layout.addWidget(constraint_group)
        
        # Control Buttons
        button_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.clicked.connect(self.start_optimization)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Optimization")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        # Checkpoint management buttons
        checkpoint_layout = QHBoxLayout()
        
        self.resume_btn = QPushButton("Resume from Checkpoint")
        self.resume_btn.clicked.connect(self.resume_from_checkpoint)
        checkpoint_layout.addWidget(self.resume_btn)
        
        self.list_checkpoints_btn = QPushButton("List Checkpoints")
        self.list_checkpoints_btn.clicked.connect(self.list_checkpoints)
        checkpoint_layout.addWidget(self.list_checkpoints_btn)
        
        button_layout.addLayout(checkpoint_layout)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        # Initialize algorithm settings to match current selection
        # Use QTimer to delay this call until after the UI is fully initialized
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, lambda: self.on_algorithm_changed(self.algorithm_combo.currentText()))
        
        return panel
    
    def add_model_row(self):
        """Add a new model/objective row"""
        model_index = len(self.model_widgets)
        
        # Create frame for this model
        model_frame = QFrame()
        model_frame.setFrameStyle(QFrame.StyledPanel)
        model_layout = QVBoxLayout(model_frame)
        
        # Header with model number and remove button
        header_layout = QHBoxLayout()
        header_label = QLabel(f"Model {model_index + 1}:")
        header_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(header_label)
        
        header_layout.addStretch()
        
        # Remove button (only show if more than 2 models)
        remove_btn = QPushButton("×")
        remove_btn.setMaximumSize(25, 25)
        remove_btn.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        remove_btn.clicked.connect(lambda: self.remove_model_row(model_index))
        header_layout.addWidget(remove_btn)
        model_layout.addLayout(header_layout)
        
        # Load model button and status
        load_btn_layout = QHBoxLayout()
        load_btn = QPushButton(f"Load Model {model_index + 1}")
        load_btn.clicked.connect(lambda: self.load_model(model_index))
        load_btn_layout.addWidget(load_btn)
        model_layout.addLayout(load_btn_layout)
        
        # Model status label
        status_label = QLabel("No model loaded")
        status_label.setStyleSheet("color: gray; font-style: italic;")
        model_layout.addWidget(status_label)
        
        # Objective name and direction
        obj_layout = QHBoxLayout()
        obj_name_label = QLabel("Not set")
        obj_layout.addWidget(QLabel("Objective:"))
        obj_layout.addWidget(obj_name_label)
        
        direction_combo = QComboBox()
        direction_combo.addItems(["Maximize", "Minimize"])
        obj_layout.addWidget(QLabel("Direction:"))
        obj_layout.addWidget(direction_combo)
        model_layout.addLayout(obj_layout)
        
        # Store widget references
        model_widget = {
            'frame': model_frame,
            'header_label': header_label,
            'remove_btn': remove_btn,
            'load_btn': load_btn,
            'status_label': status_label,
            'obj_name_label': obj_name_label,
            'direction_combo': direction_combo,
            'model_data': None
        }
        
        self.model_widgets.append(model_widget)
        self.models_data.append(None)
        self.models_layout.addWidget(model_frame)
        
        # Update remove button visibility
        self.update_remove_buttons()
        
    def remove_model_row(self, index):
        """Remove a model row"""
        if len(self.model_widgets) <= 1:
            QMessageBox.warning(self, "Warning", "At least 1 model is required for optimization.")
            return
        
        # Remove from layout and data structures
        widget = self.model_widgets.pop(index)
        self.models_data.pop(index)
        widget['frame'].setParent(None)
        
        # Update indices and labels
        for i, w in enumerate(self.model_widgets):
            w['header_label'].setText(f"Model {i + 1}:")
            w['load_btn'].setText(f"Load Model {i + 1}")
            # Reconnect signals with correct index
            w['load_btn'].clicked.disconnect()
            w['load_btn'].clicked.connect(lambda checked, idx=i: self.load_model(idx))
            w['remove_btn'].clicked.disconnect()
            w['remove_btn'].clicked.connect(lambda checked, idx=i: self.remove_model_row(idx))
        
        # Update remove button visibility
        self.update_remove_buttons()
        
        # Update features and indices after model removal
        self.update_features_and_objectives()
        
        # Update algorithm settings display
        self.update_algorithm_settings_display()
        
    def update_remove_buttons(self):
        """Update visibility of remove buttons"""
        show_remove = len(self.model_widgets) > 1  # Allow removal only if more than 1 model
        for widget in self.model_widgets:
            widget['remove_btn'].setVisible(show_remove)
    
    def update_algorithm_settings_display(self):
        """Update algorithm settings display based on number of loaded models"""
        loaded_models = [data for data in self.models_data if data is not None]
        n_models = len(loaded_models)
        
        if n_models <= 1:
            # Single-objective optimization with GA
            self.algo_group.setTitle("🎯 GA (Genetic Algorithm) Settings")
            self.algo_status_label.setText("🎯 Current Algorithm: GA (Genetic Algorithm) - Single-Objective Mode")
            self.algo_status_label.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e8;
                    border: 2px solid #4caf50;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #2e7d32;
                }
            """)
            
            # Adjust default values for single-objective if they haven't been manually changed
            if not hasattr(self, '_user_modified_params'):
                self._user_modified_params = set()
            
            if 'population_size' not in self._user_modified_params:
                current_pop = self.population_spin.value()
                if current_pop > 100:  # Only reduce if it's currently high
                    self.population_spin.setValue(50)  # Smaller population for single-objective
            
            if 'eliminate_duplicates' not in self._user_modified_params:
                self.eliminate_duplicates_check.setChecked(True)  # Better for single-objective convergence
            
            # Update tooltips for single-objective context
            self.population_spin.setToolTip(
                "Population size for GA (typically 30-100 for single-objective, smaller than multi-objective)"
            )
            self.crossover_prob_spin.setToolTip(
                "Probability of crossover in GA (0.7-0.95 recommended for single-objective)"
            )
            self.crossover_eta_spin.setToolTip(
                "Crossover distribution index for SBX (10-20 typical for single-objective)"
            )
            self.mutation_prob_spin.setToolTip(
                "Mutation probability in GA (typically 1/n_variables, auto-calculated if enabled)"
            )
            self.mutation_eta_spin.setToolTip(
                "Mutation distribution index for polynomial mutation (15-25 typical for single-objective)"
            )
            self.eliminate_duplicates_check.setToolTip(
                "Remove duplicate solutions (recommended for GA convergence)"
            )
            
            # Update diversity noise tooltip for single-objective
            self.diversity_noise_spin.setToolTip(
                "Diversity noise for GA (less critical for single-objective, can be set to 0)"
            )
            
        else:
            # Multi-objective optimization with NSGA-II
            self.algo_group.setTitle("🔄 NSGA-II (Multi-Objective) Algorithm Settings")
            self.algo_status_label.setText(f"🔄 Current Algorithm: NSGA-II - {n_models}-Objective Optimization Mode")
            self.algo_status_label.setStyleSheet("""
                QLabel {
                    background-color: #fff3e0;
                    border: 2px solid #ff9800;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #e65100;
                }
            """)
            
            # Adjust default values for multi-objective if they haven't been manually changed
            if not hasattr(self, '_user_modified_params'):
                self._user_modified_params = set()
            
            if 'population_size' not in self._user_modified_params:
                current_pop = self.population_spin.value()
                if current_pop < 50:  # Increase if it's currently too small
                    self.population_spin.setValue(max(50, n_models * 20))  # Larger population for multi-objective
            
            if 'eliminate_duplicates' not in self._user_modified_params:
                self.eliminate_duplicates_check.setChecked(False)  # Better for Pareto front diversity
            
            # Update tooltips for multi-objective context
            self.population_spin.setToolTip(
                f"Population size for NSGA-II (typically {max(50, n_models * 20)}-200 for {n_models}-objective, needs larger population for better Pareto front)"
            )
            self.crossover_prob_spin.setToolTip(
                "Probability of crossover in NSGA-II (0.8-0.95 recommended for multi-objective)"
            )
            self.crossover_eta_spin.setToolTip(
                "Crossover distribution index for SBX (5-20 typical for multi-objective, lower = more exploration)"
            )
            self.mutation_prob_spin.setToolTip(
                "Mutation probability in NSGA-II (typically 1/n_variables, critical for Pareto front diversity)"
            )
            self.mutation_eta_spin.setToolTip(
                "Mutation distribution index for polynomial mutation (10-30 typical for multi-objective)"
            )
            self.eliminate_duplicates_check.setToolTip(
                "Remove duplicate solutions (⚠️ WARNING: may severely reduce Pareto front diversity in multi-objective optimization)"
            )
            
            # Update diversity noise tooltip for multi-objective
            self.diversity_noise_spin.setToolTip(
                "Diversity noise for NSGA-II (important for Pareto front diversity, recommended: 1e-10 to 1e-8)"
            )
    
    def _mark_param_modified(self, param_name):
        """Mark a parameter as manually modified by user"""
        if not hasattr(self, '_user_modified_params'):
            self._user_modified_params = set()
        self._user_modified_params.add(param_name)
    
    def on_algorithm_changed(self, algorithm_name):
        """Handle algorithm selection change and update UI accordingly"""
        self.log_text.append(f"Algorithm changed to: {algorithm_name}")
        
        # Initialize user modified params tracking if not exists
        if not hasattr(self, '_user_modified_params'):
            self._user_modified_params = set()
        
        # Update algorithm-specific configurations
        if algorithm_name == "NSGA-III":
            self._update_nsga3_settings()
            self.log_text.append("NSGA-III: Better for many-objective problems (>3 objectives)")
        elif algorithm_name == "SPEA2":
            self._update_spea2_settings()
            if SPEA2 is not None:
                self.log_text.append("SPEA2: Alternative multi-objective algorithm with archive")
            else:
                self.log_text.append("⚠️ SPEA2: Not available in current pymoo version, will fallback to NSGA-II")
        elif algorithm_name == "MOEA/D":
            self._update_moead_settings()
            if MOEAD is not None:
                self.log_text.append("MOEA/D: Decomposition-based multi-objective algorithm")
            else:
                self.log_text.append("⚠️ MOEA/D: Not available in current pymoo version, will fallback to NSGA-II")
        elif algorithm_name == "CMA-ES":
            self._update_cmaes_settings()
            if CMA_ES is not None:
                self.log_text.append("CMA-ES: Evolution Strategy for continuous optimization")
            else:
                self.log_text.append("⚠️ CMA-ES: Not available in current pymoo version, will fallback to NSGA-II")
        elif algorithm_name == "Differential Evolution":
            self._update_de_settings()
            if DE is not None:
                self.log_text.append("DE: Differential Evolution for robust global optimization")
            else:
                self.log_text.append("⚠️ DE: Not available in current pymoo version, will fallback to NSGA-II")
        elif algorithm_name == "GA (Single-Objective)":
            self._update_ga_settings()
            self.log_text.append("GA: Forced single-objective mode (uses only first model)")
        else:  # NSGA-II
            self._update_nsga2_settings()
            self.log_text.append("NSGA-II: Standard multi-objective algorithm")
        
        # Update the visual status indicator
        self._update_algorithm_status_display(algorithm_name)
    
    def _update_algorithm_status_display(self, algorithm_name):
        """Update the visual algorithm status display"""
        loaded_models = [data for data in self.models_data if data is not None]
        n_models = len(loaded_models)
        
        if algorithm_name == "GA (Single-Objective)":
            self.algo_group.setTitle("🎯 GA (Genetic Algorithm) Settings")
            self.algo_status_label.setText("🎯 Current Algorithm: GA (Genetic Algorithm) - Single-Objective Mode")
            self.algo_status_label.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e8;
                    border: 2px solid #4caf50;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #2e7d32;
                }
            """)
        elif algorithm_name == "NSGA-III":
            self.algo_group.setTitle("🔺 NSGA-III (Many-Objective) Algorithm Settings")
            self.algo_status_label.setText(f"🔺 Current Algorithm: NSGA-III - {n_models}-Objective Optimization (Many-Objective)")
            self.algo_status_label.setStyleSheet("""
                QLabel {
                    background-color: #f3e5f5;
                    border: 2px solid #9c27b0;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #7b1fa2;
                }
            """)
        elif algorithm_name == "SPEA2":
            self.algo_group.setTitle("💎 SPEA2 (Archive-Based) Algorithm Settings")
            spea2_status = "Available" if SPEA2 is not None else "Unavailable (fallback to NSGA-II)"
            self.algo_status_label.setText(f"💎 Current Algorithm: SPEA2 - {n_models}-Objective Optimization ({spea2_status})")
            color_scheme = """
                QLabel {
                    background-color: #e0f2f1;
                    border: 2px solid #009688;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #00695c;
                }
            """ if SPEA2 is not None else """
                QLabel {
                    background-color: #ffebee;
                    border: 2px solid #f44336;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #c62828;
                }
            """
            self.algo_status_label.setStyleSheet(color_scheme)
        elif algorithm_name == "MOEA/D":
            self.algo_group.setTitle("⚡ MOEA/D (Decomposition-Based) Algorithm Settings")
            moead_status = "Available" if MOEAD is not None else "Unavailable (fallback to NSGA-II)"
            self.algo_status_label.setText(f"⚡ Current Algorithm: MOEA/D - {n_models}-Objective Optimization ({moead_status})")
            color_scheme = """
                QLabel {
                    background-color: #f3e5f5;
                    border: 2px solid #673ab7;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #4a148c;
                }
            """ if MOEAD is not None else """
                QLabel {
                    background-color: #ffebee;
                    border: 2px solid #f44336;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #c62828;
                }
            """
            self.algo_status_label.setStyleSheet(color_scheme)
        elif algorithm_name == "CMA-ES":
            self.algo_group.setTitle("🎯 CMA-ES (Evolution Strategy) Algorithm Settings")
            cmaes_status = "Available" if CMA_ES is not None else "Unavailable (fallback to NSGA-II)"
            self.algo_status_label.setText(f"🎯 Current Algorithm: CMA-ES - Continuous Optimization ({cmaes_status})")
            color_scheme = """
                QLabel {
                    background-color: #e8f5e8;
                    border: 2px solid #4caf50;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #1b5e20;
                }
            """ if CMA_ES is not None else """
                QLabel {
                    background-color: #ffebee;
                    border: 2px solid #f44336;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #c62828;
                }
            """
            self.algo_status_label.setStyleSheet(color_scheme)
        elif algorithm_name == "Differential Evolution":
            self.algo_group.setTitle("🔥 Differential Evolution Algorithm Settings")
            de_status = "Available" if DE is not None else "Unavailable (fallback to NSGA-II)"
            self.algo_status_label.setText(f"🔥 Current Algorithm: Differential Evolution - {n_models}-Objective ({de_status})")
            color_scheme = """
                QLabel {
                    background-color: #fff8e1;
                    border: 2px solid #ffc107;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #e65100;
                }
            """ if DE is not None else """
                QLabel {
                    background-color: #ffebee;
                    border: 2px solid #f44336;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #c62828;
                }
            """
            self.algo_status_label.setStyleSheet(color_scheme)
        else:  # NSGA-II
            self.algo_group.setTitle("🔄 NSGA-II (Multi-Objective) Algorithm Settings")
            self.algo_status_label.setText(f"🔄 Current Algorithm: NSGA-II - {n_models}-Objective Optimization Mode")
            self.algo_status_label.setStyleSheet("""
                QLabel {
                    background-color: #fff3e0;
                    border: 2px solid #ff9800;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                    color: #e65100;
                }
            """)
    
    def _update_ga_settings(self):
        """Update settings for GA (Single-Objective) algorithm"""
        # Optimal settings for single-objective GA
        if 'population_size' not in self._user_modified_params:
            self.population_spin.setValue(50)  # Smaller population for single-objective
        
        if 'crossover_prob' not in self._user_modified_params:
            self.crossover_prob_spin.setValue(0.8)  # Standard for GA
        
        if 'crossover_eta' not in self._user_modified_params:
            self.crossover_eta_spin.setValue(20.0)  # Higher for more exploration
        
        if 'mutation_eta' not in self._user_modified_params:
            self.mutation_eta_spin.setValue(20.0)  # Higher for single-objective
        
        if 'eliminate_duplicates' not in self._user_modified_params:
            self.eliminate_duplicates_check.setChecked(True)  # Good for convergence
        
        # Update tooltips for GA
        self.population_spin.setToolTip("Population size for GA (30-100 typical for single-objective)")
        self.crossover_prob_spin.setToolTip("Crossover probability for GA (0.7-0.9 recommended)")
        self.mutation_prob_spin.setToolTip("Mutation probability for GA (typically 1/n_variables)")
        self.eliminate_duplicates_check.setToolTip("Remove duplicates (recommended for GA convergence)")
    
    def _update_nsga2_settings(self):
        """Update settings for NSGA-II algorithm"""
        loaded_models = [data for data in self.models_data if data is not None]
        n_models = len(loaded_models)
        
        # Optimal settings for NSGA-II
        if 'population_size' not in self._user_modified_params:
            self.population_spin.setValue(max(50, n_models * 20))  # Larger for multi-objective
        
        if 'crossover_prob' not in self._user_modified_params:
            self.crossover_prob_spin.setValue(0.9)  # High for NSGA-II
        
        if 'crossover_eta' not in self._user_modified_params:
            self.crossover_eta_spin.setValue(15.0)  # Standard for NSGA-II
        
        if 'mutation_eta' not in self._user_modified_params:
            self.mutation_eta_spin.setValue(20.0)  # Standard for multi-objective
        
        if 'eliminate_duplicates' not in self._user_modified_params:
            self.eliminate_duplicates_check.setChecked(False)  # Better for Pareto diversity
        
        # Update tooltips for NSGA-II
        self.population_spin.setToolTip(f"Population size for NSGA-II ({max(50, n_models * 20)}-200 typical for {n_models}-objective)")
        self.crossover_prob_spin.setToolTip("Crossover probability for NSGA-II (0.8-0.95 recommended)")
        self.mutation_prob_spin.setToolTip("Mutation probability for NSGA-II (critical for Pareto diversity)")
        self.eliminate_duplicates_check.setToolTip("⚠️ May reduce Pareto front diversity in multi-objective")
    
    def _update_nsga3_settings(self):
        """Update settings for NSGA-III algorithm"""
        loaded_models = [data for data in self.models_data if data is not None]
        n_models = len(loaded_models)
        
        # Optimal settings for NSGA-III (many-objective)
        if 'population_size' not in self._user_modified_params:
            # NSGA-III needs larger populations for many objectives
            base_pop = max(100, n_models * 30)
            self.population_spin.setValue(base_pop)
        
        if 'crossover_prob' not in self._user_modified_params:
            self.crossover_prob_spin.setValue(0.9)  # High for many-objective
        
        if 'crossover_eta' not in self._user_modified_params:
            self.crossover_eta_spin.setValue(10.0)  # Lower for more exploration
        
        if 'mutation_eta' not in self._user_modified_params:
            self.mutation_eta_spin.setValue(15.0)  # Lower for more diversity
        
        if 'eliminate_duplicates' not in self._user_modified_params:
            self.eliminate_duplicates_check.setChecked(False)  # Critical for many-objective
        
        # Update tooltips for NSGA-III
        base_pop = max(100, n_models * 30)
        self.population_spin.setToolTip(f"Population size for NSGA-III ({base_pop}-300 typical for {n_models}-objective, needs large population)")
        self.crossover_prob_spin.setToolTip("Crossover probability for NSGA-III (0.9-0.95 for many-objective)")
        self.mutation_prob_spin.setToolTip("Mutation probability for NSGA-III (crucial for many-objective diversity)")
        self.eliminate_duplicates_check.setToolTip("⚠️ Critical: Should be FALSE for many-objective optimization")
    
    def _update_spea2_settings(self):
        """Update settings for SPEA2 algorithm"""
        loaded_models = [data for data in self.models_data if data is not None]
        n_models = len(loaded_models)
        
        # Optimal settings for SPEA2
        if 'population_size' not in self._user_modified_params:
            self.population_spin.setValue(max(60, n_models * 25))  # Medium size for SPEA2
        
        if 'crossover_prob' not in self._user_modified_params:
            self.crossover_prob_spin.setValue(0.85)  # Moderate for SPEA2
        
        if 'crossover_eta' not in self._user_modified_params:
            self.crossover_eta_spin.setValue(12.0)  # Balanced exploration/exploitation
        
        if 'mutation_eta' not in self._user_modified_params:
            self.mutation_eta_spin.setValue(18.0)  # Balanced for archive-based
        
        if 'eliminate_duplicates' not in self._user_modified_params:
            self.eliminate_duplicates_check.setChecked(True)  # SPEA2 can handle this better
        
        # Update tooltips for SPEA2
        base_pop = max(60, n_models * 25)
        self.population_spin.setToolTip(f"Population size for SPEA2 ({base_pop}-200 typical, uses external archive)")
        self.crossover_prob_spin.setToolTip("Crossover probability for SPEA2 (0.8-0.9 recommended)")
        self.mutation_prob_spin.setToolTip("Mutation probability for SPEA2 (works with archive mechanism)")
        self.eliminate_duplicates_check.setToolTip("SPEA2 can handle duplicates better than NSGA-II")
    
    def _update_moead_settings(self):
        """Update settings for MOEA/D algorithm"""
        loaded_models = [data for data in self.models_data if data is not None]
        n_models = len(loaded_models)
        
        # Optimal settings for MOEA/D
        if 'population_size' not in self._user_modified_params:
            # MOEA/D uses scalarization, needs specific population sizes
            self.population_spin.setValue(max(91, n_models * 30))  # Often uses H1+H2 formula
        
        if 'crossover_prob' not in self._user_modified_params:
            self.crossover_prob_spin.setValue(0.9)  # High for decomposition
        
        if 'crossover_eta' not in self._user_modified_params:
            self.crossover_eta_spin.setValue(20.0)  # High for local search
        
        if 'mutation_eta' not in self._user_modified_params:
            self.mutation_eta_spin.setValue(20.0)  # High for fine-tuning
        
        if 'eliminate_duplicates' not in self._user_modified_params:
            self.eliminate_duplicates_check.setChecked(False)  # Preserve decomposition diversity
        
        # Update tooltips for MOEA/D
        self.population_spin.setToolTip("Population for MOEA/D (uses decomposition, 91/105/120 are common)")
        self.crossover_prob_spin.setToolTip("Crossover for MOEA/D (0.9-0.95 for neighborhood search)")
        self.mutation_prob_spin.setToolTip("Mutation for MOEA/D (fine-tuning within neighborhoods)")
        self.eliminate_duplicates_check.setToolTip("Keep FALSE to preserve decomposition structure")
    
    def _update_cmaes_settings(self):
        """Update settings for CMA-ES algorithm"""
        loaded_models = [data for data in self.models_data if data is not None]
        n_models = len(loaded_models)
        
        # CMA-ES specific settings
        if 'population_size' not in self._user_modified_params:
            # CMA-ES has automatic population sizing, but we can suggest
            n_vars = len(self.feature_bounds_data) if hasattr(self, 'feature_bounds_data') else 10
            self.population_spin.setValue(max(30, min(100, 4 + int(3 * np.log(n_vars)))))
        
        if 'crossover_prob' not in self._user_modified_params:
            self.crossover_prob_spin.setValue(1.0)  # CMA-ES uses different operators
        
        # Update tooltips for CMA-ES
        self.population_spin.setToolTip("Population for CMA-ES (automatic sizing, 30-100 typical)")
        self.crossover_prob_spin.setToolTip("Not used in CMA-ES (uses covariance matrix adaptation)")
        self.mutation_prob_spin.setToolTip("CMA-ES uses adaptive step size, not traditional mutation")
        self.eliminate_duplicates_check.setToolTip("Duplicates less common in CMA-ES due to distribution sampling")
    
    def _update_de_settings(self):
        """Update settings for Differential Evolution algorithm"""
        loaded_models = [data for data in self.models_data if data is not None]
        n_models = len(loaded_models)
        
        # DE specific settings
        if 'population_size' not in self._user_modified_params:
            # DE typically uses 5-10 times the problem dimension
            n_vars = len(self.feature_bounds_data) if hasattr(self, 'feature_bounds_data') else 10
            self.population_spin.setValue(max(30, min(200, n_vars * 8)))
        
        if 'crossover_prob' not in self._user_modified_params:
            self.crossover_prob_spin.setValue(0.7)  # DE crossover rate (CR)
        
        if 'mutation_eta' not in self._user_modified_params:
            # In DE, this would be the differential weight (F)
            self.mutation_eta_spin.setValue(0.5)  # DE differential weight
        
        if 'eliminate_duplicates' not in self._user_modified_params:
            self.eliminate_duplicates_check.setChecked(True)  # DE can benefit from this
        
        # Update tooltips for DE
        n_vars = len(self.feature_bounds_data) if hasattr(self, 'feature_bounds_data') else 10
        self.population_spin.setToolTip(f"Population for DE (5-10x problem dimension, ~{n_vars * 8} for {n_vars} variables)")
        self.crossover_prob_spin.setToolTip("DE Crossover Rate (CR): 0.5-0.9 typical")
        self.mutation_prob_spin.setToolTip("Not used in DE (uses differential mutation)")
        self.crossover_eta_spin.setToolTip("Not directly used in DE")
        self.mutation_eta_spin.setToolTip("DE Differential Weight (F): 0.4-0.8 typical")
        self.eliminate_duplicates_check.setToolTip("DE can benefit from duplicate elimination")
    
    def resume_from_checkpoint(self):
        """Resume optimization from a checkpoint"""
        try:
            checkpoint_system = OptimizationCheckpoint()
            checkpoints = checkpoint_system.list_checkpoints()
            
            if not checkpoints:
                QMessageBox.information(self, "No Checkpoints", "No checkpoints found to resume from.")
                return
            
            # Create selection dialog
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QLabel
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Select Checkpoint")
            dialog.setModal(True)
            dialog.resize(500, 300)
            
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("Select a checkpoint to resume optimization:"))
            
            list_widget = QListWidget()
            for checkpoint in checkpoints:
                item_text = (f"Generation {checkpoint['generation']:04d} - "
                           f"{checkpoint['timestamp']} - "
                           f"{checkpoint['size_mb']:.1f} MB")
                list_widget.addItem(item_text)
            layout.addWidget(list_widget)
            
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            
            if dialog.exec_() == QDialog.Accepted:
                selected_items = list_widget.selectedItems()
                if selected_items:
                    selected_idx = list_widget.row(selected_items[0])
                    checkpoint_file = checkpoints[selected_idx]['file']
                    
                    # Load checkpoint and start optimization
                    checkpoint_data = checkpoint_system.load_checkpoint(checkpoint_file)
                    if checkpoint_data:
                        self.log_text.append(f"Loading checkpoint from generation {checkpoint_data['generation']}")
                        # This would require additional implementation to actually resume
                        QMessageBox.information(self, "Checkpoint Loaded", 
                                              f"Checkpoint from generation {checkpoint_data['generation']} loaded successfully.\n"
                                              "Note: Full resume functionality requires additional implementation.")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resume from checkpoint: {str(e)}")
            self.log_text.append(f"Resume error: {str(e)}")
    
    def list_checkpoints(self):
        """List available checkpoints"""
        try:
            checkpoint_system = OptimizationCheckpoint()
            checkpoints = checkpoint_system.list_checkpoints()
            
            if not checkpoints:
                self.log_text.append("No checkpoints found.")
                return
            
            self.log_text.append("📁 Available Checkpoints:")
            for checkpoint in checkpoints:
                self.log_text.append(
                    f"  • Generation {checkpoint['generation']:04d} - "
                    f"{checkpoint['timestamp']} - "
                    f"{checkpoint['size_mb']:.1f} MB"
                )
        
        except Exception as e:
            self.log_text.append(f"Failed to list checkpoints: {str(e)}")
    
    def extract_original_ranges(self, pipeline, feature_names):
        """Extract original feature ranges from pipeline preprocessor using smart analyzer"""
        def log_append(message):
            self.log_text.append(message)
        
        return PipelineAnalyzer.extract_feature_bounds(
            pipeline, 
            feature_names, 
            log_callback=log_append
        )

    def analyze_feature_types(self, model_data):
        """Analyze and enhance feature type information with original ranges"""
        enhanced_data = model_data.copy()
        pipeline = model_data['pipeline']
        feature_names = model_data['metadata']['feature_names']
        
        # PRIORITY 1: Check if feature bounds AND types are saved in metadata (CRITICAL FIX)
        saved_feature_bounds = model_data['metadata'].get('feature_bounds', None)
        saved_feature_types = model_data['metadata'].get('feature_types', None)
        
        if saved_feature_bounds:
            # Convert saved bounds to the format expected by PipelineAnalyzer
            original_bounds = []
            for feature_name in feature_names:
                if feature_name in saved_feature_bounds:
                    original_bounds.append(saved_feature_bounds[feature_name])
                else:
                    original_bounds.append((0.0, 1.0))  # Fallback
            self.log_text.append("✅ Using saved feature bounds from model metadata!")
            
            if saved_feature_types:
                self.log_text.append("✅ Using saved feature types from model metadata!")
                self.log_text.append(f"   📊 Feature types summary:")
                type_counts = {}
                for feature_name in feature_names:
                    ftype = saved_feature_types.get(feature_name, 'continuous')
                    type_counts[ftype] = type_counts.get(ftype, 0) + 1
                for ftype, count in type_counts.items():
                    self.log_text.append(f"      {ftype}: {count} features")
        else:
            # PRIORITY 2: Try to extract original feature ranges from pipeline
            original_bounds = self.extract_original_ranges(pipeline, feature_names)
        
        # Initialize feature type information
        feature_types = {}
        feature_bounds = {}
        feature_bounds_normalized = {}  # Store normalized bounds separately
        feature_categories = {}
        
        for i, feature_name in enumerate(feature_names):
            # PRIORITY 1: Use saved feature type if available
            if saved_feature_types and feature_name in saved_feature_types:
                feature_type = saved_feature_types[feature_name]
            else:
                # IMPORTANT FIX: Default to continuous type for ALL features unless explicitly saved otherwise
                # This prevents incorrect binary/categorical classification based on feature names
                feature_type = 'continuous'
            
            # Set bounds based on whether original ranges were detected
            if original_bounds and i < len(original_bounds):
                bounds_original = original_bounds[i]
                bounds_normalized = (0.0, 1.0)  # Default normalized
            else:
                # Smart defaults based on feature name patterns
                feature_lower = feature_name.lower()
                if any(pattern in feature_lower for pattern in ['temp', 'temperature']):
                    bounds_original = (-50, 150)  # Temperature range
                elif any(pattern in feature_lower for pattern in ['pressure']):
                    bounds_original = (0, 1000)  # Pressure range
                elif any(pattern in feature_lower for pattern in ['concentration', 'percent', 'ratio']):
                    bounds_original = (0, 100)  # Percentage/concentration
                elif any(pattern in feature_lower for pattern in ['ph']):
                    bounds_original = (0, 14)  # pH range
                elif any(pattern in feature_lower for pattern in ['time', 'duration']):
                    bounds_original = (0, 1000)  # Time range
                elif any(pattern in feature_lower for pattern in ['lattice', 'constant']):
                    bounds_original = (200, 800)  # Lattice constant (pm)
                elif any(pattern in feature_lower for pattern in ['radii', 'radius']):
                    bounds_original = (0.3, 2.5)  # Atomic/ionic radii (Å)
                elif any(pattern in feature_lower for pattern in ['weight', 'mass']):
                    bounds_original = (1, 300)  # Atomic weight
                elif any(pattern in feature_lower for pattern in ['electron']):
                    bounds_original = (0, 20)  # Electron count
                else:
                    bounds_original = (0.0, 1.0)  # Default normalized bounds
                
                bounds_normalized = (0.0, 1.0)  # Default normalized
            
            categories = None
            
            # REMOVED: Feature type inference from name patterns - this was causing the bug
            # All features default to 'continuous' unless explicitly saved as another type
            # This prevents incorrect binary/categorical classification based on feature names
            
            # Handle specific feature types based on saved information only
            if feature_type == 'binary':
                if not original_bounds:
                    bounds_original = (0, 1)
                    bounds_normalized = (0, 1)
                categories = [0, 1]
            elif feature_type == 'categorical':
                if not original_bounds:
                    # Use actual min/max for categorical features if bounds not set
                    bounds_normalized = (0, 1)
                # For categorical features, categories will be determined from actual values
                unique_vals = list(range(int(bounds_original[0]), int(bounds_original[1]) + 1)) if bounds_original else [0, 1]
                categories = unique_vals
            
            feature_types[feature_name] = feature_type
            feature_bounds[feature_name] = bounds_original
            feature_bounds_normalized[feature_name] = bounds_normalized
            if categories:
                feature_categories[feature_name] = categories
        
        # Add enhanced metadata
        enhanced_data['metadata']['feature_types'] = feature_types
        enhanced_data['metadata']['feature_bounds'] = feature_bounds
        enhanced_data['metadata']['feature_bounds_normalized'] = feature_bounds_normalized
        enhanced_data['metadata']['feature_categories'] = feature_categories
        enhanced_data['metadata']['original_ranges_detected'] = original_bounds is not None
        
        # Log the detection results and update status
        if saved_feature_bounds:
            self.log_text.append("   📊 Detailed feature information:")
            for name in feature_names:
                bounds = feature_bounds.get(name, (0, 1))
                ftype = feature_types.get(name, 'continuous')
                type_marker = {"continuous": "📈", "categorical": "📝", "binary": "🔘"}.get(ftype, "❓")
                self.log_text.append(f"      {type_marker} {name} ({ftype}): [{bounds[0]:.4f}, {bounds[1]:.4f}]")
            self.log_text.append("   You can now input physical feature values directly!")
            
            # Update status indicator
            if hasattr(self, 'range_status_label'):
                self.range_status_label.setText("✅ Metadata available")
                self.range_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif original_bounds:
            self.log_text.append("✅ Original feature ranges successfully extracted from pipeline!")
            self.log_text.append("   📊 Feature bounds summary:")
            for i, (name, bounds) in enumerate(zip(feature_names, original_bounds)):
                ftype = feature_types.get(name, 'continuous')
                type_marker = {"continuous": "📈", "categorical": "📝", "binary": "🔘"}.get(ftype, "❓")
                self.log_text.append(f"      {type_marker} {name} ({ftype}): [{bounds[0]:.4f}, {bounds[1]:.4f}]")
            self.log_text.append("   You can now input physical feature values directly.")
            
            # Update status indicator
            if hasattr(self, 'range_status_label'):
                self.range_status_label.setText("✅ Original ranges detected")
                self.range_status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.log_text.append("⚠️  Could not extract original ranges from pipeline.")
            self.log_text.append("   Using smart defaults based on feature names.")
            self.log_text.append("   🔧 All features treated as continuous unless explicitly saved otherwise.")
            self.log_text.append("   📊 Inferred feature information:")
            for name in feature_names:
                bounds = feature_bounds.get(name, (0, 1))
                ftype = feature_types.get(name, 'continuous')
                type_marker = {"continuous": "📈", "categorical": "📝", "binary": "🔘"}.get(ftype, "❓")
                self.log_text.append(f"      {type_marker} {name} ({ftype}): [{bounds[0]:.4f}, {bounds[1]:.4f}]")
            
            # Update status indicator
            if hasattr(self, 'range_status_label'):
                self.range_status_label.setText("⚠️  Using defaults")
                self.range_status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        return enhanced_data
    
    def create_results_panel(self) -> QWidget:
        """Create the enhanced right-side results panel with multiple visualizations"""
        panel = QTabWidget()
        
        # 1. Pareto Front Plot Tab
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        self.pareto_figure = Figure(figsize=(10, 6))
        self.pareto_canvas = FigureCanvas(self.pareto_figure)
        self.pareto_toolbar = NavigationToolbar(self.pareto_canvas, plot_widget)
        
        plot_layout.addWidget(self.pareto_toolbar)
        plot_layout.addWidget(self.pareto_canvas)
        
        panel.addTab(plot_widget, "🎯 Pareto Front")
        
        # 2. Convergence Tracking Tab
        convergence_widget = QWidget()
        convergence_layout = QVBoxLayout(convergence_widget)
        
        self.convergence_figure = Figure(figsize=(10, 6))
        self.convergence_canvas = FigureCanvas(self.convergence_figure)
        self.convergence_toolbar = NavigationToolbar(self.convergence_canvas, convergence_widget)
        
        convergence_layout.addWidget(self.convergence_toolbar)
        convergence_layout.addWidget(self.convergence_canvas)
        
        panel.addTab(convergence_widget, "📈 Convergence")
        
        # 3. Evolution Animation Tab
        evolution_widget = QWidget()
        evolution_layout = QVBoxLayout(evolution_widget)
        
        self.evolution_figure = Figure(figsize=(10, 6))
        self.evolution_canvas = FigureCanvas(self.evolution_figure)
        self.evolution_toolbar = NavigationToolbar(self.evolution_canvas, evolution_widget)
        
        evolution_layout.addWidget(self.evolution_toolbar)
        evolution_layout.addWidget(self.evolution_canvas)
        
        panel.addTab(evolution_widget, "🔄 Evolution")
        
        # 4. Solutions Table Tab
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        self.solutions_table = QTableWidget()
        self.solutions_table.setAlternatingRowColors(True)
        self.solutions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table_layout.addWidget(self.solutions_table)
        
        # Export button
        export_btn = QPushButton("📄 Export Results to CSV")
        export_btn.clicked.connect(self.export_results)
        table_layout.addWidget(export_btn)
        
        panel.addTab(table_widget, "📊 Solutions")
        
        # 5. Progress and Log Tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        log_layout.addWidget(self.progress_bar)
        
        # Log text
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        panel.addTab(log_widget, "📋 Progress")
        
        # Initialize visualization data storage
        self.all_progress_data = []
        
        return panel
    
    def load_model(self, model_index):
        """Load a model at the specified index"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Load Model {model_index + 1}", "", "Joblib Files (*.joblib)"
        )
        
        if file_path:
            try:
                # Load model data
                model_data = joblib.load(file_path)
                
                # Validate structure
                if not isinstance(model_data, dict) or 'pipeline' not in model_data or 'metadata' not in model_data:
                    raise ValueError("Invalid model file format")
                
                # Enhanced model type validation
                task_type = model_data['metadata'].get('task_type', 'unknown')
                if task_type == 'classification':
                    # Check if it's binary classification (might be acceptable)
                    try:
                        # Try to get number of classes from model metadata
                        pipeline = model_data['pipeline']
                        if hasattr(pipeline, 'classes_') and len(pipeline.classes_) > 2:
                            QMessageBox.critical(
                                self, "Incompatible Model", 
                                f"Multi-class classification models (with {len(pipeline.classes_)} classes) "
                                "are not suitable for multi-objective optimization. "
                                "Please use regression models or binary classification models."
                            )
                            return
                        elif hasattr(pipeline, 'classes_') and len(pipeline.classes_) == 2:
                            response = QMessageBox.question(
                                self, "Binary Classification Model", 
                                "This appears to be a binary classification model. "
                                "While not ideal, it can be used for optimization (will optimize probability outputs). "
                                "For best results, use regression models. Continue?",
                                QMessageBox.Yes | QMessageBox.No
                            )
                            if response == QMessageBox.No:
                                return
                        else:
                            # Unknown classification type
                            response = QMessageBox.question(
                                self, "Classification Model Warning", 
                                f"Model appears to be for classification, not regression. "
                                "Multi-objective optimization works best with regression models. "
                                "Continue anyway?",
                                QMessageBox.Yes | QMessageBox.No
                            )
                            if response == QMessageBox.No:
                                return
                    except Exception:
                        # If we can't determine the model details, show generic warning
                        response = QMessageBox.question(
                            self, "Model Type Warning", 
                            f"Model appears to be for {task_type}, not regression. "
                            "Multi-objective optimization typically works with regression models. "
                            "Continue anyway?",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if response == QMessageBox.No:
                            return
                elif task_type not in ['regression', 'unknown']:
                    QMessageBox.critical(
                        self, "Unsupported Model Type", 
                        f"Model type '{task_type}' is not supported for multi-objective optimization. "
                        "Please use regression models."
                    )
                    return
                
                # Analyze and enhance feature metadata
                enhanced_model_data = self.analyze_feature_types(model_data)
                
                # Store enhanced model data
                self.models_data[model_index] = enhanced_model_data
                self.model_widgets[model_index]['model_data'] = enhanced_model_data
                
                # Update UI
                model_name = os.path.basename(file_path)
                widget = self.model_widgets[model_index]
                widget['status_label'].setText(f"Loaded: {model_name}")
                widget['status_label'].setStyleSheet("color: green;")
                
                # Update objective name
                target_name = model_data['metadata'].get('target_name', f'Objective {model_index + 1}')
                widget['obj_name_label'].setText(target_name)
                
                self.log_text.append(f"Model {model_index + 1} loaded successfully: {model_name}")
                self.log_text.append(f"Features: {len(model_data['metadata']['feature_names'])}")
                self.log_text.append(f"Task type: {task_type}")
                
                # Update combined features
                self.update_features_and_objectives()
                
                # Update algorithm settings display
                self.update_algorithm_settings_display()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model {model_index + 1}: {str(e)}")
                self.log_text.append(f"Error loading model {model_index + 1}: {str(e)}")
    
    def update_features_and_objectives(self):
        """Update UI after models are loaded"""
        # Check if we have at least 1 model loaded
        loaded_models = [data for data in self.models_data if data is not None]
        if len(loaded_models) < 1:
            self.start_btn.setEnabled(False)
            # Clear features table if no models loaded
            self.combined_features = []
            self.model_indices = []
            self.feature_table.setRowCount(0)
            return
        
        try:
            # Get all feature lists
            all_features = []
            for model_data in loaded_models:
                features = model_data['metadata']['feature_names']
                all_features.extend(features)
            
            # Compute union of features
            self.combined_features = sorted(list(set(all_features)))
            
            # Compute indices for each loaded model
            self.model_indices = []
            for model_data in self.models_data:
                if model_data is not None:
                    features = model_data['metadata']['feature_names']
                    indices = [self.combined_features.index(f) for f in features if f in self.combined_features]
                    self.model_indices.append(indices)
                else:
                    self.model_indices.append([])
            
            # Update feature bounds table
            self.update_feature_bounds_table()
            
            # Enable optimization if we have at least 1 model
            self.start_btn.setEnabled(len(loaded_models) >= 1)
            
            self.log_text.append(f"Combined features: {len(self.combined_features)}")
            for i, (model_data, indices) in enumerate(zip(self.models_data, self.model_indices)):
                if model_data is not None:
                    if len(loaded_models) == 1:
                        self.log_text.append(f"Single model uses {len(indices)} features")
                    else:
                        self.log_text.append(f"Model {i+1} uses {len(indices)} features")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process features: {str(e)}")
            self.log_text.append(f"Error processing features: {str(e)}")
    
    def update_feature_bounds_table(self):
        """Update the feature bounds table with enhanced type support"""
        try:
            self.feature_table.setRowCount(len(self.combined_features))
            
            # Get combined feature metadata
            combined_feature_info = self.get_combined_feature_info()
            
            for i, feature_name in enumerate(self.combined_features):
                try:
                    # Feature name (read-only)
                    name_item = QTableWidgetItem(feature_name)
                    name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                    self.feature_table.setItem(i, 0, name_item)
                    
                    # Get feature info
                    feature_info = combined_feature_info.get(feature_name, {})
                    feature_type = feature_info.get('type', 'continuous')
                    bounds = feature_info.get('bounds', (0.0, 1.0))
                    categories = feature_info.get('categories', None)
                    
                    # Set bounds based on feature type
                    if feature_type in ['binary', 'categorical_encoded']:
                        # For binary/encoded features, use integer bounds
                        min_item = QTableWidgetItem(str(int(bounds[0])))
                        max_item = QTableWidgetItem(str(int(bounds[1])))
                        # Make them read-only since they're fixed by the encoding
                        min_item.setFlags(min_item.flags() & ~Qt.ItemIsEditable)
                        max_item.setFlags(max_item.flags() & ~Qt.ItemIsEditable)
                        min_item.setBackground(Qt.lightGray)
                        max_item.setBackground(Qt.lightGray)
                    elif feature_type == 'categorical':
                        # For categorical features, show category range
                        if categories:
                            min_item = QTableWidgetItem("0")
                            max_item = QTableWidgetItem(str(len(categories) - 1))
                        else:
                            min_item = QTableWidgetItem(str(int(bounds[0])))
                            max_item = QTableWidgetItem(str(int(bounds[1])))
                        # Make them read-only
                        min_item.setFlags(min_item.flags() & ~Qt.ItemIsEditable)
                        max_item.setFlags(max_item.flags() & ~Qt.ItemIsEditable)
                        min_item.setBackground(Qt.lightGray)
                        max_item.setBackground(Qt.lightGray)
                    else:
                        # Continuous features - editable bounds
                        min_item = QTableWidgetItem(f"{bounds[0]:.2f}")
                        max_item = QTableWidgetItem(f"{bounds[1]:.2f}")
                    
                    self.feature_table.setItem(i, 1, min_item)
                    self.feature_table.setItem(i, 2, max_item)
                    
                    # Fixed checkbox
                    fixed_checkbox = QCheckBox()
                    fixed_checkbox.stateChanged.connect(lambda state, row=i: self.on_fixed_feature_changed(row, state))
                    self.feature_table.setCellWidget(i, 3, fixed_checkbox)
                    
                    # Fixed value - use appropriate widget based on feature type
                    if feature_type in ['binary', 'categorical_encoded'] and categories and len(categories) > 0:
                        # Use combo box for categorical features
                        try:
                            fixed_combo = QComboBox()
                            # Safety check for categories content
                            valid_categories = [str(cat) for cat in categories if cat is not None]
                            if valid_categories:
                                fixed_combo.addItems(valid_categories)
                                fixed_combo.setEnabled(False)  # Initially disabled
                                self.feature_table.setCellWidget(i, 4, fixed_combo)
                            else:
                                raise ValueError("No valid categories found")
                        except Exception as e:
                            print(f"ERROR creating combo box for feature {feature_name}: {e}")
                            # Fallback to text item
                            default_value = str(int(bounds[0]))
                            fixed_value_item = QTableWidgetItem(default_value)
                            fixed_value_item.setFlags(fixed_value_item.flags() & ~Qt.ItemIsEditable)
                            fixed_value_item.setBackground(Qt.lightGray)
                            self.feature_table.setItem(i, 4, fixed_value_item)
                    elif feature_type == 'categorical' and categories and len(categories) > 0:
                        # Use combo box for categorical features
                        try:
                            fixed_combo = QComboBox()
                            # Safety check for categories content
                            valid_categories = [str(cat) for cat in categories if cat is not None]
                            if valid_categories:
                                fixed_combo.addItems(valid_categories)
                                fixed_combo.setEnabled(False)  # Initially disabled
                                self.feature_table.setCellWidget(i, 4, fixed_combo)
                            else:
                                raise ValueError("No valid categories found")
                        except Exception as e:
                            print(f"ERROR creating combo box for feature {feature_name}: {e}")
                            # Fallback to text item
                            default_value = str(int(bounds[0]))
                            fixed_value_item = QTableWidgetItem(default_value)
                            fixed_value_item.setFlags(fixed_value_item.flags() & ~Qt.ItemIsEditable)
                            fixed_value_item.setBackground(Qt.lightGray)
                            self.feature_table.setItem(i, 4, fixed_value_item)
                    else:
                        # Use text item for continuous features
                        if feature_type in ['binary', 'categorical_encoded']:
                            default_value = str(int(bounds[0]))
                        else:
                            default_value = f"{(bounds[0] + bounds[1]) / 2:.2f}"
                        
                        fixed_value_item = QTableWidgetItem(default_value)
                        fixed_value_item.setFlags(fixed_value_item.flags() & ~Qt.ItemIsEditable)
                        fixed_value_item.setBackground(Qt.lightGray)
                        self.feature_table.setItem(i, 4, fixed_value_item)
                        
                except Exception as e:
                    print(f"ERROR processing feature {feature_name} at row {i}: {e}")
                    # Continue with next feature instead of crashing
                    continue
            
            # Resize columns
            self.feature_table.resizeColumnsToContents()
            
        except Exception as e:
            print(f"CRITICAL ERROR in update_feature_bounds_table: {e}")
            # Log to text area if available
            if hasattr(self, 'log_text'):
                self.log_text.append(f"Critical error updating feature table: {str(e)}")
                self.log_text.append("Please try refreshing the feature bounds or reloading models.")
    
    def get_combined_feature_info(self):
        """Get combined feature information from all loaded models"""
        combined_info = {}
        
        # Safety check: ensure we have models_data
        if not hasattr(self, 'models_data') or not self.models_data:
            return combined_info
            
        use_original = self.use_original_ranges_check.isChecked() if hasattr(self, 'use_original_ranges_check') else True
        
        for model_data in self.models_data:
            if model_data is not None and 'metadata' in model_data:
                try:
                    metadata = model_data['metadata']
                    feature_types = metadata.get('feature_types', {})
                    feature_bounds = metadata.get('feature_bounds', {})
                    feature_bounds_normalized = metadata.get('feature_bounds_normalized', {})
                    feature_categories = metadata.get('feature_categories', {})
                    feature_names = metadata.get('feature_names', [])
                    
                    for feature_name in feature_names:
                        if feature_name not in combined_info:
                            # Choose bounds based on range mode
                            if use_original:
                                bounds = feature_bounds.get(feature_name, (0.0, 1.0))
                            else:
                                bounds = feature_bounds_normalized.get(feature_name, (0.0, 1.0))
                            
                            combined_info[feature_name] = {
                                'type': feature_types.get(feature_name, 'continuous'),
                                'bounds': bounds,
                                'bounds_original': feature_bounds.get(feature_name, (0.0, 1.0)),
                                'bounds_normalized': feature_bounds_normalized.get(feature_name, (0.0, 1.0)),
                                'categories': feature_categories.get(feature_name, None)
                            }
                except Exception as e:
                    print(f"ERROR in get_combined_feature_info for model: {e}")
                    continue
        
        return combined_info
    
    def on_fixed_feature_changed(self, row, state):
        """Handle fixed feature checkbox change"""
        try:
            is_fixed = state == Qt.Checked
            
            # Safety check: ensure we have combined_features and valid row index
            if not hasattr(self, 'combined_features') or row >= len(self.combined_features):
                print(f"ERROR: Invalid row {row} or missing combined_features")
                return
            
            # Get feature info with safety checks
            feature_name = self.combined_features[row]
            combined_feature_info = self.get_combined_feature_info()
            feature_info = combined_feature_info.get(feature_name, {})
            feature_type = feature_info.get('type', 'continuous')
            
            # Safety check: ensure table items exist
            if self.feature_table.rowCount() <= row:
                print(f"ERROR: Row {row} does not exist in feature table")
                return
                
            # Handle bounds columns with safety checks
            min_item = self.feature_table.item(row, 1)
            max_item = self.feature_table.item(row, 2)
            
            if min_item is None or max_item is None:
                print(f"ERROR: Missing bounds items for row {row}")
                return
            
            # Check if fixed value is a combo box or text item
            fixed_value_widget = self.feature_table.cellWidget(row, 4)
            fixed_value_item = self.feature_table.item(row, 4)
            
            if is_fixed:
                # Disable bounds (if they were editable)
                if feature_type == 'continuous':
                    min_item.setFlags(min_item.flags() & ~Qt.ItemIsEditable)
                    max_item.setFlags(max_item.flags() & ~Qt.ItemIsEditable)
                    min_item.setBackground(Qt.lightGray)
                    max_item.setBackground(Qt.lightGray)
                
                # Enable fixed value
                if fixed_value_widget:  # Combo box
                    fixed_value_widget.setEnabled(True)
                elif fixed_value_item:  # Text item
                    fixed_value_item.setFlags(fixed_value_item.flags() | Qt.ItemIsEditable)
                    fixed_value_item.setBackground(Qt.white)
            else:
                # Enable bounds (if they should be editable)
                if feature_type == 'continuous':
                    min_item.setFlags(min_item.flags() | Qt.ItemIsEditable)
                    max_item.setFlags(max_item.flags() | Qt.ItemIsEditable)
                    min_item.setBackground(Qt.white)
                    max_item.setBackground(Qt.white)
                
                # Disable fixed value
                if fixed_value_widget:  # Combo box
                    fixed_value_widget.setEnabled(False)
                elif fixed_value_item:  # Text item
                    fixed_value_item.setFlags(fixed_value_item.flags() & ~Qt.ItemIsEditable)
                    fixed_value_item.setBackground(Qt.lightGray)
                    
        except Exception as e:
            print(f"ERROR in on_fixed_feature_changed: {e}")
            # Log to text area if available
            if hasattr(self, 'log_text'):
                self.log_text.append(f"Error in feature fix/unfix operation: {str(e)}")
            # Don't re-raise the exception to prevent crashes
    
    def on_auto_mutation_changed(self, state):
        """Handle auto mutation probability checkbox change"""
        is_auto = state == Qt.Checked
        self.mutation_prob_spin.setEnabled(not is_auto)
        
        if is_auto:
            # Calculate 1/n_variables if we have combined features
            if hasattr(self, 'combined_features') and len(self.combined_features) > 0:
                auto_prob = 1.0 / len(self.combined_features)
                self.mutation_prob_spin.setValue(auto_prob)
            else:
                self.mutation_prob_spin.setValue(0.1)  # Default fallback
    
    def on_range_mode_changed(self, state):
        """Handle range mode change (original vs normalized)"""
        use_original = state == Qt.Checked
        self.log_text.append(f"Range mode changed to: {'original' if use_original else 'normalized'}")
        if hasattr(self, 'combined_features'):
            self.update_feature_bounds_table()
    
    def refresh_feature_bounds(self):
        """Refresh feature bounds detection"""
        self.log_text.append("🔄 Refreshing feature bounds...")
        if hasattr(self, 'models_data'):
            # Re-analyze all loaded models
            for i, model_data in enumerate(self.models_data):
                if model_data is not None:
                    self.models_data[i] = self.analyze_feature_types(model_data)
            
            # Update the feature bounds table
            if hasattr(self, 'combined_features'):
                self.update_feature_bounds_table()
                self.log_text.append("✅ Feature bounds refreshed")
        else:
            self.log_text.append("⚠️  No models loaded to refresh bounds from")
    
    def validate_optimization_inputs(self):
        """Validate all optimization inputs before starting"""
        # Check if we have at least 1 model loaded
        loaded_models = [data for data in self.models_data if data is not None]
        if len(loaded_models) < 1:
            raise ValueError("At least 1 model is required for optimization.")
        
        # Check if all loaded models are regression models
        for i, model_data in enumerate(self.models_data):
            if model_data is not None:
                task_type = model_data['metadata'].get('task_type', 'unknown')
                if task_type == 'classification':
                    # This is just a warning, not an error
                    self.log_text.append(f"Warning: Model {i+1} appears to be a classification model. "
                                       "Multi-objective optimization typically works better with regression models.")
        
        # Validate feature bounds with enhanced type support
        combined_feature_info = self.get_combined_feature_info()
        
        for i in range(self.feature_table.rowCount()):
            try:
                feature_name = self.combined_features[i]
                feature_info = combined_feature_info.get(feature_name, {})
                feature_type = feature_info.get('type', 'continuous')
                categories = feature_info.get('categories', None)
                
                # Check if feature is fixed
                fixed_checkbox = self.feature_table.cellWidget(i, 3)
                is_fixed = fixed_checkbox.isChecked()
                
                if is_fixed:
                    # Validate fixed value based on feature type
                    fixed_value_widget = self.feature_table.cellWidget(i, 4)
                    fixed_value_item = self.feature_table.item(i, 4)
                    
                    if fixed_value_widget:  # Combo box
                        fixed_val = float(fixed_value_widget.currentText())
                    elif fixed_value_item:  # Text item
                        fixed_val = float(fixed_value_item.text())
                    else:
                        raise ValueError(f"Cannot validate fixed value for feature '{feature_name}'")
                    
                    if not np.isfinite(fixed_val):
                        raise ValueError(f"Fixed value for feature '{feature_name}' must be a finite number")
                    
                    # Type-specific validation
                    if feature_type in ['binary', 'categorical_encoded', 'categorical'] and categories:
                        if fixed_val not in categories:
                            raise ValueError(f"Fixed value for feature '{feature_name}' must be one of: {categories}")
                    
                else:
                    # Validate bounds (only for continuous features)
                    if feature_type == 'continuous':
                        min_val = float(self.feature_table.item(i, 1).text())
                        max_val = float(self.feature_table.item(i, 2).text())
                        
                        if not (np.isfinite(min_val) and np.isfinite(max_val)):
                            raise ValueError(f"Bounds for feature '{feature_name}' must be finite numbers")
                        
                        if min_val >= max_val:
                            raise ValueError(f"Invalid bounds for feature '{feature_name}': minimum must be less than maximum")
                    # For categorical features, bounds are automatically set and don't need validation
                        
            except (ValueError, AttributeError) as e:
                if "could not convert string to float" in str(e):
                    raise ValueError(f"Invalid numeric value for feature '{feature_name}'. Please enter valid numbers.")
                else:
                    raise e
        
        # Check algorithm parameters
        pop_size = self.population_spin.value()
        n_gen = self.generations_spin.value()
        crossover_prob = self.crossover_prob_spin.value()
        mutation_prob = self.mutation_prob_spin.value()
        crossover_eta = self.crossover_eta_spin.value()
        mutation_eta = self.mutation_eta_spin.value()
        
        if pop_size < 10:
            raise ValueError("Population size must be at least 10")
        if n_gen < 5:
            raise ValueError("Number of generations must be at least 5")
        
        # Validate genetic algorithm parameters
        if not (0.1 <= crossover_prob <= 1.0):
            raise ValueError("Crossover probability must be between 0.1 and 1.0")
        if not (0.001 <= mutation_prob <= 0.5):
            raise ValueError("Mutation probability must be between 0.001 and 0.5")
        if not (1.0 <= crossover_eta <= 50.0):
            raise ValueError("Crossover eta must be between 1.0 and 50.0")
        if not (1.0 <= mutation_eta <= 100.0):
            raise ValueError("Mutation eta must be between 1.0 and 100.0")
            
        # Check if we have any non-fixed features
        all_fixed = True
        for i in range(self.feature_table.rowCount()):
            fixed_checkbox = self.feature_table.cellWidget(i, 3)
            if not fixed_checkbox.isChecked():
                all_fixed = False
                break
        
        if all_fixed:
            raise ValueError("At least one feature must be non-fixed for optimization")
    
    def start_optimization(self):
        """Start the optimization (single-objective or multi-objective)"""
        # Check if we have at least 1 model loaded
        loaded_models = [data for data in self.models_data if data is not None]
        if len(loaded_models) < 1:
            QMessageBox.warning(self, "Warning", "Please load at least 1 model first.")
            return
        
        try:
            # Validate inputs first
            self.validate_optimization_inputs()
            
            # Gather configuration
            config = self.gather_optimization_config()
            
            # Create and start worker
            self.worker = MultiObjectiveOptimizationWorker(config)
            self.worker.progress_updated.connect(self.on_progress_updated)
            self.worker.optimization_completed.connect(self.on_optimization_completed)
            self.worker.status_updated.connect(self.on_status_updated)
            self.worker.error_occurred.connect(self.on_error_occurred)
            
            # Update UI state
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, config['n_generations'])
            self.progress_bar.setValue(0)
            
            # Start optimization
            self.worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start optimization: {str(e)}")
            self.log_text.append(f"Error starting optimization: {str(e)}")
    
    def gather_optimization_config(self) -> Dict[str, Any]:
        """Gather all configuration parameters"""
        # Validate and collect feature bounds and fixed features
        feature_bounds = []
        fixed_features = {}  # Dictionary: feature_index -> fixed_value
        
        for i in range(self.feature_table.rowCount()):
            try:
                feature_name = self.combined_features[i]
                
                # Check if feature is fixed
                fixed_checkbox = self.feature_table.cellWidget(i, 3)
                is_fixed = fixed_checkbox.isChecked()
                
                if is_fixed:
                    # Get fixed value - handle both combo box and text item
                    fixed_value_widget = self.feature_table.cellWidget(i, 4)
                    fixed_value_item = self.feature_table.item(i, 4)
                    
                    if fixed_value_widget:  # Combo box
                        fixed_val = float(fixed_value_widget.currentText())
                    elif fixed_value_item:  # Text item
                        fixed_val = float(fixed_value_item.text())
                    else:
                        raise ValueError(f"Cannot get fixed value for feature {feature_name}")
                    
                    # Validate fixed value is within reasonable bounds
                    feature_info = self.get_combined_feature_info().get(feature_name, {})
                    feature_type = feature_info.get('type', 'continuous')
                    bounds = feature_info.get('bounds', (0.0, 1.0))
                    categories = feature_info.get('categories', None)
                    
                    if feature_type in ['binary', 'categorical_encoded', 'categorical']:
                        if categories and fixed_val not in categories:
                            raise ValueError(f"Fixed value {fixed_val} for feature '{feature_name}' "
                                           f"must be one of: {categories}")
                    else:
                        # For continuous features, check if value is reasonable
                        if not (bounds[0] <= fixed_val <= bounds[1] * 2):  # Allow some flexibility
                            self.log_text.append(f"Warning: Fixed value {fixed_val} for feature '{feature_name}' "
                                                f"is outside suggested range {bounds}")
                    
                    fixed_features[i] = fixed_val
                    # For fixed features, set bounds to the fixed value (with tiny epsilon to avoid numerical issues)
                    epsilon = 1e-8
                    feature_bounds.append((fixed_val - epsilon, fixed_val + epsilon))
                else:
                    # Get bounds
                    min_val = float(self.feature_table.item(i, 1).text())
                    max_val = float(self.feature_table.item(i, 2).text())
                    
                    if min_val >= max_val:
                        raise ValueError(f"Invalid bounds for feature {feature_name}: min >= max")
                    
                    feature_bounds.append((min_val, max_val))
                    
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid configuration for feature {self.combined_features[i]}: {str(e)}")
        
        # Get loaded models and their configurations
        models = []
        model_indices = []
        directions = []
        model_names = []
        objective_names = []
        
        for i, (model_data, widget) in enumerate(zip(self.models_data, self.model_widgets)):
            if model_data is not None:
                models.append(model_data['pipeline'])
                model_indices.append(self.model_indices[i])
                
                # Get optimization direction
                direction = -1 if widget['direction_combo'].currentText() == "Maximize" else 1
                directions.append(direction)
                
                # Get names
                model_names.append(f"Model {i+1}")
                objective_names.append(widget['obj_name_label'].text())
        
        # Collect feature type information for enhanced constraint handling
        feature_types = []
        categorical_ranges = {}
        combined_feature_info = self.get_combined_feature_info()
        
        for i, feature_name in enumerate(self.combined_features):
            feature_info = combined_feature_info.get(feature_name, {})
            feature_type = feature_info.get('type', 'continuous')
            
            # Map feature types to our constraint system
            if feature_type in ['binary', 'categorical_encoded']:
                if feature_info.get('bounds', (0, 1))[1] <= 1:
                    feature_types.append('binary')
                else:
                    feature_types.append('categorical')
                    # For categorical features, create range of valid values
                    bounds = feature_info.get('bounds', (0, 1))
                    categorical_ranges[i] = list(range(int(bounds[0]), int(bounds[1]) + 1))
            elif feature_type == 'categorical':
                feature_types.append('categorical')
                categories = feature_info.get('categories', None)
                if categories:
                    categorical_ranges[i] = categories
                else:
                    bounds = feature_info.get('bounds', (0, 1))
                    categorical_ranges[i] = list(range(int(bounds[0]), int(bounds[1]) + 1))
            else:
                feature_types.append('continuous')
        
        # Handle auto mutation probability
        if self.auto_mutation_check.isChecked():
            mutation_prob = 1.0 / len(self.combined_features)
        else:
            mutation_prob = self.mutation_prob_spin.value()
        
        config = {
            'models': models,
            'model_indices': model_indices,
            'directions': directions,
            'feature_bounds': feature_bounds,
            'fixed_features': fixed_features,
            'feature_names': self.combined_features,
            'population_size': self.population_spin.value(),
            'n_generations': self.generations_spin.value(),
            'model_names': model_names,
            'objective_names': objective_names,
            # Enhanced constraint handling
            'feature_types': feature_types,
            'categorical_ranges': categorical_ranges,
            # Algorithm selection
            'algorithm': self.algorithm_combo.currentText(),
            # NSGA-II specific parameters
            'crossover_prob': self.crossover_prob_spin.value(),
            'crossover_eta': self.crossover_eta_spin.value(),
            'mutation_prob': mutation_prob,
            'mutation_eta': self.mutation_eta_spin.value(),
            'selection_method': self.selection_combo.currentText(),
            'tournament_size': self.tournament_size_spin.value(),
            'eliminate_duplicates': self.eliminate_duplicates_check.isChecked(),
            'random_seed': self.seed_spin.value() if self.seed_spin.value() != -1 else None,
            'verbose': self.verbose_check.isChecked(),
            # Advanced diversity parameters
            'diversity_noise_scale': self.diversity_noise_spin.value(),
            # Early stopping parameters
            'enable_early_stopping': self.enable_early_stopping_check.isChecked(),
            'early_stop_patience': self.early_stop_patience_spin.value(),
            'early_stop_delta': 1e-6,  # Can be made configurable if needed
            # Checkpoint parameters
            'enable_checkpoints': self.enable_checkpoints_check.isChecked(),
            'checkpoint_interval': self.checkpoint_interval_spin.value(),
            'max_checkpoints': 5,  # Can be made configurable if needed
            
            # Robust optimization parameters
            'enable_robust_optimization': self.enable_robust_check.isChecked(),
            'robust_method': self.robust_method_combo.currentText().lower().replace('-', '_'),
            'robust_samples': self.robust_samples_spin.value(),
            'robust_noise_level': self.robust_noise_spin.value(),
            
            # Constraint handling parameters
            'enable_explicit_constraints': self.enable_constraints_check.isChecked(),
            'adaptive_penalty_coefficients': self.adaptive_penalty_check.isChecked()
        }
        
        return config
    
    def stop_optimization(self):
        """Stop the running optimization"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        
        # Update UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def on_progress_updated(self, data: Dict[str, Any]):
        """Handle enhanced progress updates from worker"""
        try:
            generation = data.get('generation', 0)
            n_solutions = data.get('n_solutions', 0)
            convergence_metrics = data.get('convergence_metrics', {})
            
            # Store progress data for visualization
            self.all_progress_data.append(data)
            
            # Update progress bar
            self.progress_bar.setValue(generation)
            
            # Update log with enhanced metrics
            min_objs = convergence_metrics.get('min_objectives', [])
            if min_objs and len(min_objs) > 0:
                min_str = ", ".join([f"{val:.4f}" for val in min_objs])
                self.log_text.append(f"Gen {generation}: {n_solutions} solutions, Best: [{min_str}]")
            else:
                self.log_text.append(f"Generation {generation}: {n_solutions} solutions")
            
            # Update all visualizations
            objectives = data.get('objectives')
            if objectives is not None and len(objectives) > 0:
                self.update_pareto_plot(objectives)
                self.update_convergence_plot()
                self.update_evolution_plot()
                
        except Exception as e:
            print(f"Progress update error: {e}")
    
    def update_pareto_plot(self, objectives):
        """Update the optimization results plot (single-objective or Pareto front)"""
        try:
            self.pareto_figure.clear()
            
            n_objectives = objectives.shape[1]
            
            if n_objectives == 1:
                # Single objective optimization - show histogram and evolution
                ax = self.pareto_figure.add_subplot(111)
                
                # Get objective name
                obj_name = "Objective"
                for widget in self.model_widgets:
                    if widget['model_data'] is not None:
                        obj_name = widget['obj_name_label'].text()
                        break
                
                # Plot histogram of objective values
                ax.hist(objectives[:, 0], bins=min(20, len(objectives)), alpha=0.7, edgecolor='black')
                ax.set_xlabel(f"{obj_name} Value")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Single-Objective Optimization Results\n{obj_name}")
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                best_val = np.min(objectives[:, 0])
                mean_val = np.mean(objectives[:, 0])
                std_val = np.std(objectives[:, 0])
                ax.axvline(best_val, color='red', linestyle='--', linewidth=2, label=f'Best: {best_val:.4f}')
                ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
                ax.legend()
                
            elif n_objectives == 2:
                # Simple 2D plot
                ax = self.pareto_figure.add_subplot(111)
                ax.scatter(objectives[:, 0], objectives[:, 1], alpha=0.6, s=50)
                
                # Get objective names
                obj_names = []
                for widget in self.model_widgets:
                    if widget['model_data'] is not None:
                        obj_names.append(widget['obj_name_label'].text())
                
                if len(obj_names) >= 2:
                    ax.set_xlabel(obj_names[0])
                    ax.set_ylabel(obj_names[1])
                else:
                    ax.set_xlabel("Objective 1")
                    ax.set_ylabel("Objective 2")
                
                ax.set_title("Pareto Front")
                ax.grid(True, alpha=0.3)
                
            elif n_objectives == 3:
                # Create 2x2 subplot layout for 3D view and 2D projections
                # Main 3D plot (top row, spanning both columns)
                ax_3d = self.pareto_figure.add_subplot(2, 3, (1, 3), projection='3d')
                scatter_3d = ax_3d.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], 
                                         alpha=0.7, s=50, c=objectives[:, 0], cmap='viridis')
                
                # Get objective names
                obj_names = []
                for widget in self.model_widgets:
                    if widget['model_data'] is not None:
                        obj_names.append(widget['obj_name_label'].text())
                
                if len(obj_names) >= 3:
                    xlabel, ylabel, zlabel = obj_names[0], obj_names[1], obj_names[2]
                else:
                    xlabel, ylabel, zlabel = "Objective 1", "Objective 2", "Objective 3"
                
                ax_3d.set_xlabel(xlabel)
                ax_3d.set_ylabel(ylabel)
                ax_3d.set_zlabel(zlabel)
                ax_3d.set_title("3D Pareto Front")
                
                # 2D projection plots (bottom row)
                # XY projection (Obj1 vs Obj2, colored by Obj3)
                ax_xy = self.pareto_figure.add_subplot(2, 3, 4)
                scatter_xy = ax_xy.scatter(objectives[:, 0], objectives[:, 1], 
                                         alpha=0.7, s=30, c=objectives[:, 2], cmap='plasma', edgecolors='black', linewidth=0.5)
                ax_xy.set_xlabel(xlabel)
                ax_xy.set_ylabel(ylabel)
                ax_xy.set_title(f'XY Projection\n(Color: {zlabel})')
                ax_xy.grid(True, alpha=0.3)
                
                # XZ projection (Obj1 vs Obj3, colored by Obj2)
                ax_xz = self.pareto_figure.add_subplot(2, 3, 5)
                scatter_xz = ax_xz.scatter(objectives[:, 0], objectives[:, 2], 
                                         alpha=0.7, s=30, c=objectives[:, 1], cmap='plasma', edgecolors='black', linewidth=0.5)
                ax_xz.set_xlabel(xlabel)
                ax_xz.set_ylabel(zlabel)
                ax_xz.set_title(f'XZ Projection\n(Color: {ylabel})')
                ax_xz.grid(True, alpha=0.3)
                
                # YZ projection (Obj2 vs Obj3, colored by Obj1)
                ax_yz = self.pareto_figure.add_subplot(2, 3, 6)
                scatter_yz = ax_yz.scatter(objectives[:, 1], objectives[:, 2], 
                                         alpha=0.7, s=30, c=objectives[:, 0], cmap='plasma', edgecolors='black', linewidth=0.5)
                ax_yz.set_xlabel(ylabel)
                ax_yz.set_ylabel(zlabel)
                ax_yz.set_title(f'YZ Projection\n(Color: {xlabel})')
                ax_yz.grid(True, alpha=0.3)
                
                # Add colorbars for 2D projections
                try:
                    # Colorbar for XY projection
                    cbar_xy = self.pareto_figure.colorbar(scatter_xy, ax=ax_xy, shrink=0.6, aspect=15, pad=0.02)
                    cbar_xy.set_label(zlabel, rotation=270, labelpad=12, fontsize=8)
                    
                    # Colorbar for XZ projection  
                    cbar_xz = self.pareto_figure.colorbar(scatter_xz, ax=ax_xz, shrink=0.6, aspect=15, pad=0.02)
                    cbar_xz.set_label(ylabel, rotation=270, labelpad=12, fontsize=8)
                    
                    # Colorbar for YZ projection
                    cbar_yz = self.pareto_figure.colorbar(scatter_yz, ax=ax_yz, shrink=0.6, aspect=15, pad=0.02)
                    cbar_yz.set_label(xlabel, rotation=270, labelpad=12, fontsize=8)
                except Exception as e:
                    print(f"Warning: Could not add colorbars: {e}")
                
                # Adjust layout to prevent overlap
                self.pareto_figure.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, 
                                                 wspace=0.3, hspace=0.4)
                
            else:
                # Multiple 2D projections for >3 objectives
                import math
                n_plots = min(6, n_objectives * (n_objectives - 1) // 2)  # Limit to 6 plots
                n_cols = min(3, int(math.ceil(math.sqrt(n_plots))))
                n_rows = int(math.ceil(n_plots / n_cols))
                
                plot_idx = 0
                obj_names = []
                for widget in self.model_widgets:
                    if widget['model_data'] is not None:
                        obj_names.append(widget['obj_name_label'].text())
                
                for i in range(n_objectives):
                    for j in range(i + 1, n_objectives):
                        if plot_idx >= n_plots:
                            break
                        
                        ax = self.pareto_figure.add_subplot(n_rows, n_cols, plot_idx + 1)
                        ax.scatter(objectives[:, i], objectives[:, j], alpha=0.6, s=30)
                        
                        x_name = obj_names[i] if i < len(obj_names) else f"Obj {i+1}"
                        y_name = obj_names[j] if j < len(obj_names) else f"Obj {j+1}"
                        ax.set_xlabel(x_name)
                        ax.set_ylabel(y_name)
                        ax.set_title(f"{x_name} vs {y_name}")
                        ax.grid(True, alpha=0.3)
                        
                        plot_idx += 1
                    
                    if plot_idx >= n_plots:
                        break
            
            self.pareto_figure.tight_layout()
            self.pareto_canvas.draw()
            
        except Exception as e:
            print(f"Plot update error: {e}")
    
    def update_convergence_plot(self):
        """Update the convergence tracking plot"""
        try:
            if len(self.all_progress_data) < 2:
                return
            
            self.convergence_figure.clear()
            
            # Extract data from progress history
            generations = [d.get('generation', 0) for d in self.all_progress_data]
            hypervolumes = [d.get('hypervolume_history', [])[-1] if d.get('hypervolume_history') else 0 
                           for d in self.all_progress_data]
            best_objectives = [d.get('best_objectives_history', [])[-1] if d.get('best_objectives_history') else [] 
                              for d in self.all_progress_data]
            
            # Create subplots
            n_objectives = len(best_objectives[-1]) if best_objectives[-1] else 2
            
            if n_objectives <= 4:
                # Show individual objective convergence + hypervolume
                ax1 = self.convergence_figure.add_subplot(2, 1, 1)
                
                # Plot best objective values over generations
                obj_names = []
                for widget in self.model_widgets:
                    if widget['model_data'] is not None:
                        obj_names.append(widget['obj_name_label'].text())
                
                colors = plt.cm.tab10(np.linspace(0, 1, n_objectives))
                for i in range(n_objectives):
                    obj_values = [objs[i] if i < len(objs) else float('inf') for objs in best_objectives]
                    obj_name = obj_names[i] if i < len(obj_names) else f'Objective {i+1}'
                    ax1.plot(generations, obj_values, '-o', color=colors[i], 
                            label=obj_name, markersize=3, linewidth=2)
                
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('Best Objective Value')
                ax1.set_title('Objective Convergence')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot hypervolume
                ax2 = self.convergence_figure.add_subplot(2, 1, 2)
                ax2.plot(generations, hypervolumes, '-o', color='red', markersize=3, linewidth=2)
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Hypervolume')
                ax2.set_title('Hypervolume Indicator')
                ax2.grid(True, alpha=0.3)
            else:
                # For many objectives, show only hypervolume
                ax = self.convergence_figure.add_subplot(1, 1, 1)
                ax.plot(generations, hypervolumes, '-o', color='red', markersize=3, linewidth=2)
                ax.set_xlabel('Generation')
                ax.set_ylabel('Hypervolume')
                ax.set_title('Convergence Progress (Hypervolume Indicator)')
                ax.grid(True, alpha=0.3)
            
            self.convergence_figure.tight_layout()
            self.convergence_canvas.draw()
            
        except Exception as e:
            print(f"Convergence plot update error: {e}")

    def update_evolution_plot(self):
        """Update the evolution animation plot showing solution distribution over time"""
        try:
            if len(self.all_progress_data) < 2:
                return
            
            self.evolution_figure.clear()
            
            # Get current and previous data
            current_data = self.all_progress_data[-1]
            current_objectives = current_data.get('objectives')
            
            if current_objectives is None or len(current_objectives) == 0:
                return
            
            n_objectives = current_objectives.shape[1]
            
            if n_objectives == 2:
                # 2D evolution plot
                ax = self.evolution_figure.add_subplot(1, 1, 1)
                
                # Plot evolution path for a subset of generations
                step = max(1, len(self.all_progress_data) // 5)  # Show ~5 generations
                colors = plt.cm.viridis(np.linspace(0, 1, len(self.all_progress_data[::step])))
                
                for i, (data, color) in enumerate(zip(self.all_progress_data[::step], colors)):
                    objs = data.get('objectives')
                    if objs is not None and len(objs) > 0:
                        gen = data.get('generation', 0)
                        alpha = 0.3 + 0.7 * (i / len(colors))  # Fade older generations
                        size = 20 + 30 * (i / len(colors))     # Bigger dots for recent generations
                        ax.scatter(objs[:, 0], objs[:, 1], alpha=alpha, s=size, c=[color], 
                                 label=f'Gen {gen}' if i % 2 == 0 else '')
                
                # Get objective names
                obj_names = []
                for widget in self.model_widgets:
                    if widget['model_data'] is not None:
                        obj_names.append(widget['obj_name_label'].text())
                
                if len(obj_names) >= 2:
                    ax.set_xlabel(obj_names[0])
                    ax.set_ylabel(obj_names[1])
                else:
                    ax.set_xlabel('Objective 1')
                    ax.set_ylabel('Objective 2')
                
                ax.set_title(f'Solution Evolution (Generation {current_data.get("generation", 0)})')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            elif n_objectives == 3:
                # 3D evolution plot with projections
                # Main 3D plot (top, spanning columns)
                ax_3d = self.evolution_figure.add_subplot(2, 2, (1, 2), projection='3d')
                
                # Show fewer generations for 3D clarity
                step = max(1, len(self.all_progress_data) // 3)
                colors = plt.cm.viridis(np.linspace(0, 1, len(self.all_progress_data[::step])))
                
                for i, (data, color) in enumerate(zip(self.all_progress_data[::step], colors)):
                    objs = data.get('objectives')
                    if objs is not None and len(objs) > 0:
                        gen = data.get('generation', 0)
                        alpha = 0.3 + 0.7 * (i / len(colors))
                        size = 20 + 30 * (i / len(colors))
                        ax_3d.scatter(objs[:, 0], objs[:, 1], objs[:, 2], alpha=alpha, s=size, c=[color],
                                     label=f'Gen {gen}' if i % 2 == 0 else '')
                
                # 2D projection evolution plots
                ax_xy = self.evolution_figure.add_subplot(2, 2, 3)
                ax_xz = self.evolution_figure.add_subplot(2, 2, 4)
                
                # Plot evolution in 2D projections
                for i, (data, color) in enumerate(zip(self.all_progress_data[::step], colors)):
                    objs = data.get('objectives')
                    if objs is not None and len(objs) > 0:
                        alpha = 0.3 + 0.7 * (i / len(colors))
                        size = 15 + 20 * (i / len(colors))
                        
                        # XY projection
                        ax_xy.scatter(objs[:, 0], objs[:, 1], alpha=alpha, s=size, c=[color])
                        
                        # XZ projection  
                        ax_xz.scatter(objs[:, 0], objs[:, 2], alpha=alpha, s=size, c=[color])
                
                # Get objective names
                obj_names = []
                for widget in self.model_widgets:
                    if widget['model_data'] is not None:
                        obj_names.append(widget['obj_name_label'].text())
                
                if len(obj_names) >= 3:
                    xlabel, ylabel, zlabel = obj_names[0], obj_names[1], obj_names[2]
                else:
                    xlabel, ylabel, zlabel = 'Objective 1', 'Objective 2', 'Objective 3'
                
                # Set labels for all plots
                ax_3d.set_xlabel(xlabel)
                ax_3d.set_ylabel(ylabel)
                ax_3d.set_zlabel(zlabel)
                ax_3d.set_title(f'3D Evolution (Gen {current_data.get("generation", 0)})')
                
                # Set labels for 2D projections
                ax_xy.set_xlabel(xlabel)
                ax_xy.set_ylabel(ylabel)
                ax_xy.set_title('XY Projection')
                ax_xy.grid(True, alpha=0.3)
                
                ax_xz.set_xlabel(xlabel)
                ax_xz.set_ylabel(zlabel)
                ax_xz.set_title('XZ Projection')
                ax_xz.grid(True, alpha=0.3)
                
            else:
                # For >3 objectives, show parallel coordinates plot
                ax = self.evolution_figure.add_subplot(1, 1, 1)
                
                # Get recent generations
                recent_data = self.all_progress_data[-3:]  # Last 3 generations
                colors = ['lightblue', 'blue', 'darkblue']
                
                for i, (data, color) in enumerate(zip(recent_data, colors)):
                    objs = data.get('objectives')
                    if objs is not None and len(objs) > 0:
                        gen = data.get('generation', 0)
                        
                        # Normalize objectives for parallel coordinates
                        obj_norm = (objs - np.min(objs, axis=0)) / (np.max(objs, axis=0) - np.min(objs, axis=0) + 1e-10)
                        
                        # Plot a subset of solutions
                        n_solutions = min(50, len(obj_norm))
                        indices = np.random.choice(len(obj_norm), n_solutions, replace=False)
                        
                        for sol in obj_norm[indices]:
                            ax.plot(range(n_objectives), sol, color=color, alpha=0.3, linewidth=1)
                
                # Get objective names
                obj_names = []
                for widget in self.model_widgets:
                    if widget['model_data'] is not None:
                        obj_names.append(widget['obj_name_label'].text())
                
                if len(obj_names) >= n_objectives:
                    ax.set_xticks(range(n_objectives))
                    ax.set_xticklabels([obj_names[i] for i in range(n_objectives)], rotation=45)
                else:
                    ax.set_xticks(range(n_objectives))
                    ax.set_xticklabels([f'Obj {i+1}' for i in range(n_objectives)], rotation=45)
                
                ax.set_ylabel('Normalized Objective Value')
                ax.set_title(f'Multi-Objective Evolution (Gen {current_data.get("generation", 0)})')
                ax.grid(True, alpha=0.3)
            
            self.evolution_figure.tight_layout()
            self.evolution_canvas.draw()
            
        except Exception as e:
            print(f"Evolution plot update error: {e}")
    
    def on_optimization_completed(self, results: Dict[str, Any]):
        """Handle optimization completion"""
        try:
            # Update UI state
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.export_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            
            # Store results and config
            self.optimization_results = results
            self.config = self.worker.config if self.worker else {}
            
            # Update final plot
            pareto_front = results['pareto_front']
            self.update_pareto_plot(pareto_front)
            
            # Update solutions table
            self.update_solutions_table(results)
            
            # Report completion
            n_solutions = results['n_solutions']
            n_objectives = results['n_objectives']
            
            self.log_text.append("=" * 50)
            self.log_text.append("OPTIMIZATION COMPLETED SUCCESSFULLY!")
            self.log_text.append("=" * 50)
            self.log_text.append(f"Found {n_solutions} Pareto-optimal solutions with {n_objectives} objectives.")
            
            # Show final convergence information
            if self.all_progress_data:
                final_data = self.all_progress_data[-1]
                final_metrics = final_data.get('convergence_metrics', {})
                if final_metrics:
                    min_objs = final_metrics.get('min_objectives', [])
                    if min_objs:
                        self.log_text.append("")
                        self.log_text.append("📊 FINAL CONVERGENCE METRICS:")
                        for i, min_val in enumerate(min_objs):
                            obj_name = f"Objective {i+1}"
                            # Try to get actual objective name
                            if i < len(self.model_widgets) and self.model_widgets[i]['model_data'] is not None:
                                obj_name = self.model_widgets[i]['obj_name_label'].text()
                            self.log_text.append(f"   • Best {obj_name}: {min_val:.6f}")
            
            self.log_text.append("=" * 50)
            
            # Generate automated report
            try:
                report_path = self.generate_optimization_report(results)
                if report_path:
                    self.log_text.append(f"📄 Detailed report generated: {report_path}")
            except Exception as e:
                self.log_text.append(f"Report generation failed: {str(e)}")
            
        except Exception as e:
            self.log_text.append(f"Error processing results: {str(e)}")
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate automated optimization report in Markdown format"""
        try:
            timestamp = datetime.datetime.now()
            pareto_front = results['pareto_front']
            n_solutions = results['n_solutions']
            n_objectives = results['n_objectives']
            feature_names = results['feature_names']
            objective_names = results['objective_names']
            fixed_features = results.get('fixed_features', {})
            
            # Create report content
            report_content = f"""# Multi-Objective Optimization Report

## Summary
- **Date**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- **Algorithm**: {self.config.get('algorithm', 'NSGA-II')}
- **Objectives**: {n_objectives} ({', '.join(objective_names)})
- **Features**: {len(feature_names)} total ({len(feature_names) - len(fixed_features)} optimized, {len(fixed_features)} fixed)
- **Solutions Found**: {n_solutions}

## Configuration
- **Population Size**: {self.config.get('population_size', 'N/A')}
- **Generations**: {self.config.get('n_generations', 'N/A')}
- **Crossover Probability**: {self.config.get('crossover_prob', 'N/A'):.3f}
- **Mutation Probability**: {self.config.get('mutation_prob', 'N/A'):.3f}
- **Early Stopping**: {"Enabled" if self.config.get('enable_early_stopping', False) else "Disabled"}
- **Checkpoints**: {"Enabled" if self.config.get('enable_checkpoints', False) else "Disabled"}

## Optimization Results

### Best Solutions (Top 5)
"""
            
            # Add top 5 solutions
            n_display = min(5, n_solutions)
            for i in range(n_display):
                report_content += f"\n#### Solution {i+1}\n"
                for j, obj_name in enumerate(objective_names):
                    report_content += f"- **{obj_name}**: {pareto_front[i, j]:.6f}\n"
                report_content += "\n"
            
            # Add fixed features information
            if fixed_features:
                report_content += "\n### Fixed Features\n"
                for feature_idx, value in fixed_features.items():
                    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
                    report_content += f"- **{feature_name}**: {value:.6f}\n"
            
            # Add statistics
            report_content += "\n### Objective Statistics\n"
            for j, obj_name in enumerate(objective_names):
                obj_values = pareto_front[:, j]
                report_content += f"\n#### {obj_name}\n"
                report_content += f"- **Best**: {np.min(obj_values):.6f}\n"
                report_content += f"- **Worst**: {np.max(obj_values):.6f}\n"
                report_content += f"- **Mean**: {np.mean(obj_values):.6f}\n"
                report_content += f"- **Std Dev**: {np.std(obj_values):.6f}\n"
            
            # Add convergence information if available
            if hasattr(self, 'all_progress_data') and self.all_progress_data:
                final_data = self.all_progress_data[-1]
                final_gen = final_data.get('generation', 'N/A')
                report_content += f"\n### Convergence\n"
                report_content += f"- **Final Generation**: {final_gen}\n"
                report_content += f"- **Total Evaluations**: {(final_gen + 1) * self.config.get('population_size', 0) if isinstance(final_gen, int) else 'N/A'}\n"
            
            # Add feature bounds information
            report_content += "\n### Feature Configuration\n"
            combined_feature_info = self.get_combined_feature_info()
            for feature_name in feature_names:
                feature_info = combined_feature_info.get(feature_name, {})
                bounds = feature_info.get('bounds', (0, 1))
                feature_type = feature_info.get('type', 'continuous')
                
                if feature_names.index(feature_name) in fixed_features:
                    fixed_value = fixed_features[feature_names.index(feature_name)]
                    report_content += f"- **{feature_name}** ({feature_type}): Fixed at {fixed_value:.6f}\n"
                else:
                    report_content += f"- **{feature_name}** ({feature_type}): [{bounds[0]:.6f}, {bounds[1]:.6f}]\n"
            
            # Write report to file
            report_filename = f"optimization_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return report_filename
            
        except Exception as e:
            print(f"Report generation error: {e}")
            return None
    
    def update_solutions_table(self, results: Dict[str, Any]):
        """Update the solutions table with results"""
        try:
            pareto_front = results['pareto_front']
            pareto_solutions = results['pareto_solutions']
            feature_names = results['feature_names']
            objective_names = results['objective_names']
            fixed_features = results.get('fixed_features', {})
            
            # Set up table
            n_solutions = len(pareto_front)
            n_features = len(feature_names)
            n_objectives = len(objective_names)
            
            headers = ['Solution ID'] + objective_names + feature_names
            self.solutions_table.setColumnCount(len(headers))
            self.solutions_table.setHorizontalHeaderLabels(headers)
            self.solutions_table.setRowCount(n_solutions)
            
            # Fill table
            for i in range(n_solutions):
                # Solution ID
                self.solutions_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                
                # Objectives
                for j in range(n_objectives):
                    value = pareto_front[i, j]
                    self.solutions_table.setItem(i, j + 1, QTableWidgetItem(f"{value:.4f}"))
                
                # Features - handle fixed features specially
                for j in range(n_features):
                    if j in fixed_features:
                        # For fixed features, show the fixed value
                        value = fixed_features[j]
                        item = QTableWidgetItem(f"{value:.4f}")
                        # Highlight fixed features with different background
                        item.setBackground(Qt.lightGray)
                        item.setToolTip(f"Fixed feature: {feature_names[j]} = {value:.4f}")
                    else:
                        # For variable features, show the optimized value
                        value = pareto_solutions[i, j]
                        item = QTableWidgetItem(f"{value:.4f}")
                    
                    self.solutions_table.setItem(i, j + 1 + n_objectives, item)
            
            # Resize columns
            self.solutions_table.resizeColumnsToContents()
            
            # Log information about fixed features
            if fixed_features:
                self.log_text.append(f"📌 Fixed features in solutions table:")
                for feature_idx, value in fixed_features.items():
                    feature_name = feature_names[feature_idx]
                    self.log_text.append(f"   • {feature_name}: {value:.4f} (highlighted in gray)")
            
        except Exception as e:
            self.log_text.append(f"Error updating solutions table: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_status_updated(self, message: str):
        """Handle status updates from worker"""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
    
    def on_error_occurred(self, error_message: str):
        """Handle errors from worker"""
        QMessageBox.critical(self, "Optimization Error", error_message)
        self.log_text.append(f"ERROR: {error_message}")
        
        # Reset UI state
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def export_results(self):
        """Export optimization results to CSV"""
        if not hasattr(self, 'optimization_results'):
            QMessageBox.warning(self, "Warning", "No results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "pareto_solutions.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                results = self.optimization_results
                pareto_front = results['pareto_front']
                pareto_solutions = results['pareto_solutions']
                feature_names = results['feature_names']
                
                # Create DataFrame
                data = {}
                objective_names = results['objective_names']
                
                # Add objective columns
                for i, obj_name in enumerate(objective_names):
                    data[obj_name] = pareto_front[:, i]
                
                # Add feature columns
                for i, feature_name in enumerate(feature_names):
                    data[feature_name] = pareto_solutions[:, i]
                
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
                self.log_text.append(f"Results exported to: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")
                self.log_text.append(f"Export error: {str(e)}")


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create and show the module
    window = MultiObjectiveOptimizationModule()
    window.show()
    
    sys.exit(app.exec_()) 