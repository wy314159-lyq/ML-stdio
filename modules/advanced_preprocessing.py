"""
Advanced Data Preprocessing Module for MatSci-ML Studio
高级数据预处理系统，包含异常值检测、智能缺失值处理等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                            QLabel, QPushButton, QTextEdit, QComboBox,
                            QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
                            QTableWidget, QTableWidgetItem, QScrollArea,
                            QProgressBar, QSplitter, QFrame, QListWidget,
                            QListWidgetItem, QDialog, QFileDialog, QMessageBox,
                            QLineEdit, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor

from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                 PowerTransformer, QuantileTransformer)
from sklearn.impute import SimpleImputer, KNNImputer
# 启用实验性功能
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

# Fix potential numpy/Qt compatibility issues and set backend BEFORE importing matplotlib
import os
os.environ['QT_API'] = 'pyqt5'

# Ensure matplotlib uses Qt5Agg backend
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not available, some visualizations may be simplified")
    sns = None

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

class PerformanceOptimizer:
    """Performance optimization utilities for large datasets"""
    
    @staticmethod
    def is_large_dataset(data: pd.DataFrame, threshold_mb: int = 100) -> bool:
        """Check if dataset is considered large"""
        if data is None:
            return False
        memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        return memory_usage > threshold_mb
    
    @staticmethod
    def optimize_dtypes(data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage"""
        optimized = data.copy()
        
        for col in optimized.columns:
            col_type = optimized[col].dtype
            
            if col_type != 'object':
                c_min = optimized[col].min()
                c_max = optimized[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized[col] = optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized[col] = optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized[col] = optimized[col].astype(np.int32)
                        
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        optimized[col] = optimized[col].astype(np.float32)
        
        return optimized
    
    @staticmethod
    def get_memory_usage(data: pd.DataFrame) -> Dict[str, float]:
        """Get detailed memory usage information"""
        if data is None:
            return {}
        
        total_memory = data.memory_usage(deep=True).sum()
        return {
            'total_mb': total_memory / (1024 * 1024),
            'rows': len(data),
            'columns': len(data.columns),
            'mb_per_row': total_memory / (len(data) * 1024 * 1024) if len(data) > 0 else 0
        }
    
    @staticmethod
    def sample_data_for_preview(data: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
        """Sample data for preview if dataset is too large"""
        if len(data) <= max_rows:
            return data
        return data.sample(n=max_rows, random_state=42)

class OutlierDetector:
    """异常值检测器"""
    
    def __init__(self):
        self.methods = {
            'IQR': self._iqr_detection,
            'Z-Score': self._zscore_detection,
            'Isolation Forest': self._isolation_forest_detection,
            'Local Outlier Factor': self._lof_detection,
            'Elliptic Envelope': self._elliptic_envelope_detection,
            'One-Class SVM': self._oneclass_svm_detection
        }
        
        self.handling_methods = {
            'Remove Rows': self._remove_outlier_rows,
            'Cap Values': self._cap_outliers,
            'Replace with NaN': self._replace_with_nan,
            'Replace with Median': self._replace_with_median,
            'Replace with Mean': self._replace_with_mean
        }
    
    def detect_outliers(self, X: pd.DataFrame, method: str = 'IQR', **kwargs) -> pd.DataFrame:
        """检测异常值"""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        
        return self.methods[method](X, **kwargs)
    
    def _iqr_detection(self, X: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
        """IQR方法检测异常值"""
        outliers = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers[col] = (X[col] < lower_bound) | (X[col] > upper_bound)
        
        return outliers
    
    def _zscore_detection(self, X: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Z-Score方法检测异常值"""
        outliers = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        for col in X.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
            outliers[col] = z_scores > threshold
        
        return outliers
    
    def _isolation_forest_detection(self, X: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        """Isolation Forest检测异常值"""
        outliers = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_numeric = X[numeric_cols].fillna(X[numeric_cols].median())
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(X_numeric)
            
            # 将结果应用到所有数值列
            outlier_mask = outlier_labels == -1
            for col in numeric_cols:
                outliers[col] = outlier_mask
        
        return outliers
    
    def _lof_detection(self, X: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.1) -> pd.DataFrame:
        """Local Outlier Factor检测异常值"""
        outliers = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_numeric = X[numeric_cols].fillna(X[numeric_cols].median())
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outlier_labels = lof.fit_predict(X_numeric)
            
            outlier_mask = outlier_labels == -1
            for col in numeric_cols:
                outliers[col] = outlier_mask
        
        return outliers
    
    def _elliptic_envelope_detection(self, X: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
        """椭圆包络检测异常值"""
        outliers = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_numeric = X[numeric_cols].fillna(X[numeric_cols].median())
            
            envelope = EllipticEnvelope(contamination=contamination, random_state=42)
            outlier_labels = envelope.fit_predict(X_numeric)
            
            outlier_mask = outlier_labels == -1
            for col in numeric_cols:
                outliers[col] = outlier_mask
        
        return outliers
    
    def _oneclass_svm_detection(self, X: pd.DataFrame, nu: float = 0.1) -> pd.DataFrame:
        """One-Class SVM检测异常值"""
        outliers = pd.DataFrame(False, index=X.index, columns=X.columns)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_numeric = X[numeric_cols].fillna(X[numeric_cols].median())
            
            svm = OneClassSVM(nu=nu, random_state=42)
            outlier_labels = svm.fit_predict(X_numeric)
            
            outlier_mask = outlier_labels == -1
            for col in numeric_cols:
                outliers[col] = outlier_mask
        
        return outliers
    
    def handle_outliers(self, X: pd.DataFrame, outliers: pd.DataFrame, method: str = 'Remove Rows', selected_columns: List[str] = None) -> pd.DataFrame:
        """处理异常值"""
        if method not in self.handling_methods:
            raise ValueError(f"Unknown handling method: {method}")
        
        # 如果指定了列，只处理这些列的异常值
        if selected_columns:
            outliers_subset = outliers[selected_columns]
        else:
            outliers_subset = outliers
            
        return self.handling_methods[method](X, outliers_subset)
    
    def _remove_outlier_rows(self, X: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """删除包含异常值的行"""
        # 标识任何列中有异常值的行
        outlier_rows = outliers.any(axis=1)
        return X[~outlier_rows].reset_index(drop=True)
    
    def _cap_outliers(self, X: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """对异常值进行盖帽处理"""
        X_processed = X.copy()
        
        for col in outliers.columns:
            if outliers[col].any() and col in X_processed.columns:
                # 计算该列的1%和99%分位数
                q01 = X_processed[col].quantile(0.01)
                q99 = X_processed[col].quantile(0.99)
                
                # 将异常值盖帽到分位数范围内
                outlier_mask = outliers[col]
                X_processed.loc[outlier_mask & (X_processed[col] < q01), col] = q01
                X_processed.loc[outlier_mask & (X_processed[col] > q99), col] = q99
                
        return X_processed
    
    def _replace_with_nan(self, X: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """将异常值替换为NaN"""
        X_processed = X.copy()
        
        for col in outliers.columns:
            if col in X_processed.columns:
                X_processed.loc[outliers[col], col] = np.nan
                
        return X_processed
    
    def _replace_with_median(self, X: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """将异常值替换为中位数"""
        X_processed = X.copy()
        
        for col in outliers.columns:
            if col in X_processed.columns and X_processed[col].dtype in ['int64', 'float64']:
                median_val = X_processed[col].median()
                X_processed.loc[outliers[col], col] = median_val
                
        return X_processed
    
    def _replace_with_mean(self, X: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
        """将异常值替换为均值"""
        X_processed = X.copy()
        
        for col in outliers.columns:
            if col in X_processed.columns and X_processed[col].dtype in ['int64', 'float64']:
                mean_val = X_processed[col].mean()
                X_processed.loc[outliers[col], col] = mean_val
                
        return X_processed

class SmartImputer:
    """智能缺失值填补器"""
    
    def __init__(self):
        self.methods = {
            'mean': lambda x, **kwargs: x.fillna(x.mean()),
            'median': lambda x, **kwargs: x.fillna(x.median()),
            'mode': lambda x, **kwargs: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
            'forward_fill': lambda x, **kwargs: x.fillna(method='ffill'),
            'backward_fill': lambda x, **kwargs: x.fillna(method='bfill'),
            'interpolate': lambda x, **kwargs: x.interpolate(),
            'constant': self._constant_impute,
            'knn': self._knn_impute,
            'iterative': self._iterative_impute
        }
    
    def impute(self, X: pd.DataFrame, method: str = 'median', **kwargs) -> pd.DataFrame:
        """填补缺失值"""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        
        return self.methods[method](X, **kwargs)
    
    def _knn_impute(self, X: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """KNN填补"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        result = X.copy()
        
        # 数值列使用KNN
        if len(numeric_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            result[numeric_cols] = knn_imputer.fit_transform(result[numeric_cols])
        
        # 分类列使用众数
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = result[col].mode()
                if len(mode_value) > 0:
                    result[col].fillna(mode_value.iloc[0], inplace=True)
        
        return result
    
    def _constant_impute(self, X: pd.DataFrame, fill_value=0, **kwargs) -> pd.DataFrame:
        """Constant value imputation"""
        result = X.copy()
        
        # Apply constant imputation
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                # For numeric columns, try to convert fill_value to numeric
                try:
                    numeric_fill_value = float(fill_value)
                    result[col].fillna(numeric_fill_value, inplace=True)
                except (ValueError, TypeError):
                    # If conversion fails, use column mean as fallback
                    result[col].fillna(result[col].mean(), inplace=True)
            else:
                # For non-numeric columns, use fill_value as string
                result[col].fillna(str(fill_value), inplace=True)
        
        return result

    def _iterative_impute(self, X: pd.DataFrame, max_iter: int = 10, tol: float = 0.001, **kwargs) -> pd.DataFrame:
        """迭代填补"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        result = X.copy()
        
        # 数值列使用迭代填补
        if len(numeric_cols) > 0:
            iterative_imputer = IterativeImputer(max_iter=max_iter, tol=tol, random_state=42)
            result[numeric_cols] = iterative_imputer.fit_transform(result[numeric_cols])
        
        # 分类列使用众数
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = result[col].mode()
                if len(mode_value) > 0:
                    result[col].fillna(mode_value.iloc[0], inplace=True)
        
        return result

class DataState:
    """Data state snapshot for undo/redo functionality"""
    
    def __init__(self, data: pd.DataFrame, operation_name: str, timestamp: str):
        self.data = data.copy() if data is not None else None
        self.operation_name = operation_name
        self.timestamp = timestamp
        self.shape = data.shape if data is not None else (0, 0)

class StateManager:
    """Manages data states for undo/redo operations"""
    
    def __init__(self, max_states: int = 20):
        self.states = []
        self.current_index = -1
        self.max_states = max_states
    
    def save_state(self, data: pd.DataFrame, operation_name: str):
        """Save current data state"""
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
        state = DataState(data, operation_name, timestamp)
        
        # Remove any states after current index (for branching)
        self.states = self.states[:self.current_index + 1]
        
        # Add new state
        self.states.append(state)
        self.current_index += 1
        
        # Maintain max states limit
        if len(self.states) > self.max_states:
            self.states.pop(0)
            self.current_index -= 1
    
    def can_undo(self) -> bool:
        """Check if undo is possible"""
        return self.current_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible"""
        return self.current_index < len(self.states) - 1
    
    def undo(self) -> Optional[DataState]:
        """Undo to previous state"""
        if self.can_undo():
            self.current_index -= 1
            return self.states[self.current_index]
        return None
    
    def redo(self) -> Optional[DataState]:
        """Redo to next state"""
        if self.can_redo():
            self.current_index += 1
            return self.states[self.current_index]
        return None
    
    def get_current_state(self) -> Optional[DataState]:
        """Get current state"""
        if 0 <= self.current_index < len(self.states):
            return self.states[self.current_index]
        return None
    
    def get_state_history(self) -> List[str]:
        """Get list of state descriptions"""
        history = []
        for i, state in enumerate(self.states):
            prefix = "→ " if i == self.current_index else "  "
            history.append(f"{prefix}[{state.timestamp}] {state.operation_name} {state.shape}")
        return history
    
    def clear(self):
        """Clear all states"""
        self.states = []
        self.current_index = -1

class SmartRecommendationEngine:
    """Intelligent recommendation system for preprocessing"""
    
    def __init__(self):
        self.recommendations = []
    
    def analyze_and_recommend(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze data and provide smart recommendations"""
        recommendations = []
        
        if data is None or data.empty:
            return recommendations
        
        # Missing value analysis
        missing_info = self._analyze_missing_values(data)
        if missing_info:
            recommendations.extend(missing_info)
        
        # Outlier analysis
        outlier_info = self._analyze_outliers(data)
        if outlier_info:
            recommendations.extend(outlier_info)
        
        # Distribution analysis
        distribution_info = self._analyze_distributions(data)
        if distribution_info:
            recommendations.extend(distribution_info)
        
        # Correlation analysis
        correlation_info = self._analyze_correlations(data)
        if correlation_info:
            recommendations.extend(correlation_info)
        
        return recommendations
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze missing values and recommend handling methods"""
        recommendations = []
        missing_counts = data.isnull().sum()
        total_rows = len(data)
        
        for col in missing_counts.index:
            missing_count = missing_counts[col]
            if missing_count > 0:
                missing_pct = (missing_count / total_rows) * 100
                
                if missing_pct < 5:
                    method = "Drop rows with missing values"
                    priority = "Low"
                elif missing_pct < 20:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        method = "Fill with median value"
                    else:
                        method = "Fill with mode value"
                    priority = "Medium"
                elif missing_pct < 50:
                    method = "Use KNN imputation"
                    priority = "High"
                else:
                    method = "Consider dropping this column"
                    priority = "Critical"
                
                recommendations.append({
                    'type': 'Missing Values',
                    'column': col,
                    'issue': f'{missing_pct:.1f}% missing values',
                    'recommendation': method,
                    'priority': priority,
                    'action': 'handle_missing_values'
                })
        
        return recommendations
    
    def _analyze_outliers(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze outliers and recommend detection methods"""
        recommendations = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].nunique() > 10:  # Skip categorical-like numeric columns
                # Simple IQR-based outlier detection for recommendation
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]
                
                if len(outliers) > 0:
                    outlier_pct = (len(outliers) / len(data)) * 100
                    
                    if outlier_pct < 1:
                        method = "Remove outlying rows"
                        priority = "Low"
                    elif outlier_pct < 5:
                        method = "Cap outliers to percentile bounds"
                        priority = "Medium"
                    else:
                        method = "Use robust scaling transformation"
                        priority = "High"
                    
                    recommendations.append({
                        'type': 'Outliers',
                        'column': col,
                        'issue': f'{outlier_pct:.1f}% outliers detected',
                        'recommendation': method,
                        'priority': priority,
                        'action': 'handle_outliers'
                    })
        
        return recommendations
    
    def _analyze_distributions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze data distributions and recommend transformations"""
        recommendations = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].nunique() > 10:
                skewness = abs(data[col].skew())
                
                if skewness > 2:
                    recommendations.append({
                        'type': 'Distribution',
                        'column': col,
                        'issue': f'Highly skewed distribution (skewness: {skewness:.2f})',
                        'recommendation': 'Apply power transformation (Box-Cox or Yeo-Johnson)',
                        'priority': 'Medium',
                        'action': 'apply_transformation'
                    })
                elif skewness > 1:
                    recommendations.append({
                        'type': 'Distribution',
                        'column': col,
                        'issue': f'Moderately skewed distribution (skewness: {skewness:.2f})',
                        'recommendation': 'Consider log transformation or standardization',
                        'priority': 'Low',
                        'action': 'apply_transformation'
                    })
        
        return recommendations
    
    def _analyze_correlations(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze correlations and recommend actions"""
        recommendations = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            
            # Find high correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.9:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        recommendations.append({
                            'type': 'Correlation',
                            'column': f'{col1} & {col2}',
                            'issue': f'Very high correlation ({corr_val:.3f})',
                            'recommendation': 'Consider removing one of the features',
                            'priority': 'High',
                            'action': 'feature_selection'
                        })
        
        return recommendations

class OperationHistory:
    """Operation history management system"""
    
    def __init__(self):
        self.operations = []
        self.operation_id = 0
    
    def add_operation(self, operation_type: str, details: Dict[str, Any], data_shape_before: Tuple[int, int], data_shape_after: Tuple[int, int]):
        """添加一个操作记录"""
        self.operation_id += 1
        operation = {
            'id': self.operation_id,
            'timestamp': pd.Timestamp.now().strftime('%H:%M:%S'),
            'type': operation_type,
            'details': details,
            'shape_before': data_shape_before,
            'shape_after': data_shape_after
        }
        self.operations.append(operation)
    
    def get_history_text(self) -> str:
        """获取操作历史的文本表示"""
        if not self.operations:
            return "无操作历史"
        
        text = "操作历史记录:\n" + "="*50 + "\n"
        for op in self.operations:
            text += f"[{op['timestamp']}] {op['type']}\n"
            text += f"  详情: {op['details']}\n"
            text += f"  数据维度: {op['shape_before']} → {op['shape_after']}\n\n"
        
        return text
    
    def clear_history(self):
        """清空操作历史"""
        self.operations = []
        self.operation_id = 0

class DataQualityAnalyzer:
    """数据质量分析器"""
    
    def analyze(self, X: pd.DataFrame) -> Dict[str, Any]:
        """分析数据质量"""
        analysis = {
            'basic_stats': self._basic_statistics(X),
            'missing_data': self._missing_data_analysis(X),
            'duplicates': self._duplicate_analysis(X),
            'data_types': self._data_type_analysis(X),
            'distribution': self._distribution_analysis(X),
            'correlation': self._correlation_analysis(X),
            'quality_score': 0
        }
        
        # 计算质量分数
        analysis['quality_score'] = self._calculate_quality_score(analysis)
        
        return analysis
    
    def _basic_statistics(self, X: pd.DataFrame) -> Dict[str, Any]:
        """基础统计信息"""
        return {
            'n_rows': len(X),
            'n_columns': len(X.columns),
            'memory_usage_mb': X.memory_usage(deep=True).sum() / (1024**2),
            'numeric_columns': len(X.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(X.select_dtypes(exclude=[np.number]).columns)
        }
    
    def _missing_data_analysis(self, X: pd.DataFrame) -> Dict[str, Any]:
        """缺失数据分析"""
        missing_counts = X.isnull().sum()
        missing_percentages = (missing_counts / len(X)) * 100
        
        return {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': (missing_counts > 0).sum(),
            'missing_by_column': {
                col: {'count': int(count), 'percentage': float(pct)}
                for col, count, pct in zip(missing_counts.index, missing_counts.values, missing_percentages.values)
                if count > 0
            },
            'worst_column': missing_percentages.idxmax() if missing_percentages.max() > 0 else None,
            'worst_percentage': float(missing_percentages.max())
        }
    
    def _duplicate_analysis(self, X: pd.DataFrame) -> Dict[str, Any]:
        """重复数据分析"""
        duplicates = X.duplicated()
        
        return {
            'total_duplicates': int(duplicates.sum()),
            'percentage': float((duplicates.sum() / len(X)) * 100),
            'unique_rows': int(len(X) - duplicates.sum())
        }
    
    def _data_type_analysis(self, X: pd.DataFrame) -> Dict[str, Any]:
        """数据类型分析"""
        type_counts = X.dtypes.value_counts()
        
        return {
            'type_distribution': {str(dtype): int(count) for dtype, count in type_counts.items()},
            'consistent_types': True,  # 可以添加更复杂的类型一致性检查
            'recommendations': []
        }
    
    def _distribution_analysis(self, X: pd.DataFrame) -> Dict[str, Any]:
        """分布分析"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        skewness = {}
        kurtosis = {}
        
        for col in numeric_cols:
            if not X[col].isnull().all():
                skewness[col] = float(X[col].skew())
                kurtosis[col] = float(X[col].kurtosis())
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'highly_skewed_columns': [col for col, skew in skewness.items() if abs(skew) > 2]
        }
    
    def _correlation_analysis(self, X: pd.DataFrame) -> Dict[str, Any]:
        """相关性分析"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr()
            
            # 找出高相关性特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.9:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            
            return {
                'max_correlation': float(corr_matrix.abs().max().max()),
                'high_correlation_pairs': high_corr_pairs,
                'correlation_matrix_shape': corr_matrix.shape
            }
        
        return {'message': 'Not enough numeric columns for correlation analysis'}
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """计算数据质量分数"""
        score = 100.0
        
        # 缺失数据扣分
        missing_info = analysis['missing_data']
        if missing_info['total_missing'] > 0:
            missing_penalty = min(30, (missing_info['total_missing'] / analysis['basic_stats']['n_rows']) * 100)
            score -= missing_penalty
        
        # 重复数据扣分
        duplicate_penalty = min(20, analysis['duplicates']['percentage'])
        score -= duplicate_penalty
        
        # 高相关性扣分
        if 'high_correlation_pairs' in analysis['correlation']:
            corr_penalty = min(15, len(analysis['correlation']['high_correlation_pairs']) * 5)
            score -= corr_penalty
        
        return max(0, score)

class AdvancedPreprocessing(QWidget):
    """高级数据预处理主界面"""
    
    # 信号
    preprocessing_completed = pyqtSignal(pd.DataFrame, pd.Series)  # X, y
    quality_analysis_completed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None
        self.original_X = None
        self.original_y = None
        self.outlier_detector = OutlierDetector()
        self.smart_imputer = SmartImputer()
        self.quality_analyzer = DataQualityAnalyzer()
        self.operation_history = OperationHistory()
        self.state_manager = StateManager()
        self.recommendation_engine = SmartRecommendationEngine()
        self.performance_optimizer = PerformanceOptimizer()
        self.detected_outliers = None
        self.current_recommendations = []
        
        # Missing value handling attributes
        self.imputation_preview = None
        self.original_missing_columns = {}
        
        # Auto-refresh control
        self.auto_refresh_enabled = True
        self._last_refresh_time = 0
        self._refresh_cooldown = 1.0  # Minimum 1 second between auto-refreshes
        
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("🧬 Advanced Data Preprocessing")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("QLabel { color: #2196F3; margin: 10px; }")
        layout.addWidget(title)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # 1. Smart Recommendations
        self.create_recommendations_tab()
        
        # 2. Data Quality Analysis
        self.create_quality_analysis_tab()
        
        # 3. Missing Values
        self.create_missing_values_tab()
        
        # 4. Outlier Detection & Visualization
        self.create_outlier_detection_tab()
        
        # 5. Data Transformation
        self.create_transformation_tab()
        
        # 6. Operation History & State Management
        self.create_history_tab()
        
        # 7. Data Export
        self.create_export_tab()
        
        # 控制按钮
        self.create_control_buttons(layout)
    
    def create_recommendations_tab(self):
        """Create smart recommendations tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("🧠 Smart Preprocessing Recommendations")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #2196F3; margin: 10px;")
        header_layout.addWidget(title)
        
        # Analyze button
        self.analyze_recommendations_btn = QPushButton("🔍 Analyze Data & Get Recommendations")
        self.analyze_recommendations_btn.clicked.connect(self.generate_recommendations)
        self.analyze_recommendations_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        header_layout.addWidget(self.analyze_recommendations_btn)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Recommendations display
        self.recommendations_table = QTableWidget()
        self.recommendations_table.setColumnCount(5)
        self.recommendations_table.setHorizontalHeaderLabels([
            "Priority", "Type", "Column", "Issue", "Recommendation"
        ])
        self.recommendations_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.recommendations_table)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.apply_selected_btn = QPushButton("✅ Apply Selected Recommendations")
        self.apply_selected_btn.clicked.connect(self.apply_selected_recommendations)
        self.apply_selected_btn.setEnabled(False)
        action_layout.addWidget(self.apply_selected_btn)
        
        self.apply_all_btn = QPushButton("🚀 Apply All High Priority")
        self.apply_all_btn.clicked.connect(self.apply_high_priority_recommendations)
        self.apply_all_btn.setEnabled(False)
        action_layout.addWidget(self.apply_all_btn)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        # Recommendation summary
        self.recommendation_summary = QTextEdit()
        self.recommendation_summary.setReadOnly(True)
        self.recommendation_summary.setMaximumHeight(150)
        layout.addWidget(self.recommendation_summary)
        
        self.tabs.addTab(tab, "Smart Recommendations")
    
    def create_quality_analysis_tab(self):
        """创建数据质量分析标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 分析按钮
        analyze_btn = QPushButton("🔍 Start Quality Analysis")
        analyze_btn.clicked.connect(self.analyze_data_quality)
        layout.addWidget(analyze_btn)
        
        # 结果显示
        self.quality_results_text = QTextEdit()
        self.quality_results_text.setReadOnly(True)
        layout.addWidget(self.quality_results_text)
        
        self.tabs.addTab(tab, "Data Quality Analysis")
    
    def create_missing_values_tab(self):
        """Enhanced Missing Value Imputation Tab with Column Selection and Visualization"""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        
        # Left Panel: Controls
        left_panel = QVBoxLayout()
        
        # Column Selection Group
        column_group = QGroupBox("Column Selection for Imputation")
        column_layout = QVBoxLayout(column_group)
        
        # Missing value summary
        self.missing_summary_label = QLabel("Missing Value Summary:")
        column_layout.addWidget(self.missing_summary_label)
        
        # Column list with missing value counts
        self.missing_columns_list = QListWidget()
        self.missing_columns_list.setSelectionMode(QListWidget.MultiSelection)
        self.missing_columns_list.setMaximumHeight(150)
        column_layout.addWidget(self.missing_columns_list)
        
        # Selection control buttons
        selection_btn_layout = QHBoxLayout()
        self.select_all_missing_btn = QPushButton("Select All")
        self.select_all_missing_btn.clicked.connect(self.select_all_missing_columns)
        self.clear_missing_btn = QPushButton("Clear Selection")
        self.clear_missing_btn.clicked.connect(self.clear_missing_selection)
        self.select_high_missing_btn = QPushButton("Select High Missing (>10%)")
        self.select_high_missing_btn.clicked.connect(self.select_high_missing_columns)
        
        selection_btn_layout.addWidget(self.select_all_missing_btn)
        selection_btn_layout.addWidget(self.clear_missing_btn)
        selection_btn_layout.addWidget(self.select_high_missing_btn)
        column_layout.addLayout(selection_btn_layout)
        
        left_panel.addWidget(column_group)
        
        # Imputation Method Group
        method_group = QGroupBox("Imputation Configuration")
        method_layout = QVBoxLayout(method_group)
        
        # Strategy selection
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Imputation Strategy:"))
        self.impute_method_combo = QComboBox()
        self.impute_method_combo.addItems([
            'median', 'mean', 'mode', 'constant', 'forward_fill', 
            'backward_fill', 'interpolate', 'knn', 'iterative'
        ])
        self.impute_method_combo.currentTextChanged.connect(self.on_impute_method_changed)
        strategy_layout.addWidget(self.impute_method_combo)
        method_layout.addLayout(strategy_layout)
        
        # Dynamic parameter widget
        self.param_widget = QWidget()
        self.param_layout = QVBoxLayout(self.param_widget)
        
        # KNN parameters
        self.knn_widget = QWidget()
        knn_layout = QHBoxLayout(self.knn_widget)
        knn_layout.addWidget(QLabel("Number of Neighbors:"))
        self.knn_neighbors_spin = QSpinBox()
        self.knn_neighbors_spin.setRange(1, 20)
        self.knn_neighbors_spin.setValue(5)
        knn_layout.addWidget(self.knn_neighbors_spin)
        knn_layout.addStretch()
        self.param_layout.addWidget(self.knn_widget)
        
        # Iterative parameters
        self.iterative_widget = QWidget()
        iter_layout = QVBoxLayout(self.iterative_widget)
        
        iter_max_layout = QHBoxLayout()
        iter_max_layout.addWidget(QLabel("Max Iterations:"))
        self.iter_max_spin = QSpinBox()
        self.iter_max_spin.setRange(1, 50)
        self.iter_max_spin.setValue(10)
        iter_max_layout.addWidget(self.iter_max_spin)
        iter_max_layout.addStretch()
        iter_layout.addLayout(iter_max_layout)
        
        iter_tol_layout = QHBoxLayout()
        iter_tol_layout.addWidget(QLabel("Tolerance:"))
        self.iter_tolerance_spin = QDoubleSpinBox()
        self.iter_tolerance_spin.setRange(0.001, 0.1)
        self.iter_tolerance_spin.setValue(0.001)
        self.iter_tolerance_spin.setDecimals(4)
        iter_tol_layout.addWidget(self.iter_tolerance_spin)
        iter_tol_layout.addStretch()
        iter_layout.addLayout(iter_tol_layout)
        
        self.param_layout.addWidget(self.iterative_widget)
        
        # Constant value parameters
        self.constant_widget = QWidget()
        const_layout = QHBoxLayout(self.constant_widget)
        const_layout.addWidget(QLabel("Fill Value:"))
        self.constant_value_edit = QLineEdit("0")
        const_layout.addWidget(self.constant_value_edit)
        const_layout.addStretch()
        self.param_layout.addWidget(self.constant_widget)
        
        method_layout.addWidget(self.param_widget)
        left_panel.addWidget(method_group)
        
        # Action Buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout(action_group)
        
        btn_layout = QHBoxLayout()
        self.preview_imputation_btn = QPushButton("📊 Preview Changes")
        self.preview_imputation_btn.clicked.connect(self.preview_imputation)
        self.apply_imputation_btn = QPushButton("✅ Apply Imputation")
        self.apply_imputation_btn.clicked.connect(self.handle_missing_values)
        
        btn_layout.addWidget(self.preview_imputation_btn)
        btn_layout.addWidget(self.apply_imputation_btn)
        action_layout.addLayout(btn_layout)
        
        left_panel.addWidget(action_group)
        
        # Results Display
        results_group = QGroupBox("Imputation Results")
        results_layout = QVBoxLayout(results_group)
        
        self.impute_results_text = QTextEdit()
        self.impute_results_text.setReadOnly(True)
        self.impute_results_text.setMaximumHeight(120)
        results_layout.addWidget(self.impute_results_text)
        
        left_panel.addWidget(results_group)
        left_panel.addStretch()
        
        # Right Panel: Visualization
        right_panel = QVBoxLayout()
        
        # Visualization Controls
        viz_controls_group = QGroupBox("Visualization Options")
        viz_controls_layout = QVBoxLayout(viz_controls_group)
        
        # Primary visualization buttons
        viz_btn_layout = QHBoxLayout()
        self.visualize_missing_pattern_btn = QPushButton("📊 Missing Patterns")
        self.visualize_missing_pattern_btn.clicked.connect(self.visualize_missing_patterns)
        self.visualize_comparison_btn = QPushButton("📈 Before/After Comparison")
        self.visualize_comparison_btn.clicked.connect(self.visualize_imputation_comparison)
        
        viz_btn_layout.addWidget(self.visualize_missing_pattern_btn)
        viz_btn_layout.addWidget(self.visualize_comparison_btn)
        viz_controls_layout.addLayout(viz_btn_layout)
        
        # Distribution analysis section
        dist_analysis_layout = QVBoxLayout()
        dist_header_layout = QHBoxLayout()
        
        self.visualize_distribution_btn = QPushButton("📉 Distribution Analysis")
        self.visualize_distribution_btn.clicked.connect(self.visualize_distribution_analysis)
        dist_header_layout.addWidget(self.visualize_distribution_btn)
        
        # Column selection for distribution analysis
        self.distribution_column_combo = QComboBox()
        self.distribution_column_combo.addItem("Auto Select (Top 6)", "auto")
        self.distribution_column_combo.addItem("All Columns", "all")
        self.distribution_column_combo.currentTextChanged.connect(self.on_distribution_column_changed)
        dist_header_layout.addWidget(QLabel("Columns:"))
        dist_header_layout.addWidget(self.distribution_column_combo)
        
        # Visualization type selection
        self.distribution_type_combo = QComboBox()
        self.distribution_type_combo.addItem("📊 Histogram", "histogram")
        self.distribution_type_combo.addItem("📦 Box Plot", "boxplot")
        self.distribution_type_combo.addItem("🌊 Density Plot", "density")
        self.distribution_type_combo.addItem("📈 Combined View", "combined")
        self.distribution_type_combo.currentTextChanged.connect(self.on_distribution_type_changed)
        dist_header_layout.addWidget(QLabel("Type:"))
        dist_header_layout.addWidget(self.distribution_type_combo)
        
        # Auto-refresh control
        auto_refresh_layout = QHBoxLayout()
        self.auto_refresh_checkbox = QCheckBox("🔄 Auto-refresh visualization")
        self.auto_refresh_checkbox.setChecked(True)  # Default enabled
        self.auto_refresh_checkbox.stateChanged.connect(self.on_auto_refresh_toggle)
        self.auto_refresh_checkbox.setToolTip("Automatically refresh distribution analysis when data or settings change")
        auto_refresh_layout.addWidget(self.auto_refresh_checkbox)
        auto_refresh_layout.addStretch()
        
        dist_analysis_layout.addLayout(dist_header_layout)
        dist_analysis_layout.addLayout(auto_refresh_layout)
        viz_controls_layout.addLayout(dist_analysis_layout)
        
        right_panel.addWidget(viz_controls_group)
        
        # Visualization Display Area
        visualization_group = QGroupBox("Data Visualization & Analysis")
        viz_layout = QVBoxLayout(visualization_group)
        
        # Create matplotlib canvas
        try:
            import matplotlib
            matplotlib.use('Qt5Agg')
            matplotlib.pyplot.ioff()  # Turn off interactive mode
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            self.missing_figure = Figure(figsize=(12, 10))
            self.missing_canvas = FigureCanvas(self.missing_figure)
            viz_layout.addWidget(self.missing_canvas)
            
        except ImportError:
            viz_placeholder = QLabel("📊 Matplotlib not available for visualization\nPlease install matplotlib for enhanced visualization features.")
            viz_placeholder.setAlignment(Qt.AlignCenter)
            viz_placeholder.setStyleSheet("color: #666; font-style: italic; padding: 20px;")
            viz_layout.addWidget(viz_placeholder)
        
        right_panel.addWidget(visualization_group)
        
        # Setup main layout
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(450)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        
        # Initialize parameter visibility
        self.on_impute_method_changed()
        
        self.tabs.addTab(tab, "Enhanced Missing Value Imputation")
    
    def create_outlier_detection_tab(self):
        """创建异常值检测标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Column selection for outlier detection
        column_group = QGroupBox("Column Selection")
        column_layout = QVBoxLayout(column_group)
        
        column_layout.addWidget(QLabel("Select columns to detect outliers:"))
        self.outlier_columns_list = QListWidget()
        self.outlier_columns_list.setSelectionMode(QListWidget.MultiSelection)
        self.outlier_columns_list.setMaximumHeight(100)
        column_layout.addWidget(self.outlier_columns_list)
        
        # Select all/none buttons for outlier detection
        outlier_btn_layout = QHBoxLayout()
        self.select_all_outlier_btn = QPushButton("Select All")
        self.select_all_outlier_btn.clicked.connect(self.select_all_outlier_columns)
        self.clear_outlier_btn = QPushButton("Clear")
        self.clear_outlier_btn.clicked.connect(self.clear_outlier_columns)
        outlier_btn_layout.addWidget(self.select_all_outlier_btn)
        outlier_btn_layout.addWidget(self.clear_outlier_btn)
        column_layout.addLayout(outlier_btn_layout)
        
        layout.addWidget(column_group)
        
        # Method selection
        method_group = QGroupBox("Outlier Detection Methods")
        method_layout = QVBoxLayout(method_group)
        
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems([
            'IQR', 'Z-Score', 'Isolation Forest', 
            'Local Outlier Factor', 'Elliptic Envelope', 'One-Class SVM'
        ])
        method_layout.addWidget(self.outlier_method_combo)
        
        # Parameter settings
        params_layout = QHBoxLayout()
        
        # IQR multiplier
        self.iqr_multiplier_spin = QDoubleSpinBox()
        self.iqr_multiplier_spin.setRange(1.0, 3.0)
        self.iqr_multiplier_spin.setValue(1.5)
        self.iqr_multiplier_spin.setSingleStep(0.1)
        params_layout.addWidget(QLabel("IQR Multiplier:"))
        params_layout.addWidget(self.iqr_multiplier_spin)
        
        # Contamination ratio
        self.contamination_spin = QDoubleSpinBox()
        self.contamination_spin.setRange(0.01, 0.5)
        self.contamination_spin.setValue(0.1)
        self.contamination_spin.setSingleStep(0.01)
        params_layout.addWidget(QLabel("Contamination:"))
        params_layout.addWidget(self.contamination_spin)
        
        method_layout.addLayout(params_layout)
        layout.addWidget(method_group)
        
        # Detection button
        detect_btn = QPushButton("🎯 Detect Outliers")
        detect_btn.clicked.connect(self.detect_outliers)
        layout.addWidget(detect_btn)
        
        # Outlier handling section
        handling_group = QGroupBox("Outlier Handling")
        handling_layout = QVBoxLayout(handling_group)
        
        self.outlier_handling_combo = QComboBox()
        self.outlier_handling_combo.addItems([
            'Remove Rows', 'Cap Values', 'Replace with NaN', 
            'Replace with Median', 'Replace with Mean'
        ])
        handling_layout.addWidget(QLabel("Handling Method:"))
        handling_layout.addWidget(self.outlier_handling_combo)
        
        # Handle outliers button
        self.handle_outliers_btn = QPushButton("🔧 Handle Outliers")
        self.handle_outliers_btn.clicked.connect(self.handle_outliers)
        self.handle_outliers_btn.setEnabled(False)  # Disabled until outliers are detected
        handling_layout.addWidget(self.handle_outliers_btn)
        
        layout.addWidget(handling_group)
        
        # Results and Visualization
        results_viz_splitter = QSplitter(Qt.Horizontal)
        
        # Results display
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.addWidget(QLabel("Detection Results:"))
        
        self.outlier_results_text = QTextEdit()
        self.outlier_results_text.setReadOnly(True)
        self.outlier_results_text.setMaximumHeight(200)
        results_layout.addWidget(self.outlier_results_text)
        
        # Visualization button
        self.visualize_outliers_btn = QPushButton("📊 Visualize Outliers")
        self.visualize_outliers_btn.clicked.connect(self.visualize_outliers)
        self.visualize_outliers_btn.setEnabled(False)
        results_layout.addWidget(self.visualize_outliers_btn)
        
        results_viz_splitter.addWidget(results_widget)
        
        # Visualization area
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.addWidget(QLabel("Outlier Visualization:"))
        
        # Create matplotlib figure
        self.outlier_figure = Figure(figsize=(10, 6))
        self.outlier_canvas = FigureCanvas(self.outlier_figure)
        viz_layout.addWidget(self.outlier_canvas)
        
        results_viz_splitter.addWidget(viz_widget)
        results_viz_splitter.setSizes([400, 600])
        
        layout.addWidget(results_viz_splitter)
        
        self.tabs.addTab(tab, "Outlier Detection & Visualization")
    
    def create_transformation_tab(self):
        """创建数据变换标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Transformation methods
        transform_group = QGroupBox("Data Transformation")
        transform_layout = QVBoxLayout(transform_group)
        
        self.transform_method_combo = QComboBox()
        self.transform_method_combo.addItems([
            'StandardScaler', 'MinMaxScaler', 'RobustScaler',
            'PowerTransformer', 'QuantileTransformer'
        ])
        transform_layout.addWidget(self.transform_method_combo)
        
        layout.addWidget(transform_group)
        
        # Execute button
        transform_btn = QPushButton("🔄 Apply Data Transformation")
        transform_btn.clicked.connect(self.apply_transformation)
        layout.addWidget(transform_btn)
        
        # Results display
        self.transform_results_text = QTextEdit()
        self.transform_results_text.setReadOnly(True)
        layout.addWidget(self.transform_results_text)
        
        self.tabs.addTab(tab, "Data Transformation")
    
    def create_history_tab(self):
        """Create operation history and state management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        title = QLabel("📋 Operation History & State Management")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #2196F3; margin: 10px;")
        layout.addWidget(title)
        
        # State management controls
        state_group = QGroupBox("State Management (Undo/Redo)")
        state_layout = QVBoxLayout(state_group)
        
        # Undo/Redo buttons
        undo_redo_layout = QHBoxLayout()
        
        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.clicked.connect(self.undo_operation)
        self.undo_btn.setEnabled(False)
        self.undo_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        undo_redo_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("↷ Redo")
        self.redo_btn.clicked.connect(self.redo_operation)
        self.redo_btn.setEnabled(False)
        self.redo_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        undo_redo_layout.addWidget(self.redo_btn)
        
        undo_redo_layout.addStretch()
        state_layout.addLayout(undo_redo_layout)
        
        # State history display
        self.state_history_text = QTextEdit()
        self.state_history_text.setReadOnly(True)
        self.state_history_text.setMaximumHeight(120)
        self.state_history_text.setFont(QFont("Courier New", 9))
        state_layout.addWidget(QLabel("State History:"))
        state_layout.addWidget(self.state_history_text)
        
        layout.addWidget(state_group)
        
        # Operation history
        history_group = QGroupBox("Detailed Operation History")
        history_layout = QVBoxLayout(history_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh History")
        refresh_btn.clicked.connect(self.refresh_history)
        btn_layout.addWidget(refresh_btn)
        
        clear_history_btn = QPushButton("🗑️ Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        btn_layout.addWidget(clear_history_btn)
        
        btn_layout.addStretch()
        history_layout.addLayout(btn_layout)
        
        # History display
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont("Courier New", 9))
        history_layout.addWidget(self.history_text)
        
        layout.addWidget(history_group)
        
        self.tabs.addTab(tab, "History & States")
    
    def create_export_tab(self):
        """创建数据导出标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 导出选项组
        export_options_group = QGroupBox("Export Options")
        export_options_layout = QVBoxLayout(export_options_group)
        
        # 数据选择
        data_selection_layout = QVBoxLayout()
        data_selection_layout.addWidget(QLabel("Select data to export:"))
        
        self.export_data_combo = QComboBox()
        self.export_data_combo.addItems([
            "Current processed data",
            "Original data", 
            "Data quality report",
            "Operation history"
        ])
        data_selection_layout.addWidget(self.export_data_combo)
        export_options_layout.addLayout(data_selection_layout)
        
        # 文件格式选择
        format_layout = QVBoxLayout()
        format_layout.addWidget(QLabel("File Format:"))
        
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems([
            "CSV (comma-separated)",
            "Excel (.xlsx)",
            "TSV (tab-separated)",
            "JSON",
            "TXT (text format)"
        ])
        format_layout.addWidget(self.export_format_combo)
        export_options_layout.addLayout(format_layout)
        
        # 导出设置
        settings_group = QGroupBox("Export Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # 包含索引
        self.include_index_cb = QCheckBox("Include row index")
        self.include_index_cb.setChecked(False)
        settings_layout.addWidget(self.include_index_cb)
        
        # 包含列名
        self.include_header_cb = QCheckBox("Include column names")
        self.include_header_cb.setChecked(True)
        settings_layout.addWidget(self.include_header_cb)
        
        # 编码选择
        encoding_layout = QHBoxLayout()
        encoding_layout.addWidget(QLabel("File Encoding:"))
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["utf-8-sig", "utf-8", "gbk", "gb2312", "latin-1"])
        self.encoding_combo.setCurrentText("utf-8-sig")  # 默认使用带BOM的UTF-8，避免中文乱码
        encoding_layout.addWidget(self.encoding_combo)
        settings_layout.addLayout(encoding_layout)
        
        export_options_layout.addWidget(settings_group)
        layout.addWidget(export_options_group)
        
        # 预览区域
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # 预览按钮
        preview_btn_layout = QHBoxLayout()
        self.preview_export_btn = QPushButton("🔍 Preview Export Data")
        self.preview_export_btn.clicked.connect(self.preview_export_data)
        preview_btn_layout.addWidget(self.preview_export_btn)
        
        self.refresh_preview_btn = QPushButton("🔄 Refresh Preview")
        self.refresh_preview_btn.clicked.connect(self.refresh_export_preview)
        preview_btn_layout.addWidget(self.refresh_preview_btn)
        
        preview_btn_layout.addStretch()
        preview_layout.addLayout(preview_btn_layout)
        
        # 预览文本区域
        self.export_preview_text = QTextEdit()
        self.export_preview_text.setReadOnly(True)
        self.export_preview_text.setMaximumHeight(200)
        self.export_preview_text.setFont(QFont("Courier New", 9))
        preview_layout.addWidget(self.export_preview_text)
        
        layout.addWidget(preview_group)
        
        # 导出按钮区域
        export_btn_layout = QHBoxLayout()
        
        self.quick_export_btn = QPushButton("⚡ Quick Export")
        self.quick_export_btn.clicked.connect(self.quick_export)
        self.quick_export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        export_btn_layout.addWidget(self.quick_export_btn)
        
        self.custom_export_btn = QPushButton("📁 Custom Export")
        self.custom_export_btn.clicked.connect(self.custom_export)
        self.custom_export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        export_btn_layout.addWidget(self.custom_export_btn)
        
        export_btn_layout.addStretch()
        layout.addLayout(export_btn_layout)
        
        # 导出状态显示
        self.export_status_text = QTextEdit()
        self.export_status_text.setReadOnly(True)
        self.export_status_text.setMaximumHeight(100)
        layout.addWidget(self.export_status_text)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Data Export")
    
    def create_control_buttons(self, parent_layout):
        """创建控制按钮"""
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_data)
        button_layout.addWidget(self.reset_btn)
        
        self.preview_btn = QPushButton("👁️ Preview Data")
        self.preview_btn.clicked.connect(self.preview_data)
        button_layout.addWidget(self.preview_btn)
        
        self.apply_btn = QPushButton("✅ Apply Processing Results")
        self.apply_btn.clicked.connect(self.apply_preprocessing)
        button_layout.addWidget(self.apply_btn)
        
        self.status_btn = QPushButton("📊 Check Data Status")
        self.status_btn.clicked.connect(self.show_data_status)
        button_layout.addWidget(self.status_btn)
        
        parent_layout.addLayout(button_layout)
    
    def set_data(self, X: pd.DataFrame, y: pd.Series = None):
        """Set data and initialize all components"""
        self.X = X.copy()
        self.original_X = X.copy()
        
        if y is not None:
            self.y = y.copy()
            self.original_y = y.copy()
        else:
            self.y = None
            self.original_y = None
            
        self.detected_outliers = None
        
        # Initialize state management
        self.state_manager.clear()
        self.state_manager.save_state(self.X, "Initial Data Load")
        
        # Check for performance optimization opportunities
        if self.performance_optimizer.is_large_dataset(self.X):
            optimized_X = self.performance_optimizer.optimize_dtypes(self.X)
            memory_saved = (self.X.memory_usage(deep=True).sum() - optimized_X.memory_usage(deep=True).sum()) / (1024 * 1024)
            if memory_saved > 1:  # If more than 1MB can be saved
                from PyQt5.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    self, 
                    "Performance Optimization",
                    f"Large dataset detected ({self.performance_optimizer.get_memory_usage(self.X)['total_mb']:.1f} MB).\n"
                    f"Optimize data types to save {memory_saved:.1f} MB?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.X = optimized_X
                    self.original_X = optimized_X.copy()
        
        # Update UI components
        self.update_column_lists()
        self.update_missing_columns_list()
        self.update_undo_redo_buttons()
        
        # Generate initial recommendations
        if hasattr(self, 'analyze_recommendations_btn'):
            self.generate_recommendations()
            
        # Auto-refresh distribution analysis for new data
        self.auto_refresh_distribution_analysis()
    
    def analyze_data_quality(self):
        """分析数据质量"""
        if self.X is None:
            return
        
        analysis = self.quality_analyzer.analyze(self.X)
        self.quality_analysis_completed.emit(analysis)
        
        # 显示结果
        result_text = self._format_quality_analysis(analysis)
        self.quality_results_text.setText(result_text)
    
    def _format_quality_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format quality analysis results with performance metrics"""
        # Get performance metrics
        performance_metrics = self.performance_optimizer.get_memory_usage(self.X)
        is_large = self.performance_optimizer.is_large_dataset(self.X)
        
        text = f"""🔍 Enhanced Data Quality Analysis Report
{'='*60}

⭐ Overall Quality Score: {analysis['quality_score']:.1f}/100

📊 Basic Statistics:
• Number of rows: {analysis['basic_stats']['n_rows']:,}
• Number of columns: {analysis['basic_stats']['n_columns']}
• Memory usage: {analysis['basic_stats']['memory_usage_mb']:.2f} MB
• Numeric columns: {analysis['basic_stats']['numeric_columns']}
• Categorical columns: {analysis['basic_stats']['categorical_columns']}

🚀 Performance Metrics:
• Dataset size: {'Large (>100MB)' if is_large else 'Optimal (<100MB)'}
• Memory per row: {performance_metrics.get('mb_per_row', 0):.6f} MB
• Optimization potential: {'High' if is_large else 'Low'}

🧩 Missing Data:
• Total missing: {analysis['missing_data']['total_missing']:,}
• Columns with missing: {analysis['missing_data']['columns_with_missing']}
• Worst column: {analysis['missing_data']['worst_column']} ({analysis['missing_data']['worst_percentage']:.1f}%)

🔄 Duplicates:
• Total duplicates: {analysis['duplicates']['total_duplicates']:,}
• Duplicate ratio: {analysis['duplicates']['percentage']:.2f}%

"""
        
        if 'high_correlation_pairs' in analysis['correlation']:
            text += f"""🔗 Correlation Analysis:
• High correlation pairs: {len(analysis['correlation']['high_correlation_pairs'])}
"""
            for pair in analysis['correlation']['high_correlation_pairs'][:5]:
                text += f"  - {pair['feature1']} vs {pair['feature2']}: {pair['correlation']:.3f}\n"
        
        # Add recommendations based on analysis
        text += f"""
💡 Smart Recommendations:
• {'Enable data type optimization' if is_large else 'Data types already optimized'}
• {'Consider sampling for visualization' if len(self.X) > 10000 else 'Dataset size is manageable'}
• {'High priority: Handle missing values' if analysis['missing_data']['total_missing'] > 0 else 'No missing values detected'}
• {'Medium priority: Remove duplicates' if analysis['duplicates']['total_duplicates'] > 0 else 'No duplicates found'}
"""
        
        return text
    
    def on_impute_method_changed(self):
        """Handle imputation method change to show/hide relevant parameters"""
        method = self.impute_method_combo.currentText()
        
        # Hide all parameter widgets first
        self.knn_widget.hide()
        self.iterative_widget.hide()
        self.constant_widget.hide()
        
        # Show relevant parameter widget
        if method == 'knn':
            self.knn_widget.show()
        elif method == 'iterative':
            self.iterative_widget.show()
        elif method == 'constant':
            self.constant_widget.show()
    
    def update_missing_columns_list(self):
        """Update the missing columns list with missing value counts"""
        if self.X is None:
            return
        
        self.missing_columns_list.clear()
        
        # Calculate missing values for each column
        missing_counts = self.X.isnull().sum()
        total_rows = len(self.X)
        
        # Store original missing column info
        self.original_missing_columns = {}
        
        # Add columns with missing values
        for col in missing_counts.index:
            missing_count = missing_counts[col]
            missing_percent = (missing_count / total_rows) * 100
            
            if missing_count > 0:
                self.original_missing_columns[col] = {
                    'count': missing_count,
                    'percentage': missing_percent
                }
                
                item_text = f"{col} - {missing_count} missing ({missing_percent:.1f}%)"
                item = QListWidgetItem(item_text)
                
                # Color code by severity
                if missing_percent > 30:
                    item.setBackground(QColor('#ffcccb'))  # Light red
                elif missing_percent > 10:
                    item.setBackground(QColor('#fff2cc'))  # Light yellow
                else:
                    item.setBackground(QColor('#d4edda'))  # Light green
                
                self.missing_columns_list.addItem(item)
        
        # Update summary label
        total_missing = missing_counts.sum()
        total_values = self.X.size
        overall_percentage = (total_missing / total_values) * 100
        
        summary_text = f"Total Missing Values: {total_missing:,} ({overall_percentage:.2f}% of all values)"
        self.missing_summary_label.setText(summary_text)
        
        # Update distribution analysis column options
        self.update_distribution_column_options()
        
        # Auto-refresh distribution analysis if tab is active
        self.auto_refresh_distribution_analysis()
    
    def update_distribution_column_options(self):
        """Update distribution analysis column dropdown with available columns"""
        if self.X is None or not hasattr(self, 'distribution_column_combo'):
            return
        
        # Save current selection
        current_selection = self.distribution_column_combo.currentData()
        
        # Clear and rebuild options
        self.distribution_column_combo.clear()
        
        # Add default options
        self.distribution_column_combo.addItem("Auto Select (Top 6)", "auto")
        self.distribution_column_combo.addItem("All Columns", "all")
        
        # Add individual columns with missing values
        missing_counts = self.X.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0].index
        
        if len(columns_with_missing) > 0:
            self.distribution_column_combo.addItem("--- Individual Columns ---", "separator")
            
            # Sort columns by missing value percentage (descending)
            missing_info = []
            total_rows = len(self.X)
            
            for col in columns_with_missing:
                if pd.api.types.is_numeric_dtype(self.X[col]):
                    missing_count = missing_counts[col]
                    missing_percent = (missing_count / total_rows) * 100
                    missing_info.append((col, missing_percent))
            
            # Sort by missing percentage (highest first)
            missing_info.sort(key=lambda x: x[1], reverse=True)
            
            for col, missing_percent in missing_info:
                display_text = f"{col} ({missing_percent:.1f}% missing)"
                self.distribution_column_combo.addItem(display_text, col)
        
        # Restore previous selection if possible
        if current_selection:
            index = self.distribution_column_combo.findData(current_selection)
            if index >= 0:
                self.distribution_column_combo.setCurrentIndex(index)
    
    def on_distribution_column_changed(self):
        """Handle distribution column selection change and auto-refresh visualization"""
        # Automatically refresh the distribution analysis when selection changes
        self.auto_refresh_distribution_analysis()
    
    def on_distribution_type_changed(self):
        """Handle distribution visualization type change and auto-refresh"""
        # Automatically refresh the distribution analysis when visualization type changes
        self.auto_refresh_distribution_analysis()
    
    def on_auto_refresh_toggle(self, state):
        """Handle auto-refresh checkbox toggle"""
        self.auto_refresh_enabled = (state == 2)  # 2 = Qt.Checked
        if self.auto_refresh_enabled:
            # Trigger immediate refresh when auto-refresh is enabled
            self.auto_refresh_distribution_analysis()
    
    def auto_refresh_distribution_analysis(self):
        """Automatically refresh distribution analysis if conditions are met"""
        try:
            # Check if auto-refresh is enabled
            if not self.auto_refresh_enabled:
                return
                
            # Check cooldown to prevent excessive refreshes
            import time
            current_time = time.time()
            if current_time - self._last_refresh_time < self._refresh_cooldown:
                return
                
            # Check if we have data
            if self.X is None:
                return
                
            # Check if the missing values tab is currently active
            if hasattr(self, 'tabs'):
                current_tab_index = self.tabs.currentIndex()
                missing_values_tab_index = None
                for i in range(self.tabs.count()):
                    if "Missing Value" in self.tabs.tabText(i):
                        missing_values_tab_index = i
                        break
                
                # Only auto-refresh if the missing values tab is active
                if missing_values_tab_index is not None and current_tab_index == missing_values_tab_index:
                    # Check if distribution visualization components exist
                    if (hasattr(self, 'missing_canvas') and 
                        hasattr(self, 'distribution_column_combo') and
                        hasattr(self, 'distribution_type_combo')):
                        
                        # Only refresh if there's actually something to visualize
                        missing_counts = self.X.isnull().sum()
                        if missing_counts.sum() > 0:
                            # Update last refresh time
                            self._last_refresh_time = current_time
                            
                            # Automatically trigger distribution analysis
                            QApplication.processEvents()  # Process any pending UI events
                            self.visualize_distribution_analysis()
                            
        except Exception as e:
            # Silently handle errors to avoid disrupting the user experience
            pass
    
    def select_all_missing_columns(self):
        """Select all columns in the missing columns list"""
        for i in range(self.missing_columns_list.count()):
            self.missing_columns_list.item(i).setSelected(True)
    
    def clear_missing_selection(self):
        """Clear all selections in the missing columns list"""
        self.missing_columns_list.clearSelection()
    
    def select_high_missing_columns(self):
        """Select columns with high missing values (>10%)"""
        self.missing_columns_list.clearSelection()
        
        for i in range(self.missing_columns_list.count()):
            item = self.missing_columns_list.item(i)
            # Extract column name from item text
            col_name = item.text().split(' - ')[0]
            
            if col_name in self.original_missing_columns:
                percentage = self.original_missing_columns[col_name]['percentage']
                if percentage > 10:
                    item.setSelected(True)
    
    def get_selected_missing_columns(self):
        """Get list of selected column names for imputation"""
        selected_columns = []
        for item in self.missing_columns_list.selectedItems():
            # Extract column name from item text
            col_name = item.text().split(' - ')[0]
            selected_columns.append(col_name)
        return selected_columns
    
    def preview_imputation(self):
        """Preview the effect of imputation on selected columns"""
        if self.X is None:
            self.impute_results_text.setText("No data available for preview.")
            return
        
        selected_columns = self.get_selected_missing_columns()
        if not selected_columns:
            self.impute_results_text.setText("Please select columns for imputation preview.")
            return
        
        method = self.impute_method_combo.currentText()
        
        try:
            # Create a copy for preview
            X_preview = self.X[selected_columns].copy()
            
            # Debug info
            print(f"Debug: Preview method: {method}")
            print(f"Debug: Selected columns: {selected_columns}")
            print(f"Debug: X_preview shape: {X_preview.shape}")
            print(f"Debug: X_preview has missing values: {X_preview.isnull().sum().sum()}")
            
            # Configure method-specific parameters
            kwargs = {}
            if method == 'knn':
                kwargs['n_neighbors'] = self.knn_neighbors_spin.value()
                print(f"Debug: KNN neighbors: {kwargs['n_neighbors']}")
                
                # Check if we have enough numeric columns for KNN
                numeric_cols = X_preview.select_dtypes(include=[np.number]).columns
                print(f"Debug: Numeric columns for KNN: {list(numeric_cols)}")
                if len(numeric_cols) == 0:
                    self.impute_results_text.setText("KNN imputation requires at least one numeric column. Please select numeric columns or use a different method.")
                    return
                    
            elif method == 'iterative':
                kwargs['max_iter'] = self.iter_max_spin.value()
                kwargs['tol'] = self.iter_tolerance_spin.value()
            elif method == 'constant':
                try:
                    kwargs['fill_value'] = float(self.constant_value_edit.text())
                except ValueError:
                    kwargs['fill_value'] = self.constant_value_edit.text()
            
            print(f"Debug: kwargs: {kwargs}")
            
            # Apply imputation to preview data
            X_imputed_preview = self.smart_imputer.impute(X_preview, method, **kwargs)
            
            print(f"Debug: X_imputed_preview type: {type(X_imputed_preview)}")
            print(f"Debug: X_imputed_preview is None: {X_imputed_preview is None}")
            
            if X_imputed_preview is None:
                self.impute_results_text.setText(f"Imputation failed: {method} method returned None. Please check your data or try a different method.")
                return
            
            # Store preview for visualization
            self.imputation_preview = {
                'original': X_preview,
                'imputed': X_imputed_preview,
                'method': method,
                'parameters': kwargs
            }
            
            # Generate preview summary
            preview_text = f"Preview - Imputation Method: {method}\n"
            preview_text += f"Selected Columns: {len(selected_columns)}\n\n"
            
            for col in selected_columns:
                missing_before = X_preview[col].isnull().sum()
                missing_after = X_imputed_preview[col].isnull().sum()
                filled = missing_before - missing_after
                
                preview_text += f"• {col}:\n"
                preview_text += f"  Missing before: {missing_before}\n"
                preview_text += f"  Missing after: {missing_after}\n"
                preview_text += f"  Values filled: {filled}\n"
                
                if filled > 0 and method in ['median', 'mean']:
                    fill_value = X_imputed_preview[col].fillna(method='ffill').iloc[-1]
                    preview_text += f"  Fill value: {fill_value:.3f}\n"
                
                preview_text += "\n"
            
            preview_text += "✓ Preview ready. Use 'Before/After Comparison' to visualize changes."
            self.impute_results_text.setText(preview_text)
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Debug: Full error trace:\n{error_detail}")
            self.impute_results_text.setText(f"Preview failed: {str(e)}\nMethod: {method}\nError details: Check console for full trace.")
    
    def visualize_missing_patterns(self):
        """Visualize missing value patterns in the data"""
        if self.X is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            self.missing_figure.clear()
            
            # Check if there are any missing values
            missing_counts = self.X.isnull().sum()
            columns_with_missing = missing_counts[missing_counts > 0]
            
            if len(columns_with_missing) == 0:
                ax = self.missing_figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No missing values found in the dataset!', 
                       ha='center', va='center', fontsize=16, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                self.missing_canvas.draw()
                return
            
            # Create subplots
            if len(columns_with_missing) <= 10:
                # Missing value heatmap
                gs = self.missing_figure.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
                
                # Heatmap of missing values
                ax1 = self.missing_figure.add_subplot(gs[0, :])
                missing_data = self.X[columns_with_missing.index].isnull()
                
                # Sample data if too large
                if len(missing_data) > 1000:
                    missing_data = missing_data.sample(1000, random_state=42)
                
                sns.heatmap(missing_data.T, cbar=True, cmap='viridis_r', 
                           ax=ax1, yticklabels=True, xticklabels=False)
                ax1.set_title('Missing Value Pattern Heatmap', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Data Samples')
                ax1.set_ylabel('Features')
                
                # Missing value count bar plot
                ax2 = self.missing_figure.add_subplot(gs[1, 0])
                y_pos = range(len(columns_with_missing))
                bars = ax2.barh(y_pos, columns_with_missing.values, 
                               color=['red' if x > len(self.X)*0.3 else 'orange' if x > len(self.X)*0.1 else 'green' 
                                     for x in columns_with_missing.values])
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(columns_with_missing.index, fontsize=8)
                ax2.set_xlabel('Missing Value Count')
                ax2.set_title('Missing Values by Column', fontsize=10, fontweight='bold')
                
                # Add value labels on bars
                for i, (bar, count) in enumerate(zip(bars, columns_with_missing.values)):
                    percentage = (count / len(self.X)) * 100
                    ax2.text(bar.get_width() + max(columns_with_missing.values) * 0.01, 
                            bar.get_y() + bar.get_height()/2, 
                            f'{count}\n({percentage:.1f}%)', 
                            ha='left', va='center', fontsize=8)
                
                # Missing value percentage pie chart
                ax3 = self.missing_figure.add_subplot(gs[1, 1])
                total_values = self.X.size
                total_missing = missing_counts.sum()
                total_present = total_values - total_missing
                
                sizes = [total_present, total_missing]
                labels = ['Present Data', 'Missing Data']
                colors = ['lightgreen', 'lightcoral']
                
                wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                                  startangle=90, textprops={'fontsize': 8})
                ax3.set_title('Overall Data Completeness', fontsize=10, fontweight='bold')
                
            else:
                # Too many columns with missing values, show summary only
                ax = self.missing_figure.add_subplot(111)
                top_missing = columns_with_missing.head(20)
                
                y_pos = range(len(top_missing))
                bars = ax.barh(y_pos, top_missing.values,
                              color=['red' if x > len(self.X)*0.3 else 'orange' if x > len(self.X)*0.1 else 'green' 
                                    for x in top_missing.values])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_missing.index, fontsize=8)
                ax.set_xlabel('Missing Value Count')
                ax.set_title(f'Top 20 Columns with Missing Values (out of {len(columns_with_missing)} total)', 
                           fontsize=12, fontweight='bold')
                
                # Add percentage labels
                for i, (bar, count) in enumerate(zip(bars, top_missing.values)):
                    percentage = (count / len(self.X)) * 100
                    ax.text(bar.get_width() + max(top_missing.values) * 0.01, 
                           bar.get_y() + bar.get_height()/2, 
                           f'{percentage:.1f}%', 
                           ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            self.missing_canvas.draw()
            
        except Exception as e:
            print(f"Error in visualize_missing_patterns: {str(e)}")
            # Create error message plot
            self.missing_figure.clear()
            ax = self.missing_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Visualization Error:\n{str(e)}', 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.missing_canvas.draw()
    
    def visualize_imputation_comparison(self):
        """Visualize before/after comparison of imputation"""
        if self.imputation_preview is None:
            self.impute_results_text.setText("Please run 'Preview Changes' first to enable comparison visualization.")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            self.missing_figure.clear()
            
            original_data = self.imputation_preview['original']
            imputed_data = self.imputation_preview['imputed']
            
            # Get numeric columns only
            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                ax = self.missing_figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No numeric columns selected for comparison.', 
                       ha='center', va='center', fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                self.missing_canvas.draw()
                return
            
            # Limit to first 6 columns for visualization
            cols_to_show = numeric_cols[:6]
            n_cols = len(cols_to_show)
            
            if n_cols == 1:
                fig_layout = (1, 2)
            elif n_cols <= 2:
                fig_layout = (1, 2)
            elif n_cols <= 4:
                fig_layout = (2, 2)
            else:
                fig_layout = (3, 2)
            
            for i, col in enumerate(cols_to_show):
                # Create subplot
                ax = self.missing_figure.add_subplot(fig_layout[0], fig_layout[1], i + 1)
                
                # Get data for this column
                original_col = original_data[col].dropna()
                imputed_col = imputed_data[col]
                
                # Get only the newly imputed values
                was_missing = original_data[col].isnull()
                newly_imputed = imputed_col[was_missing]
                
                if len(newly_imputed) > 0:
                    # Plot distributions
                    if len(original_col) > 0:
                        ax.hist(original_col, bins=20, alpha=0.7, color='blue', 
                               label=f'Original ({len(original_col)} values)', density=True)
                    
                    if len(newly_imputed) > 0:
                        ax.hist(newly_imputed, bins=20, alpha=0.7, color='red', 
                               label=f'Imputed ({len(newly_imputed)} values)', density=True)
                    
                    ax.set_title(f'{col}\nMissing: {len(newly_imputed)}', fontsize=10)
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Density')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics text
                    if len(original_col) > 0 and len(newly_imputed) > 0:
                        orig_mean = original_col.mean()
                        imp_mean = newly_imputed.mean()
                        ax.axvline(orig_mean, color='blue', linestyle='--', alpha=0.8)
                        ax.axvline(imp_mean, color='red', linestyle='--', alpha=0.8)
                else:
                    ax.text(0.5, 0.5, f'{col}\nNo missing values', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # If more than 6 columns, add a note
            if len(numeric_cols) > 6:
                self.missing_figure.suptitle(f'Distribution Comparison (showing first 6 of {len(numeric_cols)} numeric columns)', 
                                           fontsize=12, fontweight='bold')
            else:
                self.missing_figure.suptitle('Before vs After Imputation - Distribution Comparison', 
                                           fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            self.missing_canvas.draw()
            
        except Exception as e:
            print(f"Error in visualize_imputation_comparison: {str(e)}")
            self.missing_figure.clear()
            ax = self.missing_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Comparison Error:\n{str(e)}', 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.missing_canvas.draw()
    
    def visualize_distribution_analysis(self):
        """Enhanced distribution analysis with column selection"""
        if self.X is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            self.missing_figure.clear()
            
            # Get selection from dropdowns
            if hasattr(self, 'distribution_column_combo'):
                selection_data = self.distribution_column_combo.currentData()
                selection_text = self.distribution_column_combo.currentText()
            else:
                selection_data = "auto"
                selection_text = "Auto Select (Top 6)"
            
            if hasattr(self, 'distribution_type_combo'):
                viz_type = self.distribution_type_combo.currentData()
            else:
                viz_type = "histogram"
            
            # Get columns with missing values
            missing_counts = self.X.isnull().sum()
            columns_with_missing = missing_counts[missing_counts > 0].index
            
            if len(columns_with_missing) == 0:
                ax = self.missing_figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No missing values found for distribution analysis!', 
                       ha='center', va='center', fontsize=16, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                self.missing_canvas.draw()
                return
            
            # Get numeric columns with missing values
            numeric_cols_with_missing = []
            for col in columns_with_missing:
                if pd.api.types.is_numeric_dtype(self.X[col]):
                    numeric_cols_with_missing.append(col)
            
            if len(numeric_cols_with_missing) == 0:
                ax = self.missing_figure.add_subplot(111)
                ax.text(0.5, 0.5, 'No numeric columns with missing values for distribution analysis!', 
                       ha='center', va='center', fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                self.missing_canvas.draw()
                return
            
            # Determine columns to analyze based on selection
            if selection_data == "auto":
                # Auto select top 6 columns with highest missing percentages
                missing_info = []
                total_rows = len(self.X)
                for col in numeric_cols_with_missing:
                    missing_count = missing_counts[col]
                    missing_percent = (missing_count / total_rows) * 100
                    missing_info.append((col, missing_percent))
                
                # Sort by missing percentage (highest first) and take top 6
                missing_info.sort(key=lambda x: x[1], reverse=True)
                cols_to_analyze = [col for col, _ in missing_info[:6]]
                title_suffix = "(Top 6 by Missing %)"
                
            elif selection_data == "all":
                # Show all columns (but limit to first 12 for readability)
                cols_to_analyze = numeric_cols_with_missing[:12]
                title_suffix = f"(Showing {len(cols_to_analyze)} of {len(numeric_cols_with_missing)} columns)"
                
            elif selection_data == "separator":
                # Invalid selection, fallback to auto
                cols_to_analyze = numeric_cols_with_missing[:6]
                title_suffix = "(Top 6 columns)"
                
            else:
                # Single column selected
                if selection_data in numeric_cols_with_missing:
                    cols_to_analyze = [selection_data]
                    title_suffix = f"(Column: {selection_data})"
                else:
                    # Fallback to auto if selected column not found
                    cols_to_analyze = numeric_cols_with_missing[:6]
                    title_suffix = "(Top 6 columns)"
            
            n_cols = len(cols_to_analyze)
            
            # Determine layout based on number of columns
            if n_cols == 1:
                fig_layout = (1, 1)
            elif n_cols <= 2:
                fig_layout = (1, 2)
            elif n_cols <= 4:
                fig_layout = (2, 2)
            elif n_cols <= 6:
                fig_layout = (2, 3)
            elif n_cols <= 9:
                fig_layout = (3, 3)
            else:
                fig_layout = (3, 4)
            
            for i, col in enumerate(cols_to_analyze):
                if n_cols == 1:
                    ax = self.missing_figure.add_subplot(1, 1, 1)
                else:
                    ax = self.missing_figure.add_subplot(fig_layout[0], fig_layout[1], i + 1)
                
                # Get data for this column
                data_col = self.X[col].dropna()
                missing_count = self.X[col].isnull().sum()
                missing_percent = (missing_count / len(self.X)) * 100
                
                if len(data_col) > 0:
                    # Calculate statistics
                    mean_val = data_col.mean()
                    median_val = data_col.median()
                    std_val = data_col.std()
                    skew_val = data_col.skew()
                    
                    if viz_type == "histogram":
                        # Create histogram
                        ax.hist(data_col, bins=20, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
                        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                        ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                        ax.set_ylabel('Frequency')
                        
                    elif viz_type == "boxplot":
                        # Create box plot
                        box_data = ax.boxplot(data_col, patch_artist=True, notch=True)
                        box_data['boxes'][0].set_facecolor('lightblue')
                        box_data['boxes'][0].set_alpha(0.7)
                        ax.set_ylabel('Value')
                        ax.set_xticklabels([col[:15] + '...' if len(col) > 15 else col])
                        
                        # Add mean marker
                        ax.scatter([1], [mean_val], color='red', s=50, marker='D', label=f'Mean: {mean_val:.2f}', zorder=5)
                        
                    elif viz_type == "density":
                        # Create density plot
                        ax.hist(data_col, bins=30, density=True, alpha=0.3, color='skyblue', edgecolor='black', linewidth=0.5)
                        
                        # Add KDE if seaborn is available
                        if sns is not None:
                            sns.kdeplot(data=data_col, ax=ax, color='blue', linewidth=2)
                        
                        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                        ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                        ax.set_ylabel('Density')
                        
                    elif viz_type == "combined":
                        # Create combined view with histogram and overlay
                        ax.hist(data_col, bins=20, alpha=0.5, color='skyblue', edgecolor='black', linewidth=0.5, density=True, label='Distribution')
                        
                        # Add KDE overlay if seaborn is available
                        if sns is not None:
                            try:
                                sns.kdeplot(data=data_col, ax=ax, color='darkblue', linewidth=2, label='Density')
                            except:
                                pass  # Fallback if KDE fails
                        
                        # Add statistical lines
                        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
                        ax.axvline(median_val, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_val:.2f}')
                        
                        # Add quartiles
                        q1 = data_col.quantile(0.25)
                        q3 = data_col.quantile(0.75)
                        ax.axvline(q1, color='orange', linestyle=':', alpha=0.6, label=f'Q1: {q1:.2f}')
                        ax.axvline(q3, color='orange', linestyle=':', alpha=0.6, label=f'Q3: {q3:.2f}')
                        
                        ax.set_ylabel('Density')
                        
                        # Create a secondary y-axis for box plot
                        ax2 = ax.twinx()
                        box_data = ax2.boxplot(data_col, positions=[0.8], widths=0.1, vert=False, 
                                             patch_artist=True, showfliers=False)
                        box_data['boxes'][0].set_facecolor('lightcoral')
                        box_data['boxes'][0].set_alpha(0.6)
                        ax2.set_ylim(-0.2, 1.2)
                        ax2.set_yticks([])
                        ax2.set_ylabel('')
                    
                    # Common formatting for non-combined views
                    ax.set_title(f'{col}\nMissing: {missing_count} ({missing_percent:.1f}%)', fontsize=10)
                    ax.set_xlabel('Value')
                    
                    if viz_type != "boxplot":
                        ax.legend(fontsize=8)
                    
                    ax.grid(True, alpha=0.3)
                    
                    # Add distribution info as text (except for boxplot which has different layout)
                    if viz_type != "boxplot":
                        stats_text = f'Std: {std_val:.2f}\nSkew: {skew_val:.2f}'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', fontsize=8, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    else:
                        # For boxplot, show stats in legend
                        ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, f'{col}\nAll values missing', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Set appropriate title based on selection
            if selection_data == "auto":
                main_title = f'Distribution Analysis {title_suffix}'
            elif selection_data == "all":
                main_title = f'Distribution Analysis - All Columns {title_suffix}'
            elif len(cols_to_analyze) == 1:
                main_title = f'Distribution Analysis - {cols_to_analyze[0]}'
            else:
                main_title = f'Distribution Analysis {title_suffix}'
            
            self.missing_figure.suptitle(main_title, fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            self.missing_canvas.draw()
            
        except Exception as e:
            print(f"Error in visualize_distribution_analysis: {str(e)}")
            self.missing_figure.clear()
            ax = self.missing_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Distribution Analysis Error:\n{str(e)}', 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.missing_canvas.draw()

    def handle_missing_values(self):
        """Enhanced missing value imputation with column selection"""
        if self.X is None:
            self.impute_results_text.setText("No data available for imputation.")
            return
        
        selected_columns = self.get_selected_missing_columns()
        if not selected_columns:
            self.impute_results_text.setText("Please select columns for imputation.")
            return
        
        method = self.impute_method_combo.currentText()
        
        try:
            original_shape = self.X.shape
            missing_before = self.X[selected_columns].isnull().sum().sum()
            
            # Configure method-specific parameters
            kwargs = {}
            if method == 'knn':
                kwargs['n_neighbors'] = self.knn_neighbors_spin.value()
            elif method == 'iterative':
                kwargs['max_iter'] = self.iter_max_spin.value()
                kwargs['tol'] = self.iter_tolerance_spin.value()
            elif method == 'constant':
                try:
                    kwargs['fill_value'] = float(self.constant_value_edit.text())
                except ValueError:
                    kwargs['fill_value'] = self.constant_value_edit.text()
            
            # Apply imputation to selected columns only
            X_subset = self.X[selected_columns].copy()
            X_imputed_subset = self.smart_imputer.impute(X_subset, method, **kwargs)
            
            # Update the original dataframe with imputed values
            for col in selected_columns:
                self.X[col] = X_imputed_subset[col]
            
            new_shape = self.X.shape
            missing_after = self.X[selected_columns].isnull().sum().sum()
            filled_values = missing_before - missing_after
            
            # Save state and record operation history
            operation_name = f"Missing Value Imputation - {method}"
            self.save_state_and_update(operation_name)
            self.operation_history.add_operation(
                operation_name,
                {
                    "method": method,
                    "selected_columns": selected_columns,
                    "missing_filled": filled_values,
                    "parameters": kwargs
                },
                original_shape,
                new_shape
            )
            
            # Generate detailed result text
            result_text = f"""✅ Missing Value Imputation Completed
            
Method: {method}
Selected Columns: {len(selected_columns)}
Missing Values Before: {missing_before:,}
Missing Values After: {missing_after:,}
Values Filled: {filled_values:,}

Column Details:"""
            
            for col in selected_columns:
                col_missing_before = self.original_missing_columns.get(col, {}).get('count', 0)
                col_missing_after = self.X[col].isnull().sum()
                col_filled = col_missing_before - col_missing_after
                
                result_text += f"\n• {col}: {col_filled} values filled"
                
                if method in ['median', 'mean'] and col_filled > 0:
                    # Show what value was used for filling
                    if method == 'median':
                        fill_value = self.X[col].median()
                    else:  # mean
                        fill_value = self.X[col].mean()
                    result_text += f" (≈{fill_value:.3f})"
            
            if kwargs:
                result_text += f"\n\nParameters Used: {kwargs}"
            
            self.impute_results_text.setText(result_text)
            
            # Update missing columns list and clear preview
            self.update_missing_columns_list()
            self.imputation_preview = None
            
            # Refresh displays
            self.refresh_history()
            
            # Show success message
            QApplication.processEvents()  # Ensure UI updates
            
        except Exception as e:
            error_msg = f"Imputation failed: {str(e)}"
            self.impute_results_text.setText(error_msg)
            print(f"Error in handle_missing_values: {str(e)}")  # Debug output
    
    def detect_outliers(self):
        """检测异常值"""
        if self.X is None:
            return
        
        method = self.outlier_method_combo.currentText()
        selected_columns = self.get_selected_outlier_columns()
        
        if not selected_columns:
            self.outlier_results_text.setText("请先选择要检测异常值的列")
            return
        
        try:
            # 创建只包含选中列的数据副本进行检测
            X_selected = self.X[selected_columns].copy()
            
            if method == 'IQR':
                multiplier = self.iqr_multiplier_spin.value()
                outliers = self.outlier_detector.detect_outliers(X_selected, method, multiplier=multiplier)
            elif method in ['Isolation Forest', 'Local Outlier Factor', 'Elliptic Envelope']:
                contamination = self.contamination_spin.value()
                outliers = self.outlier_detector.detect_outliers(X_selected, method, contamination=contamination)
            else:
                outliers = self.outlier_detector.detect_outliers(X_selected, method)
            
            # 扩展异常值检测结果到完整数据框架
            full_outliers = pd.DataFrame(False, index=self.X.index, columns=self.X.columns)
            for col in selected_columns:
                if col in outliers.columns:
                    full_outliers[col] = outliers[col]
            
            self.detected_outliers = full_outliers
            
            # 统计异常值
            total_outliers = outliers.sum().sum()
            outlier_rows = outliers.any(axis=1).sum()
            
            result_text = f"""Outlier Detection Completed
Detection Method: {method}
Detected Columns: {', '.join(selected_columns)}
Detected Outliers: {total_outliers:,}
Affected Rows: {outlier_rows:,}
Outlier Ratio: {(outlier_rows / len(self.X)) * 100:.2f}%

By Column:
"""
            
            for col in selected_columns:
                if col in outliers.columns and outliers[col].sum() > 0:
                    result_text += f"• {col}: {outliers[col].sum()} outliers\n"
            
            self.outlier_results_text.setText(result_text)
            
            # Enable processing and visualization buttons
            self.handle_outliers_btn.setEnabled(True)
            if hasattr(self, 'visualize_outliers_btn'):
                self.visualize_outliers_btn.setEnabled(True)
            
        except Exception as e:
            self.outlier_results_text.setText(f"Failed: {str(e)}")
            self.detected_outliers = None
            self.handle_outliers_btn.setEnabled(False)
    
    def apply_transformation(self):
        """应用数据变换"""
        if self.X is None:
            return
        
        method = self.transform_method_combo.currentText()
        
        try:
            original_shape = self.X.shape
            numeric_cols = self.X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                self.transform_results_text.setText("No numeric columns to transform")
                return
            
            # 选择变换器
            if method == 'StandardScaler':
                scaler = StandardScaler()
            elif method == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif method == 'RobustScaler':
                scaler = RobustScaler()
            elif method == 'PowerTransformer':
                scaler = PowerTransformer()
            elif method == 'QuantileTransformer':
                scaler = QuantileTransformer()
            else:
                raise ValueError(f"Unknown transformer: {method}")
            
            # 应用变换
            X_transformed = self.X.copy()
            X_transformed[numeric_cols] = scaler.fit_transform(X_transformed[numeric_cols])
            self.X = X_transformed
            
            new_shape = self.X.shape
            
            # Save state and record operation history
            self.save_state_and_update(f"Data Transformation - {method}")
            self.operation_history.add_operation(
                f"Data Transformation - {method}",
                {
                    "method": method,
                    "transformed_columns": list(numeric_cols),
                    "num_columns": len(numeric_cols)
                },
                original_shape,
                new_shape
            )
            
            result_text = f"""Data Transformation Completed
Method: {method}
Transformed Columns: {len(numeric_cols)}
Transformed Column Names: {', '.join(numeric_cols)}"""
            
            self.transform_results_text.setText(result_text)
            
            # 刷新历史显示
            self.refresh_history()
            
        except Exception as e:
            self.transform_results_text.setText(f"Failed: {str(e)}")
    
    def reset_data(self):
        """重置数据"""
        if self.original_X is not None:
            self.X = self.original_X.copy()
            
        if self.original_y is not None:
            self.y = self.original_y.copy()
            
            # Reset outlier detection state
            self.detected_outliers = None
            if hasattr(self, 'handle_outliers_btn'):
                self.handle_outliers_btn.setEnabled(False)
            if hasattr(self, 'visualize_outliers_btn'):
                self.visualize_outliers_btn.setEnabled(False)
            
            # Clear operation history and state management
            self.operation_history.clear_history()
            self.state_manager.clear()
            self.state_manager.save_state(self.X, "Data Reset")
            
            # Clear imputation preview
            self.imputation_preview = None
            self.original_missing_columns = {}
            
            # Update UI components
            self.update_column_lists()
            self.update_missing_columns_list()
            self.update_undo_redo_buttons()
            self.refresh_state_history()
            
            # Clear visualizations
            if hasattr(self, 'outlier_figure'):
                self.outlier_figure.clear()
                self.outlier_canvas.draw()
            
            # 清空结果显示
            self.quality_results_text.clear()
            self.impute_results_text.clear()
            self.outlier_results_text.clear()
            self.transform_results_text.clear()
            self.history_text.clear()
            if hasattr(self, 'export_preview_text'):
                self.export_preview_text.clear()
            if hasattr(self, 'export_status_text'):
                self.export_status_text.clear()
    
    def preview_data(self):
        """预览数据"""
        if self.X is not None:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Data Preview")
            dialog.setGeometry(200, 200, 800, 600)
            
            layout = QVBoxLayout(dialog)
            
            table = QTableWidget()
            table.setRowCount(min(100, len(self.X)))
            table.setColumnCount(len(self.X.columns))
            table.setHorizontalHeaderLabels(self.X.columns.tolist())
            
            for i in range(min(100, len(self.X))):
                for j, col in enumerate(self.X.columns):
                    value = str(self.X.iloc[i, j])
                    table.setItem(i, j, QTableWidgetItem(value))
            
            layout.addWidget(table)
            dialog.exec_()
    
    def apply_preprocessing(self):
        """应用预处理结果并传递到下一个模块"""
        if self.X is not None:
            try:
                # 创建安全副本避免引用问题
                processed_data = self.X.copy()
                
                # 检查数据是否被处理过
                data_changed = False
                if self.original_X is not None:
                    try:
                        data_changed = not processed_data.equals(self.original_X)
                    except:
                        data_changed = True  # 如果无法比较，假设已更改
                
                # 发送信号传递处理后的数据
                if self.y is not None:
                    self.preprocessing_completed.emit(processed_data, self.y)
                else:
                    # 如果没有y数据，创建虚拟序列（正常工作流中不应该发生）
                    dummy_y = pd.Series([0] * len(processed_data), name='dummy_target')
                    self.preprocessing_completed.emit(processed_data, dummy_y)
                
                # 显示详细的成功消息
                operations_count = len(self.operation_history.operations)
                status_msg = f"预处理应用成功！\n\n"
                status_msg += f"处理后数据形状: {processed_data.shape}\n"
                status_msg += f"执行的操作数量: {operations_count}\n"
                
                if data_changed:
                    status_msg += f"✅ 数据已被成功处理\n"
                else:
                    status_msg += f"⚠️ 数据未发生变化，可能未执行预处理操作\n"
                
                status_msg += f"\n数据已传递到特征选择模块。"
                
                # 记录详细状态到控制台
                print("=== 预处理应用详情 ===")
                print(f"处理后数据形状: {processed_data.shape}")
                print(f"数据类型: {processed_data.dtypes.to_dict()}")
                print(f"缺失值数量: {processed_data.isnull().sum().sum()}")
                print(f"执行的操作: {operations_count}")
                print(f"数据是否改变: {data_changed}")
                if operations_count > 0:
                    print("操作历史:")
                    for i, op in enumerate(self.operation_history.operations, 1):
                        print(f"  {i}. {op['type']} ({op['timestamp']})")
                print("=== 应用完成 ===")
                
                QMessageBox.information(
                    self, 
                    "预处理应用成功", 
                    status_msg
                )
                
            except Exception as e:
                # 详细错误处理
                import traceback
                error_details = traceback.format_exc()
                print(f"ERROR in apply_preprocessing: {error_details}")
                
                QMessageBox.critical(
                    self, 
                    "预处理应用错误", 
                    f"应用预处理结果失败:\n\n{str(e)}\n\n"
                    f"请检查控制台以获取详细错误信息。"
                )
        else:
            QMessageBox.warning(
                self, 
                "无数据", 
                "没有可用的处理后数据。请确保您已经:\n"
                "1. 从数据管理模块加载了数据\n"
                "2. 执行了至少一个预处理操作\n\n"
                "提示: 请先在各个标签页中执行预处理操作（如处理缺失值、异常值检测等），\n"
                "然后再点击此按钮应用结果。"
            )
    
    def generate_recommendations(self):
        """Generate smart preprocessing recommendations"""
        if self.X is None:
            return
        
        try:
            self.current_recommendations = self.recommendation_engine.analyze_and_recommend(self.X)
            self.display_recommendations()
            
        except Exception as e:
            if hasattr(self, 'recommendation_summary'):
                self.recommendation_summary.setText(f"❌ Error generating recommendations: {str(e)}")
    
    def display_recommendations(self):
        """Display recommendations in the table"""
        if not hasattr(self, 'recommendations_table'):
            return
            
        self.recommendations_table.setRowCount(len(self.current_recommendations))
        
        priority_colors = {
            'Critical': '#F44336',
            'High': '#FF9800', 
            'Medium': '#FFC107',
            'Low': '#4CAF50'
        }
        
        for i, rec in enumerate(self.current_recommendations):
            # Priority
            priority_item = QTableWidgetItem(rec['priority'])
            priority_item.setBackground(QColor(priority_colors.get(rec['priority'], '#FFFFFF')))
            self.recommendations_table.setItem(i, 0, priority_item)
            
            # Type
            self.recommendations_table.setItem(i, 1, QTableWidgetItem(rec['type']))
            
            # Column
            self.recommendations_table.setItem(i, 2, QTableWidgetItem(rec['column']))
            
            # Issue
            self.recommendations_table.setItem(i, 3, QTableWidgetItem(rec['issue']))
            
            # Recommendation
            self.recommendations_table.setItem(i, 4, QTableWidgetItem(rec['recommendation']))
        
        # Enable action buttons
        self.apply_selected_btn.setEnabled(len(self.current_recommendations) > 0)
        self.apply_all_btn.setEnabled(len(self.current_recommendations) > 0)
        
        # Update summary
        total = len(self.current_recommendations)
        critical = len([r for r in self.current_recommendations if r['priority'] == 'Critical'])
        high = len([r for r in self.current_recommendations if r['priority'] == 'High'])
        
        summary = f"""📊 Recommendation Summary:
Total recommendations: {total}
Critical issues: {critical}
High priority issues: {high}

💡 Tip: Start with Critical and High priority recommendations for best results."""
        
        self.recommendation_summary.setText(summary)
    
    def apply_selected_recommendations(self):
        """Apply selected recommendations"""
        selected_rows = set()
        for item in self.recommendations_table.selectedItems():
            selected_rows.add(item.row())
        
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select recommendations to apply.")
            return
        
        applied_count = 0
        for row in selected_rows:
            if self.apply_recommendation(self.current_recommendations[row]):
                applied_count += 1
        
        QMessageBox.information(self, "Success", f"Applied {applied_count} recommendations successfully.")
        self.generate_recommendations()  # Refresh recommendations
    
    def apply_high_priority_recommendations(self):
        """Apply all high priority recommendations"""
        high_priority = [r for r in self.current_recommendations if r['priority'] in ['Critical', 'High']]
        
        if not high_priority:
            QMessageBox.information(self, "Info", "No high priority recommendations found.")
            return
        
        applied_count = 0
        for rec in high_priority:
            if self.apply_recommendation(rec):
                applied_count += 1
        
        QMessageBox.information(self, "Success", f"Applied {applied_count} high priority recommendations.")
        self.generate_recommendations()  # Refresh recommendations
    
    def apply_recommendation(self, recommendation: Dict[str, Any]) -> bool:
        """Apply a single recommendation"""
        try:
            action = recommendation.get('action')
            rec_type = recommendation.get('type')
            column = recommendation.get('column')
            
            if action == 'handle_missing_values':
                if 'Drop rows' in recommendation['recommendation']:
                    self.X = self.X.dropna(subset=[column])
                elif 'median' in recommendation['recommendation'].lower():
                    self.X[column] = self.X[column].fillna(self.X[column].median())
                elif 'mode' in recommendation['recommendation'].lower():
                    mode_val = self.X[column].mode()
                    if len(mode_val) > 0:
                        self.X[column] = self.X[column].fillna(mode_val.iloc[0])
                elif 'KNN' in recommendation['recommendation']:
                    self.X = self.smart_imputer.impute(self.X, 'knn')
                
                self.save_state_and_update(f"Applied missing value recommendation for {column}")
                return True
                
            elif action == 'handle_outliers':
                # Apply IQR-based outlier capping
                Q1 = self.X[column].quantile(0.25)
                Q3 = self.X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if 'Remove' in recommendation['recommendation']:
                    self.X = self.X[(self.X[column] >= lower_bound) & (self.X[column] <= upper_bound)]
                elif 'Cap' in recommendation['recommendation']:
                    self.X[column] = self.X[column].clip(lower=lower_bound, upper=upper_bound)
                
                self.save_state_and_update(f"Applied outlier recommendation for {column}")
                return True
                
            elif action == 'apply_transformation':
                if 'power' in recommendation['recommendation'].lower():
                    from sklearn.preprocessing import PowerTransformer
                    pt = PowerTransformer()
                    self.X[column] = pt.fit_transform(self.X[[column]]).flatten()
                elif 'log' in recommendation['recommendation'].lower():
                    self.X[column] = np.log1p(self.X[column])
                elif 'standard' in recommendation['recommendation'].lower():
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    self.X[column] = scaler.fit_transform(self.X[[column]]).flatten()
                
                self.save_state_and_update(f"Applied transformation recommendation for {column}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error applying recommendation: {e}")
            return False
    
    def save_state_and_update(self, operation_name: str):
        """Save current state and update UI"""
        self.state_manager.save_state(self.X, operation_name)
        self.update_undo_redo_buttons()
        # Auto-refresh distribution analysis after data changes
        self.auto_refresh_distribution_analysis()
        self.refresh_state_history()
    
    def undo_operation(self):
        """Undo last operation"""
        state = self.state_manager.undo()
        if state:
            self.X = state.data.copy()
            self.update_undo_redo_buttons()
            self.refresh_state_history()
            QMessageBox.information(self, "Undo", f"Undone: {state.operation_name}")
    
    def redo_operation(self):
        """Redo last undone operation"""
        state = self.state_manager.redo()
        if state:
            self.X = state.data.copy()
            self.update_undo_redo_buttons()
            self.refresh_state_history()
            QMessageBox.information(self, "Redo", f"Redone: {state.operation_name}")
    
    def update_undo_redo_buttons(self):
        """Update undo/redo button states"""
        if hasattr(self, 'undo_btn'):
            self.undo_btn.setEnabled(self.state_manager.can_undo())
        if hasattr(self, 'redo_btn'):
            self.redo_btn.setEnabled(self.state_manager.can_redo())
    
    def refresh_state_history(self):
        """Refresh state history display"""
        if hasattr(self, 'state_history_text'):
            history = self.state_manager.get_state_history()
            self.state_history_text.setText('\n'.join(history))
    
    def visualize_outliers(self):
        """Create outlier visualization with enhanced error handling"""
        if self.detected_outliers is None or self.X is None:
            QMessageBox.warning(self, "No Data", "Please detect outliers first before visualizing.")
            return
        
        try:
            selected_columns = self.get_selected_outlier_columns()
            if not selected_columns:
                QMessageBox.warning(self, "No Columns", "Please select columns for outlier visualization.")
                return
            
            # Clear the figure safely
            self.outlier_figure.clear()
            
            # Create subplots based on number of columns
            n_cols = min(len(selected_columns), 3)
            n_rows = (len(selected_columns) + n_cols - 1) // n_cols
            
            plot_count = 0
            for i, col in enumerate(selected_columns[:6]):  # Limit to 6 plots
                try:
                    plot_count += 1
                    ax = self.outlier_figure.add_subplot(n_rows, n_cols, plot_count)
                    
                    # Get outlier mask for this column
                    if col not in self.detected_outliers.columns:
                        continue
                        
                    outlier_mask = self.detected_outliers[col]
                    
                    # Create box plot with outliers highlighted
                    data_clean = self.X[col][~outlier_mask].dropna()
                    data_outliers = self.X[col][outlier_mask].dropna()
                    
                    if len(data_clean) == 0:
                        ax.text(0.5, 0.5, 'No clean data', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{col}\nNo data available')
                        continue
                    
                    # Box plot
                    bp = ax.boxplot([data_clean.values], patch_artist=True)
                    if bp['boxes']:
                        bp['boxes'][0].set_facecolor('lightblue')
                        bp['boxes'][0].set_alpha(0.7)
                    
                    # Scatter outliers
                    if len(data_outliers) > 0:
                        y_outliers = [1] * len(data_outliers)
                        ax.scatter([1] * len(data_outliers), data_outliers.values, 
                                 color='red', alpha=0.8, s=40, 
                                 label=f'Outliers ({len(data_outliers)})', 
                                 edgecolors='darkred', linewidths=0.5)
                        ax.legend(fontsize=8)
                    
                    ax.set_title(f'{col}\n{len(data_outliers)} outliers', fontsize=10)
                    ax.set_xticks([])
                    ax.grid(True, alpha=0.3)
                    
                except Exception as e:
                    print(f"Error plotting column {col}: {e}")
                    # Create an error plot
                    ax = self.outlier_figure.add_subplot(n_rows, n_cols, plot_count)
                    ax.text(0.5, 0.5, f'Error plotting\n{col}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{col} - Error')
            
            self.outlier_figure.suptitle('Outlier Detection Results', fontsize=12, fontweight='bold')
            self.outlier_figure.tight_layout()
            
            # Safely draw the canvas
            try:
                self.outlier_canvas.draw()
            except Exception as e:
                print(f"Error drawing canvas: {e}")
                QMessageBox.warning(self, "Visualization Error", f"Could not display outlier plots: {str(e)}")
                
        except Exception as e:
            print(f"Error creating outlier visualization: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Visualization Error", 
                               f"Failed to create outlier visualization:\n{str(e)}\n\n"
                               f"This might be due to data format issues or matplotlib configuration.")

    def update_column_lists(self):
        """更新列选择列表"""
        if self.X is None:
            return
        
        # 更新异常值检测的列列表（仅数值列）
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.outlier_columns_list.clear()
        for col in numeric_cols:
            item = QListWidgetItem(col)
            self.outlier_columns_list.addItem(item)
            item.setSelected(True)  # 默认选中所有数值列
    
    def select_all_outlier_columns(self):
        """选择所有异常值检测列"""
        for i in range(self.outlier_columns_list.count()):
            self.outlier_columns_list.item(i).setSelected(True)
    
    def clear_outlier_columns(self):
        """清空异常值检测列选择"""
        self.outlier_columns_list.clearSelection()
    
    def get_selected_outlier_columns(self):
        """获取选中的异常值检测列"""
        selected_items = self.outlier_columns_list.selectedItems()
        return [item.text() for item in selected_items]
    
    def handle_outliers(self):
        """处理检测到的异常值"""
        if self.detected_outliers is None or self.X is None:
            self.outlier_results_text.setText("Please detect outliers first")
            return
        
        method = self.outlier_handling_combo.currentText()
        selected_columns = self.get_selected_outlier_columns()
        
        if not selected_columns:
            self.outlier_results_text.setText("Please select columns to handle outliers")
            return
        
        try:
            original_shape = self.X.shape
            
            # 处理异常值
            self.X = self.outlier_detector.handle_outliers(
                self.X, self.detected_outliers, method, selected_columns
            )
            
            new_shape = self.X.shape
            
            # Save state and record operation history  
            self.save_state_and_update(f"Outlier Handling - {method}")
            self.operation_history.add_operation(
                f"Outlier Handling - {method}",
                {
                    "method": method,
                    "affected_columns": selected_columns,
                    "rows_removed": original_shape[0] - new_shape[0] if method == 'Remove Rows' else 0
                },
                original_shape,
                new_shape
            )
            
            result_text = f"""Outlier Handling Completed
Method: {method}
Affected Columns: {', '.join(selected_columns)}
Data Dimensions: {original_shape} → {new_shape}"""
             
            if method == 'Remove Rows':
                result_text += f"\nRemoved Rows: {original_shape[0] - new_shape[0]}"
            
            self.outlier_results_text.setText(result_text)
            
            # 重置异常值检测结果
            self.detected_outliers = None
            self.handle_outliers_btn.setEnabled(False)
            
            # 刷新历史显示
            self.refresh_history()
            
        except Exception as e:
            self.outlier_results_text.setText(f"Failed to handle outliers: {str(e)}")
    
    def refresh_history(self):
        """刷新操作历史显示"""
        self.history_text.setText(self.operation_history.get_history_text())
    
    def clear_history(self):
        """清空操作历史"""
        self.operation_history.clear_history()
        self.refresh_history()
    
    def preview_export_data(self):
        """预览要导出的数据"""
        try:
            data_type = self.export_data_combo.currentText()
            data_to_preview = self.get_export_data(data_type)
            
            if data_to_preview is None:
                self.export_preview_text.setText("No data available for preview")
                return
            
            # 根据数据类型生成预览
            if isinstance(data_to_preview, pd.DataFrame):
                preview_text = f"Data Dimensions: {data_to_preview.shape}\n"
                preview_text += f"Column Names: {list(data_to_preview.columns)}\n\n"
                preview_text += "Preview of first 5 rows:\n"
                preview_text += data_to_preview.head().to_string()
                
                if len(data_to_preview) > 5:
                    preview_text += f"\n\n... {len(data_to_preview) - 5} more rows"
            
            elif isinstance(data_to_preview, dict):
                preview_text = "Data Quality Report Preview:\n"
                preview_text += f"Basic Statistics: {data_to_preview.get('basic_stats', {})}\n"
                preview_text += f"Quality Score: {data_to_preview.get('quality_score', 'N/A')}\n"
                preview_text += "... (Full report will be generated during export)"
            
            elif isinstance(data_to_preview, str):
                preview_text = f"Text Content Preview (first 500 characters):\n"
                preview_text += data_to_preview[:500]
                if len(data_to_preview) > 500:
                    preview_text += "\n... (content truncated)"
            
            else:
                preview_text = f"Data Type: {type(data_to_preview).__name__}\n"
                preview_text += str(data_to_preview)[:500]
            
            self.export_preview_text.setText(preview_text)
            
        except Exception as e:
            self.export_preview_text.setText(f"Preview failed: {str(e)}")
    
    def refresh_export_preview(self):
        """刷新导出预览"""
        self.preview_export_data()
    
    def get_export_data(self, data_type):
        """根据选择获取要导出的数据"""
        if data_type == "Current processed data":
            return self.X
        elif data_type == "Original data":
            return self.original_X
        elif data_type == "Data quality report":
            if self.X is not None:
                return self.quality_analyzer.analyze(self.X)
            return None
        elif data_type == "Operation history":
            return self.operation_history.get_history_text()
        else:
            return None
    
    def quick_export(self):
        """快速导出到桌面"""
        try:
            data_type = self.export_data_combo.currentText()
            file_format = self.export_format_combo.currentText()
            
            # 获取桌面路径
            import os
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            
            # 生成文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if "CSV" in file_format:
                filename = f"processed_data_{timestamp}.csv"
                extension = ".csv"
            elif "Excel" in file_format:
                filename = f"processed_data_{timestamp}.xlsx"
                extension = ".xlsx"
            elif "TSV" in file_format:
                filename = f"processed_data_{timestamp}.tsv"
                extension = ".tsv"
            elif "JSON" in file_format:
                filename = f"processed_data_{timestamp}.json"
                extension = ".json"
            else:  # TXT
                filename = f"processed_data_{timestamp}.txt"
                extension = ".txt"
            
            filepath = os.path.join(desktop, filename)
            
            # 执行导出
            success = self.export_data_to_file(filepath, data_type, file_format)
            
            if success:
                self.export_status_text.setText(f"✅ Quick export successful!\nFile saved to: {filepath}")
            else:
                self.export_status_text.setText("❌ Quick export failed")
                
        except Exception as e:
            self.export_status_text.setText(f"❌ Quick export failed: {str(e)}")
    
    def custom_export(self):
        """自定义导出路径"""
        try:
            data_type = self.export_data_combo.currentText()
            file_format = self.export_format_combo.currentText()
            
            # 确定文件扩展名和过滤器
            if "CSV" in file_format:
                file_filter = "CSV Files (*.csv)"
                default_ext = ".csv"
            elif "Excel" in file_format:
                file_filter = "Excel Files (*.xlsx)"
                default_ext = ".xlsx"
            elif "TSV" in file_format:
                file_filter = "TSV Files (*.tsv)"
                default_ext = ".tsv"
            elif "JSON" in file_format:
                file_filter = "JSON Files (*.json)"
                default_ext = ".json"
            else:  # TXT
                file_filter = "Text Files (*.txt)"
                default_ext = ".txt"
            
            # 打开文件保存对话框
            filepath, _ = QFileDialog.getSaveFileName(
                self,
                f"Export {data_type}",
                f"processed_data{default_ext}",
                file_filter
            )
            
            if filepath:
                # 确保文件有正确的扩展名
                if not filepath.endswith(default_ext):
                    filepath += default_ext
                
                # 执行导出
                success = self.export_data_to_file(filepath, data_type, file_format)
                
                if success:
                    self.export_status_text.setText(f"✅ Export successful!\nFile saved to: {filepath}")
                    
                    # 询问是否打开文件所在文件夹
                    reply = QMessageBox.question(
                        self, "Export successful", 
                        f"File saved to:\n{filepath}\n\nOpen folder?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        import os
                        import subprocess
                        import platform
                        
                        folder_path = os.path.dirname(filepath)
                        if platform.system() == "Windows":
                            subprocess.run(["explorer", folder_path])
                        elif platform.system() == "Darwin":  # macOS
                            subprocess.run(["open", folder_path])
                        else:  # Linux
                            subprocess.run(["xdg-open", folder_path])
                else:
                    self.export_status_text.setText("❌ Export failed")
                    
        except Exception as e:
            self.export_status_text.setText(f"❌ Export failed: {str(e)}")
    
    def export_data_to_file(self, filepath, data_type, file_format):
        """执行实际的数据导出，支持中文编码和错误处理"""
        try:
            data = self.get_export_data(data_type)
            
            if data is None:
                error_msg = f"❌ 没有可用的 {data_type} 数据"
                self.export_status_text.setText(error_msg)
                return False
            
            # 获取导出选项
            include_index = self.include_index_cb.isChecked()
            include_header = self.include_header_cb.isChecked()
            encoding = self.encoding_combo.currentText()
            
            # 根据格式和数据类型导出
            if isinstance(data, pd.DataFrame):
                if "CSV" in file_format:
                    # 使用增强的编码处理，避免中文乱码
                    data.to_csv(filepath, index=include_index, header=include_header, 
                               encoding=encoding, errors='replace')
                    
                elif "Excel" in file_format:
                    # Excel自动处理编码
                    data.to_excel(filepath, index=include_index, header=include_header)
                    
                elif "TSV" in file_format:
                    data.to_csv(filepath, sep='\t', index=include_index, header=include_header, 
                               encoding=encoding, errors='replace')
                    
                elif "JSON" in file_format:
                    # 确保中文字符正确处理
                    data.to_json(filepath, orient='records', force_ascii=False, indent=2)
                    
                else:  # TXT
                    with open(filepath, 'w', encoding=encoding, errors='replace') as f:
                        f.write(data.to_string(index=include_index, header=include_header))
                    
            elif isinstance(data, dict):
                # 数据质量报告
                if "JSON" in file_format:
                    import json
                    with open(filepath, 'w', encoding=encoding, errors='replace') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                else:
                    # 转换为可读文本格式
                    report_text = self._format_quality_analysis(data)
                    with open(filepath, 'w', encoding=encoding, errors='replace') as f:
                        f.write(report_text)
                
            elif isinstance(data, str):
                # 操作历史记录
                with open(filepath, 'w', encoding=encoding, errors='replace') as f:
                    f.write(data)
                
            else:
                # 其他类型数据
                with open(filepath, 'w', encoding=encoding, errors='replace') as f:
                    f.write(str(data))
            
            # 验证文件是否成功创建且有内容
            import os
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return True
            else:
                self.export_status_text.setText("❌ 导出失败 - 文件未创建或为空")
                return False
            
        except Exception as e:
            error_msg = f"❌ 导出过程出错: {str(e)}"
            self.export_status_text.setText(error_msg)
            return False
    
    def show_data_status(self):
        """显示当前数据处理状态"""
        try:
            if self.X is None:
                QMessageBox.information(
                    self, 
                    "数据状态", 
                    "❌ 当前没有加载数据\n\n请先从数据管理模块加载数据。"
                )
                return
            
            # 检查数据是否被处理过
            data_changed = False
            if self.original_X is not None:
                try:
                    data_changed = not self.X.equals(self.original_X)
                except:
                    data_changed = True
            
            # 生成状态报告
            operations_count = len(self.operation_history.operations)
            missing_values = self.X.isnull().sum().sum()
            
            status_msg = f"📊 **当前数据状态报告**\n\n"
            status_msg += f"数据形状: {self.X.shape}\n"
            status_msg += f"缺失值数量: {missing_values}\n"
            status_msg += f"执行的操作数量: {operations_count}\n\n"
            
            if data_changed:
                status_msg += f"✅ 数据已被处理（与原始数据不同）\n"
            else:
                status_msg += f"⚠️ 数据未被处理（与原始数据相同）\n"
            
            if operations_count > 0:
                status_msg += f"\n📋 **操作历史**:\n"
                try:
                    for i, op in enumerate(self.operation_history.operations, 1):
                        # 安全地获取操作类型，提供默认值
                        op_type = op.get('type', op.get('operation_type', '未知操作'))
                        op_time = op.get('timestamp', '未知时间')
                        status_msg += f"{i}. {op_type} ({op_time})\n"
                except Exception as e:
                    status_msg += f"• 操作历史加载出错: {str(e)}\n"
            else:
                status_msg += f"\n💡 **建议**: 执行一些预处理操作，如:\n"
                status_msg += f"• 处理缺失值\n"
                status_msg += f"• 检测和处理异常值\n"
                status_msg += f"• 数据变换/标准化\n"
            
            status_msg += f"\n📋 **下一步**: 完成预处理后，点击'Apply Processing Results'按钮将数据传递到特征选择模块。"
            
            QMessageBox.information(
                self, 
                "数据处理状态", 
                status_msg
            )
            
        except Exception as e:
            # 如果出现任何错误，显示友好的错误消息
            QMessageBox.critical(
                self,
                "状态检查错误",
                f"检查数据状态时出现错误:\n\n{str(e)}\n\n请尝试重新加载数据或重启程序。"
            ) 