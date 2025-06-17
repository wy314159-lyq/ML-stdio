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
                            QProgressBar, QSplitter, QFrame)
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

class SmartImputer:
    """智能缺失值填补器"""
    
    def __init__(self):
        self.methods = {
            'mean': lambda x: x.fillna(x.mean()),
            'median': lambda x: x.fillna(x.median()),
            'mode': lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
            'forward_fill': lambda x: x.fillna(method='ffill'),
            'backward_fill': lambda x: x.fillna(method='bfill'),
            'interpolate': lambda x: x.interpolate(),
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
    
    def _iterative_impute(self, X: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        """迭代填补"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        
        result = X.copy()
        
        # 数值列使用迭代填补
        if len(numeric_cols) > 0:
            iterative_imputer = IterativeImputer(max_iter=max_iter, random_state=42)
            result[numeric_cols] = iterative_imputer.fit_transform(result[numeric_cols])
        
        # 分类列使用众数
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = result[col].mode()
                if len(mode_value) > 0:
                    result[col].fillna(mode_value.iloc[0], inplace=True)
        
        return result

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
    preprocessing_completed = pyqtSignal(pd.DataFrame)
    quality_analysis_completed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.X = None
        self.original_X = None
        self.outlier_detector = OutlierDetector()
        self.smart_imputer = SmartImputer()
        self.quality_analyzer = DataQualityAnalyzer()
        
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
        
        # 1. 数据质量分析
        self.create_quality_analysis_tab()
        
        # 2. 缺失值处理
        self.create_missing_values_tab()
        
        # 3. 异常值检测
        self.create_outlier_detection_tab()
        
        # 4. 数据变换
        self.create_transformation_tab()
        
        # 控制按钮
        self.create_control_buttons(layout)
    
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
        """创建缺失值处理标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Method selection
        method_group = QGroupBox("Imputation Method")
        method_layout = QVBoxLayout(method_group)
        
        self.impute_method_combo = QComboBox()
        self.impute_method_combo.addItems([
            'median', 'mean', 'mode', 'forward_fill', 
            'backward_fill', 'interpolate', 'knn', 'iterative'
        ])
        method_layout.addWidget(self.impute_method_combo)
        
        # KNN parameters
        self.knn_neighbors_spin = QSpinBox()
        self.knn_neighbors_spin.setRange(1, 20)
        self.knn_neighbors_spin.setValue(5)
        knn_layout = QHBoxLayout()
        knn_layout.addWidget(QLabel("KNN Neighbors:"))
        knn_layout.addWidget(self.knn_neighbors_spin)
        method_layout.addLayout(knn_layout)
        
        layout.addWidget(method_group)
        
        # Execute button
        impute_btn = QPushButton("🔧 Execute Missing Value Imputation")
        impute_btn.clicked.connect(self.handle_missing_values)
        layout.addWidget(impute_btn)
        
        # Results display
        self.impute_results_text = QTextEdit()
        self.impute_results_text.setReadOnly(True)
        layout.addWidget(self.impute_results_text)
        
        self.tabs.addTab(tab, "Missing Values")
    
    def create_outlier_detection_tab(self):
        """创建异常值检测标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Method selection
        method_group = QGroupBox("Detection Method")
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
        
        # Execute button
        detect_btn = QPushButton("🎯 Detect Outliers")
        detect_btn.clicked.connect(self.detect_outliers)
        layout.addWidget(detect_btn)
        
        # Results display
        self.outlier_results_text = QTextEdit()
        self.outlier_results_text.setReadOnly(True)
        layout.addWidget(self.outlier_results_text)
        
        self.tabs.addTab(tab, "Outlier Detection")
    
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
    
    def create_control_buttons(self, parent_layout):
        """创建控制按钮"""
        button_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 Reset Data")
        self.reset_btn.clicked.connect(self.reset_data)
        button_layout.addWidget(self.reset_btn)
        
        self.preview_btn = QPushButton("👁️ Preview Data")
        self.preview_btn.clicked.connect(self.preview_data)
        button_layout.addWidget(self.preview_btn)
        
        self.apply_btn = QPushButton("✅ Apply Processing Results")
        self.apply_btn.clicked.connect(self.apply_preprocessing)
        button_layout.addWidget(self.apply_btn)
        
        parent_layout.addLayout(button_layout)
    
    def set_data(self, X: pd.DataFrame):
        """设置数据"""
        self.X = X.copy()
        self.original_X = X.copy()
    
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
        """格式化质量分析结果"""
        text = f"""Data quality analysis report
{'='*50}

Overall quality score: {analysis['quality_score']:.1f}/100

Basic statistics:
• Number of rows: {analysis['basic_stats']['n_rows']:,}
• Number of columns: {analysis['basic_stats']['n_columns']}
• Memory usage: {analysis['basic_stats']['memory_usage_mb']:.2f} MB
• Numeric columns: {analysis['basic_stats']['numeric_columns']}
• 分类列: {analysis['basic_stats']['categorical_columns']}

Missing data:
• Total missing: {analysis['missing_data']['total_missing']:,}
• Columns with missing: {analysis['missing_data']['columns_with_missing']}
• Worst column: {analysis['missing_data']['worst_column']} ({analysis['missing_data']['worst_percentage']:.1f}%)

Duplicates:
• Total duplicates: {analysis['duplicates']['total_duplicates']:,}
• Duplicate ratio: {analysis['duplicates']['percentage']:.2f}%

"""
        
        if 'high_correlation_pairs' in analysis['correlation']:
            text += f"""Correlation analysis:
• High correlation pairs: {len(analysis['correlation']['high_correlation_pairs'])}
"""
            for pair in analysis['correlation']['high_correlation_pairs'][:5]:  # 显示前5对
                text += f"  - {pair['feature1']} vs {pair['feature2']}: {pair['correlation']:.3f}\n"
        
        return text
    
    def handle_missing_values(self):
        """处理缺失值"""
        if self.X is None:
            return
        
        method = self.impute_method_combo.currentText()
        
        try:
            if method == 'knn':
                n_neighbors = self.knn_neighbors_spin.value()
                self.X = self.smart_imputer.impute(self.X, method, n_neighbors=n_neighbors)
            else:
                self.X = self.smart_imputer.impute(self.X, method)
            
            # 显示结果
            missing_before = self.original_X.isnull().sum().sum()
            missing_after = self.X.isnull().sum().sum()
            
            result_text = f"""Missing value processing completed
Method: {method}
Before: {missing_before:,}
After: {missing_after:,}
Filled: {missing_before - missing_after:,}"""
            
            self.impute_results_text.setText(result_text)
            
        except Exception as e:
            self.impute_results_text.setText(f"Processing failed: {str(e)}")
    
    def detect_outliers(self):
        """检测异常值"""
        if self.X is None:
            return
        
        method = self.outlier_method_combo.currentText()
        
        try:
            if method == 'IQR':
                multiplier = self.iqr_multiplier_spin.value()
                outliers = self.outlier_detector.detect_outliers(self.X, method, multiplier=multiplier)
            elif method in ['Isolation Forest', 'Local Outlier Factor', 'Elliptic Envelope']:
                contamination = self.contamination_spin.value()
                outliers = self.outlier_detector.detect_outliers(self.X, method, contamination=contamination)
            else:
                outliers = self.outlier_detector.detect_outliers(self.X, method)
            
            # 统计异常值
            total_outliers = outliers.sum().sum()
            outlier_rows = outliers.any(axis=1).sum()
            
            result_text = f"""Outlier detection completed
Method: {method}
Detected outliers: {total_outliers:,}
Affected rows: {outlier_rows:,}
Outlier ratio: {(outlier_rows / len(self.X)) * 100:.2f}%

By column:
"""
            
            for col in outliers.columns:
                if outliers[col].sum() > 0:
                    result_text += f"• {col}: {outliers[col].sum()} 个异常值\n"
            
            self.outlier_results_text.setText(result_text)
            
        except Exception as e:
            self.outlier_results_text.setText(f"检测失败: {str(e)}")
    
    def apply_transformation(self):
        """应用数据变换"""
        if self.X is None:
            return
        
        method = self.transform_method_combo.currentText()
        
        try:
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
            
            result_text = f"""Data transformation completed
Method: {method}
Transformed columns: {len(numeric_cols)}
Transformed columns: {', '.join(numeric_cols)}"""
            
            self.transform_results_text.setText(result_text)
            
        except Exception as e:
            self.transform_results_text.setText(f"Transformation failed: {str(e)}")
    
    def reset_data(self):
        """重置数据"""
        if self.original_X is not None:
            self.X = self.original_X.copy()
            
            # 清空结果显示
            self.quality_results_text.clear()
            self.impute_results_text.clear()
            self.outlier_results_text.clear()
            self.transform_results_text.clear()
    
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
        """应用预处理结果"""
        if self.X is not None:
            self.preprocessing_completed.emit(self.X) 