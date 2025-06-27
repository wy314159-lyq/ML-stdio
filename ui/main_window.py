"""
Main window for MatSci-ML Studio
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                            QWidget, QMenuBar, QStatusBar, QAction, QMessageBox,
                            QFileDialog, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont

from modules.data_module import DataModule
from modules.feature_module import FeatureModule
from modules.training_module import TrainingModule
from modules.prediction_module import PredictionModule
from modules.intelligent_wizard import IntelligentWizard
from modules.performance_monitor import PerformanceMonitor
from modules.advanced_preprocessing import AdvancedPreprocessing
from modules.collaboration_version_control import CollaborationWidget
from modules.shap_analysis import SHAPAnalysisModule
from modules.target_optimization import TargetOptimizationModule
from modules.multi_objective_optimization import MultiObjectiveOptimizationModule

from ui.active_learning_window import ActiveLearningWindow


class MatSciMLStudioWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_modules()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("AutoMatFlow Studio v1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application font
        font = QFont("Segoe UI", 9)
        self.setFont(font)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget for modules
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        layout.addWidget(self.tab_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress bar for status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load Project", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_project)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        user_guide_action = QAction("User Guide", self)
        user_guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(user_guide_action)
        
    def setup_modules(self):
        """Setup all application modules"""
        # Module 1: Data Ingestion & Preprocessing
        self.data_module = DataModule()
        self.tab_widget.addTab(self.data_module, "📊 Data Management")
        
        # Module 2: Intelligent Wizard
        self.intelligent_wizard = IntelligentWizard()
        self.tab_widget.addTab(self.intelligent_wizard, "🧙‍♂️ Intelligent Wizard")
        
        # Module 3: Advanced Preprocessing
        self.advanced_preprocessing = AdvancedPreprocessing()
        self.tab_widget.addTab(self.advanced_preprocessing, "🧬 Advanced Preprocessing")
        
        # Module 4: Feature Engineering & Selection
        self.feature_module = FeatureModule()
        self.tab_widget.addTab(self.feature_module, "🎯 Feature Selection")
        
        # Module 5: Model Training & Evaluation
        self.training_module = TrainingModule()
        self.tab_widget.addTab(self.training_module, "🔬 Model Training")
        
        # Module 6: Prediction & Export
        self.prediction_module = PredictionModule()
        self.tab_widget.addTab(self.prediction_module, "🎯 Model Prediction")
        
        # Module 7: Performance Monitor
        self.performance_monitor = PerformanceMonitor()
        self.tab_widget.addTab(self.performance_monitor, "📊 Performance Monitor")
        
        # Module 8: Collaboration & Version Control
        self.collaboration_widget = CollaborationWidget()
        self.tab_widget.addTab(self.collaboration_widget, "🤝 Collaboration")
        
        # Module 9: SHAP Analysis
        self.shap_analysis = SHAPAnalysisModule()
        self.tab_widget.addTab(self.shap_analysis, "🧠 SHAP Analysis")
        
        # Module 10: Target Optimization
        self.target_optimization = TargetOptimizationModule()
        self.tab_widget.addTab(self.target_optimization, "🎯 Target Optimization")
        
        # Module 11: Single & Multi-Objective Optimization
        self.multi_objective_optimization = MultiObjectiveOptimizationModule()
        self.tab_widget.addTab(self.multi_objective_optimization, "🔄 Optimization")
        
        # Module 12: Active Learning & Optimization
        self.active_learning_window = ActiveLearningWindow()
        self.tab_widget.addTab(self.active_learning_window, "🤖 Active Learning & Optimization")
        
        # Connect modules
        self.connect_modules()
        
        # Initially disable some modules
        self.set_module_enabled(3, False)  # Feature Selection
        self.set_module_enabled(4, False)  # Model Training
        # self.set_module_enabled(5, False)  # Prediction - Keep enabled for independent use
        self.set_module_enabled(8, False)  # SHAP Analysis
        self.set_module_enabled(9, False)  # Target Optimization
        # Prediction and Multi-Objective Optimization are always enabled (independent modules)
        
    def connect_modules(self):
        """Connect signals between modules"""
        # Data module -> Intelligent Wizard and Advanced Preprocessing
        self.data_module.data_ready.connect(self.intelligent_wizard.set_data)
        self.data_module.data_ready.connect(self.advanced_preprocessing.set_data)
        self.data_module.data_ready.connect(lambda: self.set_module_enabled(3, True))  # Enable Feature Selection
        
        # Intelligent Wizard -> Other modules
        self.intelligent_wizard.configuration_ready.connect(self.apply_wizard_configuration)
        
        # Advanced Preprocessing -> Feature module (with safe wrapper)
        self.advanced_preprocessing.preprocessing_completed.connect(self.safe_set_feature_data)
        
        # Data/Preprocessing -> Feature module
        self.data_module.data_ready.connect(self.feature_module.set_data)
        
        # Feature module -> Training module (with safe wrapper)
        self.feature_module.features_ready.connect(self.safe_set_training_data)
        self.feature_module.features_ready.connect(lambda: self.set_module_enabled(4, True))  # Enable Training
        
        # Training module -> Prediction module (pass model to already-enabled prediction module)
        self.training_module.model_ready.connect(self.prediction_module.set_model)
        # Note: Prediction module is already enabled for independent use
        
        # Training module -> SHAP Analysis and Target Optimization
        self.training_module.model_ready.connect(self.shap_analysis.set_model)
        # Modified to pass training data to target optimization
        self.training_module.model_ready.connect(
            lambda model, feature_names, feature_info, X_train, y_train: 
            self.target_optimization.set_model(model, feature_names, feature_info, X_train)
        )
        self.training_module.model_ready.connect(lambda: self.set_module_enabled(8, True))  # Enable SHAP Analysis
        self.training_module.model_ready.connect(lambda: self.set_module_enabled(9, True))  # Enable Target Optimization
        # Multi-Objective Optimization is independent and doesn't need model_ready signal
        
        # Also pass training data to SHAP analysis (with safe wrapper)
        self.feature_module.features_ready.connect(self.safe_set_shap_data)
        
        # Performance Monitor connections
        self.feature_module.selection_started.connect(
            lambda: self.performance_monitor.start_task("feature_selection", "Feature Selection", 100)
        )
        self.training_module.training_started.connect(
            lambda: self.performance_monitor.start_task("model_training", "Model Training", 100)
        )
        
        # Progress updates
        self.data_module.progress_updated.connect(self.update_progress)
        self.feature_module.progress_updated.connect(self.update_progress)
        self.training_module.progress_updated.connect(self.update_progress)
        self.prediction_module.progress_updated.connect(self.update_progress)
        
        # Progress updates to performance monitor
        self.feature_module.progress_updated.connect(
            lambda value: self.performance_monitor.update_task_progress("feature_selection", value)
        )
        self.training_module.progress_updated.connect(
            lambda value: self.performance_monitor.update_task_progress("model_training", value)
        )
        
        # Status updates
        self.data_module.status_updated.connect(self.update_status)
        self.feature_module.status_updated.connect(self.update_status)
        self.training_module.status_updated.connect(self.update_status)
        self.prediction_module.status_updated.connect(self.update_status)
        self.shap_analysis.status_updated.connect(self.update_status)
        self.target_optimization.status_updated.connect(self.update_status)
        self.intelligent_wizard.wizard_completed.connect(lambda: self.update_status("Intelligent configuration applied"))
        self.advanced_preprocessing.preprocessing_completed.connect(lambda: self.update_status("Advanced preprocessing completed"))
        
        # Performance alerts
        self.performance_monitor.performance_alert.connect(self.handle_performance_alert)
    
    def safe_set_feature_data(self, X, y):
        """Safely set feature data with error handling"""
        try:
            print("=== SAFE FEATURE DATA TRANSFER FROM PREPROCESSING ===")
            print(f"Transferring data: X shape {X.shape}, y shape {y.shape}")
            self.feature_module.set_data(X, y)
            print("✓ Feature data transfer from preprocessing successful")
        except Exception as e:
            print(f"ERROR in safe_set_feature_data: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Data Transfer Error", 
                               f"Failed to transfer preprocessed data to feature module:\n{str(e)}\n\n"
                               f"Please try the following:\n"
                               f"1. Check if preprocessing was completed successfully\n"
                               f"2. Restart the application if the issue persists")
        
    def safe_set_training_data(self, X, y, config):
        """Safely set training data with error handling"""
        try:
            print("=== SAFE TRAINING DATA TRANSFER ===")
            print(f"Transferring data: X shape {X.shape}, y shape {y.shape}")
            self.training_module.set_data(X, y, config)
            print("✓ Training data transfer successful")
        except Exception as e:
            print(f"ERROR in safe_set_training_data: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Data Transfer Error", 
                               f"Failed to transfer data to training module:\n{str(e)}\n\nPlease try again or restart the application.")
    
    def safe_set_shap_data(self, X, y, config):
        """Safely set SHAP data with error handling"""
        try:
            print("=== SAFE SHAP DATA TRANSFER ===")
            print(f"Transferring data to SHAP: X shape {X.shape}, y shape {y.shape}")
            self.shap_analysis.set_data(X, y, config)
            print("✓ SHAP data transfer successful")
        except Exception as e:
            print(f"ERROR in safe_set_shap_data: {str(e)}")
            # Don't show error for SHAP as it's not critical
            print("SHAP data transfer failed, but continuing...")
    
    def set_module_enabled(self, module_index: int, enabled: bool):
        """Enable/disable a module tab"""
        self.tab_widget.setTabEnabled(module_index, enabled)
        
    def update_progress(self, value: int):
        """Update progress bar"""
        if value == 0:
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(value)
            if value >= 100:
                self.progress_bar.setVisible(False)
                
    def update_status(self, message: str):
        """Update status bar message"""
        self.status_bar.showMessage(message)
    
    def apply_wizard_configuration(self, config: dict):
        """Apply intelligent wizard configuration"""
        try:
            # Apply feature selection configuration
            if 'feature_selection' in config:
                self.feature_module.apply_wizard_config(config['feature_selection'])
            
            # Apply model configuration
            if 'selected_models' in config:
                self.training_module.apply_wizard_config(config)
            
            self.update_status("Intelligent wizard configuration applied")
            
        except Exception as e:
            self.update_status(f"Failed to apply configuration: {str(e)}")
    
    def handle_performance_alert(self, alert_type: str, message: str):
        """Handle performance alerts"""
        if alert_type == 'cpu_high':
            QMessageBox.warning(self, "Performance Warning", f"High CPU usage!\n{message}\nRecommend pausing compute-intensive tasks.")
        elif alert_type == 'memory_high':
            QMessageBox.warning(self, "Performance Warning", f"High memory usage!\n{message}\nRecommend reducing dataset size or enabling batch processing.")
        elif alert_type == 'disk_full':
            QMessageBox.critical(self, "Storage Warning", f"Low disk space!\n{message}\nPlease free up disk space.")
        
    def new_project(self):
        """Create new project"""
        reply = QMessageBox.question(
            self, "New Project", 
            "This will clear all current work. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset all modules
            self.data_module.reset()
            self.feature_module.reset()
            self.training_module.reset()
            self.prediction_module.reset()
            self.shap_analysis.reset()
            self.target_optimization.reset()
            
            # Disable modules 3-9 (except Prediction which stays enabled)
            self.set_module_enabled(3, False)  # Feature Selection
            self.set_module_enabled(4, False)  # Model Training
            # self.set_module_enabled(5, False)  # Prediction - Keep enabled for independent use
            self.set_module_enabled(8, False)  # SHAP Analysis
            self.set_module_enabled(9, False)  # Target Optimization
            
            # Switch to first tab
            self.tab_widget.setCurrentIndex(0)
            
            self.update_status("New project created")
            
    def save_project(self):
        """Save project"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "AutoMatFlow Projects (*.mml)"
        )
        
        if file_path:
            try:
                # TODO: Implement project saving
                self.update_status(f"Project saved to {file_path}")
                QMessageBox.information(self, "Success", "Project saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
                
    def load_project(self):
        """Load project"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "AutoMatFlow Projects (*.mml)"
        )
        
        if file_path:
            try:
                # TODO: Implement project loading
                self.update_status(f"Project loaded from {file_path}")
                QMessageBox.information(self, "Success", "Project loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
                
    def reset_layout(self):
        """Reset window layout"""
        self.resize(1400, 900)
        self.center_window()
        
    def center_window(self):
        """Center the window on screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
        
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>AutoMatFlow Studio v1.0</h2>
        This software is developed by Dr. Yu Wang from Sichuan University. Welcome to use! If you have any questions or bugs, please contact the email 1255201958@qq.com
        """
        
        QMessageBox.about(self, "About AutoMatFlow Studio", about_text)
        
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
        <h2>AutoMatFlow Studio User Guide</h2>
        
        <h3>Module 1: Data & Preprocessing</h3>
        <p>Import your materials data from CSV or Excel files. Explore data quality, 
        handle missing values, and prepare features and target variables.</p>
        
        <h3>Module 2: Feature Selection</h3>
        <p>Select the best features using multiple strategies: importance-based filtering,
        correlation analysis, and advanced wrapper methods.</p>
        
        <h3>Module 3: Model Training</h3>
        <p>Train machine learning models with hyperparameter optimization.
        Evaluate model performance with comprehensive metrics and visualizations.</p>
        
        <h3>Module 4: Prediction</h3>
        <p>Apply trained models to new data and export predictions.</p>
        
        <p><b>Workflow:</b> Follow the modules in order for the best experience.
        Each module builds upon the previous one.</p>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("User Guide")
        msg.setText(guide_text)
        msg.setTextFormat(Qt.RichText)
        msg.exec_()
        
    def closeEvent(self, event):
        """Handle application close event"""
        reply = QMessageBox.question(
            self, "Exit Application",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("AutoMatFlow Studio")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    window = MatSciMLStudioWindow()
    window.show()
    window.center_window()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 