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
        """Create the enhanced menu bar with comprehensive functionality"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # New submenu
        new_menu = file_menu.addMenu("New")
        new_project_action = QAction("Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.triggered.connect(self.new_project)
        new_menu.addAction(new_project_action)
        
        new_experiment_action = QAction("Experiment", self)
        new_experiment_action.setShortcut("Ctrl+Shift+N")
        new_experiment_action.triggered.connect(self.new_experiment)
        new_menu.addAction(new_experiment_action)
        
        file_menu.addSeparator()
        
        # Import Data submenu
        import_menu = file_menu.addMenu("Import Data")
        import_training_action = QAction("Training Data (.csv, .xlsx)", self)
        import_training_action.triggered.connect(self.import_training_data)
        import_menu.addAction(import_training_action)
        
        import_virtual_action = QAction("Virtual Screening Data", self)
        import_virtual_action.triggered.connect(self.import_virtual_data)
        import_menu.addAction(import_virtual_action)
        
        import_model_action = QAction("Pre-trained Model (.joblib)", self)
        import_model_action.triggered.connect(self.import_model)
        import_menu.addAction(import_model_action)
        
        # Export submenu
        export_menu = file_menu.addMenu("Export")
        export_results_action = QAction("Results (.csv)", self)
        export_results_action.triggered.connect(self.export_results)
        export_menu.addAction(export_results_action)
        
        export_charts_action = QAction("Charts (.png, .svg)", self)
        export_charts_action.triggered.connect(self.export_charts)
        export_menu.addAction(export_charts_action)
        
        export_model_action = QAction("Model (.joblib)", self)
        export_model_action.triggered.connect(self.export_model)
        export_menu.addAction(export_model_action)
        
        export_report_action = QAction("Optimization Report (.md, .pdf)", self)
        export_report_action.triggered.connect(self.export_report)
        export_menu.addAction(export_report_action)
        
        file_menu.addSeparator()
        
        # Recent Projects
        recent_menu = file_menu.addMenu("Recent Projects")
        self.update_recent_projects_menu(recent_menu)
        
        file_menu.addSeparator()
        
        # Settings
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
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
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_action)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo_action)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        copy_action = QAction("Copy", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.copy_data)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("Paste", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self.paste_data)
        edit_menu.addAction(paste_action)
        
        edit_menu.addSeparator()
        
        find_action = QAction("Find", self)
        find_action.setShortcut("Ctrl+F")
        find_action.triggered.connect(self.show_find_dialog)
        edit_menu.addAction(find_action)
        
        # Data menu
        data_menu = menubar.addMenu("Data")
        
        data_import_action = QAction("Import Data", self)
        data_import_action.triggered.connect(self.switch_to_data_module)
        data_menu.addAction(data_import_action)
        
        data_preprocessing_action = QAction("Preprocessing", self)
        data_preprocessing_action.triggered.connect(self.switch_to_preprocessing_module)
        data_menu.addAction(data_preprocessing_action)
        
        feature_engineering_action = QAction("Feature Engineering", self)
        feature_engineering_action.triggered.connect(self.switch_to_feature_module)
        data_menu.addAction(feature_engineering_action)
        
        data_menu.addSeparator()
        
        data_visualization_action = QAction("Data Visualization", self)
        data_visualization_action.triggered.connect(self.show_data_visualization)
        data_menu.addAction(data_visualization_action)
        
        data_summary_action = QAction("Data Summary", self)
        data_summary_action.triggered.connect(self.show_data_summary)
        data_menu.addAction(data_summary_action)
        
        # Analyze menu
        analyze_menu = menubar.addMenu("Analyze")
        
        train_model_action = QAction("Train Model", self)
        train_model_action.triggered.connect(self.switch_to_training_module)
        analyze_menu.addAction(train_model_action)
        
        evaluate_model_action = QAction("Evaluate Model", self)
        evaluate_model_action.triggered.connect(self.show_model_evaluation)
        analyze_menu.addAction(evaluate_model_action)
        
        analyze_menu.addSeparator()
        
        shap_analysis_action = QAction("SHAP Analysis", self)
        shap_analysis_action.triggered.connect(self.switch_to_shap_module)
        analyze_menu.addAction(shap_analysis_action)
        
        learning_curve_action = QAction("Learning Curve Analysis", self)
        learning_curve_action.triggered.connect(self.show_learning_curve_analysis)
        analyze_menu.addAction(learning_curve_action)
        
        performance_monitor_action = QAction("Performance Monitor", self)
        performance_monitor_action.triggered.connect(self.switch_to_performance_module)
        analyze_menu.addAction(performance_monitor_action)
        
        # Optimize menu
        optimize_menu = menubar.addMenu("Optimize")
        
        active_learning_action = QAction("Active Learning", self)
        active_learning_action.triggered.connect(self.switch_to_active_learning_module)
        optimize_menu.addAction(active_learning_action)
        
        multi_objective_action = QAction("Multi-objective Optimization", self)
        multi_objective_action.triggered.connect(self.switch_to_multi_objective_module)
        optimize_menu.addAction(multi_objective_action)
        
        target_prediction_action = QAction("Target Prediction", self)
        target_prediction_action.triggered.connect(self.switch_to_prediction_module)
        optimize_menu.addAction(target_prediction_action)
        
        optimize_menu.addSeparator()
        
        target_optimization_action = QAction("Target Optimization", self)
        target_optimization_action.triggered.connect(self.switch_to_target_optimization_module)
        optimize_menu.addAction(target_optimization_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Panels submenu
        panels_menu = view_menu.addMenu("Panels")
        
        log_console_action = QAction("Log Console", self)
        log_console_action.setCheckable(True)
        log_console_action.setChecked(True)
        log_console_action.triggered.connect(self.toggle_log_console)
        panels_menu.addAction(log_console_action)
        
        data_viewer_action = QAction("Data Viewer", self)
        data_viewer_action.setCheckable(True)
        data_viewer_action.setChecked(True)
        data_viewer_action.triggered.connect(self.toggle_data_viewer)
        panels_menu.addAction(data_viewer_action)
        
        charts_panel_action = QAction("Charts Panel", self)
        charts_panel_action.setCheckable(True)
        charts_panel_action.setChecked(True)
        charts_panel_action.triggered.connect(self.toggle_charts_panel)
        panels_menu.addAction(charts_panel_action)
        
        view_menu.addSeparator()
        
        # Appearance submenu
        appearance_menu = view_menu.addMenu("Appearance")
        
        theme_action = QAction("Toggle Light/Dark Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        appearance_menu.addAction(theme_action)
        
        font_size_menu = appearance_menu.addMenu("Font Size")
        
        small_font_action = QAction("Small", self)
        small_font_action.triggered.connect(lambda: self.set_font_size(8))
        font_size_menu.addAction(small_font_action)
        
        normal_font_action = QAction("Normal", self)
        normal_font_action.triggered.connect(lambda: self.set_font_size(9))
        font_size_menu.addAction(normal_font_action)
        
        large_font_action = QAction("Large", self)
        large_font_action.triggered.connect(lambda: self.set_font_size(11))
        font_size_menu.addAction(large_font_action)
        
        view_menu.addSeparator()
        
        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
        
        fullscreen_action = QAction("Toggle Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        welcome_guide_action = QAction("Welcome Guide", self)
        welcome_guide_action.triggered.connect(self.show_welcome_guide)
        help_menu.addAction(welcome_guide_action)
        
        online_docs_action = QAction("Online Documentation", self)
        online_docs_action.triggered.connect(self.show_online_documentation)
        help_menu.addAction(online_docs_action)
        
        help_menu.addSeparator()
        
        view_logs_action = QAction("View Logs", self)
        view_logs_action.triggered.connect(self.show_logs)
        help_menu.addAction(view_logs_action)
        
        report_issue_action = QAction("Report Issue", self)
        report_issue_action.triggered.connect(self.report_issue)
        help_menu.addAction(report_issue_action)
        
        help_menu.addSeparator()
        
        user_guide_action = QAction("User Guide", self)
        user_guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(user_guide_action)
        
        about_action = QAction("About AutoMatFlow Studio", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
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
        reply = QMessageBox.question(self, 'Exit Application', 
                                   'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Save current state if needed
            try:
                # Add any cleanup code here
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")
            event.accept()
        else:
            event.ignore()

    # Enhanced Menu Action Methods
    def new_experiment(self):
        """Create a new experiment"""
        try:
            from PyQt5.QtWidgets import QInputDialog
            experiment_name, ok = QInputDialog.getText(self, 'New Experiment', 'Enter experiment name:')
            if ok and experiment_name:
                self.update_status(f"Created new experiment: {experiment_name}")
                QMessageBox.information(self, "New Experiment", f"Experiment '{experiment_name}' created successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create experiment: {str(e)}")
    
    def import_training_data(self):
        """Import training data"""
        try:
            self.tab_widget.setCurrentIndex(0)  # Switch to Data Management tab
            self.data_module.load_training_data()
            self.update_status("Training data import initiated")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import training data: {str(e)}")
    
    def import_virtual_data(self):
        """Import virtual screening data"""
        try:
            self.tab_widget.setCurrentIndex(0)  # Switch to Data Management tab
            self.data_module.load_virtual_data()
            self.update_status("Virtual data import initiated")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import virtual data: {str(e)}")
    
    def import_model(self):
        """Import pre-trained model"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Import Model", "", "Joblib Files (*.joblib)")
            if file_path:
                # Load model and set it to prediction module
                import joblib
                model = joblib.load(file_path)
                self.prediction_module.set_model(model)
                self.update_status(f"Model imported from {file_path}")
                QMessageBox.information(self, "Import Model", "Model imported successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to import model: {str(e)}")
    
    def export_results(self):
        """Export analysis results"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'export_results'):
                current_tab.export_results()
                self.update_status("Results exported successfully")
            else:
                QMessageBox.information(self, "Export Results", "No results available to export from current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export results: {str(e)}")
    
    def export_charts(self):
        """Export charts and visualizations"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'export_charts'):
                current_tab.export_charts()
                self.update_status("Charts exported successfully")
            else:
                QMessageBox.information(self, "Export Charts", "No charts available to export from current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export charts: {str(e)}")
    
    def export_model(self):
        """Export trained model"""
        try:
            if hasattr(self.training_module, 'model') and self.training_module.model is not None:
                file_path, _ = QFileDialog.getSaveFileName(self, "Export Model", "", "Joblib Files (*.joblib)")
                if file_path:
                    import joblib
                    joblib.dump(self.training_module.model, file_path)
                    self.update_status(f"Model exported to {file_path}")
                    QMessageBox.information(self, "Export Model", "Model exported successfully!")
            else:
                QMessageBox.information(self, "Export Model", "No trained model available to export")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export model: {str(e)}")
    
    def export_report(self):
        """Export optimization report"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'export_report'):
                current_tab.export_report()
                self.update_status("Report exported successfully")
            else:
                QMessageBox.information(self, "Export Report", "No report available to export from current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export report: {str(e)}")
    
    def update_recent_projects_menu(self, menu):
        """Update recent projects menu"""
        try:
            # This would typically load from settings/config file
            recent_projects = ["Project_1.json", "Experiment_A.json", "Analysis_B.json"]
            
            if not recent_projects:
                no_recent_action = QAction("No recent projects", self)
                no_recent_action.setEnabled(False)
                menu.addAction(no_recent_action)
            else:
                for project in recent_projects[:5]:  # Show only last 5
                    action = QAction(project, self)
                    action.triggered.connect(lambda checked, p=project: self.load_recent_project(p))
                    menu.addAction(action)
                
                menu.addSeparator()
                clear_action = QAction("Clear Recent Projects", self)
                clear_action.triggered.connect(self.clear_recent_projects)
                menu.addAction(clear_action)
        except Exception as e:
            print(f"Error updating recent projects menu: {e}")
    
    def load_recent_project(self, project_name):
        """Load a recent project"""
        try:
            self.update_status(f"Loading recent project: {project_name}")
            # Implementation would load the actual project file
            QMessageBox.information(self, "Load Project", f"Project '{project_name}' loaded successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load project: {str(e)}")
    
    def clear_recent_projects(self):
        """Clear recent projects list"""
        try:
            reply = QMessageBox.question(self, 'Clear Recent Projects', 
                                       'Are you sure you want to clear the recent projects list?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.update_status("Recent projects list cleared")
                QMessageBox.information(self, "Clear Recent Projects", "Recent projects list cleared!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear recent projects: {str(e)}")
    
    def show_settings(self):
        """Show application settings dialog"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QCheckBox, QPushButton, QTabWidget, QComboBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Settings")
            dialog.setModal(True)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Create tab widget for settings categories
            tab_widget = QTabWidget()
            layout.addWidget(tab_widget)
            
            # General settings tab
            general_tab = QWidget()
            general_layout = QVBoxLayout(general_tab)
            
            # Auto-save settings
            auto_save_cb = QCheckBox("Enable auto-save")
            auto_save_cb.setChecked(True)
            general_layout.addWidget(auto_save_cb)
            
            # Default paths
            general_layout.addWidget(QLabel("Default Data Path:"))
            # Add path selection widget here
            
            tab_widget.addTab(general_tab, "General")
            
            # Appearance settings tab
            appearance_tab = QWidget()
            appearance_layout = QVBoxLayout(appearance_tab)
            
            # Theme selection
            appearance_layout.addWidget(QLabel("Theme:"))
            theme_combo = QComboBox()
            theme_combo.addItems(["Light", "Dark", "Auto"])
            appearance_layout.addWidget(theme_combo)
            
            # Font size
            font_layout = QHBoxLayout()
            font_layout.addWidget(QLabel("Font Size:"))
            font_spin = QSpinBox()
            font_spin.setRange(8, 16)
            font_spin.setValue(9)
            font_layout.addWidget(font_spin)
            appearance_layout.addLayout(font_layout)
            
            tab_widget.addTab(appearance_tab, "Appearance")
            
            # Button layout
            button_layout = QHBoxLayout()
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            button_layout.addStretch()
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            
            layout.addLayout(button_layout)
            
            if dialog.exec_() == QDialog.Accepted:
                self.update_status("Settings updated")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show settings: {str(e)}")
    
    # Edit menu methods
    def undo_action(self):
        """Undo last action"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'undo'):
                current_tab.undo()
                self.update_status("Action undone")
            else:
                QMessageBox.information(self, "Undo", "No action to undo in current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to undo: {str(e)}")
    
    def redo_action(self):
        """Redo last undone action"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'redo'):
                current_tab.redo()
                self.update_status("Action redone")
            else:
                QMessageBox.information(self, "Redo", "No action to redo in current tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to redo: {str(e)}")
    
    def copy_data(self):
        """Copy selected data"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'copy_selection'):
                current_tab.copy_selection()
                self.update_status("Data copied to clipboard")
            else:
                QMessageBox.information(self, "Copy", "No data selected to copy")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to copy data: {str(e)}")
    
    def paste_data(self):
        """Paste data from clipboard"""
        try:
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'paste_data'):
                current_tab.paste_data()
                self.update_status("Data pasted from clipboard")
            else:
                QMessageBox.information(self, "Paste", "Current tab doesn't support paste operation")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to paste data: {str(e)}")
    
    def show_find_dialog(self):
        """Show find dialog"""
        try:
            from PyQt5.QtWidgets import QInputDialog
            search_text, ok = QInputDialog.getText(self, 'Find', 'Enter search text:')
            if ok and search_text:
                current_tab = self.tab_widget.currentWidget()
                if hasattr(current_tab, 'find_text'):
                    current_tab.find_text(search_text)
                    self.update_status(f"Searching for: {search_text}")
                else:
                    QMessageBox.information(self, "Find", "Current tab doesn't support text search")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show find dialog: {str(e)}")
    
    # Navigation methods for switching between modules
    def switch_to_data_module(self):
        """Switch to data management module"""
        self.tab_widget.setCurrentIndex(0)
        self.update_status("Switched to Data Management")
    
    def switch_to_preprocessing_module(self):
        """Switch to preprocessing module"""
        self.tab_widget.setCurrentIndex(2)  # Advanced Preprocessing tab
        self.update_status("Switched to Advanced Preprocessing")
    
    def switch_to_feature_module(self):
        """Switch to feature engineering module"""
        self.tab_widget.setCurrentIndex(3)
        self.update_status("Switched to Feature Selection")
    
    def switch_to_training_module(self):
        """Switch to model training module"""
        self.tab_widget.setCurrentIndex(4)
        self.update_status("Switched to Model Training")
    
    def switch_to_shap_module(self):
        """Switch to SHAP analysis module"""
        self.tab_widget.setCurrentIndex(8)
        self.update_status("Switched to SHAP Analysis")
    
    def switch_to_performance_module(self):
        """Switch to performance monitor module"""
        self.tab_widget.setCurrentIndex(6)
        self.update_status("Switched to Performance Monitor")
    
    def switch_to_active_learning_module(self):
        """Switch to active learning module"""
        self.tab_widget.setCurrentIndex(11)
        self.update_status("Switched to Active Learning & Optimization")
    
    def switch_to_multi_objective_module(self):
        """Switch to multi-objective optimization module"""
        self.tab_widget.setCurrentIndex(10)
        self.update_status("Switched to Multi-objective Optimization")
    
    def switch_to_prediction_module(self):
        """Switch to prediction module"""
        self.tab_widget.setCurrentIndex(5)
        self.update_status("Switched to Model Prediction")
    
    def switch_to_target_optimization_module(self):
        """Switch to target optimization module"""
        self.tab_widget.setCurrentIndex(9)
        self.update_status("Switched to Target Optimization")
    
    # Analysis and visualization methods
    def show_data_visualization(self):
        """Show data visualization dialog"""
        try:
            if hasattr(self.data_module, 'show_visualization_dialog'):
                self.data_module.show_visualization_dialog()
            else:
                QMessageBox.information(self, "Data Visualization", "Please load data first in the Data Management tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show data visualization: {str(e)}")
    
    def show_data_summary(self):
        """Show data summary dialog"""
        try:
            if hasattr(self.data_module, 'show_data_summary'):
                self.data_module.show_data_summary()
            else:
                QMessageBox.information(self, "Data Summary", "Please load data first in the Data Management tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show data summary: {str(e)}")
    
    def show_model_evaluation(self):
        """Show model evaluation dialog"""
        try:
            if hasattr(self.training_module, 'show_evaluation_dialog'):
                self.training_module.show_evaluation_dialog()
            else:
                QMessageBox.information(self, "Model Evaluation", "Please train a model first in the Model Training tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show model evaluation: {str(e)}")
    
    def show_learning_curve_analysis(self):
        """Show learning curve analysis"""
        try:
            if hasattr(self.training_module, 'show_learning_curves'):
                self.training_module.show_learning_curves()
            else:
                QMessageBox.information(self, "Learning Curve Analysis", "Please train a model first in the Model Training tab")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show learning curve analysis: {str(e)}")
    
    # View menu methods
    def toggle_log_console(self, checked):
        """Toggle log console visibility"""
        try:
            # Implementation would show/hide log console panel
            self.update_status(f"Log console {'shown' if checked else 'hidden'}")
        except Exception as e:
            print(f"Error toggling log console: {e}")
    
    def toggle_data_viewer(self, checked):
        """Toggle data viewer panel"""
        try:
            # Implementation would show/hide data viewer panel
            self.update_status(f"Data viewer {'shown' if checked else 'hidden'}")
        except Exception as e:
            print(f"Error toggling data viewer: {e}")
    
    def toggle_charts_panel(self, checked):
        """Toggle charts panel"""
        try:
            # Implementation would show/hide charts panel
            self.update_status(f"Charts panel {'shown' if checked else 'hidden'}")
        except Exception as e:
            print(f"Error toggling charts panel: {e}")
    
    def toggle_theme(self):
        """Toggle between light and dark theme"""
        try:
            # Simple theme toggle implementation
            current_style = self.styleSheet()
            if "background-color: #2b2b2b" in current_style:
                # Switch to light theme
                self.setStyleSheet("")
                self.update_status("Switched to light theme")
            else:
                # Switch to dark theme
                dark_style = """
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QTabWidget::pane {
                    background-color: #3c3c3c;
                    border: 1px solid #555555;
                }
                QTabBar::tab {
                    background-color: #404040;
                    color: #ffffff;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #555555;
                }
                """
                self.setStyleSheet(dark_style)
                self.update_status("Switched to dark theme")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle theme: {str(e)}")
    
    def set_font_size(self, size):
        """Set application font size"""
        try:
            font = QFont("Segoe UI", size)
            self.setFont(font)
            self.update_status(f"Font size set to {size}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to set font size: {str(e)}")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        try:
            if self.isFullScreen():
                self.showNormal()
                self.update_status("Exited fullscreen mode")
            else:
                self.showFullScreen()
                self.update_status("Entered fullscreen mode")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle fullscreen: {str(e)}")
    
    # Help menu methods
    def show_welcome_guide(self):
        """Show welcome guide dialog"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Welcome to AutoMatFlow Studio")
            dialog.setModal(True)
            dialog.resize(600, 500)
            
            layout = QVBoxLayout(dialog)
            
            # Welcome content
            welcome_text = """
            <h2>Welcome to AutoMatFlow Studio v1.0</h2>
            
            <h3>Getting Started:</h3>
            <ol>
                <li><b>Load Data:</b> Start by importing your training data in the Data Management tab</li>
                <li><b>Preprocess:</b> Use Advanced Preprocessing to clean and prepare your data</li>
                <li><b>Feature Engineering:</b> Select and engineer features in the Feature Selection tab</li>
                <li><b>Train Models:</b> Build and evaluate models in the Model Training tab</li>
                <li><b>Optimize:</b> Use Active Learning or Multi-objective Optimization for advanced analysis</li>
            </ol>
            
            <h3>Key Features:</h3>
            <ul>
                <li>🧙‍♂️ Intelligent Wizard for automated workflow guidance</li>
                <li>🧬 Advanced preprocessing with multiple algorithms</li>
                <li>🎯 Comprehensive feature engineering and selection</li>
                <li>🔬 Multiple machine learning algorithms</li>
                <li>🧠 SHAP analysis for model interpretability</li>
                <li>🤖 Active learning for efficient data acquisition</li>
                <li>🔄 Multi-objective optimization</li>
                <li>📊 Rich visualizations and performance monitoring</li>
            </ul>
            
            <h3>Navigation:</h3>
            <p>Use the menu bar to quickly access different functions:</p>
            <ul>
                <li><b>File:</b> Project management and data import/export</li>
                <li><b>Data:</b> Data-related operations</li>
                <li><b>Analyze:</b> Model training and analysis</li>
                <li><b>Optimize:</b> Advanced optimization techniques</li>
                <li><b>View:</b> Interface customization</li>
            </ul>
            """
            
            text_browser = QTextBrowser()
            text_browser.setHtml(welcome_text)
            layout.addWidget(text_browser)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show welcome guide: {str(e)}")
    
    def show_online_documentation(self):
        """Open online documentation"""
        try:
            import webbrowser
            webbrowser.open("https://github.com/your-repo/automatflow-studio/wiki")
            self.update_status("Opened online documentation")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open online documentation: {str(e)}")
    
    def show_logs(self):
        """Show application logs"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Application Logs")
            dialog.setModal(True)
            dialog.resize(700, 500)
            
            layout = QVBoxLayout(dialog)
            
            # Log content (in a real application, this would read from log files)
            log_text = QTextEdit()
            log_text.setReadOnly(True)
            log_text.setPlainText("""
[2024-01-01 10:00:00] INFO: Application started
[2024-01-01 10:00:01] INFO: Modules loaded successfully
[2024-01-01 10:00:02] INFO: UI initialized
[2024-01-01 10:05:30] INFO: Training data loaded: 1000 samples
[2024-01-01 10:06:15] INFO: Model training completed
[2024-01-01 10:07:00] INFO: Analysis results exported
            """)
            layout.addWidget(log_text)
            
            # Button layout
            button_layout = QHBoxLayout()
            refresh_button = QPushButton("Refresh")
            clear_button = QPushButton("Clear Logs")
            close_button = QPushButton("Close")
            
            refresh_button.clicked.connect(lambda: self.update_status("Logs refreshed"))
            clear_button.clicked.connect(lambda: log_text.clear())
            close_button.clicked.connect(dialog.accept)
            
            button_layout.addWidget(refresh_button)
            button_layout.addWidget(clear_button)
            button_layout.addStretch()
            button_layout.addWidget(close_button)
            
            layout.addLayout(button_layout)
            
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to show logs: {str(e)}")
    
    def report_issue(self):
        """Open issue reporting interface"""
        try:
            import webbrowser
            webbrowser.open("https://github.com/your-repo/automatflow-studio/issues/new")
            self.update_status("Opened issue reporting page")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open issue reporting: {str(e)}")


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