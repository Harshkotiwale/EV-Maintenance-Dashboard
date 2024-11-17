import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error
import joblib
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Tuple, Union
import json
import time  # Added missing import
import os    # Added for file operations

class EVSensor:
    """Simulates various EV sensors and components"""
    
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.sensors = {
            'battery_temp': {'min': 20, 'max': 45, 'unit': '°C'},
            'battery_voltage': {'min': 350, 'max': 400, 'unit': 'V'},
            'motor_temp': {'min': 30, 'max': 85, 'unit': '°C'},
            'charging_cycles': {'min': 0, 'max': 2000, 'unit': 'cycles'},
            'battery_health': {'min': 70, 'max': 100, 'unit': '%'},
            'motor_vibration': {'min': 0.1, 'max': 5.0, 'unit': 'mm/s'},
            'power_output': {'min': 0, 'max': 150, 'unit': 'kW'}
        }
        
    def get_sensor_reading(self) -> Dict[str, float]:
        """Generate realistic sensor readings with some random variation"""
        readings = {}
        for sensor, limits in self.sensors.items():
            base_value = (limits['min'] + limits['max']) / 2
            variation = (limits['max'] - limits['min']) * 0.1
            reading = base_value + np.random.normal(0, variation)
            readings[sensor] = max(limits['min'], min(limits['max'], reading))
        return readings

class DataCollector:
    """Handles data collection and preprocessing from EV sensors"""
    
    def __init__(self, db_path: str = 'ev_maintenance.db'):
        self.db_path = db_path
        self.setup_logging()
        
    def setup_logging(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='logs/ev_maintenance.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def collect_data(self, ev_sensor: EVSensor, duration_minutes: int = 60) -> pd.DataFrame:
        """Collect sensor data for specified duration"""
        data_points = []
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            reading = ev_sensor.get_sensor_reading()
            reading['timestamp'] = datetime.now()
            reading['vehicle_id'] = ev_sensor.vehicle_id
            data_points.append(reading)
            # Reduced sleep time for demonstration purposes
            time.sleep(1)  # Changed from 60 to 1 second for faster execution
            
        return pd.DataFrame(data_points)

class PredictiveModel:
    """Handles training and prediction for EV maintenance"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.failure_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.maintenance_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.logger = logging.getLogger(__name__)
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess sensor data for model training"""
        feature_columns = [
            'battery_temp', 'battery_voltage', 'motor_temp',
            'charging_cycles', 'battery_health', 'motor_vibration',
            'power_output'
        ]
        
        X = data[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate synthetic labels for demonstration
        y_failure = (data['battery_health'] < 80) | (data['motor_temp'] > 80)
        y_maintenance = data['charging_cycles'] / 2000 * 100
        
        return X_scaled, y_failure, y_maintenance
        
    def train(self, data: pd.DataFrame):
        """Train the predictive models"""
        X_scaled, y_failure, y_maintenance = self.preprocess_data(data)
        
        # Split data
        X_train, X_test, y_fail_train, y_fail_test = train_test_split(
            X_scaled, y_failure, test_size=0.2, random_state=42
        )
        _, _, y_maint_train, y_maint_test = train_test_split(
            X_scaled, y_maintenance, test_size=0.2, random_state=42
        )
        
        # Train models
        self.failure_classifier.fit(X_train, y_fail_train)
        self.maintenance_predictor.fit(X_train, y_maint_train)
        
        # Evaluate models
        failure_pred = self.failure_classifier.predict(X_test)
        maintenance_pred = self.maintenance_predictor.predict(X_test)
        
        self.logger.info("Failure Classification Report:")
        self.logger.info(classification_report(y_fail_test, failure_pred))
        
        self.logger.info("Maintenance Prediction RMSE:")
        self.logger.info(np.sqrt(mean_squared_error(y_maint_test, maintenance_pred)))
        
    def predict(self, sensor_data: Dict[str, float]) -> Dict[str, Union[bool, float]]:
        """Make predictions for new sensor readings"""
        feature_columns = [
            'battery_temp', 'battery_voltage', 'motor_temp',
            'charging_cycles', 'battery_health', 'motor_vibration',
            'power_output'
        ]
        
        X = np.array([[sensor_data[col] for col in feature_columns]])
        X_scaled = self.scaler.transform(X)
        
        failure_prob = self.failure_classifier.predict_proba(X_scaled)[0][1]
        maintenance_score = self.maintenance_predictor.predict(X_scaled)[0]
        
        return {
            'failure_probability': failure_prob,
            'maintenance_score': maintenance_score,
            'maintenance_recommended': maintenance_score > 70,
            'immediate_attention_required': failure_prob > 0.7
        }

class MaintenanceSystem:
    """Main system that coordinates data collection and predictions"""
    
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.ev_sensor = EVSensor(vehicle_id)
        self.data_collector = DataCollector()
        self.model = PredictiveModel()
        self.setup_logging()
        
    def setup_logging(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=f'logs/maintenance_{self.vehicle_id}.log'
        )
        self.logger = logging.getLogger(__name__)
        
    def train_initial_model(self, duration_minutes: int = 60):
        """Collect initial data and train the model"""
        self.logger.info(f"Starting initial data collection for vehicle {self.vehicle_id}")
        training_data = self.data_collector.collect_data(self.ev_sensor, duration_minutes)
        self.model.train(training_data)
        self.logger.info("Initial model training completed")
        
    def monitor_vehicle(self, interval_minutes: int = 60):
        """Continuous monitoring of vehicle health"""
        while True:
            try:
                sensor_data = self.ev_sensor.get_sensor_reading()
                predictions = self.model.predict(sensor_data)
                
                self.log_predictions(sensor_data, predictions)
                self.handle_alerts(predictions)
                
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error during monitoring: {str(e)}")
                continue
    
    def log_predictions(self, sensor_data: Dict[str, float], predictions: Dict[str, Union[bool, float]]):
        """Log sensor readings and predictions"""
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'vehicle_id': self.vehicle_id,
            'sensor_data': sensor_data,
            'predictions': predictions
        }
        
        with open(f'data/predictions_{self.vehicle_id}.json', 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
            
    def handle_alerts(self, predictions: Dict[str, Union[bool, float]]):
        """Handle maintenance alerts based on predictions"""
        if predictions['immediate_attention_required']:
            self.send_urgent_alert()
        elif predictions['maintenance_recommended']:
            self.schedule_maintenance()
            
    def send_urgent_alert(self):
        """Send urgent maintenance alert"""
        alert_message = f"""
        URGENT MAINTENANCE REQUIRED
        Vehicle ID: {self.vehicle_id}
        Time: {datetime.now().isoformat()}
        Please schedule immediate maintenance check.
        """
        self.logger.warning(alert_message)
        # In production, implement actual alert mechanism (email, SMS, etc.)
        
    def schedule_maintenance(self):
        """Schedule routine maintenance"""
        maintenance_message = f"""
        MAINTENANCE RECOMMENDED
        Vehicle ID: {self.vehicle_id}
        Time: {datetime.now().isoformat()}
        Please schedule maintenance within the next 7 days.
        """
        self.logger.info(maintenance_message)
        # In production, implement actual scheduling mechanism

def main():
    """Main function to demonstrate system usage"""
    print("Starting EV Predictive Maintenance System...")
    
    # Initialize system for a specific vehicle
    vehicle_id = "EV001"
    print(f"Initializing maintenance system for vehicle {vehicle_id}")
    maintenance_system = MaintenanceSystem(vehicle_id)
    
    print("\nStarting initial data collection and model training (5 minutes)...")
    print("Please wait while collecting training data...")
    
    # Train initial model (reduced duration for demonstration)
    maintenance_system.train_initial_model(duration_minutes=5)  # Reduced to 5 minutes for testing
    
    print("\nInitial training completed!")
    print("Starting continuous monitoring...")
    print("Press Ctrl+C to stop the program")
    
    try:
        # Start continuous monitoring
        maintenance_system.monitor_vehicle(interval_minutes=1)  # Check every minute
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
        print("Check 'logs' directory for detailed logs")
        print("Check 'data' directory for predictions")

if __name__ == "__main__":
    main()