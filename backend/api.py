from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
from typing import List, Dict
import uvicorn
from ev_predictive_maintenance import EVSensor, MaintenanceSystem

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active websocket connections
active_connections: List[WebSocket] = []
# Store maintenance systems for each vehicle
maintenance_systems: Dict[str, MaintenanceSystem] = {}

@app.on_event("startup")
async def startup_event():
    # Initialize maintenance systems for demo vehicles
    vehicle_ids = ["EV001", "EV002", "EV003"]
    for vehicle_id in vehicle_ids:
        maintenance_systems[vehicle_id] = MaintenanceSystem(vehicle_id)
        # Train initial model
        maintenance_systems[vehicle_id].train_initial_model(duration_minutes=1)

@app.websocket("/ws/{vehicle_id}")
async def websocket_endpoint(websocket: WebSocket, vehicle_id: str):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Get maintenance system for this vehicle
            maintenance_system = maintenance_systems.get(vehicle_id)
            if not maintenance_system:
                await websocket.send_text("Vehicle not found")
                break
                
            # Get sensor readings
            sensor_data = maintenance_system.ev_sensor.get_sensor_reading()
            predictions = maintenance_system.model.predict(sensor_data)
            
            # Prepare data packet
            data_packet = {
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": vehicle_id,
                "sensor_data": sensor_data,
                "predictions": predictions
            }
            
            # Send data to client
            await websocket.send_json(data_packet)
            await asyncio.sleep(2)  # Send data every 2 seconds
            
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        active_connections.remove(websocket)

@app.get("/vehicles")
async def get_vehicles():
    return list(maintenance_systems.keys())

@app.get("/vehicle/{vehicle_id}/status")
async def get_vehicle_status(vehicle_id: str):
    maintenance_system = maintenance_systems.get(vehicle_id)
    if not maintenance_system:
        return {"error": "Vehicle not found"}
        
    sensor_data = maintenance_system.ev_sensor.get_sensor_reading()
    predictions = maintenance_system.model.predict(sensor_data)
    
    return {
        "vehicle_id": vehicle_id,
        "sensor_data": sensor_data,
        "predictions": predictions,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)