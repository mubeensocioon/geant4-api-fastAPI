from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import subprocess
import json
import os
import tempfile
import uuid
from datetime import datetime
import asyncio
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Geant4 Nuclear Simulation API",
    description="API wrapper for Geant4 nuclear fusion and decay simulations",
    version="1.0.0"
)

# Pydantic models for request/response
class SimulationRequest(BaseModel):
    Z: int = Field(..., description="Atomic number of target nucleus", ge=1, le=118)
    N: int = Field(..., description="Number of neutrons in target nucleus", ge=0)
    fusion_reaction: str = Field(..., description="Fusion reaction path (e.g., 'Ca-48 + Bk-249')")
    beam_energy_mev: float = Field(..., description="Beam energy in MeV", gt=0)
    simulate_decay_chain: bool = Field(default=True, description="Flag to simulate decay chain")
    max_events: int = Field(default=10000, description="Maximum number of events to simulate", ge=100, le=100000)
    
class DecayStep(BaseModel):
    parent_nucleus: str
    daughter_nucleus: str
    decay_mode: str
    half_life: float
    half_life_unit: str
    branching_ratio: float

class SimulationResult(BaseModel):
    simulation_id: str
    status: str
    target_nucleus: str
    projectile: str
    beam_energy_mev: float
    cross_section: Optional[float] = None
    cross_section_error: Optional[float] = None
    primary_products: List[str] = []
    decay_chain: List[DecayStep] = []
    half_life: Optional[float] = None
    half_life_unit: Optional[str] = None
    confidence: Optional[float] = None
    total_events: int = 0
    computation_time: Optional[float] = None
    created_at: datetime
    error_message: Optional[str] = None

# In-memory storage for simulation results
simulation_results: Dict[str, SimulationResult] = {}

def parse_fusion_reaction(reaction_str: str) -> tuple:
    """Parse fusion reaction string like 'Ca-48 + Bk-249' into components"""
    try:
        parts = reaction_str.replace(" ", "").split("+")
        if len(parts) != 2:
            raise ValueError("Invalid reaction format")
        
        projectile = parts[0].strip()
        target = parts[1].strip()
        
        return projectile, target
    except Exception as e:
        raise ValueError(f"Could not parse reaction: {reaction_str}. Expected format: 'Element-Mass + Element-Mass'")

def simulate_nuclear_physics(sim_request: SimulationRequest) -> Dict[str, Any]:
    """Simulate nuclear physics calculations (simplified for now)"""
    
    # Parse the reaction
    projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
    
    # Simulate some processing time
    processing_time = min(sim_request.max_events / 1000.0, 10.0)  # Max 10 seconds
    time.sleep(processing_time)
    
    # Calculate some mock physics results based on input parameters
    # This is where you would integrate actual Geant4 calculations
    
    # Mock cross section calculation (simplified Coulomb barrier)
    coulomb_barrier = 1.44 * sim_request.Z * (sim_request.Z + sim_request.N) / (1.2 * ((sim_request.Z + sim_request.N)**(1/3) + 48**(1/3)))
    
    if sim_request.beam_energy_mev > coulomb_barrier:
        # Above barrier - calculate cross section
        excess_energy = sim_request.beam_energy_mev - coulomb_barrier
        cross_section = max(0.1, 10.0 * excess_energy / (excess_energy + 50.0))  # mb
        confidence = min(0.9, 0.3 + excess_energy / 100.0)
    else:
        # Below barrier - tunneling probability
        cross_section = 0.01 * max(0.001, sim_request.beam_energy_mev / coulomb_barrier)
        confidence = 0.2
    
    # Generate some mock decay products
    mass_number = sim_request.Z + sim_request.N + 48  # projectile mass
    if mass_number > 250:
        products = [f"Element-{sim_request.Z + 20}", f"Element-{sim_request.Z + 19}"]
    else:
        products = [f"Element-{sim_request.Z + 20}"]
    
    # Mock decay chain
    decay_chain = []
    if sim_request.simulate_decay_chain and products:
        decay_chain = [
            {
                "parent_nucleus": products[0],
                "daughter_nucleus": f"Element-{sim_request.Z + 19}",
                "decay_mode": "alpha",
                "half_life": max(0.001, cross_section * 100),
                "half_life_unit": "seconds",
                "branching_ratio": 0.85
            }
        ]
    
    return {
        "cross_section": round(cross_section, 3),
        "cross_section_error": round(cross_section * 0.1, 3),
        "primary_products": products,
        "decay_chain": decay_chain,
        "total_events": sim_request.max_events,
        "confidence": round(confidence, 2),
        "half_life": decay_chain[0]["half_life"] if decay_chain else None,
        "half_life_unit": "seconds" if decay_chain else None
    }

async def run_simulation_async(sim_request: SimulationRequest, sim_id: str):
    """Run simulation asynchronously"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting simulation {sim_id}")
        
        # Update status to running
        projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
        simulation_results[sim_id].status = "running"
        simulation_results[sim_id].target_nucleus = target
        simulation_results[sim_id].projectile = projectile
        
        # Run the simulation
        results = simulate_nuclear_physics(sim_request)
        
        # Calculate computation time
        computation_time = (datetime.now() - start_time).total_seconds()
        
        # Update the simulation result
        simulation_result = simulation_results[sim_id]
        simulation_result.status = "completed"
        simulation_result.cross_section = results.get("cross_section")
        simulation_result.cross_section_error = results.get("cross_section_error")
        simulation_result.primary_products = results.get("primary_products", [])
        simulation_result.total_events = results.get("total_events", 0)
        simulation_result.confidence = results.get("confidence")
        simulation_result.computation_time = computation_time
        simulation_result.half_life = results.get("half_life")
        simulation_result.half_life_unit = results.get("half_life_unit")
        
        # Add decay chain
        decay_data = results.get("decay_chain", [])
        simulation_result.decay_chain = [
            DecayStep(**decay) for decay in decay_data
        ]
        
        logger.info(f"Simulation {sim_id} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Simulation {sim_id} failed: {error_msg}")
        
        # Update with error
        simulation_results[sim_id].status = "failed"
        simulation_results[sim_id].error_message = error_msg

@app.post("/simulate", response_model=Dict[str, str])
async def start_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """Start a nuclear fusion simulation"""
    
    # Validate fusion reaction format
    try:
        parse_fusion_reaction(request.fusion_reaction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Generate unique simulation ID
    sim_id = str(uuid.uuid4())
    
    # Initialize simulation status
    simulation_results[sim_id] = SimulationResult(
        simulation_id=sim_id,
        status="queued",
        target_nucleus="",
        projectile="",
        beam_energy_mev=request.beam_energy_mev,
        created_at=datetime.now()
    )
    
    # Start simulation in background
    background_tasks.add_task(run_simulation_async, request, sim_id)
    
    return {
        "simulation_id": sim_id,
        "status": "started",
        "message": "Simulation started. Use the simulation_id to check status."
    }

@app.get("/simulation/{simulation_id}", response_model=SimulationResult)
async def get_simulation_result(simulation_id: str):
    """Get simulation result by ID"""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return simulation_results[simulation_id]

@app.get("/simulations", response_model=List[Dict[str, Any]])
async def list_simulations():
    """List all simulations"""
    return [
        {
            "simulation_id": sim_id,
            "status": result.status,
            "created_at": result.created_at,
            "fusion_reaction": f"{result.projectile} + {result.target_nucleus}" if result.projectile and result.target_nucleus else "N/A"
        }
        for sim_id, result in simulation_results.items()
    ]

@app.delete("/simulation/{simulation_id}")
async def delete_simulation(simulation_id: str):
    """Delete a simulation result"""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del simulation_results[simulation_id]
    return {"message": f"Simulation {simulation_id} deleted"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if Geant4 is available
        geant4_config = "/root/geant4-v11.3.2-install/bin/geant4-config"
        if os.path.exists(geant4_config):
            result = subprocess.run([geant4_config, "--version"], capture_output=True, text=True, timeout=10)
            geant4_version = result.stdout.strip() if result.returncode == 0 else "Available but version check failed"
        else:
            geant4_version = "Not found"
        
        return {
            "status": "healthy",
            "geant4_version": geant4_version,
            "active_simulations": len([r for r in simulation_results.values() if r.status == "running"]),
            "total_simulations": len(simulation_results)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
