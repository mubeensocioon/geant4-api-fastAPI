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

# In-memory storage for simulation results (use Redis in production)
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

def create_geant4_macro(sim_request: SimulationRequest, output_dir: str) -> str:
    """Create Geant4 macro file for the simulation"""
    projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
    
    macro_content = f"""
# Geant4 Nuclear Fusion Simulation Macro
# Generated for: {sim_request.fusion_reaction}
# Beam Energy: {sim_request.beam_energy_mev} MeV

/run/verbose 1
/event/verbose 0
/tracking/verbose 0

# Physics list - use QGSP_BERT_HP for nuclear reactions
/physics/addPhysics QGSP_BERT_HP

# Geometry setup
/detector/setTargetMaterial G4_{target.split('-')[0]}
/detector/setTargetThickness 1 mm

# Primary particle gun setup  
/gun/particle ion
/gun/ion {sim_request.Z} {sim_request.Z + sim_request.N} 0 0  # Z A Q E
/gun/energy {sim_request.beam_energy_mev} MeV
/gun/direction 0 0 1

# Analysis setup
/analysis/setFileName {output_dir}/simulation_output
/analysis/h1/create 1 "Energy Spectrum" 100 0 {sim_request.beam_energy_mev * 2} MeV
/analysis/h1/create 2 "Angular Distribution" 180 0 180 deg

# Run simulation
/run/beamOn {sim_request.max_events}

# Enable decay chain tracking if requested
"""
    
    if sim_request.simulate_decay_chain:
        macro_content += """
/process/had/rdm/nucleusLimits 1 250 1 250
/process/had/rdm/analogueMC true
/process/had/rdm/decayHalfLife 1e-6 s 1e12 s
"""

    return macro_content

def run_geant4_simulation(macro_content: str, output_dir: str) -> Dict[str, Any]:
    """Execute Geant4 simulation with the given macro"""
    macro_file = os.path.join(output_dir, "simulation.mac")
    
    try:
        # Write macro file
        with open(macro_file, 'w') as f:
            f.write(macro_content)
        
        # Run Geant4 simulation
        # Adjust the command based on your Geant4 installation
        cmd = [
            "geant4-config", "--prefix",  # Get Geant4 installation path
        ]
        
        # Get Geant4 path
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError("Geant4 not found or not properly configured")
        
        geant4_path = result.stdout.strip()
        
        # Run the simulation (adjust executable name as needed)
        simulation_cmd = [
            f"{geant4_path}/bin/exampleB1",  # Replace with your Geant4 executable
            macro_file
        ]
        
        logger.info(f"Running Geant4 simulation: {' '.join(simulation_cmd)}")
        
        # Execute simulation with timeout
        result = subprocess.run(
            simulation_cmd,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Geant4 simulation failed: {result.stderr}")
        
        # Parse output
        return parse_geant4_output(output_dir, result.stdout)
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Simulation timed out")
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise RuntimeError(f"Simulation failed: {str(e)}")

def parse_geant4_output(output_dir: str, stdout: str) -> Dict[str, Any]:
    """Parse Geant4 simulation output and extract results"""
    results = {
        "cross_section": None,
        "cross_section_error": None,
        "primary_products": [],
        "decay_chain": [],
        "total_events": 0,
        "raw_output": stdout
    }
    
    try:
        # Parse stdout for key information
        lines = stdout.split('\n')
        for line in lines:
            if "Cross section" in line:
                # Extract cross section value
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "mb" or part == "barn":
                        try:
                            results["cross_section"] = float(parts[i-1])
                        except (ValueError, IndexError):
                            pass
            
            elif "Events processed" in line:
                # Extract number of events
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        results["total_events"] = int(part)
                        break
        
        # Look for output files
        output_files = [f for f in os.listdir(output_dir) if f.endswith('.root') or f.endswith('.csv')]
        
        # Parse nuclear data if available
        for filename in output_files:
            if "nuclei" in filename.lower():
                # Parse nuclear data file for decay information
                results["primary_products"] = parse_nuclear_products(os.path.join(output_dir, filename))
        
        return results
        
    except Exception as e:
        logger.error(f"Error parsing output: {str(e)}")
        return results

def parse_nuclear_products(filepath: str) -> List[str]:
    """Parse nuclear products from output file"""
    products = []
    try:
        if filepath.endswith('.csv'):
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        products.append(parts[0].strip())
        # Add more parsers for different file formats as needed
    except Exception as e:
        logger.error(f"Error parsing nuclear products: {str(e)}")
    
    return products

def calculate_confidence(results: Dict[str, Any]) -> float:
    """Calculate confidence based on simulation statistics"""
    total_events = results.get("total_events", 0)
    if total_events < 1000:
        return 0.3
    elif total_events < 5000:
        return 0.6
    elif total_events < 10000:
        return 0.8
    else:
        return 0.9

async def run_simulation_async(sim_request: SimulationRequest, sim_id: str):
    """Run simulation asynchronously"""
    start_time = datetime.now()
    
    try:
        # Create temporary directory for simulation
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Starting simulation {sim_id} in {temp_dir}")
            
            # Create Geant4 macro
            macro_content = create_geant4_macro(sim_request, temp_dir)
            
            # Run simulation
            results = run_geant4_simulation(macro_content, temp_dir)
            
            # Calculate computation time
            computation_time = (datetime.now() - start_time).total_seconds()
            
            # Parse fusion reaction
            projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
            
            # Create simulation result
            simulation_result = SimulationResult(
                simulation_id=sim_id,
                status="completed",
                target_nucleus=target,
                projectile=projectile,
                beam_energy_mev=sim_request.beam_energy_mev,
                cross_section=results.get("cross_section"),
                cross_section_error=results.get("cross_section_error"),
                primary_products=results.get("primary_products", []),
                decay_chain=[],  # Would need more sophisticated parsing
                total_events=results.get("total_events", 0),
                confidence=calculate_confidence(results),
                computation_time=computation_time,
                created_at=start_time
            )
            
            # Store result
            simulation_results[sim_id] = simulation_result
            logger.info(f"Simulation {sim_id} completed successfully")
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Simulation {sim_id} failed: {error_msg}")
        
        # Store error result
        simulation_results[sim_id] = SimulationResult(
            simulation_id=sim_id,
            status="failed",
            target_nucleus="",
            projectile="",
            beam_energy_mev=sim_request.beam_energy_mev,
            created_at=start_time,
            error_message=error_msg
        )

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
        status="running",
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
        result = subprocess.run(["geant4-config", "--version"], capture_output=True, text=True, timeout=10)
        geant4_version = result.stdout.strip() if result.returncode == 0 else "Not available"
        
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
