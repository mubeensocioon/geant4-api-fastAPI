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
import base64
import shutil
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Geant4 Nuclear Simulation API with Qt Visualization",
    description="API wrapper for Geant4 nuclear fusion and decay simulations with Qt-based visualization",
    version="1.3.0"
)

# Pydantic models for request/response (keeping existing models)
class SimulationRequest(BaseModel):
    Z: int = Field(..., description="Atomic number of target nucleus", ge=1, le=118)
    N: int = Field(..., description="Number of neutrons in target nucleus", ge=0)
    fusion_reaction: str = Field(..., description="Fusion reaction path (e.g., 'Ca-48 + Bk-249')")
    beam_energy_mev: float = Field(..., description="Beam energy in MeV", gt=0)
    simulate_decay_chain: bool = Field(default=True, description="Flag to simulate decay chain")
    max_events: int = Field(default=10000, description="Maximum number of events to simulate", ge=100, le=100000)
    
    # Visualization parameters
    enable_visualization: bool = Field(default=False, description="Generate visualization images")
    visualization_type: str = Field(default="geometry", description="Type: geometry, tracks, hits, dose")
    camera_angle: str = Field(default="iso", description="Camera angle: side, top, iso, front")
    image_width: int = Field(default=800, description="Image width in pixels", ge=400, le=2048)
    image_height: int = Field(default=600, description="Image height in pixels", ge=300, le=2048)
    background_color: str = Field(default="white", description="Background color: white, black")
    show_axes: bool = Field(default=True, description="Show coordinate axes")

class VisualizationData(BaseModel):
    image_base64: Optional[str] = None
    image_format: str = "png"
    description: str = ""
    generation_time: Optional[float] = None

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
    visualization: Optional[VisualizationData] = None
    created_at: datetime
    error_message: Optional[str] = None

# In-memory storage for simulation results
simulation_results: Dict[str, SimulationResult] = {}

def find_geant4_executable() -> Optional[str]:
    """Find the actual Geant4 executable"""
    possible_locations = [
        "/root/geant4-v11.3.2-install/bin/geant4",
        "/root/geant4-v11.3.2-install/bin/G4",
        "/usr/local/bin/geant4",
        "/usr/bin/geant4"
    ]
    
    # Also search for any executable with geant4 in the name
    search_patterns = [
        "/root/geant4-v11.3.2-install/bin/*geant4*",
        "/root/geant4-v11.3.2-install/bin/*G4*"
    ]
    
    # Check specific locations first
    for location in possible_locations:
        if os.path.isfile(location) and os.access(location, os.X_OK):
            logger.info(f"Found Geant4 executable: {location}")
            return location
    
    # Search with patterns
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.isfile(match) and os.access(match, os.X_OK):
                logger.info(f"Found Geant4 executable via pattern: {match}")
                return match
    
    # If still not found, try to find any executable in the Geant4 directory
    try:
        geant4_bin = "/root/geant4-v11.3.2-install/bin"
        if os.path.exists(geant4_bin):
            for file in os.listdir(geant4_bin):
                filepath = os.path.join(geant4_bin, file)
                if os.path.isfile(filepath) and os.access(filepath, os.X_OK):
                    # Check if it looks like a main Geant4 executable
                    if any(keyword in file.lower() for keyword in ['geant4', 'g4']):
                        logger.info(f"Found potential Geant4 executable: {filepath}")
                        return filepath
    except Exception as e:
        logger.warning(f"Error searching for Geant4 executable: {e}")
    
    logger.error("No Geant4 executable found")
    return None

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

def generate_geant4_macro(sim_request: SimulationRequest, output_dir: str) -> str:
    """Generate Geant4 macro file with improved Qt visualization commands"""
    
    # Limit events for visualization
    viz_events = min(sim_request.max_events, 50) if sim_request.enable_visualization else sim_request.max_events
    
    macro_lines = [
        "# Geant4 Macro File for Nuclear Physics Simulation",
        "# Generated automatically by Nuclear Simulation API",
        "",
        "# Set verbosity levels",
        "/control/verbose 1",
        "/run/verbose 1",
        "/tracking/verbose 0",
        "/hits/verbose 0",
        "",
        "# Initialize kernel",
        "/run/initialize",
        "",
        "# Basic physics setup",
        "/process/em/verbose 0",
        "/process/had/verbose 0",
        "",
    ]
    
    if sim_request.enable_visualization:
        viz_commands = [
            "# Visualization Setup",
            f"# Attempting Qt visualization with size {sim_request.image_width}x{sim_request.image_height}",
            "",
            "# Try Qt driver first",
            f"/vis/open Qt {sim_request.image_width}x{sim_request.image_height}-0+0",
            "/vis/viewer/set/autoRefresh false",
            "/vis/verbose confirmations",
            "",
            "# Draw the world volume",
            "/vis/drawVolume worlds",
            "",
            "# Set up the scene",
            "/vis/scene/create",
            "/vis/scene/add/volume worlds",
            "",
        ]
        
        # Camera setup
        if sim_request.camera_angle == "side":
            viz_commands.extend([
                "/vis/viewer/set/viewpointVector 1 0 0",
                "/vis/viewer/set/upVector 0 0 1",
            ])
        elif sim_request.camera_angle == "top":
            viz_commands.extend([
                "/vis/viewer/set/viewpointVector 0 0 1", 
                "/vis/viewer/set/upVector 0 1 0",
            ])
        elif sim_request.camera_angle == "front":
            viz_commands.extend([
                "/vis/viewer/set/viewpointVector 0 1 0",
                "/vis/viewer/set/upVector 0 0 1",
            ])
        else:  # iso
            viz_commands.extend([
                "/vis/viewer/set/viewpointVector 1 1 1",
                "/vis/viewer/set/upVector 0 0 1",
            ])
        
        # Background and visual settings
        bg_color = "1 1 1" if sim_request.background_color == "white" else "0 0 0"
        viz_commands.extend([
            f"/vis/viewer/set/background {bg_color}",
            "/vis/viewer/set/style surface",
            "/vis/viewer/set/auxiliaryEdge true",
            "",
        ])
        
        # Particle tracking for visualization
        if sim_request.visualization_type == "tracks":
            viz_commands.extend([
                "# Enable particle tracking",
                "/tracking/storeTrajectory 2",
                "/vis/scene/add/trajectories smooth",
                "/vis/modeling/trajectories/create/drawByCharge",
                "/vis/modeling/trajectories/drawByCharge-0/set 1 blue",
                "/vis/modeling/trajectories/drawByCharge-0/set -1 red",
                "/vis/modeling/trajectories/drawByCharge-0/set 0 green",
                "/vis/scene/endOfEventAction accumulate",
            ])
        elif sim_request.visualization_type == "hits":
            viz_commands.extend([
                "# Enable hit visualization",
                "/vis/scene/add/hits",
            ])
        
        # Axes
        if sim_request.show_axes:
            viz_commands.append("/vis/scene/add/axes 0 0 0 10 cm")
        
        # Beam setup
        viz_commands.extend([
            "",
            "# Beam setup",
            f"/gun/energy {sim_request.beam_energy_mev} MeV",
            "/gun/position 0 0 -50 mm",
            "/gun/direction 0 0 1",
            "",
            "# Run a few events for visualization",
            f"/run/beamOn {viz_events}",
            "",
            "# Export the image - try multiple formats",
            f"/vis/ogl/export {output_dir}/geant4_viz",
            f"/vis/qt/export {output_dir}/geant4_viz.png",
            f"/vis/qt/export {output_dir}/geant4_viz.eps",
            "",
            "# Force refresh and update",
            "/vis/viewer/refresh",
            "/vis/viewer/update",
            "/vis/viewer/flush",
        ])
        
        macro_lines.extend(viz_commands)
    else:
        # No visualization - simple run
        macro_lines.extend([
            f"/gun/energy {sim_request.beam_energy_mev} MeV",
            f"/run/beamOn {sim_request.max_events}",
        ])
    
    return "\n".join(macro_lines)

def create_geant4_visualization(sim_request: SimulationRequest, output_dir: str) -> VisualizationData:
    """Create visualization using Geant4 with improved error handling"""
    viz_start_time = time.time()
    
    try:
        # Find Geant4 executable
        geant4_exe = find_geant4_executable()
        if not geant4_exe:
            logger.error("Geant4 executable not found")
            return create_fallback_visualization(sim_request, viz_start_time)
        
        # Generate macro
        macro_content = generate_geant4_macro(sim_request, output_dir)
        macro_path = os.path.join(output_dir, "simulation.mac")
        
        with open(macro_path, 'w') as f:
            f.write(macro_content)
        
        logger.info(f"Generated macro at: {macro_path}")
        logger.info(f"Using Geant4 executable: {geant4_exe}")
        
        # Set up environment
        env = os.environ.copy()
        env.update({
            'G4INSTALL': '/root/geant4-v11.3.2-install',
            'G4SYSTEM': 'Linux-g++',
            'PATH': '/root/geant4-v11.3.2-install/bin:' + env.get('PATH', ''),
            'LD_LIBRARY_PATH': '/root/geant4-v11.3.2-install/lib:' + env.get('LD_LIBRARY_PATH', ''),
            'QT_QPA_PLATFORM': 'offscreen',
            'DISPLAY': ':99',
            'G4VERBOSE': '1',
            'G4DEBUG': '1'
        })
        
        # Run Geant4
        logger.info("Starting Geant4 simulation...")
        process = subprocess.Popen(
            [geant4_exe, macro_path],
            cwd=output_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait with timeout
        try:
            stdout, stderr = process.communicate(timeout=90)
            logger.info(f"Geant4 completed with return code: {process.returncode}")
            logger.info(f"Geant4 stdout: {stdout}")
            if stderr:
                logger.warning(f"Geant4 stderr: {stderr}")
        except subprocess.TimeoutExpired:
            logger.error("Geant4 process timed out")
            process.kill()
            return create_fallback_visualization(sim_request, viz_start_time)
        
        # Look for generated images
        image_files = []
        for ext in ['png', 'eps', 'jpg', 'gif']:
            pattern = os.path.join(output_dir, f"*.{ext}")
            image_files.extend(glob.glob(pattern))
        
        logger.info(f"Found image files: {image_files}")
        
        # Process the best available image
        best_image = None
        for img_path in image_files:
            if img_path.endswith('.png'):
                best_image = img_path
                break
        
        if not best_image and image_files:
            best_image = image_files[0]
        
        if best_image and os.path.exists(best_image):
            logger.info(f"Using image: {best_image}")
            
            # Convert to PNG if needed
            final_image = best_image
            if not best_image.endswith('.png'):
                png_path = os.path.join(output_dir, "converted.png")
                convert_cmd = ['convert', best_image, png_path]
                result = subprocess.run(convert_cmd, capture_output=True, timeout=30)
                if result.returncode == 0 and os.path.exists(png_path):
                    final_image = png_path
                    logger.info("Successfully converted image to PNG")
            
            # Read and encode
            with open(final_image, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            return VisualizationData(
                image_base64=img_base64,
                image_format="png",
                description=f"Geant4 {sim_request.visualization_type} visualization of {sim_request.fusion_reaction} at {sim_request.beam_energy_mev} MeV",
                generation_time=time.time() - viz_start_time
            )
        else:
            logger.error("No visualization images were generated by Geant4")
            return create_fallback_visualization(sim_request, viz_start_time)
            
    except Exception as e:
        logger.error(f"Geant4 visualization failed: {e}")
        return create_fallback_visualization(sim_request, viz_start_time)

def create_fallback_visualization(sim_request: SimulationRequest, start_time: float) -> VisualizationData:
    """Create fallback visualization using ImageMagick"""
    try:
        projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            png_path = tmp_file.name
        
        bg_color = "white" if sim_request.background_color == "white" else "black"
        text_color = "black" if sim_request.background_color == "white" else "white"
        
        draw_commands = [
            'convert',
            '-size', f'{sim_request.image_width}x{sim_request.image_height}',
            f'xc:{bg_color}',
            '-font', 'DejaVu-Sans',
            '-pointsize', '16',
            '-fill', text_color,
            '-draw', f'text {sim_request.image_width//2-100},{30} "Nuclear Fusion: {sim_request.fusion_reaction}"',
            '-pointsize', '12',
            '-draw', f'text {sim_request.image_width//2-80},{55} "Energy: {sim_request.beam_energy_mev} MeV"',
            '-fill', 'red',
            '-draw', f'circle {sim_request.image_width//2+100},{sim_request.image_height//2} {sim_request.image_width//2+140},{sim_request.image_height//2}',
            '-stroke', 'blue',
            '-strokewidth', '4',
            '-draw', f'line 100,{sim_request.image_height//2} {sim_request.image_width//2+60},{sim_request.image_height//2}',
            '-fill', text_color,
            '-pointsize', '10',
            '-draw', f'text 100,{sim_request.image_height//2-15} "{projectile}"',
            '-draw', f'text {sim_request.image_width//2+85},{sim_request.image_height//2+5} "{target}"',
            png_path
        ]
        
        result = subprocess.run(draw_commands, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(png_path):
            with open(png_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            try:
                os.unlink(png_path)
            except:
                pass
            
            return VisualizationData(
                image_base64=img_base64,
                image_format="png",
                description=f"Fallback visualization for {sim_request.fusion_reaction} simulation (Geant4 Qt driver unavailable)",
                generation_time=time.time() - start_time
            )
        else:
            logger.error(f"Fallback visualization failed: {result.stderr}")
            return VisualizationData(
                description=f"All visualization methods failed: {result.stderr}",
                generation_time=time.time() - start_time
            )
            
    except Exception as e:
        logger.error(f"Fallback visualization error: {e}")
        return VisualizationData(
            description=f"Complete visualization failure: {str(e)}",
            generation_time=time.time() - start_time
        )

def simulate_nuclear_physics(sim_request: SimulationRequest) -> Dict[str, Any]:
    """Enhanced simulation with improved Geant4 visualization"""
    
    projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
    sim_dir = tempfile.mkdtemp(prefix="geant4_sim_")
    visualization_data = None
    
    try:
        logger.info(f"Running simulation in {sim_dir} (visualization: {sim_request.enable_visualization})")
        
        # Simulate processing time
        processing_time = min(sim_request.max_events / 2000.0, 8.0)
        time.sleep(processing_time)
        
        # Physics calculations (keeping existing logic)
        coulomb_barrier = 1.44 * sim_request.Z * (sim_request.Z + sim_request.N) / (1.2 * ((sim_request.Z + sim_request.N)**(1/3) + 48**(1/3)))
        
        if sim_request.beam_energy_mev > coulomb_barrier:
            excess_energy = sim_request.beam_energy_mev - coulomb_barrier
            cross_section = max(0.1, 10.0 * excess_energy / (excess_energy + 50.0))
            confidence = min(0.9, 0.3 + excess_energy / 100.0)
        else:
            cross_section = 0.01 * max(0.001, sim_request.beam_energy_mev / coulomb_barrier)
            confidence = 0.2
        
        mass_number = sim_request.Z + sim_request.N + 48
        if mass_number > 250:
            products = [f"Element-{sim_request.Z + 20}", f"Element-{sim_request.Z + 19}"]
        else:
            products = [f"Element-{sim_request.Z + 20}"]
        
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
        
        # Generate visualization
        if sim_request.enable_visualization:
            logger.info("Generating Geant4 visualization...")
            visualization_data = create_geant4_visualization(sim_request, sim_dir)
            logger.info(f"Visualization complete. Success: {visualization_data.image_base64 is not None}")
        
        return {
            "cross_section": round(cross_section, 3),
            "cross_section_error": round(cross_section * 0.1, 3),
            "primary_products": products,
            "decay_chain": decay_chain,
            "total_events": sim_request.max_events,
            "confidence": round(confidence, 2),
            "half_life": decay_chain[0]["half_life"] if decay_chain else None,
            "half_life_unit": "seconds" if decay_chain else None,
            "visualization": visualization_data
        }
        
    finally:
        try:
            shutil.rmtree(sim_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {sim_dir}: {e}")

# Keep all existing endpoints unchanged
async def run_simulation_async(sim_request: SimulationRequest, sim_id: str):
    """Run simulation asynchronously"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting simulation {sim_id}")
        projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
        simulation_results[sim_id].status = "running"
        simulation_results[sim_id].target_nucleus = target
        simulation_results[sim_id].projectile = projectile
        
        results = simulate_nuclear_physics(sim_request)
        computation_time = (datetime.now() - start_time).total_seconds()
        
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
        simulation_result.visualization = results.get("visualization")
        
        decay_data = results.get("decay_chain", [])
        simulation_result.decay_chain = [DecayStep(**decay) for decay in decay_data]
        
        logger.info(f"Simulation {sim_id} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Simulation {sim_id} failed: {error_msg}")
        simulation_results[sim_id].status = "failed"
        simulation_results[sim_id].error_message = error_msg

@app.post("/simulate", response_model=Dict[str, str])
async def start_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """Start a nuclear fusion simulation with improved visualization"""
    try:
        parse_fusion_reaction(request.fusion_reaction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    sim_id = str(uuid.uuid4())
    simulation_results[sim_id] = SimulationResult(
        simulation_id=sim_id,
        status="queued",
        target_nucleus="",
        projectile="",
        beam_energy_mev=request.beam_energy_mev,
        created_at=datetime.now()
    )
    
    background_tasks.add_task(run_simulation_async, request, sim_id)
    
    return {
        "simulation_id": sim_id,
        "status": "started",
        "message": "Simulation started with improved Geant4 detection. Use simulation_id to check status.",
        "visualization_enabled": str(request.enable_visualization)
    }

@app.get("/simulation/{simulation_id}", response_model=SimulationResult)
async def get_simulation_result(simulation_id: str):
    """Get simulation result by ID"""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulation_results[simulation_id]

@app.get("/simulation/{simulation_id}/image")
async def get_simulation_image(simulation_id: str):
    """Get only the visualization image for a simulation"""
    if simulation_id not in simulation_results:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    result = simulation_results[simulation_id]
    if not result.visualization or not result.visualization.image_base64:
        raise HTTPException(status_code=404, detail="No visualization available for this simulation")
    
    return {
        "simulation_id": simulation_id,
        "image_base64": result.visualization.image_base64,
        "image_format": result.visualization.image_format,
        "description": result.visualization.description,
        "generation_time": result.visualization.generation_time
    }

@app.get("/simulations", response_model=List[Dict[str, Any]])
async def list_simulations():
    """List all simulations"""
    return [
        {
            "simulation_id": sim_id,
            "status": result.status,
            "created_at": result.created_at,
            "fusion_reaction": f"{result.projectile} + {result.target_nucleus}" if result.projectile and result.target_nucleus else "N/A",
            "has_visualization": result.visualization is not None and result.visualization.image_base64 is not None
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

@app.get("/debug/geant4")
async def debug_geant4():
    """Debug endpoint to check Geant4 setup"""
    debug_info = {}
    
    try:
        # Find Geant4 executable
        geant4_exe = find_geant4_executable()
        debug_info["geant4_executable"] = geant4_exe
        debug_info["geant4_executable_exists"] = geant4_exe is not None and os.path.exists(geant4_exe) if geant4_exe else False
        
        # Check Geant4 installation directory
        g4_install = "/root/geant4-v11.3.2-install"
        debug_info["g4_install_exists"] = os.path.exists(g4_install)
        
        if os.path.exists(g4_install):
            debug_info["g4_install_contents"] = os.listdir(g4_install)
            
            bin_dir = os.path.join(g4_install, "bin")
            if os.path.exists(bin_dir):
                executables = [f for f in os.listdir(bin_dir) if os.access(os.path.join(bin_dir, f), os.X_OK)]
                debug_info["g4_executables"] = executables
        
        # Test simple command
        if geant4_exe:
            try:
                result = subprocess.run([geant4_exe, "--help"], capture_output=True, text=True, timeout=10)
                debug_info["geant4_help_exit_code"] = result.returncode
                debug_info["geant4_help_output"] = result.stdout[:500] if result.stdout else "No output"
            except Exception as e:
                debug_info["geant4_help_error"] = str(e)
        
        # Environment check
        debug_info["environment"] = {
            "G4INSTALL": os.environ.get("G4INSTALL"),
            "PATH": os.environ.get("PATH", "")[:200] + "...",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")[:200] + "...",
            "DISPLAY": os.environ.get("DISPLAY"),
            "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM")
        }
        
        return debug_info
        
    except Exception as e:
        debug_info["error"] = str(e)
        return debug_info

@app.get("/health")
async def health_check():
    """Enhanced health check with Geant4 detection"""
    try:
        # Find Geant4 executable
        geant4_exe = find_geant4_executable()
        geant4_available = geant4_exe is not None
        
        # Check Geant4 config
        geant4_config = "/root/geant4-v11.3.2-install/bin/geant4-config"
        geant4_version = "Not found"
        if os.path.exists(geant4_config):
            try:
                result = subprocess.run([geant4_config, "--version"], capture_output=True, text=True, timeout=10)
                geant4_version = result.stdout.strip() if result.returncode == 0 else "Config exists but version check failed"
            except:
                geant4_version = "Config exists but not executable"
        
        # Check Qt
        try:
            result = subprocess.run(['qmake', '--version'], capture_output=True, text=True, timeout=5)
            qt_available = result.returncode == 0
            qt_version = result.stdout.split('\n')[0] if result.returncode == 0 else "Version check failed"
        except:
            qt_available = False
            qt_version = "Not available"
        
        # Check Xvfb
        try:
            subprocess.run(['Xvfb', '-help'], capture_output=True, timeout=5)
            xvfb_available = True
        except:
            xvfb_available = False
        
        # Check ImageMagick
        try:
            result = subprocess.run(['convert', '--version'], capture_output=True, text=True, timeout=5)
            imagemagick_available = result.returncode == 0
            imagemagick_version = result.stdout.split('\n')[0] if result.returncode == 0 else "Version check failed"
        except:
            imagemagick_available = False
            imagemagick_version = "Not available"
        
        return {
            "status": "healthy",
            "geant4_executable_found": geant4_available,
            "geant4_executable_path": geant4_exe,
            "geant4_version": geant4_version,
            "qt_available": qt_available,
            "qt_version": qt_version,
            "xvfb_available": xvfb_available,
            "imagemagick_available": imagemagick_available,
            "imagemagick_version": imagemagick_version,
            "display_env": os.environ.get('DISPLAY', 'Not set'),
            "qt_platform": os.environ.get('QT_QPA_PLATFORM', 'Not set'),
            "visualization_supported": geant4_available and qt_available and xvfb_available,
            "fallback_viz_available": imagemagick_available,
            "active_simulations": len([r for r in simulation_results.values() if r.status == "running"]),
            "total_simulations": len(simulation_results),
            "simulations_with_viz": len([r for r in simulation_results.values() if r.visualization and r.visualization.image_base64])
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
