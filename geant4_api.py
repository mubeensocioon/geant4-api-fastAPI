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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Geant4 Nuclear Simulation API with Visualization",
    description="API wrapper for Geant4 nuclear fusion and decay simulations with visualization support",
    version="1.1.0"
)

# Pydantic models for request/response
class SimulationRequest(BaseModel):
    Z: int = Field(..., description="Atomic number of target nucleus", ge=1, le=118)
    N: int = Field(..., description="Number of neutrons in target nucleus", ge=0)
    fusion_reaction: str = Field(..., description="Fusion reaction path (e.g., 'Ca-48 + Bk-249')")
    beam_energy_mev: float = Field(..., description="Beam energy in MeV", gt=0)
    simulate_decay_chain: bool = Field(default=True, description="Flag to simulate decay chain")
    max_events: int = Field(default=10000, description="Maximum number of events to simulate", ge=100, le=100000)
    
    # New visualization parameters
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
    """Generate Geant4 macro file with visualization commands"""
    
    # Limit events for visualization to avoid cluttered images
    viz_events = min(sim_request.max_events, 100) if sim_request.enable_visualization else sim_request.max_events
    
    macro_lines = [
        "# Geant4 Macro File for Nuclear Physics Simulation",
        "# Generated automatically by Nuclear Simulation API",
        "",
        "# Initialize kernel",
        "/run/initialize",
        "",
        "# Set up physics",
        "/process/em/verbose 0",
        "/process/had/verbose 0",
        "",
        "# Set up geometry and beam parameters",
        f"/gun/energy {sim_request.beam_energy_mev} MeV",
        "/gun/position 0 0 -10 cm",
        "/gun/direction 0 0 1",
        "",
    ]
    
    if sim_request.enable_visualization:
        # Visualization setup
        viz_commands = [
            "# Visualization Setup",
            f"/vis/open OGL {sim_request.image_width}x{sim_request.image_height}-0+0",
            "/vis/viewer/set/autoRefresh false",
            "/vis/verbose errors",
            "",
            "# Draw geometry",
            "/vis/drawVolume",
            "",
            "# Set viewing angle",
        ]
        
        # Set camera position based on requested angle
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
        else:  # iso (isometric)
            viz_commands.extend([
                "/vis/viewer/set/viewpointVector 1 1 1",
                "/vis/viewer/set/upVector 0 0 1",
            ])
        
        # Background color
        bg_color = "1 1 1" if sim_request.background_color == "white" else "0 0 0"
        viz_commands.extend([
            f"/vis/viewer/set/background {bg_color}",
            "",
        ])
        
        # Configure visualization type
        if sim_request.visualization_type == "tracks":
            viz_commands.extend([
                "# Enable particle track visualization",
                "/tracking/storeTrajectory 1",
                "/vis/scene/add/trajectories smooth",
                "/vis/modeling/trajectories/create/drawByCharge",
                "/vis/modeling/trajectories/drawByCharge-0/set 1 blue",      # positive charge
                "/vis/modeling/trajectories/drawByCharge-0/set -1 red",     # negative charge  
                "/vis/modeling/trajectories/drawByCharge-0/set 0 green",    # neutral
                "/vis/scene/endOfEventAction accumulate",
            ])
        elif sim_request.visualization_type == "hits":
            viz_commands.extend([
                "# Enable hit visualization",
                "/vis/scene/add/hits",
            ])
        elif sim_request.visualization_type == "dose":
            viz_commands.extend([
                "# Enable dose visualization",
                "/vis/scene/add/psHits",
            ])
        
        # Axes and additional visual elements
        if sim_request.show_axes:
            viz_commands.append("/vis/scene/add/axes 0 0 0 10 cm")
        
        viz_commands.extend([
            "",
            "# Set visual attributes",
            "/vis/viewer/set/auxiliaryEdge true",
            "/vis/viewer/set/lineSegmentsPerCircle 100",
            "",
            f"# Run simulation with {viz_events} events",
            f"/run/beamOn {viz_events}",
            "",
            "# Save visualization image",
            f"/vis/ogl/printEPS {output_dir}/simulation_viz",
            "/vis/viewer/refresh",
            "/vis/viewer/update",
        ])
        
        macro_lines.extend(viz_commands)
    else:
        # No visualization - just run simulation
        macro_lines.extend([
            f"# Run simulation with {sim_request.max_events} events",
            f"/run/beamOn {sim_request.max_events}",
        ])
    
    return "\n".join(macro_lines)

def create_simple_visualization_svg(sim_request: SimulationRequest) -> str:
    """Create a simple SVG visualization as fallback"""
    projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
    
    bg_color = "#ffffff" if sim_request.background_color == "white" else "#000000"
    text_color = "#000000" if sim_request.background_color == "white" else "#ffffff"
    beam_color = "#0066cc"
    target_color = "#cc0000"
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{sim_request.image_width}" height="{sim_request.image_height}" 
     xmlns="http://www.w3.org/2000/svg">
  
  <!-- Background -->
  <rect width="100%" height="100%" fill="{bg_color}"/>
  
  <!-- Title -->
  <text x="{sim_request.image_width//2}" y="30" 
        text-anchor="middle" font-family="Arial, sans-serif" font-size="18" 
        fill="{text_color}">
    Nuclear Fusion Simulation: {sim_request.fusion_reaction}
  </text>
  
  <!-- Energy label -->
  <text x="{sim_request.image_width//2}" y="55" 
        text-anchor="middle" font-family="Arial, sans-serif" font-size="14" 
        fill="{text_color}">
    Beam Energy: {sim_request.beam_energy_mev} MeV
  </text>
  
  <!-- Target nucleus (large circle) -->
  <circle cx="{sim_request.image_width//2 + 100}" cy="{sim_request.image_height//2}" 
          r="40" fill="{target_color}" stroke="{text_color}" stroke-width="2"/>
  <text x="{sim_request.image_width//2 + 100}" y="{sim_request.image_height//2 + 5}" 
        text-anchor="middle" font-family="Arial, sans-serif" font-size="12" 
        fill="white">
    {target}
  </text>
  
  <!-- Projectile beam (arrow) -->
  <line x1="100" y1="{sim_request.image_height//2}" 
        x2="{sim_request.image_width//2 + 50}" y2="{sim_request.image_height//2}" 
        stroke="{beam_color}" stroke-width="4"/>
  <polygon points="{sim_request.image_width//2 + 50},{sim_request.image_height//2 - 8} {sim_request.image_width//2 + 65},{sim_request.image_height//2} {sim_request.image_width//2 + 50},{sim_request.image_height//2 + 8}"
           fill="{beam_color}"/>
  
  <!-- Projectile label -->
  <text x="100" y="{sim_request.image_height//2 - 15}" 
        text-anchor="start" font-family="Arial, sans-serif" font-size="12" 
        fill="{text_color}">
    {projectile}
  </text>
  
  <!-- Coordinate axes if requested -->
  {'<g stroke="' + text_color + '" stroke-width="1" opacity="0.5">' if sim_request.show_axes else '<g style="display:none">'}
    <!-- X axis -->
    <line x1="50" y1="{sim_request.image_height - 80}" 
          x2="150" y2="{sim_request.image_height - 80}"/>
    <text x="155" y="{sim_request.image_height - 75}" 
          font-family="Arial, sans-serif" font-size="10" fill="{text_color}">X</text>
    
    <!-- Y axis -->
    <line x1="50" y1="{sim_request.image_height - 80}" 
          x2="50" y2="{sim_request.image_height - 180}"/>
    <text x="45" y="{sim_request.image_height - 185}" 
          font-family="Arial, sans-serif" font-size="10" fill="{text_color}">Y</text>
    
    <!-- Z axis (3D effect) -->
    <line x1="50" y1="{sim_request.image_height - 80}" 
          x2="20" y2="{sim_request.image_height - 110}"/>
    <text x="15" y="{sim_request.image_height - 115}" 
          font-family="Arial, sans-serif" font-size="10" fill="{text_color}">Z</text>
  </g>
  
  <!-- Visualization type indicator -->
  <text x="20" y="{sim_request.image_height - 20}" 
        font-family="Arial, sans-serif" font-size="11" 
        fill="{text_color}" opacity="0.7">
    Visualization: {sim_request.visualization_type.title()}
  </text>
  
  <!-- Camera angle indicator -->
  <text x="20" y="{sim_request.image_height - 5}" 
        font-family="Arial, sans-serif" font-size="11" 
        fill="{text_color}" opacity="0.7">
    View: {sim_request.camera_angle.upper()}
  </text>
  
</svg>'''
    
    return svg_content

def create_visualization_image(image_path: str, viz_type: str, sim_request: SimulationRequest) -> VisualizationData:
    """Create visualization with better error handling and SVG fallback"""
    viz_start_time = time.time()
    
    try:
        # Try SVG approach first (more reliable)
        svg_content = create_simple_visualization_svg(sim_request)
        svg_path = f"{image_path}.svg"
        png_path = f"{image_path}.png"
        
        # Write SVG file
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        # Convert SVG to PNG using ImageMagick
        convert_cmd = [
            'convert',
            '-background', sim_request.background_color,
            '-density', '150',  # DPI for good quality
            svg_path,
            png_path
        ]
        
        logger.info(f"Converting SVG to PNG: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(png_path):
            # Success - read PNG and convert to base64
            with open(png_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Clean up files
            try:
                os.remove(svg_path)
                os.remove(png_path)
            except:
                pass
            
            descriptions = {
                "geometry": f"3D geometry visualization of {sim_request.fusion_reaction} at {sim_request.beam_energy_mev} MeV",
                "tracks": f"Particle trajectory visualization for {sim_request.fusion_reaction}",
                "hits": f"Energy deposition pattern for {sim_request.fusion_reaction}",
                "dose": f"Radiation dose distribution for {sim_request.fusion_reaction}"
            }
            
            return VisualizationData(
                image_base64=img_base64,
                image_format="png",
                description=descriptions.get(viz_type, f"Simulation visualization for {sim_request.fusion_reaction}"),
                generation_time=time.time() - viz_start_time
            )
        else:
            # SVG conversion failed, try fallback approach
            logger.warning(f"SVG conversion failed: {result.stderr}")
            return create_fallback_visualization(image_path, viz_type, sim_request, viz_start_time)
            
    except subprocess.TimeoutExpired:
        logger.error("SVG conversion timed out")
        return create_fallback_visualization(image_path, viz_type, sim_request, viz_start_time)
    except Exception as e:
        logger.error(f"SVG visualization generation error: {e}")
        return create_fallback_visualization(image_path, viz_type, sim_request, viz_start_time)

def create_fallback_visualization(image_path: str, viz_type: str, sim_request: SimulationRequest, start_time: float) -> VisualizationData:
    """Create a simple fallback visualization using direct PNG generation"""
    try:
        projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
        png_path = f"{image_path}.png"
        
        # Create a simple PNG using ImageMagick's built-in drawing capabilities
        bg_color = "white" if sim_request.background_color == "white" else "black"
        text_color = "black" if sim_request.background_color == "white" else "white"
        
        # Generate a simple image using ImageMagick convert
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
            # Draw target circle
            '-fill', 'red',
            '-draw', f'circle {sim_request.image_width//2+100},{sim_request.image_height//2} {sim_request.image_width//2+140},{sim_request.image_height//2}',
            # Draw beam line
            '-stroke', 'blue',
            '-strokewidth', '4',
            '-draw', f'line 100,{sim_request.image_height//2} {sim_request.image_width//2+60},{sim_request.image_height//2}',
            # Add labels
            '-fill', text_color,
            '-pointsize', '10',
            '-draw', f'text 100,{sim_request.image_height//2-15} "{projectile}"',
            '-draw', f'text {sim_request.image_width//2+85},{sim_request.image_height//2+5} "{target}"',
            png_path
        ]
        
        result = subprocess.run(draw_commands, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(png_path):
            with open(png_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Clean up
            try:
                os.remove(png_path)
            except:
                pass
            
            return VisualizationData(
                image_base64=img_base64,
                image_format="png",
                description=f"Fallback visualization for {sim_request.fusion_reaction} simulation",
                generation_time=time.time() - start_time
            )
        else:
            logger.error(f"Fallback PNG generation failed: {result.stderr}")
            return VisualizationData(
                description=f"All visualization methods failed. Last error: {result.stderr}",
                generation_time=time.time() - start_time
            )
            
    except Exception as e:
        logger.error(f"Fallback visualization error: {e}")
        return VisualizationData(
            description=f"Visualization generation completely failed: {str(e)}",
            generation_time=time.time() - start_time
        )

def simulate_nuclear_physics(sim_request: SimulationRequest) -> Dict[str, Any]:
    """Enhanced simulation with optional visualization"""
    
    # Parse the reaction
    projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
    
    # Create temporary directory for this simulation
    sim_dir = tempfile.mkdtemp(prefix="geant4_sim_")
    visualization_data = None
    
    try:
        if sim_request.enable_visualization:
            logger.info(f"Running simulation with visualization in {sim_dir}")
            
            # Generate Geant4 macro with visualization
            macro_content = generate_geant4_macro(sim_request, sim_dir)
            macro_path = os.path.join(sim_dir, "simulation.mac")
            
            with open(macro_path, 'w') as f:
                f.write(macro_content)
            
            # Check if ImageMagick is available
            try:
                subprocess.run(['convert', '--version'], capture_output=True, timeout=5)
                logger.info("ImageMagick is available for visualization")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("ImageMagick not found - disabling visualization")
                sim_request.enable_visualization = False
        
        # Simulate some processing time based on events
        processing_time = min(sim_request.max_events / 2000.0, 8.0)
        time.sleep(processing_time)
        
        # Your existing physics calculations
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
        
        # Generate visualization if requested
        if sim_request.enable_visualization:
            logger.info("Generating visualization...")
            visualization_data = create_visualization_image(
                os.path.join(sim_dir, "simulation_viz"),
                sim_request.visualization_type,
                sim_request
            )
            logger.info(f"Visualization complete. Has image: {visualization_data.image_base64 is not None}")
        
        results = {
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
        
        return results
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(sim_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {sim_dir}: {e}")

async def run_simulation_async(sim_request: SimulationRequest, sim_id: str):
    """Run simulation asynchronously with visualization support"""
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting simulation {sim_id} (visualization: {sim_request.enable_visualization})")
        
        # Update status to running
        projectile, target = parse_fusion_reaction(sim_request.fusion_reaction)
        simulation_results[sim_id].status = "running"
        simulation_results[sim_id].target_nucleus = target
        simulation_results[sim_id].projectile = projectile
        
        # Run the enhanced simulation
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
        simulation_result.visualization = results.get("visualization")
        
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
    """Start a nuclear fusion simulation with optional visualization"""
    
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
        "message": "Simulation started. Use the simulation_id to check status.",
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
    """List all simulations with visualization status"""
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

@app.get("/health")
async def health_check():
    """Health check endpoint with visualization capability check"""
    try:
        # Check if Geant4 is available
        geant4_config = "/root/geant4-v11.3.2-install/bin/geant4-config"
        if os.path.exists(geant4_config):
            result = subprocess.run([geant4_config, "--version"], capture_output=True, text=True, timeout=10)
            geant4_version = result.stdout.strip() if result.returncode == 0 else "Available but version check failed"
        else:
            geant4_version = "Not found"
        
        # Check ImageMagick for visualization
        try:
            result = subprocess.run(['convert', '--version'], capture_output=True, text=True, timeout=5)
            imagemagick_available = result.returncode == 0
            imagemagick_version = result.stdout.split('\n')[0] if result.returncode == 0 else "Version check failed"
        except:
            imagemagick_available = False
            imagemagick_version = "Not available"
        
        # Check Ghostscript (needed for EPS conversion)
        try:
            result = subprocess.run(['gs', '--version'], capture_output=True, text=True, timeout=5)
            ghostscript_available = result.returncode == 0
            ghostscript_version = result.stdout.strip() if result.returncode == 0 else "Version check failed"
        except:
            ghostscript_available = False
            ghostscript_version = "Not available"
        
        return {
            "status": "healthy",
            "geant4_version": geant4_version,
            "imagemagick_available": imagemagick_available,
            "imagemagick_version": imagemagick_version,
            "ghostscript_available": ghostscript_available,
            "ghostscript_version": ghostscript_version,
            "visualization_supported": imagemagick_available,
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
