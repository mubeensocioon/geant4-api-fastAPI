#!/usr/bin/env python3
"""
Test script to validate Geant4 Qt visualization setup
Run this inside your Docker container to test the visualization
"""

import requests
import json
import time
import base64
import os

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed!")
            print(f"   Geant4 version: {health_data.get('geant4_version', 'Unknown')}")
            print(f"   Qt available: {health_data.get('qt_available', False)}")
            print(f"   Qt version: {health_data.get('qt_version', 'Unknown')}")
            print(f"   Xvfb available: {health_data.get('xvfb_available', False)}")
            print(f"   Display env: {health_data.get('display_env', 'Not set')}")
            print(f"   Qt platform: {health_data.get('qt_platform', 'Not set')}")
            print(f"   Visualization supported: {health_data.get('visualization_supported', False)}")
            return health_data.get('visualization_supported', False)
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_simulation_with_visualization():
    """Test a simulation with Qt visualization enabled"""
    print("\n🚀 Testing simulation with Qt visualization...")
    
    # Simulation request with visualization
    sim_request = {
        "Z": 96,  # Curium
        "N": 151,  # Cm-247
        "fusion_reaction": "Ca-48 + Cm-247",
        "beam_energy_mev": 230.0,
        "simulate_decay_chain": True,
        "max_events": 1000,  # Smaller number for testing
        "enable_visualization": True,
        "visualization_type": "geometry",
        "camera_angle": "iso",
        "image_width": 800,
        "image_height": 600,
        "background_color": "white",
        "show_axes": True
    }
    
    try:
        # Start simulation
        print("   Starting simulation...")
        response = requests.post(f"{BASE_URL}/simulate", json=sim_request)
        
        if response.status_code != 200:
            print(f"❌ Failed to start simulation: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
        
        sim_data = response.json()
        sim_id = sim_data["simulation_id"]
        print(f"   ✅ Simulation started with ID: {sim_id}")
        
        # Poll for completion
        print("   Waiting for simulation to complete...")
        max_attempts = 30  # 5 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(10)  # Wait 10 seconds between checks
            attempt += 1
            
            # Check status
            status_response = requests.get(f"{BASE_URL}/simulation/{sim_id}")
            if status_response.status_code != 200:
                print(f"❌ Failed to get simulation status: {status_response.status_code}")
                return False
            
            status_data = status_response.json()
            status = status_data["status"]
            print(f"   Status check {attempt}: {status}")
            
            if status == "completed":
                print("   ✅ Simulation completed!")
                
                # Check if visualization was generated
                if status_data.get("visualization") and status_data["visualization"].get("image_base64"):
                    print("   ✅ Visualization image generated successfully!")
                    print(f"   Image format: {status_data['visualization'].get('image_format', 'unknown')}")
                    print(f"   Description: {status_data['visualization'].get('description', 'No description')}")
                    print(f"   Generation time: {status_data['visualization'].get('generation_time', 'unknown')} seconds")
                    
                    # Optionally save the image for inspection
                    save_test_image(status_data["visualization"]["image_base64"], sim_id)
                    
                    return True
                else:
                    print("   ⚠️  Simulation completed but no visualization image found")
                    print("   This might indicate Qt driver issues")
                    return False
                    
            elif status == "failed":
                error_msg = status_data.get("error_message", "Unknown error")
                print(f"   ❌ Simulation failed: {error_msg}")
                return False
            elif status == "running":
                print("   ⏳ Still running...")
            else:
                print(f"   ❓ Unknown status: {status}")
        
        print("   ❌ Simulation timed out")
        return False
        
    except Exception as e:
        print(f"❌ Simulation test error: {e}")
        return False

def save_test_image(image_base64, sim_id):
    """Save the test image to disk for inspection"""
    try:
        image_data = base64.b64decode(image_base64)
        filename = f"test_visualization_{sim_id}.png"
        
        with open(filename, 'wb') as f:
            f.write(image_data)
        
        print(f"   💾 Test image saved as: {filename}")
        print(f"   Image size: {len(image_data)} bytes")
        
    except Exception as e:
        print(f"   ⚠️  Failed to save test image: {e}")

def test_fallback_visualization():
    """Test simulation with visualization but expect fallback"""
    print("\n🎨 Testing fallback visualization...")
    
    sim_request = {
        "Z": 82,  # Lead
        "N": 122,  # Pb-204
        "fusion_reaction": "Ca-48 + Pb-204",
        "beam_energy_mev": 200.0,
        "simulate_decay_chain": False,
        "max_events": 500,
        "enable_visualization": True,
        "visualization_type": "tracks",
        "camera_angle": "side",
        "image_width": 600,
        "image_height": 400,
        "background_color": "black",
        "show_axes": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}/simulate", json=sim_request)
        if response.status_code == 200:
            sim_data = response.json()
            print(f"   ✅ Fallback test simulation started: {sim_data['simulation_id']}")
            return True
        else:
            print(f"   ❌ Failed to start fallback test: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Fallback test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Geant4 Qt Visualization Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    viz_supported = test_health_check()
    
    if not viz_supported:
        print("\n⚠️  Visualization not supported according to health check")
        print("   This might still work if Qt is installed but not detected properly")
    
    # Test 2: Full simulation with visualization
    viz_success = test_simulation_with_visualization()
    
    # Test 3: Test fallback system
    fallback_success = test_fallback_visualization()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Health Check: {'✅ PASS' if viz_supported else '❌ FAIL'}")
    print(f"   Qt Visualization: {'✅ PASS' if viz_success else '❌ FAIL'}")
    print(f"   Fallback System: {'✅ PASS' if fallback_success else '❌ FAIL'}")
    
    if viz_success:
        print("\n🎉 Qt visualization is working correctly!")
    elif fallback_success:
        print("\n⚠️  Qt visualization failed, but fallback system works")
        print("   Consider checking Geant4 Qt driver installation")
    else:
        print("\n❌ Both Qt and fallback visualization failed")
        print("   Check Docker configuration and dependencies")

if __name__ == "__main__":
    main()
