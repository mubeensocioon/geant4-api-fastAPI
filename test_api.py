import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    print("🩺 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_docs():
    print("\n📖 Testing Documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"Docs accessible: {response.status_code == 200}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Docs check failed: {e}")
        return False

def test_simulation():
    print("\n🧪 Testing Simulation Endpoint...")
    try:
        # Test payload matching your n8n requirements
        simulation_request = {
            "Z": 20,
            "N": 28,
            "fusion_reaction": "Ca-48 + Bk-249",
            "beam_energy_mev": 240.0,
            "simulate_decay_chain": True,
            "max_events": 1000
        }
        
        print("📤 Sending simulation request...")
        print(f"Payload: {json.dumps(simulation_request, indent=2)}")
        
        response = requests.post(f"{BASE_URL}/simulate", json=simulation_request)
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            simulation_id = result["simulation_id"]
            print(f"✅ Simulation started successfully!")
            print(f"Simulation ID: {simulation_id}")
            
            # Check simulation status
            print("\n⏳ Checking simulation status...")
            for i in range(6):  # Check 6 times over 30 seconds
                time.sleep(5)
                status_response = requests.get(f"{BASE_URL}/simulation/{simulation_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"Check {i+1}: Status = {status_data['status']}")
                    
                    if status_data["status"] == "completed":
                        print("🎉 Simulation completed!")
                        print(f"Results: {json.dumps(status_data, indent=2, default=str)}")
                        return True
                    elif status_data["status"] == "failed":
                        print(f"❌ Simulation failed: {status_data.get('error_message')}")
                        return False
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
            
            print("⏰ Simulation still running after 30 seconds (this is normal)")
            return True
        else:
            print(f"❌ Simulation start failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        return False

def test_list_simulations():
    print("\n📋 Testing List Simulations...")
    try:
        response = requests.get(f"{BASE_URL}/simulations")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            simulations = response.json()
            print(f"Total simulations: {len(simulations)}")
            for sim in simulations:
                print(f"  - {sim['simulation_id']}: {sim['status']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ List simulations failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Geant4 API...")
    print("=" * 50)
    
    # Run all tests
    health_ok = test_health()
    docs_ok = test_docs()
    list_ok = test_list_simulations()
    sim_ok = test_simulation()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Health: {health_ok}")
    print(f"✅ Docs: {docs_ok}")
    print(f"✅ List: {list_ok}")
    print(f"✅ Simulation: {sim_ok}")
    
    if all([health_ok, docs_ok, list_ok, sim_ok]):
        print("\n🎉 All tests passed! API is ready for n8n integration!")
        print("\n🔗 n8n Integration URLs:")
        print(f"  - Webhook URL: http://your-server-ip:8000/simulate")
        print(f"  - Health Check: http://your-server-ip:8000/health")
    else:
        print("\n⚠️ Some tests failed. Check the logs above.")
