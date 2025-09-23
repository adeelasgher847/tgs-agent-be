#!/usr/bin/env python3
"""
Test script for VICIdial-compatible API
Demonstrates how to use the API endpoints similar to VICIdial
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000/api/v1/vicidial"
API_USER = "apiuser"
API_PASS = "apipass"

def test_api_status():
    """Test API status endpoint"""
    print("🔍 Testing API Status...")
    
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_add_call():
    """Test adding a call via VICIdial API"""
    print("📞 Testing Add Call...")
    
    params = {
        "source": "test_app",
        "user": API_USER,
        "pass": API_PASS,
        "function": "add_call",
        "phone_number": "1234567890",
        "campaign_id": "OUTBOUND1",
        "call_type": "outbound",
        "notes": "Test call from VICIdial API"
    }
    
    response = requests.get(f"{BASE_URL}/api.php", params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()
    
    return response.json().get("call_id") if response.status_code == 200 else None

def test_get_calls():
    """Test retrieving calls via VICIdial API"""
    print("📋 Testing Get Calls...")
    
    params = {
        "source": "test_app",
        "user": API_USER,
        "pass": API_PASS,
        "function": "get_calls",
        "campaign_id": "OUTBOUND1"
    }
    
    response = requests.get(f"{BASE_URL}/api.php", params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_get_stats():
    """Test getting call statistics via VICIdial API"""
    print("📊 Testing Get Stats...")
    
    params = {
        "source": "test_app",
        "user": API_USER,
        "pass": API_PASS,
        "function": "get_stats",
        "campaign_id": "OUTBOUND1"
    }
    
    response = requests.get(f"{BASE_URL}/api.php", params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_post_method():
    """Test POST method for VICIdial API"""
    print("📤 Testing POST Method...")
    
    data = {
        "source": "test_app",
        "user": API_USER,
        "pass": API_PASS,
        "function": "add_call",
        "phone_number": "9876543210",
        "campaign_id": "INBOUND1",
        "call_type": "inbound",
        "notes": "Test POST call from VICIdial API"
    }
    
    response = requests.post(f"{BASE_URL}/api.php", data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Run all tests"""
    print("🚀 VICIdial API Integration Test")
    print("=" * 50)
    print(f"Testing against: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    try:
        # Test API status
        test_api_status()
        
        # Test adding calls
        call_id = test_add_call()
        
        # Test retrieving calls
        test_get_calls()
        
        # Test getting statistics
        test_get_stats()
        
        # Test POST method
        test_post_method()
        
        print("✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
