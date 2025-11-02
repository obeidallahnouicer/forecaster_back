"""
Test the session listing API endpoint.

Usage:
    python tests/test_session_api.py

Make sure the server is running on http://localhost:8000
"""
import requests
import json


def test_list_sessions():
    """Test the GET /api/forecasts/sessions endpoint."""
    url = "http://localhost:8000/api/forecasts/sessions"
    
    print("Testing GET /api/forecasts/sessions")
    print(f"URL: {url}")
    print("-" * 60)
    
    try:
        response = requests.get(url)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse:")
            print(json.dumps(data, indent=2))
            
            print(f"\n✅ Total Sessions: {data.get('count', 0)}")
            
            sessions = data.get('sessions', [])
            if sessions:
                print("\nSession Details:")
                for session in sessions:
                    print(f"  - ID: {session.get('session_id')}")
                    print(f"    Frequency: {session.get('frequency')}")
                    print(f"    Rows: {session.get('rows')}")
                    print(f"    Status: {session.get('status')}")
                    print(f"    File: {session.get('file_path')}")
                    print()
            else:
                print("\nℹ️  No active sessions found")
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server")
        print("   Make sure the server is running: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")


def test_upload_deduplication():
    """Test that uploading the same file twice doesn't create duplicate sessions."""
    url = "http://localhost:8000/api/forecasts/upload"
    
    print("\n" + "=" * 60)
    print("Testing Upload Deduplication")
    print("-" * 60)
    
    # Check if test file exists
    import os
    test_file = "ventes_cleann.csv"
    if not os.path.exists(test_file):
        print(f"⚠️  Test file '{test_file}' not found, skipping upload test")
        return
    
    try:
        # Get initial session count
        initial_response = requests.get("http://localhost:8000/api/forecasts/sessions")
        initial_count = initial_response.json().get('count', 0)
        print(f"Initial session count: {initial_count}")
        
        # Upload file first time
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f)}
            data = {'frequency': 'yearly'}
            response1 = requests.post(url, files=files, data=data)
        
        if response1.status_code == 200:
            session1 = response1.json().get('session_id')
            print(f"\n1st Upload: {session1}")
        else:
            print(f"❌ 1st upload failed: {response1.text}")
            return
        
        # Upload same file second time
        with open(test_file, 'rb') as f:
            files = {'file': (test_file, f)}
            data = {'frequency': 'yearly'}
            response2 = requests.post(url, files=files, data=data)
        
        if response2.status_code == 200:
            session2 = response2.json().get('session_id')
            print(f"2nd Upload: {session2}")
        else:
            print(f"❌ 2nd upload failed: {response2.text}")
            return
        
        # Check final session count
        final_response = requests.get("http://localhost:8000/api/forecasts/sessions")
        final_count = final_response.json().get('count', 0)
        print(f"\nFinal session count: {final_count}")
        print(f"New sessions created: {final_count - initial_count}")
        
        # Verify deduplication
        if session1 == session2:
            print("\n✅ SUCCESS: Same session returned (deduplication working)")
        else:
            print(f"\n⚠️  WARNING: Different sessions created")
            print(f"   This might be expected if file content changed")
        
        if final_count - initial_count <= 1:
            print("✅ Session count increased by 0 or 1 (good)")
        else:
            print(f"⚠️  Session count increased by {final_count - initial_count}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server")
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("SESSION API TESTS")
    print("=" * 60)
    print()
    
    test_list_sessions()
    test_upload_deduplication()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)
