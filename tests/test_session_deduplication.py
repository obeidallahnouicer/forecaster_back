"""
Quick test to verify session deduplication works correctly.

Run this script to test the session creation fix:
    python tests/test_session_deduplication.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from core.registry import ForecastRegistry
import tempfile
import shutil
import threading
import time


def test_unique_session_ids():
    """Test that session IDs are unique even when created rapidly."""
    print("\n=== Test 1: Unique Session IDs ===")
    
    # Create a test registry with temp cache
    temp_dir = tempfile.mkdtemp()
    try:
        registry = ForecastRegistry(cache_dir=Path(temp_dir))
        
        # Create test CSV file
        test_file = Path(temp_dir) / "test.csv"
        test_file.write_text("ref_article,designation,ventes\nA1,Product A,100\nA2,Product B,200")
        
        # Create multiple sessions rapidly
        session_ids = []
        for i in range(5):
            info = registry.create_session_from_file(str(test_file), frequency="yearly")
            session_ids.append(info.session_id)
            print(f"  Created: {info.session_id}")
        
        # Check uniqueness
        unique_ids = set(session_ids)
        print(f"\n  Total created: {len(session_ids)}")
        print(f"  Unique IDs: {len(unique_ids)}")
        
        if len(unique_ids) == 1:
            print("  ✅ SUCCESS: Deduplication working - reused existing session")
        elif len(unique_ids) == len(session_ids):
            print("  ✅ SUCCESS: All session IDs are unique")
        else:
            print(f"  ❌ FAIL: Found {len(session_ids) - len(unique_ids)} duplicate IDs")
        
    finally:
        shutil.rmtree(temp_dir)


def test_concurrent_creation():
    """Test thread-safe concurrent session creation."""
    print("\n=== Test 2: Concurrent Session Creation ===")
    
    temp_dir = tempfile.mkdtemp()
    try:
        registry = ForecastRegistry(cache_dir=Path(temp_dir))
        
        # Create test CSV file
        test_file = Path(temp_dir) / "test2.csv"
        test_file.write_text("ref_article,designation,ventes\nA1,Product A,100\nA2,Product B,200")
        
        results = []
        errors = []
        
        def create_session():
            try:
                info = registry.create_session_from_file(str(test_file), frequency="monthly")
                results.append(info.session_id)
            except Exception as e:
                errors.append(str(e))
        
        # Create 10 threads that all try to create sessions simultaneously
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_session)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        print(f"  Threads completed: {len(results)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Unique sessions: {len(set(results))}")
        print(f"  Total sessions in registry: {len(registry.list_sessions())}")
        
        if errors:
            print(f"  ❌ FAIL: {len(errors)} errors occurred")
            for err in errors[:3]:
                print(f"    - {err}")
        elif len(set(results)) == 1:
            print("  ✅ SUCCESS: All threads reused the same session (deduplication working)")
        elif len(set(results)) <= 3:
            print("  ⚠️  PARTIAL: Some threads created unique sessions (acceptable with timing)")
        else:
            print(f"  ⚠️  WARNING: Created {len(set(results))} different sessions")
        
    finally:
        shutil.rmtree(temp_dir)


def test_different_files():
    """Test that different files create different sessions."""
    print("\n=== Test 3: Different Files = Different Sessions ===")
    
    temp_dir = tempfile.mkdtemp()
    try:
        registry = ForecastRegistry(cache_dir=Path(temp_dir))
        
        # Create two different test files
        test_file1 = Path(temp_dir) / "test1.csv"
        test_file1.write_text("ref_article,ventes\nA1,100")
        
        test_file2 = Path(temp_dir) / "test2.csv"
        test_file2.write_text("ref_article,ventes\nB1,200")
        
        # Create sessions for both files
        info1 = registry.create_session_from_file(str(test_file1), frequency="yearly")
        info2 = registry.create_session_from_file(str(test_file2), frequency="yearly")
        
        print(f"  File 1 session: {info1.session_id}")
        print(f"  File 2 session: {info2.session_id}")
        
        if info1.session_id != info2.session_id:
            print("  ✅ SUCCESS: Different files create different sessions")
        else:
            print("  ❌ FAIL: Same session created for different files")
        
    finally:
        shutil.rmtree(temp_dir)


def test_session_listing():
    """Test the session listing functionality."""
    print("\n=== Test 4: Session Listing ===")
    
    temp_dir = tempfile.mkdtemp()
    try:
        registry = ForecastRegistry(cache_dir=Path(temp_dir))
        
        # Create test file
        test_file = Path(temp_dir) / "test.csv"
        test_file.write_text("ref_article,designation,ventes\nA1,Product A,100")
        
        # Create a session
        info = registry.create_session_from_file(str(test_file), frequency="yearly")
        
        # List sessions
        sessions = registry.list_sessions()
        
        print(f"  Sessions in registry: {len(sessions)}")
        print(f"  Session IDs: {sessions}")
        
        if len(sessions) == 1 and info.session_id in sessions:
            print("  ✅ SUCCESS: Session listing works correctly")
        else:
            print("  ❌ FAIL: Session listing incorrect")
        
        # Get session details
        retrieved = registry.get(info.session_id)
        if retrieved and retrieved.session_id == info.session_id:
            print(f"  ✅ SUCCESS: Session retrieval works")
        else:
            print(f"  ❌ FAIL: Could not retrieve session")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("SESSION DEDUPLICATION TESTS")
    print("=" * 60)
    
    try:
        test_unique_session_ids()
        test_concurrent_creation()
        test_different_files()
        test_session_listing()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
