"""
Example usage of the production-grade cache system.

This file demonstrates how to use the new caching system.
"""

import time
from pathlib import Path
from cache import CacheManager, UploadCacheManager, ForecastCacheManager


def example_basic_cache():
    """Example: Basic cache operations"""
    print("\n=== Basic Cache Example ===\n")
    
    cache_dir = Path("./cache/examples")
    cache = CacheManager(
        cache_dir=cache_dir,
        default_ttl=60,  # 60 seconds
        version="v1"
    )
    
    # Set a value
    cache.set("user:123", {"name": "John", "email": "john@example.com"})
    print("✓ Cached user data")
    
    # Get the value
    user = cache.get("user:123")
    print(f"✓ Retrieved from cache: {user}")
    
    # Check if exists
    exists = cache.exists("user:123")
    print(f"✓ Key exists: {exists}")
    
    # Get metadata
    metadata = cache.get_metadata("user:123")
    print(f"✓ Cache metadata: version={metadata.version}, hits={metadata.hit_count}")
    
    # Delete
    cache.delete("user:123")
    print("✓ Deleted from cache")
    
    # Verify deletion
    user = cache.get("user:123")
    print(f"✓ After deletion: {user}")


def example_upload_cache():
    """Example: Upload cache with retry logic"""
    print("\n=== Upload Cache Example ===\n")
    
    cache_dir = Path("./cache/examples")
    upload_cache = UploadCacheManager(
        cache_dir=cache_dir,
        max_retries=3,
        retry_delay=5,
        ttl=3600
    )
    
    # Create upload entry
    entry = upload_cache.create_upload(
        upload_id="upload_001",
        file_path="/tmp/data.csv",
        file_name="data.csv",
        file_size=1024,
        frequency="monthly"
    )
    print(f"✓ Created upload: {entry.upload_id}, status={entry.status}")
    
    # Mark as processing
    upload_cache.mark_processing("upload_001")
    print("✓ Marked as processing")
    
    # Simulate processing completion
    upload_cache.mark_completed("upload_001", session_id="session_123")
    print("✓ Marked as completed")
    
    # Get upload status
    entry = upload_cache.get_upload("upload_001")
    print(f"✓ Upload status: {entry.status}, session={entry.session_id}")
    
    # List all uploads
    uploads = upload_cache.list_uploads()
    print(f"✓ Total uploads: {len(uploads)}")
    
    # Get statistics
    stats = upload_cache.get_stats()
    print(f"✓ Upload stats: {stats}")


def example_forecast_cache():
    """Example: Forecast cache with invalidation"""
    print("\n=== Forecast Cache Example ===\n")
    
    cache_dir = Path("./cache/examples")
    forecast_cache = ForecastCacheManager(
        cache_dir=cache_dir,
        default_ttl=3600,
        model_version="v1"
    )
    
    # Simulate forecast data
    forecast_data = {
        "ref_article": "ART001",
        "avg_forecast": 150.5,
        "trend": "uptrend",
        "methods": {
            "sma": 145.0,
            "exponential": 155.0
        }
    }
    
    # Cache parameters
    params = {
        "period": 3,
        "alpha": 0.3,
        "methods": ["SMA", "ExpSmoothing"]
    }
    
    # Compute data fingerprint (in real usage, this comes from DataFrame)
    data_fingerprint = forecast_cache._compute_fingerprint("sample_data")
    
    # Cache forecast
    success = forecast_cache.set_article_forecast(
        ref_article="ART001",
        forecast_data=forecast_data,
        data_fingerprint=data_fingerprint,
        frequency="monthly",
        params=params,
        computation_time_ms=245.0
    )
    print(f"✓ Cached forecast for ART001: {success}")
    
    # Retrieve from cache
    cached = forecast_cache.get_article_forecast(
        ref_article="ART001",
        data_fingerprint=data_fingerprint,
        params=params
    )
    print(f"✓ Retrieved from cache: avg_forecast={cached['avg_forecast']}")
    
    # Try with different parameters (cache miss)
    different_params = {**params, "period": 5}
    cached = forecast_cache.get_article_forecast(
        ref_article="ART001",
        data_fingerprint=data_fingerprint,
        params=different_params
    )
    print(f"✓ Different parameters (cache miss): {cached}")
    
    # Invalidate article cache
    count = forecast_cache.invalidate_article("ART001")
    print(f"✓ Invalidated {count} cache entries for ART001")
    
    # Verify invalidation
    cached = forecast_cache.get_article_forecast(
        ref_article="ART001",
        data_fingerprint=data_fingerprint,
        params=params
    )
    print(f"✓ After invalidation: {cached}")


def example_ttl_expiration():
    """Example: TTL and automatic expiration"""
    print("\n=== TTL Expiration Example ===\n")
    
    cache_dir = Path("./cache/examples")
    cache = CacheManager(
        cache_dir=cache_dir,
        default_ttl=2,  # 2 seconds
        version="v1"
    )
    
    # Set with short TTL
    cache.set("temp_key", "temp_value", ttl=2)
    print("✓ Cached with 2s TTL")
    
    # Get immediately (should work)
    value = cache.get("temp_key")
    print(f"✓ Retrieved immediately: {value}")
    
    # Wait for expiration
    print("⏳ Waiting 3 seconds for expiration...")
    time.sleep(3)
    
    # Try to get after expiration (should be None)
    value = cache.get("temp_key")
    print(f"✓ After expiration: {value}")


def example_cache_stats():
    """Example: Cache statistics"""
    print("\n=== Cache Statistics Example ===\n")
    
    cache_dir = Path("./cache/examples")
    cache = CacheManager(
        cache_dir=cache_dir,
        default_ttl=3600,
        version="v1"
    )
    
    # Add some entries
    for i in range(5):
        cache.set(f"key_{i}", f"value_{i}", tags={"category": "test"})
    
    print("✓ Created 5 cache entries")
    
    # Get statistics
    stats = cache.get_stats()
    print(f"✓ Cache stats:")
    print(f"  - Total entries: {stats['total_entries']}")
    print(f"  - Total size: {stats['total_size_bytes']} bytes")
    print(f"  - By status: {stats['by_status']}")
    
    # List keys
    keys = cache.list_keys()
    print(f"✓ All keys: {keys}")
    
    # Filter by tags
    test_keys = cache.list_keys(tags={"category": "test"})
    print(f"✓ Keys with category=test: {test_keys}")


def example_cleanup():
    """Example: Cache cleanup"""
    print("\n=== Cache Cleanup Example ===\n")
    
    cache_dir = Path("./cache/examples")
    
    # Upload cache cleanup
    upload_cache = UploadCacheManager(cache_dir=cache_dir)
    
    # Create some old uploads
    upload_cache.create_upload(
        upload_id="old_upload_1",
        file_path="/tmp/old.csv",
        file_name="old.csv",
        file_size=1024,
        frequency="yearly"
    )
    
    # Mark as processing (will become stale)
    upload_cache.mark_processing("old_upload_1")
    
    print("✓ Created test upload in processing state")
    
    # Cleanup stale uploads (threshold is 1 hour by default)
    # In real usage, this would find uploads stuck in processing
    count = upload_cache.cleanup_stale()
    print(f"✓ Cleaned up {count} stale uploads")
    
    # Clear all
    cleared = upload_cache.cache.clear()
    print(f"✓ Cleared {cleared} entries from cache")


if __name__ == "__main__":
    """Run all examples"""
    print("=" * 60)
    print("Production-Grade Cache System - Examples")
    print("=" * 60)
    
    try:
        example_basic_cache()
        example_upload_cache()
        example_forecast_cache()
        example_ttl_expiration()
        example_cache_stats()
        example_cleanup()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
