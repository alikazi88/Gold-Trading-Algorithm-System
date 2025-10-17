#!/usr/bin/env python3
"""
Debug script to test AllTick API connection and response format.
Run this to see what the API actually returns.
"""
import requests
import json
from datetime import datetime, timedelta, timezone

# Configuration
TOKEN = "180dc29f5cd599cb355387e73d3d808e-c-app"
BASE_URL = "https://quote.alltick.co/quote-b-api"

def test_kline_api():
    """Test the kline endpoint with different query formats."""
    
    print("=" * 80)
    print("TESTING ALLTICK API - KLINE ENDPOINT")
    print("=" * 80)
    
    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=1)
    
    # Test different query formats
    test_cases = [
        {
            "name": "Format 1: prod_code with period_type",
            "query": {
                "prod_code": "XAUUSD",
                "period_type": "5m",
                "adjust_type": "1",
                "start_time": str(int(start_time.timestamp() * 1000)),
                "end_time": str(int(end_time.timestamp() * 1000))
            }
        },
        {
            "name": "Format 2: symbol with interval",
            "query": {
                "symbol": "XAUUSD",
                "interval": "5m",
                "start_time": int(start_time.timestamp() * 1000),
                "end_time": int(end_time.timestamp() * 1000)
            }
        },
        {
            "name": "Format 3: Just prod_code and period_type (latest)",
            "query": {
                "prod_code": "XAUUSD",
                "period_type": "5m"
            }
        },
        {
            "name": "Format 4: XAUUSD.FOREX symbol",
            "query": {
                "prod_code": "XAUUSD.FOREX",
                "period_type": "5m",
                "adjust_type": "1"
            }
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'=' * 80}")
        
        url = f"{BASE_URL}/kline"
        params = {
            'token': TOKEN,
            'query': json.dumps(test['query'])
        }
        
        print(f"\nRequest URL: {url}")
        print(f"Query params: {json.dumps(test['query'], indent=2)}")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse JSON:")
                print(json.dumps(data, indent=2))
                
                # Check response structure
                if 'code' in data:
                    print(f"\n✓ Response has 'code': {data['code']}")
                if 'message' in data or 'msg' in data:
                    print(f"✓ Message: {data.get('message', data.get('msg', 'N/A'))}")
                if 'data' in data:
                    print(f"✓ Has 'data' field")
                    if isinstance(data['data'], list):
                        print(f"  - Data is a list with {len(data['data'])} items")
                        if len(data['data']) > 0:
                            print(f"  - First item: {data['data'][0]}")
                    elif isinstance(data['data'], dict):
                        print(f"  - Data is a dict with keys: {list(data['data'].keys())}")
            else:
                print(f"\n✗ Error: {response.text}")
                
        except Exception as e:
            print(f"\n✗ Exception: {e}")
        
        print(f"\n{'=' * 80}")
        
        # Wait between requests to avoid rate limiting
        if i < len(test_cases):
            print("Waiting 7 seconds before next test...")
            import time
            time.sleep(7)

def test_quote_api():
    """Test the quote endpoint."""
    
    print("\n\n" + "=" * 80)
    print("TESTING ALLTICK API - QUOTE ENDPOINT")
    print("=" * 80)
    
    symbols = ["XAUUSD", "XAUUSD.FOREX", "GOLD"]
    
    for symbol in symbols:
        print(f"\n--- Testing symbol: {symbol} ---")
        
        url = f"{BASE_URL}/quote"
        params = {
            'token': TOKEN,
            'query': json.dumps({"prod_code": symbol})
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error: {e}")
        
        import time
        time.sleep(2)

if __name__ == "__main__":
    print("\n🔍 AllTick API Debug Tool\n")
    print("This will test different API query formats to find the correct one.")
    print("Please wait, this will take about 30 seconds...\n")
    
    test_kline_api()
    test_quote_api()
    
    print("\n\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)
    print("\nCheck the output above to see which format works.")
    print("Look for responses with 'code': 0 and actual data.\n")
