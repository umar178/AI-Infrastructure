# Import necessary modules
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pandas as pd
import os

# Create FastAPI application instance
app = FastAPI(title="District Data API", description="API for accessing district information")

# Set up security scheme for API key authentication
security = HTTPBearer()

# Your secret API key - using your custom name
API_KEY = "MY_UNIFIED_API_KEY_123"

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


# Function to verify API key
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    This function checks if the provided API key matches our expected key
    It runs automatically before each endpoint execution
    """
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return True


# Load CSV data when the application starts
def load_data():
    """
    Load the CSV file - try multiple possible locations
    """
    # Try all possible paths
    possible_paths = [
        "data/processed/pakistan_unified_district_data.csv",  # From scripts folder
        "../data/processed/pakistan_unified_district_data.csv",  # From root folder
        "../../data/processed/pakistan_unified_district_data.csv",  # If nested deeper
        "processed/pakistan_unified_district_data.csv",  # If in scripts/processed
        "../pakistan_unified_district_data.csv"  # If in root folder
    ]

    for csv_path in possible_paths:
        try:
            print(f"Trying to load CSV from: {csv_path}")
            df = pd.read_csv(csv_path)
            print(f"SUCCESS! Loaded {len(df)} records from: {csv_path}")
            print(f"Columns: {df.columns.tolist()}")

            # Convert DataFrame to list of dictionaries
            data = df.to_dict(orient="records")
            return data

        except FileNotFoundError:
            print(f"Not found: {csv_path}")
            continue
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

    print("ERROR: Could not find CSV file in any location")
    return []


# Load data into memory
district_data = load_data()


# Root endpoint - basic health check
@app.get("/")
async def root():
    """Simple endpoint to check if API is running"""
    return {"message": "District Data API is running!", "records_loaded": len(district_data)}


# Endpoint 1: Get all records
@app.get("/data", dependencies=[Depends(verify_api_key)])
@limiter.limit("100/minute")
async def get_all_data(request: Request):
    """
    Returns all records from the CSV file as JSON
    Requires valid API key in Authorization header
    """
    if not district_data:
        raise HTTPException(status_code=404, detail="No data found - CSV file may be missing")
    return district_data


# Endpoint 2: Get single record by index
@app.get("/data/{id}", dependencies=[Depends(verify_api_key)])
@limiter.limit("50/minute")
async def get_single_data(request: Request, id: int):
    """
    Returns a single record by its index position (0-based)
    Example: /data/0 returns the first record
    """
    if id < 0 or id >= len(district_data):
        raise HTTPException(
            status_code=404,
            detail=f"Record with id {id} not found. Available IDs: 0 to {len(district_data) - 1}"
        )
    return district_data[id]


# Endpoint 3: Filter records by district name
@app.get("/filter", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def filter_by_district(request: Request, district: str):
    """
    Filters records by district name (case-insensitive partial match)
    Example: /filter?district=central
    """
    if not district_data:
        raise HTTPException(status_code=404, detail="No data available")

    # Filter data - convert both to lowercase for case-insensitive search
    filtered_data = [
        record for record in district_data
        if district.lower() in record.get("district", "").lower()
    ]

    if not filtered_data:
        raise HTTPException(
            status_code=404,
            detail=f"No districts found matching '{district}'"
        )

    return filtered_data


# Special endpoint to check data structure (no auth required)
@app.get("/info")
async def get_info():
    """Returns information about the loaded data structure"""
    if not district_data:
        return {"message": "No data loaded", "columns": []}

    # Get column names from first record
    sample_record = district_data[0]
    return {
        "total_records": len(district_data),
        "available_columns": list(sample_record.keys()),
        "sample_record": sample_record
    }


# Error handler for rate limiting
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded. Please try again later."
    )