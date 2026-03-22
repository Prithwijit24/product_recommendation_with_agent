"""
Pytest configuration and shared fixtures for test suite.

This file provides:
- Shared pytest configuration
- Common fixtures for all tests
- Pytest hooks for test lifecycle
- Helper functions for testing
"""

import pytest
import json
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_configure(config):
    """Initial pytest configuration"""
    # Register custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "mock: mark test as using mocking"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords and not config.option.markexpr:
            item.add_marker(pytest.mark.skip(reason="slow test"))


# ============================================================================
# GLOBAL FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data (session-scoped)"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="function")
def temp_file(tmp_path):
    """Create a temporary file for testing file operations"""
    temp_file = tmp_path / "test_file.txt"
    temp_file.write_text("test content")
    return temp_file


# ============================================================================
# PRODUCT & RECOMMENDATION FIXTURES
# ============================================================================

@pytest.fixture
def valid_product_dict():
    """Standard valid product dictionary"""
    return {
        "product_name": "Premium Cotton Shirt",
        "creator": "BrandX",
        "product_price": "₹1500",
        "product_link": "https://example.com/shirt",
        "product_description": "High-quality cotton shirt for casual wear",
        "relevance_score": 0.95,
        "reason": "Perfect for your age and style"
    }


@pytest.fixture
def multiple_products():
    """Multiple product variations for testing"""
    return [
        {
            "product_name": "Cotton Shirt",
            "creator": "BrandX",
            "product_price": "₹1500",
            "product_link": "https://example.com/shirt1",
            "product_description": "Casual wear",
            "relevance_score": 0.9,
            "reason": "Suitable for you"
        },
        {
            "product_name": "Denim Jeans",
            "creator": "BrandY",
            "product_price": "₹2500",
            "product_link": "https://example.com/jeans1",
            "product_description": "Versatile denim",
            "relevance_score": 0.85,
            "reason": "Great fit"
        },
        {
            "product_name": "Casual Shoes",
            "creator": "BrandZ",
            "product_price": "₹3000",
            "product_link": "https://example.com/shoes1",
            "product_description": "Comfortable casual shoes",
            "relevance_score": 0.88,
            "reason": "Stylish option"
        }
    ]


@pytest.fixture
def valid_recommendation_response_dict(multiple_products):
    """Standard valid recommendation response"""
    return {
        "product_category": "Clothing",
        "recommended_products": multiple_products
    }


@pytest.fixture
def edge_case_products():
    """Products with edge case values"""
    return {
        "zero_relevance": {
            "product_name": "Niche Product",
            "creator": "SmallBrand",
            "product_price": "₹100",
            "product_link": "https://example.com/niche",
            "product_description": "Very niche product",
            "relevance_score": 0.0,
            "reason": "Not relevant"
        },
        "perfect_relevance": {
            "product_name": "Perfect Product",
            "creator": "PopularBrand",
            "product_price": "₹5000",
            "product_link": "https://example.com/perfect",
            "product_description": "Exactly what you want",
            "relevance_score": 1.0,
            "reason": "Perfect match"
        },
        "no_price": {
            "product_name": "Price TBD",
            "creator": "Brand",
            "product_price": "Price not available",
            "product_link": "https://example.com/unknown",
            "product_description": "Coming soon",
            "relevance_score": 0.5,
            "reason": "Upcoming product"
        }
    }


# ============================================================================
# DATA & DATAFRAME FIXTURES
# ============================================================================

@pytest.fixture
def sample_data_small():
    """Small sample DataFrame for testing"""
    return pd.DataFrame({
        'age': [25, 30, 45],
        'gender': [0, 1, 0],
        'race': ['0', '1', '2'],
        'is_bw': [False, False, True],
        'age_interval': ['21 to 25', '26 to 30', '41 to 45']
    })


@pytest.fixture
def sample_data_medium():
    """Medium-sized sample DataFrame"""
    return pd.DataFrame({
        'age': list(range(20, 70, 5)) * 2,
        'gender': [0, 1] * 10,
        'race': ['0', '1', '2', '3'] * 5,
        'is_bw': [False, True] * 10,
        'age_interval': [f'{i} to {i+5}' for i in range(20, 70, 5)] * 2
    })


@pytest.fixture
def sample_data_large():
    """Large sample DataFrame for performance testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 10000),
        'gender': np.random.randint(0, 2, 10000),
        'race': [str(np.random.randint(0, 4)) for _ in range(10000)],
        'is_bw': np.random.choice([True, False], 10000),
        'age_interval': [f'{i*5} to {(i+1)*5}' for i in np.random.randint(0, 15, 10000)]
    })


@pytest.fixture
def image_path_samples():
    """Sample image file paths for testing"""
    return [
        '/data/25_0_2_photo.jpg',
        '/data/30_1_3_photo.jpg',
        '/data/45_0_1_photo.jpg',
        '/data/50_1_2_photo.jpg',
    ]


# ============================================================================
# MOCK & API FIXTURES
# ============================================================================

@pytest.fixture
def mock_chatOpenAI():
    """Mock ChatOpenAI instance"""
    with patch('recommendation_agent.ChatOpenAI') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_create_agent():
    """Mock create_agent function"""
    with patch('recommendation_agent.create_agent') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_agent_executor():
    """Mock agent executor with invoke method"""
    mock = MagicMock()
    mock.invoke.return_value = {
        'output': json.dumps({
            "product_category": "Clothing",
            "recommended_products": []
        })
    }
    return mock


# ============================================================================
# JSON & SERIALIZATION FIXTURES
# ============================================================================

@pytest.fixture
def valid_json_response(valid_recommendation_response_dict):
    """Valid JSON response string"""
    return json.dumps(valid_recommendation_response_dict)


@pytest.fixture
def json_with_markdown_block(valid_recommendation_response_dict):
    """JSON wrapped in markdown code block"""
    return f"```json\n{json.dumps(valid_recommendation_response_dict)}\n```"


@pytest.fixture
def malformed_json_strings():
    """Collection of malformed JSON strings"""
    return {
        "missing_brace": "{'key': 'value'",
        "single_quotes": "{'key': 'value'}",
        "incomplete": "{\"key\": ",
        "trailing_comma": "{\"key\": \"value\",}",
        "empty": "",
        "null": "null",
        "not_json": "This is not JSON"
    }


# ============================================================================
# CURRENCY & LOCALE FIXTURES
# ============================================================================

@pytest.fixture
def currency_variants():
    """Different currency format variants"""
    return {
        'india': '₹500',
        'usa': '$50',
        'europe': '€45',
        'uk': '£40',
        'japan': '¥5000',
        'text': 'Price not available',
        'range': '₹500-₹1000'
    }


@pytest.fixture
def location_variants():
    """Different location/demographic variants"""
    return {
        'india': {
            'location': 'India',
            'currency': '₹',
            'age_bracket': 25,
            'gender': 'Male',
            'race': 'Indian'
        },
        'usa': {
            'location': 'USA',
            'currency': '$',
            'age_bracket': 30,
            'gender': 'Female',
            'race': 'American'
        },
        'uk': {
            'location': 'UK',
            'currency': '£',
            'age_bracket': 28,
            'gender': 'Male',
            'race': 'British'
        }
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_product_list(count, start_relevance=0.5):
    """Helper to create multiple products with varying relevance"""
    products = []
    for i in range(count):
        relevance = min(1.0, start_relevance + (i * 0.05))
        products.append({
            "product_name": f"Product {i}",
            "creator": f"Brand {i}",
            "product_price": f"₹{100 * (i+1)}",
            "product_link": f"https://example.com/product/{i}",
            "product_description": f"Description for product {i}",
            "relevance_score": relevance,
            "reason": f"Reason for product {i}"
        })
    return products


def create_recommendation_response(category, product_count=3):
    """Helper to create complete recommendation response"""
    return {
        "product_category": category,
        "recommended_products": create_product_list(product_count)
    }


# ============================================================================
# PYTEST PARAMETRIZE HELPERS
# ============================================================================

# Valid age intervals for parametrize
AGE_INTERVAL_PARAMS = [
    (3, '1 to 5'),
    (7, '6 to 10'),
    (15, '11 to 15'),
    (23, '21 to 25'),
    (50, '46 to 50'),
]

# Valid currency formats for parametrize
CURRENCY_PARAMS = [
    '₹500',
    '$50',
    '€45',
    '£40',
    'Price not available',
]

# Invalid relevance scores for parametrize
INVALID_RELEVANCE_SCORES = [
    -0.1,
    -1.0,
    1.1,
    1.5,
    2.0,
    100.0,
]

# Valid relevance scores for parametrize
VALID_RELEVANCE_SCORES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
]
