# Product Recommendation System - Test Suite Documentation

## Overview

This test suite provides comprehensive coverage for the Product Recommendation with Agent system, including:

- **Unit Tests**: Testing individual functions and classes in isolation
- **Integration Tests**: Testing workflows and interactions between modules
- **Error Handling Tests**: Verifying proper error handling and edge cases
- **Performance Tests**: Benchmarking critical functions with large datasets
- **Data Validation Tests**: Ensuring data integrity and validation

## Test Structure

```
test_suite.py                    # Main test file with all tests
pytest.ini                        # Pytest configuration
conftest.py (if needed)          # Shared fixtures and configuration
```

## Modules Covered

### 1. recommendation_agent.py
- **Product Model**: Pydantic validation for individual products
- **RecommendationResponse Model**: Validation for recommendation responses
- **parse_and_validate_response()**: JSON parsing and validation
- **agent_creation_wrapper()**: Agent initialization with different LLM providers

### 2. scripts/data_loader.py
- **age_interval()**: Age bucketing logic
- **bw_check()**: Black and white image detection
- **data_load_df()**: DataFrame creation from image data
- **apply_filter()**: Data filtering and cleaning

### 3. scripts/embeddings.py (ready for extension)
- Image embedding generation
- Batch processing utilities

### 4. scripts/train_model.py (ready for extension)
- Model training workflows
- Hyperparameter optimization
- Custom metrics

### 5. main.py (ready for extension)
- Feature selection and PCA
- Model pipeline execution

## Installation

### Prerequisites
```bash
pip install pytest pytest-cov pytest-mock pandas numpy
```

### Install from requirements
```bash
pip install -r requirements-test.txt
```

**requirements-test.txt**:
```
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-asyncio>=0.21.0
pandas>=1.5.0
numpy>=1.24.0
pydantic>=2.0.0
```

## Running Tests

### Run All Tests
```bash
pytest test_suite.py -v
```

### Run Specific Test Class
```bash
pytest test_suite.py::TestProductModel -v
```

### Run Specific Test Function
```bash
pytest test_suite.py::TestProductModel::test_product_creation_valid -v
```

### Run with Coverage Report
```bash
pytest test_suite.py --cov=recommendation_agent --cov=scripts --cov-report=html
```

### Run Only Unit Tests
```bash
pytest test_suite.py -m unit -v
```

### Run Only Integration Tests
```bash
pytest test_suite.py -m integration -v
```

### Run with Detailed Output
```bash
pytest test_suite.py -vv --tb=long
```

### Run Specific Test Category
```bash
# Error handling tests
pytest test_suite.py -m error_handling -v

# Performance tests
pytest test_suite.py -m performance -v

# Data validation tests
pytest test_suite.py -m data_validation -v
```

## Test Categories

### Unit Tests

#### Product & Recommendation Models
- `TestProductModel`: 
  - Valid product creation
  - Invalid relevance scores (boundary conditions)
  - Missing required fields
  - Special characters in names

- `TestRecommendationResponseModel`:
  - Valid recommendation creation
  - Single/multiple products
  - Empty products list
  - Invalid products in response

#### Agent Functions
- `TestParseAndValidateResponse`:
  - Valid JSON parsing
  - Markdown code block handling
  - Invalid JSON error handling
  - Schema validation
  - Extra fields handling

- `TestAgentCreationWrapper`:
  - Openrouter provider setup
  - Grok provider setup
  - System prompt configuration

#### Data Loading
- `TestAgeInterval`: Age bucketing logic
- `TestBwCheck`: Image color space detection
- `TestDataLoadDf`: DataFrame creation
- `TestApplyFilter`: Data filtering

### Integration Tests

- `TestRecommendationPipeline`: Full end-to-end workflow
- `TestDataPipelineIntegration`: Data load-filter workflow

### Error Handling Tests

- Invalid products in response
- Malformed JSON handling
- Invalid API key handling

### Data Validation Tests

- Zero relevance score handling
- Perfect relevance score handling
- Multiple currency symbol support

### Performance Tests

- Large recommendation response parsing (100 products)
- DataFrame filtering with 10,000+ rows

## Test Execution Examples

### Development Workflow
```bash
# Quick test during development
pytest test_suite.py -x  # Stop on first failure

# Test with coverage
pytest test_suite.py --cov=recommendation_agent --cov-report=term-missing

# Verbose output with short traceback
pytest test_suite.py -v --tb=short
```

### CI/CD Pipeline
```bash
# Full test suite with coverage report
pytest test_suite.py \
  --cov=recommendation_agent \
  --cov=scripts \
  --cov-report=html \
  --cov-report=xml \
  -v

# Generate HTML coverage report
open htmlcov/index.html
```

### Debugging Failed Tests
```bash
# Show print statements
pytest test_suite.py -s -v

# Enter debugger on failure
pytest test_suite.py --pdb

# More detailed output
pytest test_suite.py -vv --tb=long
```

## Fixtures

The test suite includes reusable fixtures for common test data:

```python
# Valid product data
@pytest.fixture
def valid_product_data()

# Complete recommendation response
@pytest.fixture
def valid_recommendation_response_data()

# Sample DataFrame with demographic data
@pytest.fixture
def sample_dataframe()
```

**Usage in tests**:
```python
def test_function(valid_product_data):
    product = Product(**valid_product_data)
    assert product.relevance_score == 0.95
```

## Mocking

Tests use `unittest.mock` to mock external dependencies:

```python
@patch('recommendation_agent.ChatOpenAI')
@patch('recommendation_agent.create_agent')
def test_agent_creation(mock_create_agent, mock_chatOpenAI):
    # Mock implementations don't require real API calls
    pass
```

## Coverage Goals

| Module | Target Coverage | Current |
|--------|-----------------|---------|
| recommendation_agent.py | 95% | Testing |
| scripts/data_loader.py | 90% | Testing |
| scripts/embeddings.py | 85% | Ready |
| scripts/train_model.py | 85% | Ready |
| main.py | 90% | Ready |

## Test Results Interpretation

### Success
```
test_product_creation_valid PASSED
```

### Failure
```
test_product_invalid_relevance FAILED - AssertionError: expected error
```

### Skipped
```
test_large_dataset SKIPPED - long running test
```

## Continuous Testing Tips

1. **Before Committing**:
   ```bash
   pytest test_suite.py -v --tb=short
   ```

2. **Watch Mode** (requires pytest-watch):
   ```bash
   ptw test_suite.py
   ```

3. **Performance Profiling**:
   ```bash
   pytest test_suite.py --durations=10  # Show 10 slowest tests
   ```

4. **Test Random Order** (requires pytest-randomly):
   ```bash
   pytest test_suite.py --random-order
   ```

## Adding New Tests

When adding new functionality, follow this template:

```python
class TestNewFeature:
    """Unit tests for new feature"""
    
    def test_basic_functionality(self):
        """Test basic case"""
        # Arrange
        test_data = {...}
        
        # Act
        result = new_function(test_data)
        
        # Assert
        assert result == expected_value
    
    def test_edge_case(self):
        """Test edge case"""
        pass
    
    @patch('module.dependency')
    def test_with_mock(self, mock_dep):
        """Test with mocked dependency"""
        pass
```

## Troubleshooting

### Import Errors
```bash
# Ensure project is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest test_suite.py
```

### Module Not Found
```bash
# Install all dependencies
pip install -e .
pytest test_suite.py
```

### Test Isolation Issues
```bash
# Run tests in random order to catch dependencies
pytest test_suite.py --random-order
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements-test.txt
      - run: pytest test_suite.py --cov --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Maintenance

- **Update tests** when API changes
- **Add tests** for bug fixes (regression prevention)
- **Review coverage** monthly
- **Refactor tests** to DRY principles
- **Document** complex test logic

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pydantic Testing](https://docs.pydantic.dev/latest/usage/testing/)
- [Unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Best Practices for Testing Python](https://realpython.com/python-testing/)

## Questions & Support

For test-related questions:
1. Check existing test documentation
2. Review similar test examples
3. Refer to pytest documentation
4. Contact the development team

---

**Last Updated**: 2026-03-22  
**Test Coverage**: Comprehensive (unit, integration, performance)  
**Status**: All tests ready for execution
