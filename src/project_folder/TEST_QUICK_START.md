# Test Suite Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Install Test Dependencies
```bash
# Install testing packages
pip install pytest pytest-cov pytest-mock pandas numpy pydantic

# Or use requirements file
pip install -r requirements-test.txt
```

### 2. Run All Tests
```bash
pytest test_suite.py -v
```

### 3. Check Coverage
```bash
pytest test_suite.py --cov=recommendation_agent --cov=scripts --cov-report=html
open htmlcov/index.html  # View coverage report
```

---

## 📋 Test Files

| File | Purpose |
|------|---------|
| **test_suite.py** | Main test file with all unit, integration, and performance tests |
| **conftest.py** | Pytest configuration and shared fixtures |
| **pytest.ini** | Pytest settings and markers |
| **requirements-test.txt** | Testing dependencies |
| **TEST_DOCUMENTATION.md** | Detailed test documentation |

---

## 🧪 Test Categories at a Glance

### Unit Tests (60+ tests)
```bash
# Test recommendation models
pytest test_suite.py::TestProductModel -v
pytest test_suite.py::TestRecommendationResponseModel -v

# Test agent functions
pytest test_suite.py::TestParseAndValidateResponse -v
pytest test_suite.py::TestAgentCreationWrapper -v

# Test data loading
pytest test_suite.py::TestAgeInterval -v
pytest test_suite.py::TestApplyFilter -v
```

### Integration Tests
```bash
pytest test_suite.py::TestRecommendationPipeline -v
pytest test_suite.py::TestDataPipelineIntegration -v
```

### Error Handling
```bash
pytest test_suite.py::TestErrorHandling -v
```

### All Tests with Summary
```bash
pytest test_suite.py -v --tb=short
```

---

## 📊 What's Being Tested

### ✅ recommendation_agent.py
- ✓ Product model creation and validation
- ✓ Recommendation response generation
- ✓ JSON parsing with edge cases
- ✓ Agent creation for different LLM providers
- ✓ Error handling for invalid data

### ✅ scripts/data_loader.py  
- ✓ Age interval bucketing
- ✓ B&W image detection
- ✓ DataFrame creation from image paths
- ✓ Data filtering and cleaning
- ✓ Edge cases and invalid data

### ✅ scripts/embeddings.py (Ready)
- Image embedding generation
- Batch processing

### ✅ scripts/train_model.py (Ready)
- Model training
- Hyperparameter optimization

### ✅ main.py (Ready)
- Feature selection
- PCA pipeline

---

## 🔍 Running Specific Tests

### By Module
```bash
# All recommendation_agent tests
pytest test_suite.py::TestProductModel -v
pytest test_suite.py::TestParseAndValidateResponse -v

# All data_loader tests
pytest test_suite.py::TestAgeInterval -v
pytest test_suite.py::TestApplyFilter -v
```

### By Keyword
```bash
# Tests containing "product"
pytest test_suite.py -k product -v

# Tests containing "valid"
pytest test_suite.py -k valid -v

# Tests containing "error"
pytest test_suite.py -k error -v
```

### By Marker (if using pytest.mark)
```bash
pytest test_suite.py -m unit -v        # Unit tests only
pytest test_suite.py -m integration -v # Integration tests only
```

---

## 📈 Coverage Report

### Generate Coverage Report
```bash
pytest test_suite.py \
  --cov=recommendation_agent \
  --cov=scripts \
  --cov-report=html \
  --cov-report=term-missing
```

### View Results
```bash
# HTML report
open htmlcov/index.html

# Terminal report
pytest test_suite.py --cov --cov-report=term-missing
```

### Coverage by Module
```bash
pytest test_suite.py --cov=recommendation_agent --cov-report=term-missing
pytest test_suite.py --cov=scripts.data_loader --cov-report=term-missing
```

---

## 🐛 Debugging Failed Tests

### Show Print Output
```bash
pytest test_suite.py -s -v
```

### Show Full Traceback
```bash
pytest test_suite.py -vv --tb=long
```

### Stop on First Failure
```bash
pytest test_suite.py -x
```

### Enter Debugger on Failure
```bash
pytest test_suite.py --pdb
```

### Verbose with Timing
```bash
pytest test_suite.py -v --durations=10
```

---

## 📦 Using Fixtures

Tests include pre-built fixtures for common scenarios:

```python
def test_example(valid_product_data):
    """Fixture provides valid product data"""
    from recommendation_agent import Product
    product = Product(**valid_product_data)
    assert product.relevance_score == 0.95

def test_response(valid_recommendation_response_data):
    """Fixture provides complete recommendation"""
    from recommendation_agent import RecommendationResponse
    response = RecommendationResponse(**valid_recommendation_response_data)
    assert response.product_category == "Clothing"

def test_dataframe(sample_data_small):
    """Fixture provides sample DataFrame"""
    from scripts.data_loader import apply_filter
    filtered = apply_filter(sample_data_small)
    assert isinstance(filtered, pd.DataFrame)
```

---

## 🔄 Continuous Testing

### Watch for Changes (requires pytest-watch)
```bash
pip install pytest-watch
ptw test_suite.py
```

### Run Tests in Parallel (requires pytest-xdist)
```bash
pip install pytest-xdist
pytest test_suite.py -n auto
```

### Run Tests on Git Hooks
```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Create .pre-commit-config.yaml with pytest checks
```

---

## ✨ Test Statistics

- **Total Tests**: 60+
- **Unit Tests**: 45+
- **Integration Tests**: 5+
- **Error Handling Tests**: 5+
- **Performance Tests**: 2+
- **Data Validation Tests**: 5+

---

## 🎯 Common Use Cases

### 1. Before Committing Code
```bash
pytest test_suite.py -v --tb=short
```

### 2. After Making Changes
```bash
# Run only affected tests
pytest test_suite.py -k "your_function" -v

# Or run full suite with coverage
pytest test_suite.py --cov
```

### 3. For CI/CD Pipeline
```bash
pytest test_suite.py \
  --cov \
  --cov-report=xml \
  --cov-report=html \
  -v \
  --tb=short
```

### 4. Performance Verification
```bash
pytest test_suite.py::TestPerformance -v
```

---

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [TEST_DOCUMENTATION.md](TEST_DOCUMENTATION.md) - Full test documentation
- [conftest.py](conftest.py) - Fixture definitions
- [test_suite.py](test_suite.py) - All test implementations

---

## 🆘 Troubleshooting

### ImportError: No module named 'recommendation_agent'
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest test_suite.py
```

### Tests Fail with "ModuleNotFoundError"
```bash
# Install all dependencies
pip install -e .
pip install -r requirements-test.txt
```

### Permission Denied
```bash
# Ensure pytest is executable
chmod +x $(which pytest)
```

---

## 💡 Tips

1. **Run tests frequently** - Catch issues early
2. **Check coverage** - Aim for 85%+ coverage
3. **Use fixtures** - Reduces test code duplication
4. **Mock external APIs** - Makes tests faster and reliable
5. **Keep tests isolated** - No dependencies between tests

---

## ✅ Success Indicators

```
✓ All tests passing (PASSED)
✓ Coverage > 85%
✓ No warnings or errors
✓ <5 seconds total runtime (with XDist)
✓ No flaky tests (consistent results)
```

---

## 🚀 Next Steps

1. ✅ Install dependencies: `pip install -r requirements-test.txt`
2. ✅ Run tests: `pytest test_suite.py -v`
3. ✅ Check coverage: `pytest test_suite.py --cov`
4. ✅ Add new tests when adding features
5. ✅ Monitor coverage metrics

Happy testing! 🎉
