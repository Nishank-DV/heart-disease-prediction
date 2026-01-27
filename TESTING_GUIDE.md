# Complete Testing Guide

## Testing Strategy for Federated Learning Heart Disease Prediction

---

## 📋 Overview

This document provides a comprehensive testing strategy for the federated learning heart disease prediction project. The testing approach covers unit tests, integration tests, edge cases, and future API/UI tests.

---

## 🧪 Test Structure

```
tests/
├── __init__.py
├── test_model.py              # Model unit tests
├── test_data_preprocessing.py  # Preprocessing unit tests
├── test_dataset_loader.py     # Dataset loading tests
├── test_federated_learning.py  # Federated learning tests
├── test_evaluation.py          # Evaluation tests
├── test_integration.py         # Integration tests
└── test_edge_cases.py          # Edge case tests
```

---

## 🚀 Running Tests

### Install Test Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
# Using pytest directly
pytest tests/ -v

# Using test runner script
python run_tests.py
```

### Run Specific Test File
```bash
pytest tests/test_model.py -v
pytest tests/test_data_preprocessing.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=client --cov=utils --cov-report=html
python run_tests.py --coverage
```

### Run Specific Test Case
```bash
pytest tests/test_model.py::TestHeartDiseaseMLP::test_model_forward_pass -v
```

---

## 📊 Test Categories

### 1. Unit Tests

**Purpose**: Test individual components in isolation

**Files**:
- `test_model.py` - Model architecture and operations
- `test_data_preprocessing.py` - Data preprocessing functions
- `test_dataset_loader.py` - Dataset loading and analysis
- `test_evaluation.py` - Evaluation metrics calculation

**Coverage**:
- ✅ Model initialization and forward pass
- ✅ Data preprocessing pipeline
- ✅ Feature normalization
- ✅ Metric calculations
- ✅ Parameter handling

### 2. Integration Tests

**Purpose**: Test component interactions

**Files**:
- `test_federated_learning.py` - Federated learning workflow
- `test_integration.py` - End-to-end workflows

**Coverage**:
- ✅ Client-server communication
- ✅ Federated averaging
- ✅ Complete preprocessing-to-training pipeline
- ✅ Dataset splitting workflow

### 3. Edge Case Tests

**Purpose**: Test boundary conditions and medical edge cases

**Files**:
- `test_edge_cases.py` - Medical and data edge cases

**Coverage**:
- ✅ Extreme medical values
- ✅ Missing data
- ✅ Boundary conditions
- ✅ Invalid inputs

---

## 🏥 Medical Edge Cases Tested

### Realistic Medical Scenarios

1. **Extreme Age Values**
   - Very young patients (age 15-25)
   - Very old patients (age 90-100)
   - **Expected**: System handles gracefully

2. **Critical Cholesterol Levels**
   - Very high cholesterol (>600 mg/dl)
   - **Expected**: Processed without error

3. **High-Risk Patient Profile**
   - Multiple risk factors present
   - **Expected**: High disease probability prediction

4. **Normal Healthy Patient**
   - All values in normal range
   - **Expected**: Low disease probability

5. **Missing Critical Features**
   - Missing blood pressure, cholesterol
   - **Expected**: Values filled appropriately

6. **Boundary Values**
   - Min/max values for all features
   - **Expected**: Handled correctly

---

## 📈 Test Results Interpretation

### Success Criteria

**Unit Tests**:
- ✅ All model tests pass
- ✅ Preprocessing handles all cases
- ✅ Metrics calculated correctly

**Integration Tests**:
- ✅ Federated learning workflow completes
- ✅ Client-server communication works
- ✅ End-to-end pipeline successful

**Edge Cases**:
- ✅ No crashes on extreme values
- ✅ Missing data handled
- ✅ Boundary conditions work

---

## 🔍 Test Case Examples

### Example 1: Model Forward Pass Test
```python
def test_model_forward_pass(self):
    model = HeartDiseaseMLP(input_size=13)
    input_tensor = torch.randn(10, 13)
    output = model(input_tensor)
    
    assert output.shape == (10, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)
```

### Example 2: Missing Values Handling
```python
def test_handle_missing_values_numerical(self):
    df = create_dataset_with_missing_values()
    preprocessor = DataPreprocessor(client_id=1)
    cleaned_df = preprocessor.handle_missing_values(df)
    
    assert cleaned_df.isnull().sum().sum() == 0
```

### Example 3: Medical Edge Case
```python
def test_critical_cholesterol_levels(self):
    # Test with cholesterol > 600 mg/dl
    data = create_high_cholesterol_data()
    preprocessor = DataPreprocessor(client_id=1)
    X, y = preprocessor.separate_features_and_target(data)
    
    assert X.shape[0] > 0  # Should process successfully
```

---

## 📝 Test Documentation

### Test Case Format

Each test case includes:
- **Test ID**: Unique identifier
- **Test Case**: Description
- **Input**: Test data
- **Expected Result**: What should happen
- **Actual Result**: What actually happened
- **Status**: Pass/Fail

### Test Coverage Report

After running tests with coverage:
```bash
pytest tests/ --cov=client --cov=utils --cov-report=html
```

Open `htmlcov/index.html` to view coverage report.

---

## ⚠️ Important Notes

### Current Project Limitations

1. **No REST API**: API tests are documented for future implementation
2. **No Database**: Database tests are for future SQL database integration
3. **No UI**: UI tests are manual test cases for future web interface

### What is Tested

✅ **Fully Tested**:
- Model architecture and operations
- Data preprocessing pipeline
- Dataset loading and splitting
- Evaluation metrics
- Federated learning components
- Edge cases and boundary conditions

⏳ **Future Tests** (when components added):
- REST API endpoints
- Database operations
- Web UI functionality

---

## 🎯 Test Execution Checklist

Before submission:
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Edge case tests pass
- [ ] Coverage > 80%
- [ ] Test documentation updated
- [ ] No test warnings

---

## 📚 Additional Resources

- **Test Cases Document**: `TEST_CASES_DOCUMENTATION.md`
- **Pytest Documentation**: https://docs.pytest.org/
- **Coverage Reports**: `htmlcov/index.html` (after running with coverage)

---

**Test Framework**: pytest
**Coverage Tool**: pytest-cov
**Target Coverage**: 90%+

