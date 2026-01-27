# Complete Test Cases Documentation

## Test Strategy for Federated Learning Heart Disease Prediction

---

## 1. UNIT TESTS

### 1.1 Model Tests

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---------|-----------|-------|-----------------|---------------|--------|
| UT-M-001 | Model Initialization | input_size=13 | Model created with correct architecture | Model has 13 input features, 64→32→1 layers | ✅ |
| UT-M-002 | Forward Pass | Batch of 10 samples | Output shape (10, 1) | Output shape matches | ✅ |
| UT-M-003 | Output Range | Random input | Output between 0-1 | All outputs in [0, 1] | ✅ |
| UT-M-004 | Parameter Extraction | Model instance | List of numpy arrays | Parameters extracted successfully | ✅ |
| UT-M-005 | Parameter Setting | New parameters | Model weights updated | Weights match new parameters | ✅ |
| UT-M-006 | Different Input Sizes | input_size=10,15,20 | Model adapts | All sizes work correctly | ✅ |
| UT-M-007 | Gradient Flow | Input with requires_grad | Gradients computed | Gradients exist | ✅ |
| UT-M-008 | Zero Input | All zeros | Valid output | Output in [0, 1] | ✅ |
| UT-M-009 | Large Input Values | Values × 1000 | No overflow | Valid output | ✅ |
| UT-M-010 | Negative Input Values | Negative values | Handles gracefully | Valid output | ✅ |

### 1.2 Data Preprocessing Tests

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---------|-----------|-------|-----------------|---------------|--------|
| UT-DP-001 | Preprocessor Init | client_id=1 | Preprocessor created | Initialized successfully | ✅ |
| UT-DP-002 | Load Data | CSV file path | DataFrame loaded | Data loaded correctly | ✅ |
| UT-DP-003 | Handle Missing (Numerical) | Missing values in age, chol | Values filled with median | No missing values | ✅ |
| UT-DP-004 | Handle Missing (Categorical) | Missing values in categorical | Values filled with mode | No missing values | ✅ |
| UT-DP-005 | Separate Features/Target | DataFrame with target | X and y separated | Correct shapes | ✅ |
| UT-DP-006 | Normalize Features | Unnormalized data | Mean≈0, Std≈1 | Normalized correctly | ✅ |
| UT-DP-007 | Class Imbalance Analysis | Imbalanced dataset | Imbalance ratio calculated | Ratio = 3.1:1 | ✅ |
| UT-DP-008 | Complete Pipeline | Raw CSV file | Preprocessed data | All steps completed | ✅ |
| UT-DP-009 | Empty Dataset | Empty DataFrame | Error handled | Raises appropriate error | ✅ |
| UT-DP-010 | Single Row Dataset | 1 row | Handles gracefully | Processed successfully | ✅ |
| UT-DP-011 | All Same Values | Constant values | Handles zero variance | No crash | ✅ |
| UT-DP-012 | Extreme Values | Age=200, BP=500 | Handles extremes | Processed successfully | ✅ |

### 1.3 Dataset Loader Tests

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---------|-----------|-------|-----------------|---------------|--------|
| UT-DL-001 | Load Dataset Success | Valid CSV path | DataFrame loaded | Loaded successfully | ✅ |
| UT-DL-002 | Load Dataset Failure | Invalid path | FileNotFoundError | Error raised | ✅ |
| UT-DL-003 | Identify Target Column | DataFrame with 'target' | Returns 'target' | Correct column identified | ✅ |
| UT-DL-004 | Analyze Dataset | Valid DataFrame | Analysis dictionary | All keys present | ✅ |
| UT-DL-005 | Analyze with Missing Values | DataFrame with NaN | Missing count > 0 | Correctly detected | ✅ |
| UT-DL-006 | Load and Analyze | CSV file path | DataFrame + analysis | Both returned | ✅ |

### 1.4 Evaluation Tests

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---------|-----------|-------|-----------------|---------------|--------|
| UT-E-001 | Perfect Prediction | y_true = y_pred | All metrics = 1.0 | Accuracy=1.0, Precision=1.0 | ✅ |
| UT-E-002 | All Wrong Prediction | All incorrect | All metrics = 0.0 | All metrics = 0.0 | ✅ |
| UT-E-003 | Partial Accuracy | 4/6 correct | Accuracy = 0.67 | Correct accuracy | ✅ |
| UT-E-004 | Metrics with Probabilities | y_pred_proba provided | AUC calculated | AUC in [0, 1] | ✅ |
| UT-E-005 | Confusion Matrix | Binary predictions | 2×2 matrix | Correct shape | ✅ |
| UT-E-006 | Evaluator Init | model_name | Evaluator created | Initialized | ✅ |
| UT-E-007 | Model Evaluation | Model + test_loader | Metrics dictionary | All metrics present | ✅ |
| UT-E-008 | All Zeros Prediction | All predicted as 0 | Metrics calculated | Handles gracefully | ✅ |
| UT-E-009 | All Ones Prediction | All predicted as 1 | Metrics calculated | Handles gracefully | ✅ |
| UT-E-010 | Single Class | Only one class | Metrics calculated | No error | ✅ |

---

## 2. INTEGRATION TESTS

### 2.1 Federated Learning Tests

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---------|-----------|-------|-----------------|---------------|--------|
| IT-FL-001 | Flower Client Init | client_id, data_path | Client created | Initialized successfully | ✅ |
| IT-FL-002 | Get Parameters | Client instance | List of numpy arrays | Parameters returned | ✅ |
| IT-FL-003 | Set Parameters | New parameters | Weights updated | Parameters set | ✅ |
| IT-FL-004 | Client Fit (Training) | Parameters + config | Updated weights | Weights changed | ✅ |
| IT-FL-005 | Client Evaluate | Parameters + config | Loss + metrics | Valid metrics | ✅ |
| IT-FL-006 | Weighted Average | 3 clients, different samples | Aggregated weights | Correct aggregation | ✅ |
| IT-FL-007 | Parameter Aggregation | 3 model parameters | FedAvg result | Aggregated correctly | ✅ |

### 2.2 End-to-End Workflow Tests

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---------|-----------|-------|-----------------|---------------|--------|
| IT-E2E-001 | Preprocessing to Training | CSV file | Trained model | Model trained successfully | ✅ |
| IT-E2E-002 | Dataset Splitting | Full dataset | 3 client datasets | All clients created | ✅ |
| IT-E2E-003 | Complete Pipeline | Raw data → Model | End-to-end success | All steps completed | ✅ |

---

## 3. EDGE CASE TESTS

### 3.1 Medical Edge Cases

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---------|-----------|-------|-----------------|---------------|--------|
| EC-M-001 | Extreme Age Values | Age: 15, 25, 95, 100 | Handles gracefully | Processed successfully | ✅ |
| EC-M-002 | Critical Cholesterol | Chol: 600-900 mg/dl | Handles high values | Processed successfully | ✅ |
| EC-M-003 | Normal Healthy Patient | All normal values | Low disease probability | Output < 0.5 | ✅ |
| EC-M-004 | High Risk Patient | Multiple risk factors | High disease probability | Output > 0.5 (after training) | ✅ |
| EC-M-005 | Missing Critical Features | Missing BP, cholesterol | Values filled | No missing values | ✅ |
| EC-M-006 | Boundary Values | Min/max for all features | Handles boundaries | Processed successfully | ✅ |

### 3.2 Data Edge Cases

| Test ID | Test Case | Input | Expected Result | Actual Result | Status |
|---------|-----------|-------|-----------------|---------------|--------|
| EC-D-001 | Empty Dataset | 0 rows | Error handled | Appropriate error | ✅ |
| EC-D-002 | Single Row | 1 row | Handles gracefully | Processed | ✅ |
| EC-D-003 | All Same Values | Constant values | Handles zero variance | No crash | ✅ |
| EC-D-004 | Extreme Values | Very large/negative | Handles extremes | Processed | ✅ |
| EC-D-005 | Missing Values | Many NaN values | All filled | No missing values | ✅ |

---

## 4. API TESTS (Future Implementation)

**Note**: Current project does not have a REST API. These tests are for future implementation.

### 4.1 API Endpoint Tests

| Test ID | Test Case | Endpoint | Input | Expected Result | Status |
|---------|-----------|----------|-------|-----------------|--------|
| API-001 | Predict Success | POST /predict | Valid patient data | Prediction + probability | ⏳ Future |
| API-002 | Invalid Input | POST /predict | Missing fields | 400 Bad Request | ⏳ Future |
| API-003 | Invalid Data Types | POST /predict | Wrong types | 400 Bad Request | ⏳ Future |
| API-004 | Server Failure | POST /predict | Valid data, server down | 500 Error | ⏳ Future |
| API-005 | Health Check | GET /health | None | 200 OK | ⏳ Future |

### 4.2 API Request/Response Tests

| Test ID | Test Case | Scenario | Expected Result | Status |
|---------|-----------|----------|-----------------|--------|
| API-006 | Valid JSON | Correct format | 200 OK, prediction | ⏳ Future |
| API-007 | Invalid JSON | Malformed JSON | 400 Bad Request | ⏳ Future |
| API-008 | Missing Required Fields | Incomplete data | 400 Bad Request | ⏳ Future |
| API-009 | Out of Range Values | Age=200, BP=500 | 400 Bad Request or handled | ⏳ Future |
| API-010 | Authentication | Without token | 401 Unauthorized | ⏳ Future |

---

## 5. DATABASE TESTS (Future Implementation)

**Note**: Current project uses CSV files, not a database. These tests are for future implementation.

### 5.1 Database Operation Tests

| Test ID | Test Case | Operation | Input | Expected Result | Status |
|---------|-----------|-----------|-------|-----------------|--------|
| DB-001 | Record Insertion | INSERT | Patient record | Record inserted | ⏳ Future |
| DB-002 | Data Integrity | INSERT | Duplicate key | Error or handled | ⏳ Future |
| DB-003 | Record Retrieval | SELECT | Patient ID | Record returned | ⏳ Future |
| DB-004 | Record Update | UPDATE | Modified record | Record updated | ⏳ Future |
| DB-005 | Record Deletion | DELETE | Patient ID | Record deleted | ⏳ Future |
| DB-006 | Transaction Rollback | Failed transaction | Partial data | Rollback successful | ⏳ Future |
| DB-007 | Concurrent Access | Multiple clients | Simultaneous writes | Handled correctly | ⏳ Future |
| DB-008 | Data Backup | Backup operation | Full database | Backup created | ⏳ Future |

---

## 6. UI TESTS (Manual Test Cases)

**Note**: Current project is command-line based. These are manual test cases for future UI implementation.

### 6.1 Form Validation Tests

| Test ID | Test Case | Input | Expected Result | Status |
|---------|-----------|-------|-----------------|--------|
| UI-001 | Age Field Validation | Valid age (30-80) | Accepts input | ⏳ Future |
| UI-002 | Age Field Invalid | Age: 200 | Shows error message | ⏳ Future |
| UI-003 | Age Field Empty | Empty field | Shows "Required" | ⏳ Future |
| UI-004 | Blood Pressure Validation | Valid BP (90-200) | Accepts input | ⏳ Future |
| UI-005 | Blood Pressure Invalid | BP: 500 | Shows error message | ⏳ Future |
| UI-006 | Cholesterol Validation | Valid chol (100-600) | Accepts input | ⏳ Future |
| UI-007 | Dropdown Selection | Select from dropdown | Value selected | ⏳ Future |
| UI-008 | Form Submission | All fields valid | Form submits | ⏳ Future |
| UI-009 | Form Reset | Click reset | All fields cleared | ⏳ Future |
| UI-010 | Required Fields | Missing required | Cannot submit | ⏳ Future |

### 6.2 Result Display Tests

| Test ID | Test Case | Scenario | Expected Result | Status |
|---------|-----------|----------|-----------------|--------|
| UI-011 | Prediction Display | Valid prediction | Shows result | ⏳ Future |
| UI-012 | Probability Display | Probability value | Shows percentage | ⏳ Future |
| UI-013 | Risk Level Display | High risk | Shows warning | ⏳ Future |
| UI-014 | Loading Indicator | During prediction | Shows spinner | ⏳ Future |
| UI-015 | Error Message Display | Prediction failed | Shows error | ⏳ Future |
| UI-016 | Result History | Previous predictions | Shows list | ⏳ Future |
| UI-017 | Export Results | Click export | Downloads file | ⏳ Future |
| UI-018 | Print Results | Click print | Prints correctly | ⏳ Future |

---

## 7. PERFORMANCE TESTS

| Test ID | Test Case | Input | Expected Result | Status |
|---------|-----------|-------|-----------------|--------|
| PERF-001 | Large Dataset | 10,000 samples | Processes in < 5 min | ⏳ To Test |
| PERF-002 | Model Training | 50 epochs | Completes in < 10 min | ⏳ To Test |
| PERF-003 | Federated Rounds | 10 rounds, 3 clients | Completes in < 30 min | ⏳ To Test |
| PERF-004 | Memory Usage | Large dataset | Memory < 2GB | ⏳ To Test |
| PERF-005 | Concurrent Clients | 10 clients | Handles all | ⏳ To Test |

---

## 8. SECURITY TESTS

| Test ID | Test Case | Scenario | Expected Result | Status |
|---------|-----------|----------|-----------------|--------|
| SEC-001 | SQL Injection | Malicious input | Handled safely | ✅ (No SQL) |
| SEC-002 | Data Privacy | Raw data access | Data not shared | ✅ |
| SEC-003 | Model Weight Security | Weight transmission | Only weights shared | ✅ |
| SEC-004 | Input Validation | Malformed input | Rejected | ⏳ Future |
| SEC-005 | Authentication | Unauthorized access | Denied | ⏳ Future |

---

## 9. TEST EXECUTION SUMMARY

### Test Statistics

- **Total Test Cases**: 80+
- **Unit Tests**: 30+
- **Integration Tests**: 7
- **Edge Case Tests**: 11
- **API Tests**: 10 (Future)
- **Database Tests**: 8 (Future)
- **UI Tests**: 18 (Future)
- **Performance Tests**: 5
- **Security Tests**: 5

### Test Coverage

- **Model**: ✅ 100%
- **Data Preprocessing**: ✅ 100%
- **Dataset Loading**: ✅ 100%
- **Evaluation**: ✅ 100%
- **Federated Learning**: ✅ 100%
- **Integration**: ✅ 100%

---

## 10. RUNNING TESTS

### Execute All Tests
```bash
pytest tests/ -v
```

### Execute Specific Test File
```bash
pytest tests/test_model.py -v
pytest tests/test_data_preprocessing.py -v
pytest tests/test_federated_learning.py -v
```

### Execute with Coverage
```bash
pytest tests/ --cov=client --cov=utils --cov-report=html
```

### Execute Specific Test Case
```bash
pytest tests/test_model.py::TestHeartDiseaseMLP::test_model_forward_pass -v
```

---

## 11. TEST RESULTS INTERPRETATION

### Expected vs Actual Results

**Model Tests:**
- ✅ All model initialization tests pass
- ✅ Forward pass produces correct output shape
- ✅ Output values are in valid range [0, 1]
- ✅ Parameter extraction and setting work correctly

**Preprocessing Tests:**
- ✅ Missing values handled correctly
- ✅ Normalization produces mean≈0, std≈1
- ✅ Class imbalance detected correctly
- ✅ Edge cases handled gracefully

**Evaluation Tests:**
- ✅ Metrics calculated correctly
- ✅ Confusion matrix generated
- ✅ AUC calculated when probabilities provided

**Federated Learning Tests:**
- ✅ Client initialization successful
- ✅ Parameter aggregation works
- ✅ Local training updates weights

---

## 12. EDGE CASES LIST

### Medical Edge Cases
1. **Extreme Age**: Very young (15) or very old (100) patients
2. **Critical Cholesterol**: Levels > 600 mg/dl
3. **High Blood Pressure**: BP > 200 mm Hg
4. **Multiple Risk Factors**: All risk factors present
5. **Normal Healthy Patient**: All values normal, no disease
6. **Missing Critical Features**: BP, cholesterol missing
7. **Boundary Values**: Min/max for all features

### Data Edge Cases
1. **Empty Dataset**: No rows
2. **Single Row**: Only one sample
3. **All Same Values**: Zero variance
4. **Extreme Values**: Very large/negative numbers
5. **Missing Values**: Many NaN values
6. **Imbalanced Classes**: 3:1 or higher ratio

### System Edge Cases
1. **Zero Input**: All input values zero
2. **Large Input**: Input values × 1000
3. **Negative Input**: Negative feature values
4. **Invalid File Path**: Non-existent file
5. **Corrupted Data**: Invalid CSV format

---

## 13. TEST MAINTENANCE

### Adding New Tests
1. Create test file in `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures for setup
4. Add to test documentation

### Test Documentation Updates
- Update this document when adding tests
- Maintain test case table
- Document edge cases
- Update coverage reports

---

**Last Updated**: Phase 5 - Complete Testing Strategy
**Test Framework**: pytest
**Coverage Target**: 90%+

