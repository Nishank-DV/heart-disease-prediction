# Edge Cases and Medical Scenarios

## Comprehensive Edge Case Documentation

---

## 1. MEDICAL DATA EDGE CASES

### 1.1 Age Edge Cases

| Case | Age Value | Medical Context | Expected Behavior |
|------|-----------|-----------------|-------------------|
| EC-Age-001 | 1 | Infant | Validation error (unrealistic) |
| EC-Age-002 | 25 | Very young adult | Valid, low risk expected |
| EC-Age-003 | 120 | Maximum age | Valid, but may need special handling |
| EC-Age-004 | 0 | Invalid | Validation error |
| EC-Age-005 | 150 | Invalid | Validation error |

### 1.2 Blood Pressure Edge Cases

| Case | BP Value | Medical Context | Expected Behavior |
|------|----------|-----------------|-------------------|
| EC-BP-001 | 50 | Hypotension | Valid, but unusual |
| EC-BP-002 | 250 | Severe hypertension | Valid, high risk indicator |
| EC-BP-003 | 30 | Critical hypotension | Validation error (too low) |
| EC-BP-004 | 300 | Extreme hypertension | Validation error (too high) |

### 1.3 Cholesterol Edge Cases

| Case | Chol Value | Medical Context | Expected Behavior |
|------|------------|-----------------|-------------------|
| EC-Chol-001 | 100 | Low cholesterol | Valid |
| EC-Chol-002 | 600 | Very high cholesterol | Valid, high risk |
| EC-Chol-003 | 50 | Unrealistically low | Validation error |
| EC-Chol-004 | 700 | Unrealistically high | Validation error |

### 1.4 Heart Rate Edge Cases

| Case | HR Value | Medical Context | Expected Behavior |
|------|----------|-----------------|-------------------|
| EC-HR-001 | 50 | Low resting HR | Valid (athlete) |
| EC-HR-002 | 200 | Very high HR | Valid, may indicate stress |
| EC-HR-003 | 250 | Maximum valid | Valid |
| EC-HR-004 | 300 | Unrealistic | Validation error |

---

## 2. COMBINATION EDGE CASES

### 2.1 High Risk Combinations

| Scenario | Features | Expected Result |
|----------|----------|-----------------|
| EC-Comb-001 | Age 80 + BP 200 + Chol 400 | High probability (>0.8) |
| EC-Comb-002 | All risk factors maximum | Very high probability (>0.9) |
| EC-Comb-003 | Age 70 + Chest Pain + ST Depression | High probability |

### 2.2 Low Risk Combinations

| Scenario | Features | Expected Result |
|----------|----------|-----------------|
| EC-Comb-004 | Age 30 + Normal BP + Normal Chol | Low probability (<0.3) |
| EC-Comb-005 | All values normal/minimal | Very low probability (<0.2) |

### 2.3 Conflicting Indicators

| Scenario | Features | Expected Behavior |
|----------|----------|-------------------|
| EC-Comb-006 | Young age but high risk factors | Medium probability |
| EC-Comb-007 | Old age but healthy values | Medium-Low probability |

---

## 3. SYSTEM EDGE CASES

### 3.1 Model Edge Cases

| Case | Scenario | Expected Behavior |
|------|----------|-------------------|
| EC-Model-001 | Model file not found | Use random initialization, warn user |
| EC-Model-002 | Corrupted model file | Error message, graceful failure |
| EC-Model-003 | Model version mismatch | Error or compatibility handling |

### 3.2 Database Edge Cases

| Case | Scenario | Expected Behavior |
|------|----------|-------------------|
| EC-DB-001 | Database file locked | Retry or error message |
| EC-DB-002 | Disk full | Error message, no data loss |
| EC-DB-003 | Concurrent writes | Database handles locking |
| EC-DB-004 | Very large dataset | Pagination works correctly |

### 3.3 API Edge Cases

| Case | Scenario | Expected Behavior |
|------|----------|-------------------|
| EC-API-001 | Missing required field | 422 Validation Error |
| EC-API-002 | Extra unknown fields | Accepted (ignored) |
| EC-API-003 | Malformed JSON | 422 Validation Error |
| EC-API-004 | Empty request body | 422 Validation Error |
| EC-API-005 | Very large payload | 413 Payload Too Large |

---

## 4. BOUNDARY VALUE TESTING

### 4.1 Minimum Values

| Field | Min Value | Valid? | Notes |
|-------|-----------|--------|-------|
| age | 1 | ✅ | Minimum realistic age |
| trestbps | 50 | ✅ | Minimum valid BP |
| chol | 100 | ✅ | Minimum valid cholesterol |
| oldpeak | 0.0 | ✅ | No ST depression |

### 4.2 Maximum Values

| Field | Max Value | Valid? | Notes |
|-------|-----------|--------|-------|
| age | 120 | ✅ | Maximum realistic age |
| trestbps | 250 | ✅ | Maximum valid BP |
| chol | 600 | ✅ | Maximum valid cholesterol |
| oldpeak | 10.0 | ✅ | Maximum ST depression |

### 4.3 Boundary +1/-1 Tests

| Field | Boundary-1 | Boundary | Boundary+1 |
|-------|------------|----------|------------|
| age | 0 (invalid) | 1 (valid) | 2 (valid) |
| age | 119 (valid) | 120 (valid) | 121 (invalid) |
| sex | -1 (invalid) | 0 (valid) | 1 (valid) |
| sex | 0 (valid) | 1 (valid) | 2 (invalid) |

---

## 5. DATA TYPE EDGE CASES

| Case | Input Type | Expected Behavior |
|------|------------|-------------------|
| EC-Type-001 | String instead of int | 422 Validation Error |
| EC-Type-002 | Float instead of int | Coerced or error |
| EC-Type-003 | Null/None values | 422 Validation Error |
| EC-Type-004 | Boolean instead of int | Coerced or error |
| EC-Type-005 | Array instead of single value | 422 Validation Error |

---

## 6. MEDICAL REALITY CHECKS

### 6.1 Realistic Medical Scenarios

| Scenario | Description | Expected Probability Range |
|-----------|-------------|---------------------------|
| Healthy 30-year-old | All normal values | 0.0 - 0.2 |
| 50-year-old with mild risk | Slightly elevated BP/Chol | 0.3 - 0.5 |
| 65-year-old with risk factors | High BP, high Chol, chest pain | 0.6 - 0.8 |
| 75-year-old with multiple risks | All risk factors present | 0.8 - 1.0 |

### 6.2 Unusual but Valid Cases

| Case | Description | Handling |
|------|-------------|----------|
| Athlete with low HR | HR=50, but healthy | Model should handle |
| Very old but healthy | Age=85, all normal | Medium probability |
| Young with genetic risk | Age=30, high Chol | Medium probability |

---

## 7. ERROR SCENARIOS

### 7.1 Input Errors

| Error Type | Example | Expected Response |
|------------|---------|-------------------|
| Missing field | No 'age' field | 422 with field name |
| Wrong type | age="sixty" | 422 with type error |
| Out of range | age=200 | 422 with range error |
| Negative value | age=-10 | 422 validation error |

### 7.2 System Errors

| Error Type | Scenario | Expected Response |
|------------|----------|-------------------|
| Model not loaded | No model file | 503 Service Unavailable |
| Database error | Connection failed | 500 with error message |
| Prediction failure | Model error | 500 with details |

---

## 8. PERFORMANCE EDGE CASES

| Case | Scenario | Expected Performance |
|------|----------|---------------------|
| PERF-001 | Single prediction | < 100ms |
| PERF-002 | 100 concurrent requests | All complete < 10s |
| PERF-003 | Large database (10K records) | Query < 1s |
| PERF-004 | Batch prediction (1000) | All complete < 60s |

---

## 9. SECURITY EDGE CASES

| Case | Scenario | Expected Behavior |
|------|----------|-------------------|
| SEC-001 | SQL injection attempt | Input sanitized, safe |
| SEC-002 | XSS in input | Input validated |
| SEC-003 | Path traversal | Request rejected |
| SEC-004 | Large payload attack | Request size limited |

---

## 10. TEST EXECUTION RESULTS

### Expected Test Outcomes

```
test_model.py::TestModel::test_model_creation PASSED
test_model.py::TestModel::test_forward_pass PASSED
test_api.py::TestAPI::test_predict_success PASSED
test_api.py::TestAPI::test_predict_invalid_age PASSED
test_database.py::TestDatabase::test_create_prediction PASSED
...

======================== 28 passed in 5.23s ========================
```

---

**All edge cases documented and testable!**

