#!/bin/bash
#
# Compile Test Script
# Verifies that all Python files compile successfully
#

echo "======================================================================"
echo "SAM3-TensorRT Compilation Test"
echo "======================================================================"
echo ""

FAILED=0
PASSED=0

# Test each Python file
for file in $(find . -name "*.py" -type f | grep -v __pycache__ | grep -v ".pyc"); do
    echo -n "Testing $file... "
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo "✓ PASS"
        ((PASSED++))
    else
        echo "✗ FAIL"
        ((FAILED++))
    fi
done

echo ""
echo "======================================================================"
echo "Compilation Summary"
echo "======================================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "======================================================================"

if [ $FAILED -eq 0 ]; then
    echo "✓ All files compile successfully!"
    exit 0
else
    echo "✗ Some files failed to compile"
    exit 1
fi
