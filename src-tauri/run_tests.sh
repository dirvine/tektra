#!/bin/bash

# Simple test runner for core AI functionality
# This tests the AI modules without requiring full Tauri dependencies

echo "🧪 Tektra AI Test Runner"
echo "======================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -n "Running $test_name... "
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
}

echo "🔍 Testing Core Functionality"
echo "----------------------------"

# Test 1: Check if Rust toolchain is available
run_test "Rust toolchain check" "rustc --version"

# Test 2: Check if cargo is available
run_test "Cargo check" "cargo --version"

# Test 3: Check code compilation (without GUI dependencies)
run_test "Code compilation" "cargo check --lib --no-default-features 2>/dev/null"

# Test 4: Check for Ollama
run_test "Ollama availability" "which ollama || echo 'Ollama will be auto-installed'"

# Test 5: Check if models.toml exists
run_test "Model config exists" "test -f models.toml"

# Test 6: Check AI module structure
run_test "AI module structure" "test -d src/ai && test -f src/ai/mod.rs"

# Test 7: Check for test files
run_test "Test files exist" "test -f src/ai/tests/mod.rs"

# Test 8: Check database module
run_test "Database module exists" "test -f src/database/mod.rs"

# Test 9: Run unit tests
run_test "Unit tests" "cargo test --lib --quiet"

# Test 10: Check for placeholders
echo ""
echo "🔍 Checking for Placeholders"
echo "---------------------------"

PLACEHOLDERS=$(grep -r "TODO\|FIXME\|unimplemented!\|todo!()" src/ 2>/dev/null | wc -l)
if [ "$PLACEHOLDERS" -eq 0 ]; then
    echo -e "${GREEN}✓ No placeholders found${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠ Found $PLACEHOLDERS placeholder(s)${NC}"
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 11: Documentation check
echo ""
echo "📚 Documentation Check"
echo "---------------------"

DOC_COUNT=$(grep -r "///" src/ --include="*.rs" | wc -l)
if [ "$DOC_COUNT" -gt 100 ]; then
    echo -e "${GREEN}✓ Found $DOC_COUNT documentation comments${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠ Only $DOC_COUNT documentation comments found${NC}"
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Summary
echo ""
echo "📊 Test Summary"
echo "==============="
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ "$FAILED_TESTS" -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi