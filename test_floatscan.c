#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <float.h>

// Test helper function
int test_double(const char* input, double expected, const char* description) {
    errno = 0;
    char* endptr;
    double result = strtod(input, &endptr);

    int passed = 0;
    if (isnan(expected) && isnan(result)) {
        passed = 1;
    } else if (isinf(expected) && isinf(result) && signbit(expected) == signbit(result)) {
        passed = 1;
    } else if (result == expected) {
        passed = 1;
    } else if (fabs(result - expected) < 1e-15 * fabs(expected)) {
        passed = 1; // Allow small floating-point errors
    }

    printf("%-40s: %s (got %.17g, expected %.17g)\n",
           description, passed ? "PASS" : "FAIL", result, expected);

    if (!passed) {
        printf("  Input: \"%s\", errno: %d, endptr offset: %ld\n",
               input, errno, endptr - input);
    }

    return passed;
}

// Test scanf-style parsing (pok=0) vs strtod-style parsing (pok=1)
void test_partial_parsing() {
    printf("\n=== Testing partial parsing differences (scanf vs strtod) ===\n");

    // Test cases where scanf and strtod should behave differently
    const char* test_cases[] = {
        "123abc",      // Valid number followed by invalid chars
        "inf_extra",   // "inf" followed by invalid chars
        "nanxyz",      // "nan" followed by invalid chars
        "1.23e",       // Number with incomplete exponent
        "0x1.23g",     // Hex number with invalid hex digit
        "1.23e+",      // Exponent with sign but no digits
    };

    for (int i = 0; i < 6; i++) {
        char* endptr;
        double strtod_result = strtod(test_cases[i], &endptr);

        double scanf_result;
        int scanf_count = sscanf(test_cases[i], "%lf", &scanf_result);

        printf("Input: %-12s | strtod: %g (consumed %ld chars) | scanf: ",
               test_cases[i], strtod_result, endptr - test_cases[i]);

        if (scanf_count == 1) {
            printf("%g (success)\n", scanf_result);
        } else {
            printf("failed\n");
        }
    }
}

int main() {
    int total_tests = 0;
    int passed_tests = 0;

    printf("=== Testing basic decimal parsing ===\n");
    total_tests++; passed_tests += test_double("123.456", 123.456, "Simple decimal");
    total_tests++; passed_tests += test_double("1.23e10", 1.23e10, "Scientific notation");
    total_tests++; passed_tests += test_double("-456.789e-5", -456.789e-5, "Negative with negative exp");
    total_tests++; passed_tests += test_double("0.000123", 0.000123, "Small decimal");
    total_tests++; passed_tests += test_double("999999999999999", 999999999999999.0, "Large integer");

    printf("\n=== Testing hexadecimal parsing ===\n");
    total_tests++; passed_tests += test_double("0x1.23p4", 0x1.23p4, "Basic hex float");
    total_tests++; passed_tests += test_double("0x1.0p-1022", 0x1.0p-1022, "Very small hex");
    total_tests++; passed_tests += test_double("0XA.Bp3", 0XA.Bp3, "Uppercase hex");
    total_tests++; passed_tests += test_double("0x1p0", 1.0, "Simple hex power");
    total_tests++; passed_tests += test_double("0x0.1p4", 0x0.1p4, "Fractional hex");

    printf("\n=== Testing special values ===\n");
    total_tests++; passed_tests += test_double("inf", INFINITY, "Positive infinity");
    total_tests++; passed_tests += test_double("-infinity", -INFINITY, "Negative infinity");
    total_tests++; passed_tests += test_double("INF", INFINITY, "Uppercase infinity");
    total_tests++; passed_tests += test_double("nan", NAN, "NaN");
    total_tests++; passed_tests += test_double("NAN", NAN, "Uppercase NaN");
    total_tests++; passed_tests += test_double("-nan", NAN, "Negative NaN (still NaN)");

    printf("\n=== Testing edge cases ===\n");
    total_tests++; passed_tests += test_double("0", 0.0, "Zero");
    total_tests++; passed_tests += test_double("-0", -0.0, "Negative zero");
    total_tests++; passed_tests += test_double("0.0", 0.0, "Zero with decimal");
    total_tests++; passed_tests += test_double("1e308", 1e308, "Large number");
    total_tests++; passed_tests += test_double("4.9e-324", 4.9e-324, "Very small number");
    total_tests++; passed_tests += test_double("2.2250738585072014e-308", DBL_MIN, "DBL_MIN");
    total_tests++; passed_tests += test_double("1.7976931348623157e+308", DBL_MAX, "DBL_MAX");

    printf("\n=== Testing precision and rounding ===\n");
    total_tests++; passed_tests += test_double("0.1", 0.1, "Decimal 0.1");
    total_tests++; passed_tests += test_double("0.3333333333333333", 1.0/3.0, "1/3 approximation");
    total_tests++; passed_tests += test_double("1.23456789012345678901234567890", 1.23456789012345678901234567890, "Long decimal");

    // Test different float types
    printf("\n=== Testing different precision types ===\n");
    float f = strtof("3.14159", NULL);
    double d = strtod("3.14159", NULL);
    long double ld = strtold("3.14159", NULL);
    printf("strtof(\"3.14159\") = %.10g\n", f);
    printf("strtod(\"3.14159\") = %.17g\n", d);
    printf("strtold(\"3.14159\") = %.21Lg\n", ld);

    // Test partial parsing behavior
    test_partial_parsing();

    printf("\n=== Summary ===\n");
    printf("Passed: %d/%d tests\n", passed_tests, total_tests);

    if (passed_tests == total_tests) {
        printf("All tests PASSED! ✓\n");
        return 0;
    } else {
        printf("Some tests FAILED! ✗\n");
        return 1;
    }
}
