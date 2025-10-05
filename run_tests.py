#!/usr/bin/env python3
"""
Test runner for all CUDA fixes and training pipeline tests.

Runs unit tests and integration tests with proper reporting and logging.
"""

import sys
import os
import unittest
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_unit_tests(verbose=2):
    """Run all unit tests."""
    print("=" * 80)
    print("RUNNING UNIT TESTS")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add unit test modules
    test_modules = [
        'tests.test_cuda_initialization',
        'tests.test_sparse_embedding',
        'tests.test_dataset_validation',
    ]

    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTests(tests)
            print(f"Loaded tests from {module}")
        except Exception as e:
            print(f"Failed to load {module}: {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_integration_tests(verbose=2):
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add integration test modules
    test_modules = [
        'tests.test_integration_model',
        'tests.test_integration_training',
    ]

    for module in test_modules:
        try:
            tests = loader.loadTestsFromName(module)
            suite.addTests(tests)
            print(f"Loaded tests from {module}")
        except Exception as e:
            print(f"Failed to load {module}: {e}")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_validation_script():
    """Run the CUDA validation script."""
    print("\n" + "=" * 80)
    print("RUNNING CUDA VALIDATION SCRIPT")
    print("=" * 80)

    import validate_cuda_fixes
    return validate_cuda_fixes.main()


def run_all_tests(verbose=2, skip_validation=False):
    """Run all tests and validation."""
    results = []

    # Run unit tests
    unit_success = run_unit_tests(verbose=verbose)
    results.append(("Unit Tests", unit_success))

    # Run integration tests
    integration_success = run_integration_tests(verbose=verbose)
    results.append(("Integration Tests", integration_success))

    # Run validation script
    if not skip_validation:
        validation_success = run_validation_script() == 0
        results.append(("CUDA Validation", validation_success))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name}: {status}")
        all_passed = all_passed and success

    print("=" * 80)

    if all_passed:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run tests for CUDA fixes")
    parser.add_argument(
        '--type',
        choices=['all', 'unit', 'integration', 'validation'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='Verbosity level (0=quiet, 1=normal, 2=verbose)'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip CUDA validation script'
    )

    args = parser.parse_args()

    if args.type == 'all':
        return run_all_tests(verbose=args.verbose, skip_validation=args.skip_validation)
    elif args.type == 'unit':
        success = run_unit_tests(verbose=args.verbose)
        return 0 if success else 1
    elif args.type == 'integration':
        success = run_integration_tests(verbose=args.verbose)
        return 0 if success else 1
    elif args.type == 'validation':
        return run_validation_script()


if __name__ == "__main__":
    sys.exit(main())
