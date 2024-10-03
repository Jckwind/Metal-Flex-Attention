import unittest

def run_tests():
    # Discover and run all unit tests in the 'tests' directory
    print("Discovering and running unit tests...")
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='*_tests.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\nAll unit tests passed successfully.")
    else:
        print("\nSome unit tests failed. Please review the errors above.")

if __name__ == "__main__":
    run_tests()
