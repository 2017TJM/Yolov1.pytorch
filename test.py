import unittest


if __name__ == '__main__':
    suite = unittest.TestSuite()
    cases = unittest.defaultTestLoader.discover('.', '*_test.py')
    for case in cases:
        suite.addTests(case)
    
    unittest.TextTestRunner().run(suite)