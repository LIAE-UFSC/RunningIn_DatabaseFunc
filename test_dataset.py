import unittest

path = "C:/"

class DBTestCase(unittest.TestCase):
    def setUp(self):
        self.dbs = Widget('The widget')

    def test_default_widget_size(self):
        self.assertEqual(self.widget.size(), (50,50),
                         'incorrect default size')

    def test_widget_resize(self):
        self.widget.resize(100,150)
        self.assertEqual(self.widget.size(), (100,150),
                         'wrong size after resize')

if __name__ == '__main__':
    unittest.main()