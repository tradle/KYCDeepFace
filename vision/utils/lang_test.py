from time import sleep
from vision.utils import lang
import unittest

class LazyTest(unittest.TestCase):

    def test_lazy_api(self):
        lazy = lang.lazy(lambda: 1)
        self.assertEqual(lazy(), 1)
        self.assertEqual(lazy(), 1)

    def test_lazy(self):
        class run():
            def __init__(self):
                self.i = 0
            def __call__(self):
                self.i += 1
                return self.i

        regular = run()
        lazy = lang.lazy(regular)
        self.assertEqual(regular(), 1)
        self.assertEqual(regular(), 2)
        self.assertEqual(lazy(), 3)
        self.assertEqual(lazy(), 3)
        self.assertEqual(regular(), 4)
        self.assertEqual(lazy(), 3)

    def test_lazy_timed(self):
        class run():
            def __init__(self):
                self.i = 0
            def __call__(self):
                self.i += 1
                sleep(0.05)
                return self.i
        
        regular = run()
        timings = {}
        lazy = lang.lazy_timed(timings, 'test', regular)
        self.assertNotIn('test', timings)
        self.assertEqual(regular(), 1)
        self.assertEqual(regular(), 2)
        self.assertEqual(lazy(), 3)
        self.assertIn('test', timings)
        self.assertAlmostEqual(timings['test'], 0.05, 1)
        sleep(0.05)
        self.assertEqual(lazy(), 3)
        self.assertAlmostEqual(timings['test'], 0.05, 1)
        self.assertEqual(regular(), 4)

    def test_timed(self):
        class regular():
            def __call__(self):
                sleep(0.05)
                return 1
        
        timings = {}
        run = regular()
        self.assertNotIn('test', timings)
        self.assertEqual(lang.timed(timings, 'test', run), 1)
        self.assertIn('test', timings)
        self.assertAlmostEqual(timings['test'], 0.05, 1)

    def test_timed_gen(self):
        def generator(len):
            for i in range(0, len):
                sleep(0.05)
                yield i

        self.assertListEqual([x for x in generator(3)], [0, 1, 2])
        timings = {}
        self.assertListEqual([x for x in lang.timed_iter(timings, 'test', generator(3))], [0, 1, 2])
        self.assertIn('test', timings)
        self.assertAlmostEqual(timings['test'], 0.15, 1)

class LangTest(unittest.TestCase):

    def test_optional(self):
        self.assertEqual(lang.optional_bool_value({}, 'test'), False)
        self.assertEqual(lang.optional_bool_value({'test': False}, 'test'), False)
        self.assertEqual(lang.optional_bool_value({'test': False}, 'test'), False)
        self.assertEqual(lang.optional_bool_value({'test': True}, 'test'), True)
        self.assertEqual(lang.optional_bool_value({'test': 1}, 'test'), True)
        self.assertEqual(lang.optional_bool_value({'test': 'true'}, 'test'), False)
        self.assertEqual(lang.optional_bool_value({'test': 'True'}, 'test'), False)

    def test_tuple(self):
        self.assertListEqual(list(lang.to_tuple(x for x in (1, 2, 3, 4, 5, 6))), [[1, 2], [3, 4], [5, 6]])
