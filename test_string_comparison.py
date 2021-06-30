import unittest

from string_comparison import StringComparator


class StringComparatorTestCase(unittest.TestCase):
    def test_one_char_symbol(self):
        dataset = ['01001', '11010', '01110', '10110']
        x = StringComparator('10110', dataset, symbol_length=1)
        actual_hd = x.run()['hamming_distances']
        expected_hd = [5, 2, 2, 0]
        self.assertEqual(expected_hd, actual_hd, 'incorrect hamming distance')

    def test_two_char_symbol(self):
        dataset = ['1001', '1000', '1011', '0001', '1101', '1111', '0110']
        x = StringComparator('1001', dataset, symbol_length=2)
        actual_hd = x.run()['hamming_distances']
        expected_hd = [0, 1, 1, 1, 1, 2, 2]
        self.assertEqual(expected_hd, actual_hd, 'incorrect hamming distance')

    def test_two_char_symbol_convertor(self):
        dataset = [['foo', 'bar', 'foo'], ['foo', 'bar', 'bar'], ['foo', 'quux', 'bar'], ['bar', 'foo', 'foo']]
        target = ['foo', 'quux', 'foo']
        x = StringComparator(target, dataset, is_binary=False, shots=10000)
        actual_hd = x.run()['hamming_distances']
        expected_hd = [1, 2, 1, 2]
        self.assertEqual(expected_hd, actual_hd, 'incorrect hamming distance')

    def test_three_char_symbol_convertor(self):
        dataset = [['foo', 'bar', 'foo'],
                   ['foo', 'bar', 'quux'],
                   ['foo', 'quux', 'foo'],
                   ['quux', 'bar', 'foo'],
                   ['quux', 'baz', 'foo'],
                   ['quux', 'baz', 'qux']]
        target = ['foo', 'bar', 'foo']
        x = StringComparator(target, dataset, is_binary=False, shots=100000)
        actual_hd = x.run()['hamming_distances']
        expected_hd = [0, 1, 1, 1, 2, 3]
        self.assertEqual(expected_hd, actual_hd, 'incorrect hamming distance')

    def test_three_char_binary(self):
        # this one is identical to test_three_char_symbol_convertor, just in the binary form
        dataset = ['000001000', '000001010', '000010000', '010001000', '010011000', '010011100']
        x = StringComparator('000001000', dataset, symbol_length=3, shots=100000)
        actual_hd = x.run()['hamming_distances']
        expected_hd = [0, 1, 1, 1, 2, 3]
        self.assertEqual(expected_hd, actual_hd, 'incorrect hamming distance')

    def test_all_different_two_char_symbol(self):
        dataset = ['0101', '1010', '1111', '1011', '1110', '1101']
        x = StringComparator('0000', dataset, symbol_length=2)
        actual_hd = x.run()['hamming_distances']
        expected_hd = [2, 2, 2, 2, 2, 2]
        self.assertEqual(expected_hd, actual_hd, 'incorrect hamming distance')

if __name__ == '__main__':
    unittest.main()
