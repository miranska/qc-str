from string_comparison import StringComparator

# Listing 1
target = '10110'
db =    ['10110', '11010', '01110', '01001']
x = StringComparator(target, db)
results = x.run()
print(f"D = {results['hamming_distances']}")
assert(results['hamming_distances'] == [0, 2, 2, 5])

# Listing 2
target = ['foo', 'quux', 'foo']
db =    [['foo', 'quux', 'bar'],
         ['foo', 'bar',  'foo'],
         ['bar', 'foo',  'foo'],
         ['foo', 'bar',  'bar']]
x = StringComparator(target, db, is_binary=False)
results = x.run()
print(f"D = {results['hamming_distances']}")
assert(results['hamming_distances'] == [1, 1, 2, 2])

# Listing 3
target = ['C', 'G', 'A', 'A', 'T', 'T']
db =    [['C', 'G', 'A', 'A', 'T', 'T'],
         ['C', 'C', 'A', 'A', 'C', 'C'],
         ['G', 'A', 'A', 'A', 'G', 'A'],
         ['C', 'G', 'A', 'T', 'A', 'T']]
x = StringComparator(target, db,
                     is_binary=False, shots=10000)
results = x.run()
print(f"D = {results['hamming_distances']}")
assert(results['hamming_distances'] == [0, 3, 4, 2])

# Listing 4
target = ['AUG', 'ACG', 'CCC']
db =    [['AUG', 'ACG', 'CUU'],
         ['GAG', 'CGC', 'CCC'],
         ['AAA', 'ACG', 'UUU'],
         ['AGA', 'GAG', 'UUU']]
x = StringComparator(target, db, is_binary=False)
results = x.run()
print(f"D = {results['hamming_distances']}")
assert(results['hamming_distances'] == [1, 2, 2, 3])
