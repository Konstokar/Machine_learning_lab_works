from data.patterns import get_patterns
from hopfield import HopfieldNetwork
from utils import add_noise, print_pattern

patterns = get_patterns()

net = HopfieldNetwork()
net.train(patterns)

original = patterns[0]

print("Original:")
print_pattern(original)

noisy = add_noise(original, noise_level=0.3)

print("Noisy:")
print_pattern(noisy)

restored = net.predict(noisy)

print("Restored:")
print_pattern(restored)