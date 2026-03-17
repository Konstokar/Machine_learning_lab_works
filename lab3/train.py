from data.patterns import get_patterns
from hopfield import HopfieldNetwork

patterns = get_patterns()

net = HopfieldNetwork()
net.train(patterns)

print("Training complete!")