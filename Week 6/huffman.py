import heapq
import sys
from collections import Counter


def count_bytes(path):
    with open(path, "rb") as f:
        return Counter(f.read())


def build_tree(freq):
    # Heap items: (frequency, tie_breaker, node)
    # A node is either an int (a byte value = leaf) or a tuple (left, right).
    heap = [(count, i, byte) for i, (byte, count) in enumerate(sorted(freq.items()))]
    heapq.heapify(heap)
    next_id = len(heap)
    if len(heap) == 1:                       # only one distinct byte
        count, _, byte = heap[0]
        return (byte, None)
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)      # least frequent
        f2, _, n2 = heapq.heappop(heap)      # second least frequent
        heapq.heappush(heap, (f1 + f2, next_id, (n1, n2)))
        next_id += 1
    return heap[0][2]


def make_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}
    if isinstance(node, tuple):
        left, right = node
        if left is not None:
            make_codes(left, prefix + "0", codes)
        if right is not None:
            make_codes(right, prefix + "1", codes)
    else:
        codes[node] = prefix or "0"
    return codes


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 huffman.py <file>")
        sys.exit(1)

    freq = count_bytes(sys.argv[1])
    if not freq:
        print("File is empty.")
        return

    codes = make_codes(build_tree(freq))

    print(f"{'Byte':>6} {'Char':>5} {'Count':>8}  Code")
    print("-" * 40)
    for byte, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        ch = chr(byte) if 32 < byte < 127 else ("SP" if byte == 32 else ".")
        print(f"  0x{byte:02X} {ch:>5} {count:>8}  {codes[byte]}")

    total = sum(freq.values())
    huff_bits = sum(freq[b] * len(codes[b]) for b in freq)
    fixed_len = max(1, (len(freq) - 1).bit_length())
    print("-" * 40)
    print(f"Distinct bytes:           {len(freq)}")
    print(f"Total bytes:              {total}")
    print(f"Original size (8 bit):    {total * 8} bits")
    print(f"Fixed-length ({fixed_len} bit):     {total * fixed_len} bits")
    print(f"Huffman coded:            {huff_bits} bits "
          f"({huff_bits / total:.3f} bits/byte, {huff_bits / (total * 8):.1%} of original)")


if __name__ == "__main__":
    main()

