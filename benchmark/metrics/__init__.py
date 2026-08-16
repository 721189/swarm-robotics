"""Benchmark metric calculators.

Implements the "Big Four" swarm-evaluation metrics:

    1. SCI - Swarm Cohesion Index   (normalised mean inter-agent distance)
    2. CE  - Coverage Efficiency    (fraction of objectives under assignment)
    3. PDR - Packet Delivery Ratio  (received vs sent packets)
    4. MTC - Mean Time to Convergence (frames-to-consensus * dt)
"""