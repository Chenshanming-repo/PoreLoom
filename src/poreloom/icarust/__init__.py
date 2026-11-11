"""Closed-loop simulation with Icarust: the simulated selective-sequencing runs of the paper.

Icarust (LooseLab) serves the MinKNOW read-until API and simulates an R10 flow cell from reference FASTAs. Each run
starts the patched simulator (icarust.patch: seeded reads and a truth audit log), streams every channel to a live
strategy (No-AS, PoreLoom, or a single-window baseline), sends unblock / stop_further_data, and joins the decisions
with the simulator's audit log. See ``python -m poreloom.icarust --help`` and the repository README.
"""
