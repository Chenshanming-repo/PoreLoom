"""MinKNOW live-read adapter for PoreLoom (template).

Connects to a MinKNOW sequencing position through the manager service, streams uncalibrated raw signal for every
read with ``DataService.get_live_reads``, runs a trained PoreLoom bundle on the growing prefixes, and sends
``unblock`` (reject) or ``stop_further_data`` (accept) for each terminal decision.

Status: exercised end to end against the Icarust simulator, which serves the same MinKNOW gRPC services. It has
not been run on a sequencing device. See ``python -m poreloom.minknow --help`` and the repository README.
"""
