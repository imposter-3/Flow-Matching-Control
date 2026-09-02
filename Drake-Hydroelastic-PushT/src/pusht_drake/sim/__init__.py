"""Simulation tier: the Drake rig, rollout, harness and command guards.

This tier may import pydrake and may not import torch; the policy enters only
through a factory spec resolved inside each worker process.
"""
