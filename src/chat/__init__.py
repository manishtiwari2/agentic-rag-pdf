"""Conversation and chat interface, layered above the pipeline (DD-065, DD-066).

``session.py`` turns a follow-up into a standalone question before calling
``pipeline.ask``; ``ui.py`` builds the notebook's Gradio app from plain,
testable functions. Nothing below this package imports it, and the benchmark
never uses it, so single-turn behaviour and every stored result are exactly
what they would be without it. ``tests/test_architecture.py`` checks both.
"""
