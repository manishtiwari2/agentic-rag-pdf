"""Conversation support layered above the pipeline.

``session.py`` turns a follow-up into a standalone question before calling
``pipeline.ask``. The benchmark calls the pipeline directly.
"""
