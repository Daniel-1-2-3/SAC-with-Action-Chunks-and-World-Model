# COMBINED PATCH: upstream also registers ACRLPDAgent (agents/acrlpd.py),
# which is not vendored -- the combined arm only needs acfql.
from agents.acfql import ACFQLAgent

agents = dict(
    acfql=ACFQLAgent,
)
