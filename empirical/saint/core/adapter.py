from empirical.saint.core import geometry
from empirical.saint.core import structural_adapter as structural

def adapt(parent, rule):
    if rule != "ordered":
        raise ValueError(rule)
    return structural.adapt(parent)

def specification(rule):
    return {}
