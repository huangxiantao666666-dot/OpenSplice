# Lazy imports to avoid loading heavy deps on package import
def get_agent():
    from .workflow import ImageStitchAgent, run_agent
    return ImageStitchAgent, run_agent
