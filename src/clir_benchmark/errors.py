class ValidationError(ValueError):
    """Invalid or ambiguous benchmark input; never silently repair it."""


class BlockedSettingsError(ValidationError):
    """Some settings lack the required multilingual annotations."""
