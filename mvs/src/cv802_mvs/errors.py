"""Domain-specific exceptions with concise CLI messages."""


class MVSError(RuntimeError):
    """Base error raised for a safely rejected operation."""


class StoragePolicyError(MVSError):
    """A path violates the permanent storage layout."""


class ConfigurationError(MVSError):
    """The JSON configuration is invalid or inconsistent."""


class InputValidationError(MVSError):
    """Staged images, masks, or calibration are invalid."""


class StageConflictError(MVSError):
    """Existing state needs an explicit resume or recoverable overwrite."""


class CommandExecutionError(MVSError):
    """A COLMAP subprocess failed."""

