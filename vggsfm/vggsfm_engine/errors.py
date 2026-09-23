"""Domain-specific failures with actionable messages for CLI users."""


class VGGSfMError(RuntimeError):
    """Base class for expected, user-facing engine errors."""


class StoragePolicyError(VGGSfMError):
    """A path would violate the permanent code/data separation."""


class ConfigurationError(VGGSfMError):
    """An inference profile or run request is invalid."""


class ProvenanceError(VGGSfMError):
    """The official checkout does not match the reviewed revision."""


class IncompleteRunError(VGGSfMError):
    """A partial experiment exists and requires an explicit safe retry."""


class OutputValidationError(VGGSfMError):
    """Official output is absent, ambiguous, or malformed."""


class ExecutionError(VGGSfMError):
    """CUDA preflight or official VGGSfM execution failed."""
