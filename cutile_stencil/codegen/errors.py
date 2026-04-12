"""Custom warning and error types for code generation."""


class CodegenWarning(UserWarning):
    """Raised when codegen falls back to a non-AST expression."""
    pass
