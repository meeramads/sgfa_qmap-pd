"""Parameter resolution utilities for handling multiple config sources."""

from typing import Any, Dict, Optional, Union, List


class ParameterResolver:
    """Resolve parameters from multiple sources with fallback chain.

    Handles the common pattern of checking args -> hypers -> config -> default
    in a clean, explicit way.

    Examples
    --------
    >>> params = ParameterResolver(args, hypers, config)
    >>> K = params.get('K', default=10)
    >>> percW = params.get('percW', default=25.0)
    """

    def __init__(self, *sources):
        """Initialize with multiple parameter sources.

        Parameters
        ----------
        *sources : Any
            Parameter sources to check in order (args, hypers, config, etc.)
            Can be objects with attributes, dictionaries, or None
        """
        self.sources = [s for s in sources if s is not None]

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value from first source that has it.

        Parameters
        ----------
        key : str
            Parameter name to look for
        default : Any, optional
            Default value if not found in any source

        Returns
        -------
        Any
            Parameter value from first source that has it, or default

        Examples
        --------
        >>> params = ParameterResolver(args, hypers, {'K': 10})
        >>> params.get('K', default=5)
        10
        """
        for source in self.sources:
            # Try attribute access (args objects)
            if hasattr(source, key):
                value = getattr(source, key)
                if value is not None:
                    return value

            # Try dictionary access (hypers, config dicts)
            if isinstance(source, dict) and key in source:
                value = source[key]
                if value is not None:
                    return value

        return default

    def get_required(self, key: str) -> Any:
        """Get parameter value, raise error if not found.

        Parameters
        ----------
        key : str
            Parameter name to look for

        Returns
        -------
        Any
            Parameter value from first source that has it

        Raises
        ------
        KeyError
            If parameter not found in any source

        Examples
        --------
        >>> params = ParameterResolver(args, hypers)
        >>> K = params.get_required('K')  # Raises KeyError if not found
        """
        value = self.get(key, default=None)
        if value is None:
            source_names = [type(s).__name__ for s in self.sources]
            raise KeyError(
                f"Required parameter '{key}' not found in any source: {source_names}"
            )
        return value

    def get_all(self, keys: List[str], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get multiple parameters at once.

        Parameters
        ----------
        keys : List[str]
            Parameter names to get
        defaults : Dict[str, Any], optional
            Default values for each key

        Returns
        -------
        Dict[str, Any]
            Dictionary of resolved parameter values

        Examples
        --------
        >>> params = ParameterResolver(args, hypers)
        >>> values = params.get_all(['K', 'percW'], defaults={'K': 10, 'percW': 25.0})
        {'K': 5, 'percW': 33.0}
        """
        defaults = defaults or {}
        return {
            key: self.get(key, default=defaults.get(key))
            for key in keys
        }

    def has(self, key: str) -> bool:
        """Check if parameter exists in any source.

        Parameters
        ----------
        key : str
            Parameter name to check

        Returns
        -------
        bool
            True if parameter found in any source
        """
        return self.get(key, default=None) is not None

    def __repr__(self) -> str:
        """String representation."""
        source_names = [type(s).__name__ for s in self.sources]
        return f"ParameterResolver(sources={source_names})"
