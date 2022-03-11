Module bitorch.util
===================

Functions
---------

    
`build_lookup_dictionary(current_module_name: str, class_strings: List[str], filter_by_superclass: Any = None, filter_fn: Callable[[Any], bool] = None, key_fn: Callable[[Any], str] = <function <lambda>>) ‑> Dict[str, Any]`
:   Builds a lookup dictionary based on a list of strings of class names.
    
    Args:
        current_module_name (str): the module from where the classes are available
        class_strings (List[str]): the list of strings
        filter_by_superclass (Any): if filter should be based on a common super class
        filter_fn (Callable[[Any], bool]): a custom filter function
        key_fn (Callable[[Any], str]): a function that provides a mapping from Class to the desired key
    
    Returns:
        Dict[str, Any]: the lookup dictionary