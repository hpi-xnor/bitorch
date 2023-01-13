import typing
import importlib
from typing import Optional, Callable, List, Any, Dict


@typing.no_type_check
def build_lookup_dictionary(
    current_module_name: str,
    class_strings: List[str],
    filter_by_superclass: Optional[Any] = None,
    filter_fn: Optional[Callable[[Any], bool]] = None,
    key_fn: Callable[[Any], str] = lambda x: x.name,
) -> Dict[str, Any]:
    """Builds a lookup dictionary based on a list of strings of class names.

    Args:
        current_module_name (str): the module from where the classes are available
        class_strings (List[str]): the list of strings
        filter_by_superclass (Any): if filter should be based on a common super class
        filter_fn (Callable[[Any], bool]): a custom filter function
        key_fn (Callable[[Any], str]): a function that provides a mapping from Class to the desired key

    Returns:
        Dict[str, Any]: the lookup dictionary
    """
    assert filter_fn is not None or filter_by_superclass is not None, "one of the filter options must be given"
    if filter_fn is None:

        def filter_fn(x: Any) -> bool:
            return isinstance(x, type) and issubclass(x, filter_by_superclass) and x != filter_by_superclass

    lookup = {}
    current_module = importlib.import_module(current_module_name)
    for class_name in class_strings:
        if not hasattr(current_module, class_name):
            continue
        class_ = getattr(current_module, class_name)
        if filter_fn(class_):
            transformed_key = key_fn(class_)
            lookup[transformed_key] = class_

    return lookup
