"""
Config class for bitorch configurations. These configs can be used to specify key default values which benefit
from beeing changed easily via argparse e.g. for training scripts.
"""

from argparse import ArgumentParser, Namespace


class Config:
    """
    Config superclass that implements functionality to create argparse arguments for class attributes of
    subclasses.
    """

    name: str

    def __init__(self) -> None:
        """collects all attributes of class that are not the name as configurable attributes."""
        configurable_attributes = [
            attribute
            for attribute in dir(self)
            if not attribute.startswith("__") and not callable(getattr(self, attribute)) and not attribute == "name"
        ]

        self._configurable_attributes = configurable_attributes
        for attribute in self._configurable_attributes:
            self._add_getter_setter_methods(attribute)

    def _add_getter_setter_methods(self, attribute: str) -> None:
        def getter(self_):  # type: ignore
            return getattr(self_, attribute)

        def setter(self_, value):  # type: ignore
            setattr(self_, attribute, value)

        setattr(self, f"get_{attribute}", getter)
        setattr(self, f"set_{attribute}", setter)

    def add_config_arguments(self, parser: ArgumentParser) -> None:
        """iterates over this classes configurable attributes and adds an argparse argument. in case of a boolean
        value, the value can then be toggled by either placing or leaving out the according flag.

        Args:
            parser (ArgumentParser): parser to add the arguments to.
        """
        config = parser.add_argument_group(self.name, f"{self.name} configuration settings")
        for attribute in self._configurable_attributes:
            attribute_value = getattr(self, attribute)
            if isinstance(attribute_value, bool):
                config.add_argument(
                    f"--{attribute.replace('_', '-')}",
                    dest=attribute,
                    default=attribute_value,
                    action=f"store_{'false' if attribute_value else 'true'}",
                    required=False,
                )
            else:
                config.add_argument(
                    f"--{attribute.replace('_', '-')}",
                    dest=attribute,
                    default=attribute_value,
                    type=type(attribute_value),
                    required=False,
                )

    def apply_args_to_configuration(self, args: Namespace) -> None:
        """loads the cli set values of configurable attributes.

        Args:
            args (Namespace): cli arguments
        """
        for attribute in self._configurable_attributes:
            setattr(self, attribute, getattr(args, attribute))
