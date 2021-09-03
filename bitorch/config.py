"""Config class for quantization layers. This file should be imported before the other layers."""

from argparse import ArgumentParser, Namespace


class Config():
    def __init__(self) -> None:
        configurable_attributes = [
            attribute for attribute in dir(self)
            if not attribute.startswith('__') and not callable(getattr(self, attribute)) and not attribute == "name"]

        self._configurable_attributes = configurable_attributes
        for attribute in self._configurable_attributes:
            self._add_getter_setter_methods(attribute)

    def _add_getter_setter_methods(self, attribute: str) -> None:
        # Todo: implement this
        pass

    def add_config_arguments(self, parser: ArgumentParser) -> None:
        config = parser.add_argument_group(self.name, f"{self.name} configuration settings")
        for attribute in self._configurable_attributes:
            attribute_value = getattr(self, attribute)
            if isinstance(attribute_value, bool):
                config.add_argument(f"--{attribute.replace('_', '-')}", dest=attribute, default=attribute_value,
                                    action=f"store_{'false' if attribute_value else 'true'}", required=False)
            else:
                config.add_argument(f"--{attribute.replace('_', '-')}", dest=attribute, default=attribute_value,
                                    type=type(attribute_value), required=False)

    def apply_args_to_configuration(self, args: Namespace) -> None:
        for attribute in self._configurable_attributes:
            setattr(self, attribute, getattr(args, attribute))
