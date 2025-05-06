#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

from revng.support.location import Location


@dataclass(frozen=True)
class ActionMapping:
    # Given a PipelineLocation, return the TupleTreePath for the model
    rename: Optional[Callable[[Location], str]]
    # Given a PipelineLocation, return the TupleTreePath for the model
    comment: Optional[Callable[[Location], str]]


def _model_path_generator(*args: Union[str, Tuple[int, bool]]) -> Callable[[Location], str]:
    """Generates a function that given a location generates a model path. The
    arguments are a list of "parts" that tell the generator how to glue
    together the location path components into the model path. The assembly
    logic is as follows:
    * If a string is encountered, then add the string to the path
    * If a tuple is encountered, grab tuple[0]-th path component from the
      location and:
      * If tuple[1] is false append it as-is to the generated model path
      * If tuple[1] is true append `path[tuple[0]]/<kind>::`, where `<kind>` is
        the string of `path[tuple[0]]` from the last '-' onward
    """

    def generator(location: Location):
        if len(args) == 0:
            return ""

        result = ""
        for element in args:
            if not result.endswith("::"):
                result += "/"

            if isinstance(element, str):
                result += element
            else:
                if element[1]:
                    # Path needs to be upcasted
                    part = location.path_components[element[0]]
                    kind = part.split("-")[-1]
                    result += f"{part}/{kind}::"
                else:
                    result += location.path_components[element[0]]
        return result

    return generator


def _common(*args: Union[str, Tuple[int, bool]]) -> ActionMapping:
    return ActionMapping(
        _model_path_generator(*args, "Name"),
        _model_path_generator(*args, "Comment"),
    )


ACTION_MAPPING: Dict[str, ActionMapping] = {
    "function": _common("Functions", (0, False)),
    "type-definition": _common("TypeDefinitions", (0, True)),
    "struct-field": _common("TypeDefinitions", (0, True), "Fields", (1, False)),
    "union-field": _common("TypeDefinitions", (0, True), "Fields", (1, False)),
    "enum-entry": _common("TypeDefinitions", (0, True), "Entries", (1, False)),
    "cabi-argument": _common("TypeDefinitions", (0, True), "Arguments", (1, False)),
    "raw-argument": _common("TypeDefinitions", (0, True), "Arguments", (1, False)),
    "return-value": ActionMapping(
        None, _model_path_generator("TypeDefinitions", (0, True), "ReturnValueComment")
    ),
    "return-register": _common("TypeDefinitions", (0, True), "ReturnValues", (1, False)),
    "segment": _common("Segments", (0, False)),
    "dynamic-function": _common("ImportedDynamicFunctions", (0, False)),
}


def model_path_for_rename(location: Location) -> Optional[str]:
    mapping = ACTION_MAPPING.get(location.type)
    if mapping is None or mapping.rename is None:
        return None
    return mapping.rename(location)


def model_path_for_comment(location: Location) -> Optional[str]:
    mapping = ACTION_MAPPING.get(location.type)
    if mapping is None or mapping.comment is None:
        return None
    return mapping.comment(location)
