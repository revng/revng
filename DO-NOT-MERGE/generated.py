@dataclass(**dataclass_kwargs)
class Binary(StructBase, BinaryMixin):
    # . . .

    def _get_next_available_ID_for_TypeDefinitions(self):
        # TODO: if there is a constant-complexity way of getting the element of
        # `TypeDefinitions` with the highest index, we should use it instead.

        if not hasattr(self, "_next_TypeDefinitions_index"):
            self._next_TypeDefinitions_index = (
                max(element.ID for element in self.TypeDefinitions)
                if len(self.TypeDefinitions)
                else 0
            )

        self._next_TypeDefinitions_index = self._next_TypeDefinitions_index + 1
        return self._next_TypeDefinitions_index - 1

    def _record_new_TypeDefinition(self, element_to_record: TypeDefinition):
        """
        Record a new element into `TypeDefinitions` while ensuring its autoincrement field is set
        correctly.
        """

        if element_to_record.ID != -1:
            raise ValueError(
                "Autoincrement fields must not have a non-default value before they are inserted.\n"
                f"{element_to_record}"
            )

        # Assign progressive ID
        element_to_record.ID = self._get_next_available_ID_for_TypeDefinitions()

        self.TypeDefinitions.append(element_to_record)
        return self.TypeDefinitions[element_to_record.key()]

    def makeTypedefDefinition(self, *args, **kwargs):
        """
        Record a newly created `TypedefDefinition` into `TypeDefinitions` while ensuring its
        autoincrement field is set correctly.

        Notice that this helper passes all its arguments directly to
        the constructor of `TypedefDefinition`.
        """

        result = TypedefDefinition(*args, **kwargs)
        return self._record_new_TypeDefinition(result)

    def makeEnumDefinition(self, *args, **kwargs):
        """
        Record a newly created `EnumDefinition` into `TypeDefinitions` while ensuring its
        autoincrement field is set correctly.

        Notice that this helper passes all its arguments directly to
        the constructor of `EnumDefinition`.
        """

        result = EnumDefinition(*args, **kwargs)
        return self._record_new_TypeDefinition(result)

    # and so on for other TypeDefinition kinds...
