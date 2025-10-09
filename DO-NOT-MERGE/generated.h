struct model::generated::Binary {
  // . . .

public:
  //
  // Helpers for an autoincrement field: `TypeDefinitions().ID()`.
  //

  uint64_t getNextAvailableIDForTypeDefinitions() const {
    static_assert(
        StrictSpecializationOf<TrackingSortedVector<UpcastablePointer<
                                   model::TypeDefinition>>::UnderlyingContainer,
                               SortedVector>,
        "As of now, auto-increment types are only allowed in "
        "`SortedVector` fields");
    if (TypeDefinitions().empty())
      return 0;

    return TypeDefinitions().rbegin()->get()->ID() + 1;
  }

  /// Record a new element into `TypeDefinitions` while ensuring its
  /// autoincrement field is set correctly.
  ///
  /// \returns a reference to the newly inserted element.
  model::TypeDefinition &
  recordNewTypeDefinition(model::UpcastableTypeDefinition &&T) {
    revng_assert(!T.isEmpty());
    model::TypeDefinition &V = *T;

    if (V.ID() != uint64_t(-1)) {
      std::string Error = "Autoincrement fields must not have a non-default"
                          "value before they are inserted.\n" +
                          ::toString(T);
      revng_abort(Error.c_str());
    }

    // Assign progressive ID
    V.ID() = getNextAvailableIDForTypeDefinitions();

    auto &&[It, Success] = TypeDefinitions().insert(std::move(T));
    revng_assert(Success);

    return **It;
  }

  /// Record a newly created element into `TypeDefinitions()` while ensuring
  /// its autoincrement field is set correctly.
  ///
  /// Notice that this helper passes all its arguments directly to
  /// the constructor of the given type.
  template <derived_from<model::TypeDefinition> NewType,
            typename... ArgumentTypes>
  NewType &makeTypeDefinition(ArgumentTypes &&...Arguments) {
    using U = model::UpcastableTypeDefinition;
    U Result = U::make<NewType>(std::forward<ArgumentTypes>(Arguments)...);
    return llvm::cast<NewType>(recordNewTypeDefinition(std::move(Result)));
  }

  // . . .
};
