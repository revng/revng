#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompiledCCodeIndentation.h"
#include "revng/Support/PTMLC.h"
#include "revng/TypeNames/DependencyGraph.h"

namespace ptml {

class CTypeBuilder : public CBuilder {
public:
  using OutStream = ptml::IndentedOstream;

public:
  const model::Binary &Binary;
  model::NameBuilder NameBuilder;

protected:
  /// The stream to print the result to.
  std::unique_ptr<OutStream> Out;

public:
  struct ConfigurationOptions {
    /// When set to true, the types that are only ever depended on by a single
    /// other type (for example, a nested struct), are printed inside the parent
    /// type definition instead of separately.
    bool EnableTypeInlining = true;

    /// When set to true, function stack frame types are printed inside the
    /// struct body as opposed to the type header.
    bool EnableStackFrameInlining = true;

    /// Because we are emitting C11, we cannot specify underlying enum type
    /// (the feature was only backported from C++ in C23), which means that
    /// when we need to preserve enum size across the compilation boundary
    /// (for example, when editing a type), we need to resolve to a trick:
    /// we print the maximum value possible - and then use it to figure out
    /// the original size.
    /// Setting this flag to true enables printing of such a value.
    bool EnablePrintingOfTheMaximumEnumValue = false;

    /// All of our structs are packed by definition. Which means every single
    /// byte has to be occupied. In explicit padding mode, the padding is
    /// printed as arrays of bytes. In implicit mode, a `_START_AT` annotation
    /// is printed instead.
    bool EnableExplicitPaddingMode = true;

    /// When this flag is set to true, a special `_SIZE($bytes)` annotation is
    /// printed with every struct definition.
    bool EnableStructSizeAnnotation = false;

    /// Sometimes you don't want to print everything. This lets you specify
    /// a set of types that will be ignored by \ref typeDefinitions.
    std::set<model::TypeDefinition::Key> TypesToOmit = {};
  };
  const ConfigurationOptions Configuration;

private:
  /// This is the cache containing the names of artificial types (like array
  /// wrappers).
  std::map<model::UpcastableType, std::string> ArtificialNameCache = {};

  /// This is the cache containing the dependency data for the types.
  /// It is here so that we don't have to recompute it with multiple invocations
  std::optional<DependencyGraph> DependencyCache = std::nullopt;

  /// This is the cache containing the keys of the types that should be inlined
  /// into their only user.
  std::set<model::TypeDefinition::Key> TypesToInlineCache = {};

  /// This is the cache containing the keys of the stack frame types that
  /// should be inlined into the functions they belong to.
  std::set<model::TypeDefinition::Key> StackFrameTypeCache = {};

  /// Is only set to true if \ref collectInlinableTypes was invoked.
  bool InlinableCacheIsReady = false;

public:
  /// Gather (and store internally) the list of types that can (and should)
  /// be inlined. This list is then later used by the invocations of
  /// \ref printTypeDefinition.
  void collectInlinableTypes();

  bool shouldInline(model::TypeDefinition::Key Key) const {
    revng_assert(InlinableCacheIsReady,
                 "`shouldInline` must not be called before "
                 "`collectInlinableTypes`.");

    if (not TypesToInlineCache.contains(Key)) {
      // This type is not allowed be inlined.
      return false;
    }

    if (StackFrameTypeCache.contains(Key)) {
      // This is a stack frame.
      return Configuration.EnableStackFrameInlining;

    } else {
      return Configuration.EnableTypeInlining;
    }
  }

  bool shouldInline(const model::TypeDefinition &Type) const {
    return shouldInline(Type.key());
  }

public:
  CTypeBuilder(llvm::raw_ostream &OutputStream,
               const model::Binary &Binary,
               CBuilder B,
               ConfigurationOptions &&Configuration) :
    CBuilder(B),
    Binary(Binary),
    NameBuilder(Binary),
    Out(std::make_unique<OutStream>(OutputStream,
                                    *this,
                                    DecompiledCCodeIndentation)),
    Configuration(std::move(Configuration)) {}

  CTypeBuilder(llvm::raw_ostream &OutputStream,
               const model::Binary &Binary,
               CBuilder B) :
    CTypeBuilder(OutputStream, Binary, B, ConfigurationOptions{}) {}

  CTypeBuilder(llvm::raw_ostream &OutputStream,
               const model::Binary &Binary,
               bool EnableTaglessMode,
               ConfigurationOptions &&Configuration) :
    CTypeBuilder(OutputStream,
                 Binary,
                 ptml::MarkupBuilder{ .IsInTaglessMode = EnableTaglessMode },
                 std::move(Configuration)) {}

  CTypeBuilder(llvm::raw_ostream &OutputStream,
               const model::Binary &Binary,
               bool EnableTaglessMode) :
    CTypeBuilder(OutputStream,
                 Binary,
                 ptml::MarkupBuilder{ .IsInTaglessMode = EnableTaglessMode }) {}

  CTypeBuilder(llvm::raw_ostream &OutputStream,
               const model::Binary &Binary,
               ConfigurationOptions &&Configuration) :
    CTypeBuilder(OutputStream, Binary, {}, std::move(Configuration)) {}

  CTypeBuilder(llvm::raw_ostream &OutputStream,
               const model::Binary &Binary = {}) :
    CTypeBuilder(OutputStream, Binary, {}, {}) {}

public:
  void setOutputStream(llvm::raw_ostream &OutputStream) {
    Out = std::make_unique<OutStream>(OutputStream,
                                      *this,
                                      DecompiledCCodeIndentation);
  }

public:
  void append(std::string &&Text) { *Out << std::move(Text); }
  void appendLineComment(std::string &&Text) {
    append(getLineComment(std::move(Text)));
  }

public:
  ScopeTag getIndentedTag(const llvm::StringRef AttributeName,
                          bool NewLine = false) {
    return getTag(AttributeName).scope(*Out, NewLine);
  }
  ScopeTag getIndentedScope(const Scopes Scope, bool NewLine = false) {
    return getScope(Scope).scope(*Out, NewLine);
  }
  ptml::IndentedOstream::Scope getSimpleScope() { return Out->scope(); }
  Scope
  getCurvedBracketScope(std::string &&Attribute = ptml::c::scopes::Scope) {
    return Scope(*Out, Attribute);
  }
  helpers::BlockComment getBlockCommentScope() {
    return helpers::BlockComment(*Out, *this);
  }
  helpers::LineComment getLineCommentScope() {
    return helpers::LineComment(*Out, *this);
  }

public:
  auto getNameTag(const model::TypeDefinition &T) {
    return tokenTag(NameBuilder.name(T), ptml::c::tokens::Type);
  }
  auto getNameTag(const model::Segment &S) {
    return tokenTag(NameBuilder.name(S), ptml::c::tokens::Variable);
  }
  auto getNameTag(const model::EnumDefinition &Enum,
                  const model::EnumEntry &Entry) {
    return tokenTag(NameBuilder.name(Enum, Entry), ptml::c::tokens::Field);
  }
  template<class Aggregate, class Field>
  auto getNameTag(const Aggregate &A, const Field &F) {
    return tokenTag(NameBuilder.name(A, F), c::tokens::Field);
  }

private:
  template<typename... Ts>
  std::string locationStringImpl(Ts &&...Vs) const {
    if (IsInTaglessMode)
      return "";
    return pipeline::locationString(std::forward<Ts>(Vs)...);
  }

public:
  std::string locationString(const model::TypeDefinition &T) const {
    return locationStringImpl(revng::ranks::TypeDefinition, T.key());
  }
  std::string locationString(const model::Segment &T) const {
    return locationStringImpl(revng::ranks::Segment, T.key());
  }
  std::string locationString(const model::EnumDefinition &Enum,
                             const model::EnumEntry &Entry) const {
    return locationStringImpl(revng::ranks::EnumEntry, Enum.key(), Entry.key());
  }
  std::string locationString(const model::StructDefinition &Struct,
                             const model::StructField &Field) const {
    return locationStringImpl(revng::ranks::StructField,
                              Struct.key(),
                              Field.key());
  }
  std::string locationString(const model::UnionDefinition &Union,
                             const model::UnionField &Field) const {
    return locationStringImpl(revng::ranks::UnionField,
                              Union.key(),
                              Field.key());
  }

public:
  constexpr const char *getLocationAttribute(bool IsDefinition) const {
    return IsDefinition ? ptml::attributes::LocationDefinition :
                          ptml::attributes::LocationReferences;
  }

  std::string getLocation(bool IsDefinition,
                          const model::TypeDefinition &T,
                          llvm::ArrayRef<std::string> AllowedActions) {
    auto Result = getNameTag(T);
    if (IsInTaglessMode)
      return Result.toString();

    std::string Location = locationString(T);
    Result.addAttribute(getLocationAttribute(IsDefinition), Location);
    Result.addAttribute(attributes::ActionContextLocation, Location);

    if (not AllowedActions.empty())
      Result.addListAttribute(attributes::AllowedActions, AllowedActions);

    return Result.toString();
  }

  std::string getLocation(bool IsDefinition, const model::Segment &S) {
    std::string Location = locationString(S);
    return getNameTag(S)
      .addAttribute(getLocationAttribute(IsDefinition), Location)
      .addAttribute(ptml::attributes::ActionContextLocation, Location)
      .toString();
  }

  std::string getLocation(bool IsDefinition,
                          const model::EnumDefinition &Enum,
                          const model::EnumEntry &Entry) {
    std::string Location = locationString(Enum, Entry);
    return getNameTag(Enum, Entry)
      .addAttribute(getLocationAttribute(IsDefinition), Location)
      .addAttribute(ptml::attributes::ActionContextLocation, Location)
      .toString();
  }

  template<typename Aggregate, typename Field>
  std::string
  getLocation(bool IsDefinition, const Aggregate &A, const Field &F) {
    std::string Location = locationString(A, F);
    return getNameTag(A, F)
      .addAttribute(getLocationAttribute(IsDefinition), Location)
      .addAttribute(attributes::ActionContextLocation, Location)
      .toString();
  }

public:
  std::string
  getLocationDefinition(const model::TypeDefinition &T,
                        llvm::ArrayRef<std::string> AllowedActions = {}) {
    return getLocation(true, T, AllowedActions);
  }

  std::string getLocationDefinition(const model::PrimitiveType &P) const {
    std::string CName = P.getCName();
    auto Result = tokenTag(CName, ptml::c::tokens::Type);
    if (IsInTaglessMode)
      return Result.toString();

    std::string L = pipeline::locationString(revng::ranks::PrimitiveType,
                                             P.getCName());
    Result.addAttribute(getLocationAttribute(true), L);
    Result.addAttribute(attributes::ActionContextLocation, L);

    return Result.toString();
  }

  std::string getLocationDefinition(const model::Segment &S) {
    return getLocation(true, S);
  }

  template<typename Aggregate, typename Field>
  std::string getLocationDefinition(const Aggregate &A, const Field &F) {
    return getLocation(true, A, F);
  }

public:
  std::string
  getLocationReference(const model::TypeDefinition &T,
                       llvm::ArrayRef<std::string> AllowedActions = {}) {
    return getLocation(false, T, AllowedActions);
  }

  std::string getPrimitiveTypeLocationReference(llvm::StringRef CName) const {
    auto Result = tokenTag(CName, ptml::c::tokens::Type);
    if (IsInTaglessMode)
      return Result.toString();

    std::string L = pipeline::locationString(revng::ranks::PrimitiveType,
                                             CName.str());
    Result.addAttribute(getLocationAttribute(false), L);
    Result.addAttribute(attributes::ActionContextLocation, L);

    return Result.toString();
  }

  std::string getLocationReference(const model::PrimitiveType &P) const {
    return getPrimitiveTypeLocationReference(P.getCName());
  }

  std::string getLocationReference(const model::Segment &S) {
    return getLocation(false, S);
  }

  template<typename Aggregate, typename Field>
  std::string getLocationReference(const Aggregate &A, const Field &F) {
    return getLocation(false, A, F);
  }

  std::string getLocationReference(const model::Function &F);
  std::string getLocationReference(const model::DynamicFunction &F);

public:
  template<model::EntityWithComment Type>
  std::string getModelComment(const Type &T) const {
    const model::Configuration &Configuration = Binary.Configuration();
    uint64_t LineWidth = Configuration.commentLineWidth();

    // TODO: do not rely on `Out`'s indentation, since there's no guarantee it's
    //       the same stream (even if it usually is).
    uint64_t Width = LineWidth - Out->currentIndentation();

    return ptml::comment(*this, T, "///", 0, Width);
  }

  std::string getFunctionComment(const model::Function &Function) const {
    const model::Configuration &Configuration = Binary.Configuration();
    uint64_t LineWidth = Configuration.commentLineWidth();

    // TODO: do not rely on `Out`'s indentation, since there's no guarantee it's
    //       the same stream (even if it usually is).
    uint64_t Width = LineWidth - Out->currentIndentation();

    return ptml::functionComment(*this, Function, Binary, "///", 0, Width);
  }

  std::string getStatementComment(const model::StatementComment &Text,
                                  llvm::StringRef EmittedAt) const {
    const model::Configuration &Configuration = Binary.Configuration();
    uint64_t LineWidth = Configuration.commentLineWidth();

    // TODO: do not rely on `Out`'s indentation, since there's no guarantee it's
    //       the same stream (even if it usually is).
    uint64_t Width = LineWidth - Out->currentIndentation();
    return ptml::statementComment(*this, Text, EmittedAt, "//", 0, Width);
  }

public:
  /// Obtain a line representing a typical usage of a type (how it appears
  /// when used to declare a struct field or a local variable)
  ///
  /// \param Type the type to serialize
  /// \param InstanceName the name of the field/variable
  /// \param AllowedActions the list of ptml actions (for example, renaming),
  ///        locations for which should be embedded into the string.
  ///        \note this option does not affect anything in tagless mode.
  /// \param OmitInnerTypeName a flag to allow omission of the inner type name,
  ///        for example:
  ///        ```cpp
  ///        // OmitInnerTypeName = false
  ///        my_struct *my_variable
  ///
  ///        // OmitInnerTypeName = true
  ///         *my_variable
  ///        ```
  tokenDefinition::types::TypeString
  getNamedCInstance(const model::Type &Type,
                    llvm::StringRef InstanceName,
                    llvm::ArrayRef<std::string> AllowedActions = {},
                    bool OmitInnerTypeName = false);

  tokenDefinition::types::TypeString getTypeName(const model::Type &Type) {
    return getNamedCInstance(Type, "");
  }

public:
  /// Return the name of the array wrapper that wraps \a ArrayType
  tokenDefinition::types::TypeString
  getArrayWrapper(const model::ArrayType &ArrayType);

  /// Return a string containing the C Type name of the return type of
  /// \a FunctionType, and a (possibly empty) \a InstanceName.
  /// \note If F returns more than one value, the name of the wrapping struct
  ///       will be returned instead.
  tokenDefinition::types::TypeString
  getNamedInstanceOfReturnType(const model::TypeDefinition &FunctionType,
                               llvm::StringRef InstanceName,
                               bool IsDefinition);

  tokenDefinition::types::TypeString
  getReturnTypeName(const model::TypeDefinition &FunctionType,
                    bool IsDefinition) {
    return getNamedInstanceOfReturnType(FunctionType, "", IsDefinition);
  }

public:
  /// Print the function prototype (without any trailing ';') of \a Function.
  /// \note If the return value or any of the arguments needs a wrapper, print
  ///       it with the corresponding wrapper type. The definition of such
  ///       wrappers should have already been printed before this function is
  ///       called.
  void printFunctionPrototype(const model::TypeDefinition &FunctionType,
                              const model::Function &Function,
                              bool SingleLine = false);
  void printFunctionPrototype(const model::TypeDefinition &FunctionType,
                              const model::DynamicFunction &Function,
                              bool SingleLine = false);
  void printFunctionPrototype(const model::TypeDefinition &FunctionType);

  void printSegmentType(const model::Segment &Segment);

public:
  std::string getArgumentLocationReference(llvm::StringRef ArgumentName,
                                           const model::Function &F) const;
  std::string getVariableLocationDefinition(llvm::StringRef VariableName,
                                            const model::Function &F) const;
  std::string getVariableLocationReference(llvm::StringRef VariableName,
                                           const model::Function &F) const;
  std::string getGotoLabelLocationDefinition(llvm::StringRef GotoLabelName,
                                             const model::Function &F) const;
  std::string getGotoLabelLocationReference(llvm::StringRef GotoLabelName,
                                            const model::Function &F) const;

public:
  /// Returns true for types we never print definitions for a give type
  /// (like typedefs, for example)
  ///
  /// \note As a side-effect this also determines whether type can be inlined:
  ///       there's no point inlining a type for which we only ever emit
  ///       a declaration
  static bool isDeclarationTheSameAsDefinition(const model::TypeDefinition &T) {
    return not llvm::isa<model::StructDefinition>(&T)
           and not llvm::isa<model::UnionDefinition>(&T)
           and not llvm::isa<model::EnumDefinition>(&T);
  }

  /// Generates the definition of a new struct type that wraps all the return
  /// values of \a F.
  void generateReturnValueWrapper(const model::RawFunctionDefinition &F);

  /// Generates the definition of a new struct type that wraps \a ArrayType.
  /// This is used to wrap array arguments or array return values of
  /// CABI functions.
  void generateArrayWrapper(const model::ArrayType &ArrayType);

  void printFunctionWrappers(const model::RawFunctionDefinition &F);
  void printFunctionWrappers(const model::CABIFunctionDefinition &F);

  void printPadding(uint64_t FieldOffset, uint64_t NextOffset);

public:
  void printForwardDeclaration(const model::TypeDefinition &T);

  void printDeclaration(const model::TypedefDefinition &Typedef);
  void printDeclaration(const model::RawFunctionDefinition &F);
  void printDeclaration(const model::CABIFunctionDefinition &F);
  void printDeclaration(const model::TypeDefinition &T);

private:
  using TypeSet = std::set<model::TypeDefinition::Key>;

public:
  void printTypeDefinition(const model::EnumDefinition &E,
                           std::string &&Suffix = "");
  void printTypeDefinition(const model::StructDefinition &S,
                           std::string &&Suffix = "");
  void printTypeDefinition(const model::UnionDefinition &U,
                           std::string &&Suffix = "");
  void printTypeDefinition(const model::TypeDefinition &T);

  void printInlineDefinition(llvm::StringRef Name, const model::Type &T);

public:
  /// Print all the type definitions in the model.
  ///
  /// Please use this instead of calling \ref typeDefinition
  /// on every type, as types can depend on each other.
  /// This method ensures they are printed in a valid order.
  void printTypeDefinitions();
};

} // namespace ptml

inline std::string getPlainTypeName(const model::Type &Type) {
  ptml::CTypeBuilder B(llvm::nulls(), {}, /* EnableTaglessMode = */ true);
  return B.getTypeName(Type).str().str();
}
