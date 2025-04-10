#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompiledCCodeIndentation.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Support/PTMLC.h"
#include "revng/TypeNames/DependencyGraph.h"

namespace ptml {

class CTypeBuilder : public CBuilder {
public:
  using OutStream = ptml::IndentedOstream;

public:
  const model::Binary &Binary;
  model::CNameBuilder NameBuilder;

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

  CTypeBuilder(llvm::raw_ostream &OutputStream, const model::Binary &Binary) :
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
  Scope getCurvedBracketScope(std::string &&Attribute =
                                ptml::c::scopes::Scope.str()) {
    return Scope(*Out, Attribute);
  }
  helpers::BlockComment getBlockCommentScope() {
    return helpers::BlockComment(*Out, *this);
  }
  helpers::LineComment getLineCommentScope() {
    return helpers::LineComment(*Out, *this);
  }

private:
  static constexpr llvm::StringRef tokenType(const model::TypeDefinition &) {
    return ptml::c::tokens::Type;
  }
  static constexpr llvm::StringRef tokenType(const model::PrimitiveType &) {
    return ptml::c::tokens::Type;
  }
  static constexpr llvm::StringRef tokenType(const model::Segment &) {
    return ptml::c::tokens::Variable;
  }
  static constexpr llvm::StringRef tokenType(const model::Function &) {
    return ptml::c::tokens::Function;
  }
  static constexpr llvm::StringRef tokenType(const model::DynamicFunction &) {
    return ptml::c::tokens::Function;
  }
  template<class Aggregate, class Field>
  static constexpr llvm::StringRef tokenType(const Aggregate &, const Field &) {
    return ptml::c::tokens::Field;
  }
  static constexpr llvm::StringRef
  tokenType(const model::CABIFunctionDefinition &, const model::Argument &) {
    return ptml::c::tokens::FunctionParameter;
  }
  static constexpr llvm::StringRef
  tokenType(const model::RawFunctionDefinition &,
            const model::NamedTypedRegister &) {
    return ptml::c::tokens::FunctionParameter;
  }

private:
  std::string locationString(const model::TypeDefinition &T) const {
    return pipeline::locationString(revng::ranks::TypeDefinition, T.key());
  }
  std::string locationString(const model::PrimitiveType &Primitive) const {
    return pipeline::locationString(revng::ranks::PrimitiveType,
                                    Primitive.getCName());
  }
  std::string locationString(const model::Binary &,
                             const model::Segment &T) const {
    return pipeline::locationString(revng::ranks::Segment, T.key());
  }
  std::string locationString(const model::Function &Function) const {
    return pipeline::locationString(revng::ranks::Function, Function.key());
  }
  std::string locationString(const model::DynamicFunction &Function) const {
    return pipeline::locationString(revng::ranks::DynamicFunction,
                                    Function.key());
  }
  std::string locationString(const model::EnumDefinition &Enum,
                             const model::EnumEntry &Entry) const {
    return pipeline::locationString(revng::ranks::EnumEntry,
                                    Enum.key(),
                                    Entry.key());
  }
  std::string locationString(const model::StructDefinition &Struct,
                             const model::StructField &Field) const {
    return pipeline::locationString(revng::ranks::StructField,
                                    Struct.key(),
                                    Field.key());
  }
  std::string locationString(const model::UnionDefinition &Union,
                             const model::UnionField &Field) const {
    return pipeline::locationString(revng::ranks::UnionField,
                                    Union.key(),
                                    Field.key());
  }
  std::string locationString(const model::DynamicFunction &Function,
                             llvm::StringRef Name) const {
    return pipeline::locationString(revng::ranks::DynamicFunctionArgument,
                                    Function.key(),
                                    Name.str());
  }
  std::string locationString(const model::CABIFunctionDefinition &Function,
                             const model::Argument &Argument) const {
    return pipeline::locationString(revng::ranks::CABIArgument,
                                    Function.key(),
                                    Argument.key());
  }
  std::string locationString(const model::RawFunctionDefinition &Function,
                             const model::NamedTypedRegister &Argument) const {
    return pipeline::locationString(revng::ranks::RawArgument,
                                    Function.key(),
                                    Argument.key());
  }

private:
  std::string variableLocationString(const model::Function &Function,
                                     llvm::StringRef Variable) const {
    return pipeline::locationString(revng::ranks::LocalVariable,
                                    Function.key(),
                                    Variable.str());
  }
  std::string
  returnValueLocationString(const model::RawFunctionDefinition &Function,
                            const model::NamedTypedRegister &Argument) const {
    return pipeline::locationString(revng::ranks::ReturnRegister,
                                    Function.key(),
                                    Argument.key());
  }

private:
  template<bool IsDefinition>
  std::string
  getNameTagImpl(ptml::Tag &&Tag,
                 llvm::StringRef Location,
                 llvm::StringRef ActionLocation,
                 RangeOf<llvm::StringRef> auto const &Actions) const {
    if (IsInTaglessMode)
      return Tag.toString();

    llvm::StringRef LocationAttribute = IsDefinition ?
                                          ptml::attributes::LocationDefinition :
                                          ptml::attributes::LocationReferences;
    Tag.addAttribute(LocationAttribute, Location.str());

    if (not std::ranges::empty(Actions)) {
      Tag.addAttribute(attributes::ActionContextLocation, ActionLocation.str());
      Tag.addListAttribute(attributes::AllowedActions, Actions);
    }

    return std::move(Tag).toString();
  }

  template<bool IsDefinition>
  std::string
  getNameTagImpl(ptml::Tag &&Tag,
                 llvm::StringRef /* Location */ L,
                 RangeOf<llvm::StringRef> auto const &Actions) const {
    return getNameTagImpl<IsDefinition>(std::move(Tag), L, L, Actions);
  }

  template<bool IsDefinition, typename AnyType>
  std::string getNameTag(const AnyType &Value,
                         RangeOf<llvm::StringRef> auto const &Actions) const {
    // TODO: build a warning based on `NameBuilder.warning(T)`
    //       if it's not `std::nullopt`.
    return getNameTagImpl<IsDefinition>(tokenTag(NameBuilder.name(Value),
                                                 tokenType(Value)),
                                        locationString(Value),
                                        Actions);
  }
  template<bool IsDefinition, typename ParentType, typename AnyType>
  std::string getNameTag(const ParentType &Parent,
                         const AnyType &Value,
                         RangeOf<llvm::StringRef> auto const &Actions) const {
    // TODO: build a warning based on `NameBuilder.warning(T)`
    //       if it's not `std::nullopt`.
    return getNameTagImpl<IsDefinition>(tokenTag(NameBuilder.name(Parent,
                                                                  Value),
                                                 tokenType(Parent, Value)),
                                        locationString(Parent, Value),
                                        Actions);
  }

public:
  std::string getDefinitionTag(const model::TypeDefinition &T) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::Comment,
                                     ptml::actions::EditType };
    return getNameTag<true>(T, Actions);
  }
  std::string getReferenceTag(const model::TypeDefinition &T) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::EditType };
    return getNameTag<false>(T, Actions);
  }

  std::string getDefinitionTag(const model::Function &F) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::Comment,
                                     ptml::actions::EditType };
    return getNameTag<true>(F, Actions);
  }
  std::string getReferenceTag(const model::Function &F) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::EditType };
    return getNameTag<false>(F, Actions);
  }

  std::string getDefinitionTag(const model::DynamicFunction &F) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::Comment,
                                     ptml::actions::EditType };
    return getNameTag<true>(F, Actions);
  }
  std::string getReferenceTag(const model::DynamicFunction &F) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::EditType };
    return getNameTag<false>(F, Actions);
  }

  std::string getDefinitionTag(const model::Segment &S) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::Comment,
                                     ptml::actions::EditType };
    return getNameTag<true>(Binary, S, Actions);
  }
  std::string getReferenceTag(const model::Segment &S) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::EditType };
    return getNameTag<false>(Binary, S, Actions);
  }

  template<typename Aggregate, typename Field>
  std::string getDefinitionTag(const Aggregate &A, const Field &F) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::Comment };
    return getNameTag<true>(A, F, Actions);
  }
  template<typename Aggregate, typename Field>
  std::string getReferenceTag(const Aggregate &A, const Field &F) const {
    constexpr std::array Actions = { ptml::actions::Rename };
    return getNameTag<false>(A, F, Actions);
  }

  std::string getDefinitionTag(const model::PrimitiveType &P) const {
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    return getNameTag<true>(P, Actions);
  }
  std::string getReferenceTag(const model::PrimitiveType &P) const {
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    return getNameTag<false>(P, Actions);
  }

  std::string
  getStackArgumentDefinitionTag(const model::RawFunctionDefinition &RFT) const {
    constexpr std::array Actions = { ptml::actions::Comment,
                                     ptml::actions::EditType };

    // For control-clicking the variable
    auto VarLoc = pipeline::locationString(revng::ranks::RawStackArguments,
                                           RFT.key());

    // For commenting on the struct
    revng_assert(RFT.stackArgumentsType());
    auto TypeLoc = pipeline::locationString(revng::ranks::TypeDefinition,
                                            RFT.stackArgumentsType()->key());

    llvm::StringRef Name = NameBuilder.Configuration.rawStackArgumentName();
    return getNameTagImpl<true>(tokenTag(Name, ptml::c::tokens::Variable),
                                VarLoc,
                                TypeLoc,
                                Actions);
  }

  std::string
  getStackArgumentReferenceTag(const model::RawFunctionDefinition &RFT) const {
    constexpr std::array Actions = { ptml::actions::EditType };
    auto VarLoc = pipeline::locationString(revng::ranks::RawStackArguments,
                                           RFT.key());

    llvm::StringRef Name = NameBuilder.Configuration.rawStackArgumentName();
    return getNameTagImpl<false>(tokenTag(Name, ptml::c::tokens::Variable),
                                 VarLoc,
                                 Actions);
  }

  std::string
  getReturnValueDefinitionTag(const model::RawFunctionDefinition &RFT,
                              const model::NamedTypedRegister Register) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::Comment,
                                     ptml::actions::EditType };
    std::string Location = returnValueLocationString(RFT, Register);

    return getNameTagImpl<true>(tokenTag(NameBuilder.name(RFT, Register),
                                         ptml::c::tokens::Field),
                                Location,
                                Actions);
  }

  std::string
  getReturnValueReferenceTag(const model::RawFunctionDefinition &RFT,
                             const model::NamedTypedRegister Register) const {
    constexpr std::array Actions = { ptml::actions::Rename,
                                     ptml::actions::EditType };
    std::string Location = returnValueLocationString(RFT, Register);

    return getNameTagImpl<false>(tokenTag(NameBuilder.name(RFT, Register),
                                          ptml::c::tokens::Field),
                                 Location,
                                 Actions);
  }

  std::string getVariableDefinitionTag(const model::Function &F,
                                       llvm::StringRef VariableName) const {
    // TODO: add the actions, at least rename!
    constexpr std::array<llvm::StringRef, 0> Actions = {};

    std::string Location = variableLocationString(F, VariableName);
    return getNameTagImpl<true>(tokenTag(VariableName,
                                         ptml::c::tokens::Variable),
                                Location,
                                Actions);
  }

  std::string getVariableReferenceTag(const model::Function &F,
                                      llvm::StringRef VariableName) const {
    constexpr std::array<llvm::StringRef, 0> Actions = {};

    std::string Location = variableLocationString(F, VariableName);
    return getNameTagImpl<false>(tokenTag(VariableName,
                                          ptml::c::tokens::Variable),
                                 Location,
                                 Actions);
  }

  std::string getGotoLabelDefinitionTag(const model::Function &F,
                                        llvm::StringRef GotoLabelName) const {
    // TODO: add the actions, at least rename!
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    std::string Location = pipeline::locationString(revng::ranks::GotoLabel,
                                                    F.key(),
                                                    GotoLabelName.str());
    return getNameTagImpl<true>(tokenTag(GotoLabelName,
                                         ptml::c::tokens::Variable),
                                Location,
                                Actions);
  }

  std::string getGotoLabelReferenceTag(const model::Function &F,
                                       llvm::StringRef GotoLabelName) const {
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    std::string Location = pipeline::locationString(revng::ranks::GotoLabel,
                                                    F.key(),
                                                    GotoLabelName.str());
    return getNameTagImpl<false>(tokenTag(GotoLabelName,
                                          ptml::c::tokens::Variable),
                                 Location,
                                 Actions);
  }

public:
  std::string getPrimitiveTag(model::PrimitiveKind::Values Kind,
                              uint64_t Size) const {
    return getReferenceTag(model::PrimitiveType(/* IsConst = */ false,
                                                Kind,
                                                Size));
  }
  std::string getVoidTag() const {
    return getReferenceTag(model::PrimitiveType(/* IsConst = */ false,
                                                model::PrimitiveKind::Void,
                                                0));
  }

public:
  template<bool IsDefinition>
  std::string
  getArtificialStructTag(const model::RawFunctionDefinition &RFT) const {
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    auto Location = pipeline::locationString(revng::ranks::ArtificialStruct,
                                             RFT.key());

    std::string Name = NameBuilder.artificialReturnValueWrapperName(RFT);
    return getNameTagImpl<IsDefinition>(tokenTag(Name, ptml::c::tokens::Type),
                                        Location,
                                        Actions);
  }

  template<bool IsDefinition>
  std::string getArrayWrapperTag(const model::ArrayType &Array) const {
    // TODO: currently there's no location dedicated to these, as such no action
    //       is possible.
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    auto Location = "";

    auto Name = NameBuilder.artificialArrayWrapperName(Array);
    return getNameTagImpl<IsDefinition>(tokenTag(std::move(Name),
                                                 ptml::c::tokens::Type),
                                        Location,
                                        Actions);
  }

  template<bool IsDefinition>
  std::string getHelperFunctionTag(llvm::StringRef Name) const {
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    auto Location = pipeline::locationString(revng::ranks::HelperFunction,
                                             Name.str());

    std::string Sanitized = model::sanitizeHelperName(Name);
    return getNameTagImpl<IsDefinition>(tokenTag(std::move(Sanitized),
                                                 ptml::c::tokens::Function),
                                        Location,
                                        Actions);
  }

  template<bool IsDefinition>
  std::string getHelperStructTag(llvm::StringRef Name) const {
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    auto Location = pipeline::locationString(revng::ranks::HelperStructType,
                                             Name.str());
    return getNameTagImpl<IsDefinition>(tokenTag(Name, ptml::c::tokens::Type),
                                        Location,
                                        Actions);
  }

  template<bool IsDefinition>
  std::string getHelperStructFieldTag(llvm::StringRef StructName,
                                      llvm::StringRef FieldName) const {
    constexpr std::array<llvm::StringRef, 0> Actions = {};
    auto Location = pipeline::locationString(revng::ranks::HelperStructField,
                                             StructName.str(),
                                             FieldName.str());
    return getNameTagImpl<IsDefinition>(tokenTag(FieldName,
                                                 ptml::c::tokens::Field),
                                        Location,
                                        Actions);
  }

  std::string
  getReturnValueTag(std::string &&Wrapped,
                    const model::TypeDefinition &FunctionType) const {
    constexpr std::array Actions = { ptml::actions::Comment };
    auto Location = pipeline::locationString(revng::ranks::ReturnValue,
                                             FunctionType.key());

    return getNameTagImpl<false>(getTag(ptml::tags::Span, std::move(Wrapped)),
                                 std::move(Location),
                                 Actions);
  }

  std::string getDebugInfoTag(std::string &&Wrapped,
                              std::string &&Location) const {
    constexpr std::array Actions = { ptml::actions::CodeSwitch,
                                     ptml::actions::Comment };

    return getNameTagImpl<false>(getTag(ptml::tags::Span, std::move(Wrapped)),
                                 "",
                                 std::move(Location),
                                 Actions);
  }

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

    return ptml::functionComment(*this,
                                 Function,
                                 Binary,
                                 "///",
                                 0,
                                 Width,
                                 NameBuilder);
  }

  std::string getStatementComment(const model::StatementComment &Text,
                                  const std::string &CommentLocation,
                                  llvm::StringRef EmittedAt) const {
    const model::Configuration &Configuration = Binary.Configuration();
    uint64_t LineWidth = Configuration.commentLineWidth();

    // TODO: do not rely on `Out`'s indentation, since there's no guarantee it's
    //       the same stream (even if it usually is).
    uint64_t Width = LineWidth - Out->currentIndentation();
    return ptml::statementComment(*this,
                                  Text,
                                  CommentLocation,
                                  EmittedAt,
                                  "//",
                                  0,
                                  Width);
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
  std::string getNamedCInstance(const model::Type &Type,
                                llvm::StringRef InstanceName = "",
                                bool OmitInnerTypeName = false) const;

  std::string getTypeName(const model::Type &Type) const {
    return getNamedCInstance(Type, "");
  }

  /// Return a string containing the C Type name of the return type of
  /// \a FunctionType, and a (possibly empty) \a InstanceName.
  /// \note If F returns more than one value, the name of the wrapping struct
  ///       will be returned instead.
  std::string
  getNamedInstanceOfReturnType(const model::TypeDefinition &FunctionType,
                               llvm::StringRef InstanceName,
                               bool IsDefinition) const;

  std::string getReturnTypeName(const model::TypeDefinition &FunctionType,
                                bool IsDefinition) const {
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
  void printForwardTypeDeclaration(const model::TypeDefinition &T);

  void printTypeDeclaration(const model::TypedefDefinition &Typedef);
  void printTypeDeclaration(const model::RawFunctionDefinition &F);
  void printTypeDeclaration(const model::CABIFunctionDefinition &F);
  void printTypeDeclaration(const model::TypeDefinition &T);

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
