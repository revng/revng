//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/Support/Annotations.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"

using T = model::TypeDefinition;
void ptml::CTypeBuilder::printForwardTypeDeclaration(const T &Type) {
  revng_assert(not isDeclarationTheSameAsDefinition(Type));

  auto TypeNameReference = getLocationReference(Type);
  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getTypeKeyword(Type) << " "
       << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " "
       << TypeNameReference << " " << TypeNameReference << ";\n";
}

void ptml::CTypeBuilder::printTypeDefinition(const model::EnumDefinition &E,
                                             std::string &&Suffix) {
  // We have to make the enum of the correct size of the underlying type
  auto ByteSize = *E.size();
  revng_assert(ByteSize <= 8);
  size_t FullMask = std::numeric_limits<size_t>::max();
  size_t MaxBitPatternInEnum = (ByteSize == 8) ?
                                 FullMask :
                                 ((FullMask) xor (FullMask << (8 * ByteSize)));

  std::string Underlying = E.underlyingType().getCName();
  *Out << getModelComment(E) << getKeyword(ptml::CBuilder::Keyword::Enum) << " "
       << ptml::AttributeRegistry::getAnnotation<"_ENUM_UNDERLYING">(Underlying)
       << " " << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " "
       << getLocationDefinition(E) << " ";

  {
    Scope Scope(*Out);

    using COperator = ptml::CBuilder::Operator;
    for (const auto &Entry : E.Entries()) {
      *Out << getModelComment(Entry) << getLocationDefinition(E, Entry) << " "
           << getOperator(COperator::Assign) << " " << getHex(Entry.Value())
           << ",\n";
    }

    if (Configuration.EnablePrintingOfTheMaximumEnumValue) {
      // This ensures the enum is exactly as large as the Underlying type
      *Out << tokenTag(("_enum_max_value_" + NameBuilder.name(E)).str(),
                       ptml::c::tokens::Field)
           << " " + getOperator(COperator::Assign) + " "
           << getHex(MaxBitPatternInEnum) << ",\n";
    }
  }

  *Out << std::move(Suffix) << ";\n";
}

void ptml::CTypeBuilder::printPadding(uint64_t FieldOffset,
                                      uint64_t NextOffset) {
  revng_assert(FieldOffset <= NextOffset);
  if (FieldOffset == NextOffset)
    return; // There is no padding

  if (Configuration.EnableExplicitPaddingMode) {
    *Out << tokenTag("uint8_t", ptml::c::tokens::Type) << " "
         << tokenTag(NameBuilder.paddingFieldName(FieldOffset),
                     ptml::c::tokens::Field)
         << "[" << getNumber(NextOffset - FieldOffset) << "];\n";
  } else {
    *Out << ptml::AttributeRegistry::getAnnotation<"_START_AT">(NextOffset)
         << "\n";
  }
}

void ptml::CTypeBuilder::printTypeDefinition(const model::StructDefinition &S,
                                             std::string &&Suffix) {

  *Out << getModelComment(S) << getKeyword(ptml::CBuilder::Keyword::Struct)
       << " " << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " ";

  if (S.CanContainCode())
    *Out << ptml::AttributeRegistry::getAttribute<"_CAN_CONTAIN_CODE">() << " ";

  if (Configuration.EnableStructSizeAnnotation)
    *Out << ptml::AttributeRegistry::getAnnotation<"_SIZE">(S.Size()) << " ";

  *Out << getLocationDefinition(S) << " ";

  {
    Scope Scope(*Out, ptml::c::scopes::StructBody);

    size_t PreviousOffset = 0ULL;
    for (const auto &Field : S.Fields()) {
      printPadding(PreviousOffset, Field.Offset());

      auto *Definition = Field.Type()->skipToDefinition();
      if (not Definition or not shouldInline(*Definition)) {
        auto F = getLocationDefinition(S, Field);
        *Out << getModelComment(Field) << getNamedCInstance(*Field.Type(), F)
             << ";\n";
      } else {
        printInlineDefinition(NameBuilder.name(S, Field).str(), *Field.Type());
      }

      PreviousOffset = Field.Offset() + Field.Type()->size().value();
    }

    if (Configuration.EnableExplicitPaddingMode)
      printPadding(PreviousOffset, S.Size());
  }

  *Out << std::move(Suffix) << ";\n";
}

void ptml::CTypeBuilder::printTypeDefinition(const model::UnionDefinition &U,
                                             std::string &&Suffix) {
  *Out << getModelComment(U) << getKeyword(ptml::CBuilder::Keyword::Union)
       << " " << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " ";
  *Out << getLocationDefinition(U) << " ";

  {
    Scope Scope(*Out, ptml::c::scopes::UnionBody);
    for (const auto &Field : U.Fields()) {
      auto *Definition = Field.Type()->skipToDefinition();
      if (not Definition or not shouldInline(*Definition)) {
        auto F = getLocationDefinition(U, Field);
        *Out << getModelComment(Field) << getNamedCInstance(*Field.Type(), F)
             << ";\n";
      } else {
        printInlineDefinition(NameBuilder.name(U, Field).str(), *Field.Type());
      }
    }
  }

  *Out << std::move(Suffix) << ";\n";
}

using TD = model::TypedefDefinition;
void ptml::CTypeBuilder::printTypeDeclaration(const TD &Typedef) {
  if (isDeclarationTheSameAsDefinition(Typedef))
    *Out << getModelComment(Typedef);

  auto Type = getLocationDefinition(Typedef);
  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getNamedCInstance(*Typedef.UnderlyingType(), Type) << ";\n";
}

/// Generate the definition of a new struct type that wraps all the return
/// values of \a F. The name of the struct type is provided by the caller.
using RFT = model::RawFunctionDefinition;
void ptml::CTypeBuilder::generateReturnValueWrapper(const RFT &F) {
  revng_assert(F.ReturnValues().size() > 1);
  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getKeyword(ptml::CBuilder::Keyword::Struct) << " "
       << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " ";

  {
    Scope Scope(*Out, ptml::c::scopes::StructBody);
    for (auto &[Index, ReturnValue] : llvm::enumerate(F.ReturnValues())) {
      std::string
        ActionLocation = pipeline::locationString(revng::ranks::ReturnRegister,
                                                  F.key(),
                                                  ReturnValue.key());

      std::string
        FieldString = tokenTag(NameBuilder.returnValueName(F, ReturnValue),
                               ptml::c::tokens::Field)
                        .addAttribute(ptml::attributes::ActionContextLocation,
                                      ActionLocation)
                        .toString();
      *Out << getNamedCInstance(*ReturnValue.Type(), FieldString) << ";\n";
    }
  }

  *Out << " " << getReturnTypeName(F, true) << ";\n";
}

/// If the function has more than one return value, generate a wrapper struct
/// that contains them.
void ptml::CTypeBuilder::printFunctionWrappers(const RFT &F) {
  if (F.ReturnValues().size() > 1)
    generateReturnValueWrapper(F);

  for (auto &Arg : F.Arguments())
    revng_assert(Arg.Type()->isScalar());
}

/// Print a typedef for a RawFunctionDefinition, that can be used when you have
/// a variable that is a pointer to a function.
void ptml::CTypeBuilder::printTypeDeclaration(const RFT &F) {
  printFunctionWrappers(F);

  *Out << getModelComment(F) << getKeyword(ptml::CBuilder::Keyword::Typedef)
       << " ";
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionPrototype(F);
  *Out << ";\n";
}

/// Generate the definition of a new struct type that wraps \a ArrayType.
/// This is used to wrap array arguments or array return values of
/// CABI functions.
void ptml::CTypeBuilder::generateArrayWrapper(const model::ArrayType
                                                &ArrayType) {
  // Check if the wrapper was already added
  auto &&[It, IsNew] = ArtificialNameCache.emplace(ArrayType,
                                                   getArrayWrapper(ArrayType));
  if (not IsNew)
    return;

  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getKeyword(ptml::CBuilder::Keyword::Struct) << " "
       << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " ";
  {
    Scope Scope(*Out, ptml::c::scopes::StructBody);
    *Out << getNamedCInstance(ArrayType,
                              NameBuilder.artificialArrayWrapperFieldName())
         << ";\n";
  }
  *Out << " " << tokenTag(It->second, ptml::c::tokens::Type) << ";\n";
}

/// If the return value or any of the arguments is an array, generate a wrapper
/// struct for each of them, if it's not already in the cache.
using CFT = model::CABIFunctionDefinition;
void ptml::CTypeBuilder::printFunctionWrappers(const CFT &F) {
  if (not F.ReturnType().isEmpty())
    if (auto *Array = F.ReturnType()->getArray())
      generateArrayWrapper(*Array);

  for (auto &Arg : F.Arguments())
    if (auto *Array = Arg.Type()->getArray())
      generateArrayWrapper(*Array);
}

/// Print a typedef for a CABI function, that can be used when you have
/// a variable that is a pointer to a function.
void ptml::CTypeBuilder::printTypeDeclaration(const CFT &F) {
  printFunctionWrappers(F);

  *Out << getModelComment(F) << getKeyword(ptml::CBuilder::Keyword::Typedef)
       << " ";

  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionPrototype(F);
  *Out << ";\n";
}

void ptml::CTypeBuilder::printTypeDeclaration(const model::TypeDefinition &T) {
  if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&T))
    printForwardTypeDeclaration(*Enum);

  else if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(&T))
    printForwardTypeDeclaration(*Struct);

  else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(&T))
    printForwardTypeDeclaration(*Union);

  else if (auto *Typedef = llvm::dyn_cast<model::TypedefDefinition>(&T))
    printTypeDeclaration(*Typedef);

  else if (auto *RFT = llvm::dyn_cast<model::RawFunctionDefinition>(&T))
    printTypeDeclaration(*RFT);

  else if (auto *CFT = llvm::dyn_cast<model::CABIFunctionDefinition>(&T))
    printTypeDeclaration(*CFT);

  else
    revng_abort("Unsupported type definition.");
}

void ptml::CTypeBuilder::printTypeDefinition(const model::TypeDefinition &T) {
  if (isDeclarationTheSameAsDefinition(T))
    printTypeDeclaration(T);

  else if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(&T))
    printTypeDefinition(*Struct);

  else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(&T))
    printTypeDefinition(*Union);

  else if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&T))
    printTypeDefinition(*Enum);

  else
    revng_abort("Unsupported type definition.");
}

void ptml::CTypeBuilder::printInlineDefinition(llvm::StringRef Name,
                                               const model::Type &T) {
  const model::TypeDefinition *Definition = T.skipToDefinition();
  revng_assert(Definition, "Primitives cannot be printed inline.");

  auto Suffix = getNamedCInstance(T, Name, {}, true).str().str();
  if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(Definition)) {
    printTypeDefinition(*Struct, std::move(Suffix));

  } else if (auto *U = llvm::dyn_cast<model::UnionDefinition>(Definition)) {
    printTypeDefinition(*U, std::move(Suffix));

  } else if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(Definition)) {
    printTypeDefinition(*Enum, std::move(Suffix));

  } else {
    revng_abort("Only enums, structs, and unions can be printed inline.");
  }
}

static Logger<> InlineTypeLog{ "inline-type-selection" };

void ptml::CTypeBuilder::collectInlinableTypes() {
  if (not DependencyCache.has_value())
    DependencyCache = buildDependencyGraph(Binary.TypeDefinitions());

  StackFrameTypeCache = {};
  for (const model::Function &Function : Binary.Functions())
    if (auto *StackFrame = Function.stackFrameType())
      StackFrameTypeCache.insert(StackFrame->key());

  if (Configuration.EnableTypeInlining
      || Configuration.EnableStackFrameInlining) {
    std::map<model::TypeDefinition::Key, uint64_t> DependentTypeCount;
    for (const auto *Node : DependencyCache->nodes()) {
      if (isDeclarationTheSameAsDefinition(*Node->T)) {
        // Skip types that never produce a definition since there's no point
        // inlining them.
        continue;
      }

      auto &&[Iterator, _] = DependentTypeCount.try_emplace(Node->T->key(), 0);
      Iterator->second += Node->predecessorCount();
      if (Node->K == TypeNode::Kind::Declaration) {
        // Ignore a reference from a type definition to its own declaration.
        // But only do so if there is exactly one. If there are more, keep it in
        // order to ensure it is never marked for inlining.
        auto SelfEdgeCounter = [Key = Node->T->key()](auto *N) {
          return N->T->key() == Key;
        };
        if (llvm::count_if(Node->predecessors(), SelfEdgeCounter) == 1)
          --Iterator->second;

        // Since dependency graph does not take functions into account,
        // explicitly add one "use" to each struct that appears as a function
        // stack frame.
        if (StackFrameTypeCache.contains(Node->T->key()))
          ++Iterator->second;
      }

      if (InlineTypeLog.isEnabled()) {
        InlineTypeLog << getNodeLabel(Node) << "' is depended on by: {\n";

        for (auto *Predecessor : Node->predecessors())
          InlineTypeLog << "- " << getNodeLabel(Predecessor) << '\n';

        InlineTypeLog << "}\n" << DoLog;
      }
    }

    auto SingleDependencyFilter = std::views::filter([](const auto &Pair) {
      return Pair.second == 1;
    });
    TypesToInlineCache = DependentTypeCount | SingleDependencyFilter
                         | std::views::keys
                         | revng::to<std::set<model::TypeDefinition::Key>>();

    if (InlineTypeLog.isEnabled()) {
      revng_log(InlineTypeLog, "Final list of types that can be inlined: {");
      {
        LoggerIndent Indent{ InlineTypeLog };
        for (const model::TypeDefinition::Key &T : TypesToInlineCache)
          revng_log(InlineTypeLog, ::toString(T));
      }
      revng_log(InlineTypeLog, "}");
    }
  }

  if (Configuration.EnableStackFrameInlining && InlineTypeLog.isEnabled()) {
    revng_log(InlineTypeLog, "Which also includes stack frames: {");
    {
      LoggerIndent Indent{ InlineTypeLog };
      for (const model::TypeDefinition::Key &T : StackFrameTypeCache)
        if (TypesToInlineCache.contains(T))
          revng_log(InlineTypeLog, ::toString(T));
    }
    revng_log(InlineTypeLog, "}");
  }

  InlinableCacheIsReady = true;
}

static Logger<> TypePrinterLog{ "type-definition-printer" };

void ptml::CTypeBuilder::printTypeDefinitions() {
  if (not DependencyCache.has_value())
    DependencyCache = buildDependencyGraph(Binary.TypeDefinitions());

  const auto &TypeNodes = DependencyCache->TypeNodes();

  std::set<const TypeDependencyNode *> Defined;
  for (const auto *Root : DependencyCache->nodes()) {
    revng_log(TypePrinterLog, "PostOrder from Root:" << getNodeLabel(Root));

    for (const auto *Node : llvm::post_order_ext(Root, Defined)) {

      LoggerIndent PostOrderIndent{ TypePrinterLog };
      revng_log(TypePrinterLog, "post_order visiting: " << getNodeLabel(Node));

      const model::TypeDefinition *NodeT = Node->T;
      const auto DeclKind = Node->K;

      if (Configuration.TypesToOmit.contains(NodeT->key())) {
        revng_log(TypePrinterLog, "Omitted");
        continue;
      }

      constexpr auto Declaration = TypeNode::Kind::Declaration;

      if (DeclKind == Declaration) {
        revng_log(TypePrinterLog, "Declaration");

        // Print the declaration. Notice that the forward declarations are
        // emitted even for inlined types, because it's only the full definition
        // that will be inlined.
        printTypeDeclaration(*NodeT);

      } else {
        revng_log(TypePrinterLog, "Definition");

        revng_assert(Defined.contains(TypeNodes.at({ NodeT, Declaration })));
        if (isDeclarationTheSameAsDefinition(*NodeT) or shouldInline(*NodeT)) {
          continue;
        }

        revng_log(TypePrinterLog, "printTypeDefinition");
        printTypeDefinition(*NodeT);
      }

      *Out << "\n";
    }
    revng_log(TypePrinterLog, "PostOrder DONE");
  }
}
