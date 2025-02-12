//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/CommonTypeMethods.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Annotations.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"

using T = model::TypeDefinition;
void ptml::CTypeBuilder::printForwardDeclaration(const T &Type) {
  revng_assert(not isDeclarationTheSameAsDefinition(Type));

  auto TypeNameReference = getLocationReference(Type);
  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getTypeKeyword(Type) << " "
       << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " "
       << TypeNameReference << " " << TypeNameReference << ";\n";
}

void ptml::CTypeBuilder::printDefinition(const model::EnumDefinition &E,
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

void ptml::CTypeBuilder::printDefinition(const model::StructDefinition &S,
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

void ptml::CTypeBuilder::printDefinition(const model::UnionDefinition &U,
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
void ptml::CTypeBuilder::printDeclaration(const TD &Typedef) {
  if (isDeclarationTheSameAsDefinition(Typedef))
    *Out << getModelComment(Typedef);

  auto Type = getLocationDefinition(Typedef);
  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getNamedCInstance(*Typedef.UnderlyingType(), Type) << ";\n";
}

using RFT = model::RawFunctionDefinition;

void ptml::CTypeBuilder::printReturnValueWrapperImpl(const RFT &F,
                                                     bool IsDefinition) {
  revng_assert(F.ReturnValues().size() > 1);
  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getKeyword(ptml::CBuilder::Keyword::Struct) << " "
       << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " "
       << getReturnTypeName(F, false) << " ";

  if (IsDefinition) {
    {
      Scope Scope(*Out, ptml::c::scopes::StructBody);
      for (auto &[Index, ReturnValue] : llvm::enumerate(F.ReturnValues())) {
        using pipeline::locationString;
        std::string
          ActionLocation = locationString(revng::ranks::ReturnRegister,
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
      *Out << " ";
    }
  }

  *Out << getReturnTypeName(F, IsDefinition) << ";\n";
}

/// Print a typedef for a RawFunctionDefinition, that can be used when you
/// have a variable that is a pointer to a function.
void ptml::CTypeBuilder::printDeclaration(const RFT &F) {

  *Out << getModelComment(F) << getKeyword(ptml::CBuilder::Keyword::Typedef)
       << " ";
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the
  // type.
  printFunctionPrototype(F);
  *Out << ";\n";
}

void ptml::CTypeBuilder::printArrayWrapperImpl(const model::ArrayType
                                                 &ArrayType,
                                               bool IsDefinition) {
  // Check if the wrapper was already added
  auto &&[It, IsNew] = ArtificialNameCache.emplace(ArrayType,
                                                   getArrayWrapper(ArrayType));
  revng_assert(IsNew or It->second == getArrayWrapper(ArrayType));

  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getKeyword(ptml::CBuilder::Keyword::Struct) << " "
       << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " "
       << tokenTag(It->second, ptml::c::tokens::Type) << " ";

  if (IsDefinition) {
    {
      Scope Scope(*Out, ptml::c::scopes::StructBody);
      *Out << getNamedCInstance(ArrayType,
                                NameBuilder.artificialArrayWrapperFieldName())
           << ";\n";
    }
    *Out << " ";
  }
  *Out << tokenTag(It->second, ptml::c::tokens::Type) << ";\n";
}

using CFT = model::CABIFunctionDefinition;

/// Print a typedef for a CABI function, that can be used when you have
/// a variable that is a pointer to a function.
void ptml::CTypeBuilder::printDeclaration(const CFT &F) {

  *Out << getModelComment(F) << getKeyword(ptml::CBuilder::Keyword::Typedef)
       << " ";

  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the
  // type.
  printFunctionPrototype(F);
  *Out << ";\n";
}

void ptml::CTypeBuilder::printDeclaration(const model::TypeDefinition &T) {
  if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&T))
    printForwardDeclaration(*Enum);

  else if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(&T))
    printForwardDeclaration(*Struct);

  else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(&T))
    printForwardDeclaration(*Union);

  else if (auto *Typedef = llvm::dyn_cast<model::TypedefDefinition>(&T))
    printDeclaration(*Typedef);

  else if (auto *RFT = llvm::dyn_cast<model::RawFunctionDefinition>(&T))
    printDeclaration(*RFT);

  else if (auto *CFT = llvm::dyn_cast<model::CABIFunctionDefinition>(&T))
    printDeclaration(*CFT);

  else
    revng_abort("Unsupported type definition.");
}

void ptml::CTypeBuilder::printDefinition(const model::TypeDefinition &T) {
  if (isDeclarationTheSameAsDefinition(T))
    printDeclaration(T);

  else if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(&T))
    printDefinition(*Struct);

  else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(&T))
    printDefinition(*Union);

  else if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&T))
    printDefinition(*Enum);

  else
    revng_abort("Unsupported type definition.");
}

void ptml::CTypeBuilder::printInlineDefinition(llvm::StringRef Name,
                                               const model::Type &T) {
  const model::TypeDefinition *Definition = T.skipToDefinition();
  revng_assert(Definition, "Primitives cannot be printed inline.");

  auto Suffix = getNamedCInstance(T, Name, {}, true).str().str();
  if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(Definition)) {
    printDefinition(*Struct, std::move(Suffix));

  } else if (auto *U = llvm::dyn_cast<model::UnionDefinition>(Definition)) {
    printDefinition(*U, std::move(Suffix));

  } else if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(Definition)) {
    printDefinition(*Enum, std::move(Suffix));

  } else {
    revng_abort("Only enums, structs, and unions can be printed inline.");
  }
}

static Logger<> InlineTypeLog{ "inline-type-selection" };

void ptml::CTypeBuilder::collectInlinableTypes() {
  if (not DependencyCache.has_value())
    DependencyCache = DependencyGraph::make(Binary);

  StackFrameTypeCache = {};
  for (const model::Function &Function : Binary.Functions())
    if (auto *StackFrame = Function.stackFrameType())
      StackFrameTypeCache.insert(StackFrame->key());

  if (Configuration.EnableTypeInlining
      or Configuration.EnableStackFrameInlining) {
    std::map<model::TypeDefinition::Key, uint64_t> DependentTypeCount;
    for (const auto *Node : DependencyCache->nodes()) {
      const model::TypeDefinition &T = *Node->T->tryGetAsDefinition();

      // Skip stuff we never want to inline.
      if (Node->isArtificial() or isDeclarationTheSameAsDefinition(T)) {
        // Skip types that never produce a definition since there's no point
        // inlining them.
        // Artificial nodes are always struct, so in principle they could be
        // inlined. However, there are various reasons why we don't want to
        // ever inline artificial structs.
        //
        // Artificial structs are currently emitted in C for 3 reasons.
        // 1. For representing return types of RawFunctionDefinitions that
        // return a RegisterSet.
        // 2. For wrapping array arguments types and array return types for
        // CABIFunctionDefinitions.
        // 3. For wrapping individual instances of subtypes whose type is a
        // pointer-to-array, to break potentially infinite loops.
        //
        // In each of these case we have reasons no to inline them.
        // Respectively for the three cases above.
        //
        // 1. inlining a struct wrapper in the declaration of a
        // RawFunctionDefinition (called f) would look something like
        //
        //     struct f_return_wrapper { uint64_t rax; uint64_t rdx; } f();
        //
        // This is ugly and very verbose. It's better not to inline this.
        //
        // 2. inlining a struct wrapper for an array argument type or return
        // value type would look like
        //
        //     struct f_array_return_wrapper { uint8_t array[4]; }
        //     f(struct f_arg_0_type_wrapper { uint8_t array[8]; });
        //
        // This is ugly and very verbose. It's better not to inline this.
        //
        // 3. inlining a struct wrapper for specific instance of an array
        // subtype would look like
        //
        //     struct external {
        //       struct wrapper {
        //         int32_t array[8];
        //       } *pointer_to_array;
        //     };
        //
        // This in principle could be desirable, but doing this requires us to
        // track not only the keys of the TypeDefinitions that must always be
        // inlined, but pairs of <parent_type, subtype> such that the given
        // instance of the subtype needs a struct wrapper in the parent_type.
        // This is feasible but requires sensible redesign of how we track the
        // types to be inlined, so we don't do it for now and just ban
        // inlining for artificial wrapper types.
        continue;
      }

      auto [Iterator, _] = DependentTypeCount.try_emplace(T.key(), 0);
      Iterator->second += Node->predecessorCount();
      if (Node->isDeclaration()) {
        // Ignore a reference from a type definition to its own declaration.
        // But only do so if there is exactly one. If there are more, keep it
        // in order to ensure it is never marked for inlining.
        using TDN = TypeDependencyNode;
        const TDN *DefNode = DependencyCache->getDefinition(&T);
        if (1 == llvm::count(Node->predecessors(), DefNode))
          --Iterator->second;

        // Since dependency graph does not take functions into account,
        // explicitly add one "use" to each struct that appears as a function
        // stack frame.
        if (StackFrameTypeCache.contains(T.key()))
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
    DependencyCache = DependencyGraph::make(Binary);

  std::set<const TypeDependencyNode *> Defined;
  for (const auto *Root : DependencyCache->nodes()) {
    revng_log(TypePrinterLog, "PostOrder from Root:" << getNodeLabel(Root));

    for (const auto *Node : llvm::post_order_ext(Root, Defined)) {

      LoggerIndent PostOrderIndent{ TypePrinterLog };
      revng_log(TypePrinterLog, "post_order visiting: " << getNodeLabel(Node));

      if (Node->isArtificial()) {
        using namespace llvm; // For isa, dyn_cast

        const model::Type *T = Node->T.get();
        const model::ArrayType *Array = dyn_cast<model::ArrayType>(T);

        bool ReturnsRegisterSet = false;
        const model::RawFunctionDefinition *RF = nullptr;
        if (const auto *DT = dyn_cast<model::DefinedType>(T)) {
          RF = dyn_cast<model::RawFunctionDefinition>(&DT->unwrap());
          if (RF) {

            using Layout = abi::FunctionType::Layout;
            auto RetMethod = Layout::make(*RF).returnMethod();

            using namespace abi::FunctionType::ReturnMethod;
            ReturnsRegisterSet = RetMethod == RegisterSet;
          }
        }

        revng_assert(Array or ReturnsRegisterSet);
        revng_log(TypePrinterLog,
                  "Warning: ignoring unexpected artificial TypeDependencyNode"
                    << getNodeLabel(Node));

        if (Array) {
          printArrayWrapperImpl(*Array, Node->isDefinition());
        } else {
          revng_assert(ReturnsRegisterSet and RF);
          printReturnValueWrapperImpl(*RF, Node->isDefinition());
        }

        *Out << "\n";

        continue;
      }

      const model::TypeDefinition *NodeT = Node->T->tryGetAsDefinition();

      if (Configuration.TypesToOmit.contains(NodeT->key())) {
        revng_log(TypePrinterLog, "Omitted");
        continue;
      }

      constexpr auto Declaration = TypeNode::Kind::Declaration;

      if (Node->isDeclaration()) {
        revng_log(TypePrinterLog, "Declaration");

        // Print the declaration. Notice that the forward declarations are
        // emitted even for inlined types, because it's only the full
        // definition that will be inlined.
        printDeclaration(*NodeT);

      } else {
        revng_log(TypePrinterLog, "Definition");

        revng_assert(Defined.contains(DependencyCache->getDeclaration(NodeT)));
        if (isDeclarationTheSameAsDefinition(*NodeT) or shouldInline(*NodeT)) {
          continue;
        }

        revng_log(TypePrinterLog, "printDefinition");
        printDefinition(*NodeT);
      }

      *Out << "\n";
    }
    revng_log(TypePrinterLog, "PostOrder DONE");
  }
}
