//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Support/Annotations.h"
#include "revng/TypeNames/ModelCBuilder.h"

using T = model::TypeDefinition;
void ptml::ModelCBuilder::printForwardDeclaration(const T &Type) {
  revng_assert(not isDeclarationTheSameAsDefinition(Type));

  auto TypeNameReference = getReferenceTag(Type);
  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getTypeKeyword(Type) << " "
       << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " "
       << TypeNameReference << " " << TypeNameReference << ";\n";
}

void ptml::ModelCBuilder::printDefinition(const model::EnumDefinition &E,
                                          std::string &&Suffix) {
  // We have to make the enum of the correct size of the underlying type
  auto ByteSize = *E.size();
  revng_assert(ByteSize <= 8);
  size_t FullMask = std::numeric_limits<size_t>::max();
  size_t MaxBitPatternInEnum = (ByteSize == 8) ?
                                 FullMask :
                                 ((FullMask) xor (FullMask << (8 * ByteSize)));

  std::string UndTag = getReferenceTag(E.underlyingType());
  auto U = ptml::AttributeRegistry::getAnnotation<"_ENUM_UNDERLYING">(UndTag);
  std::string EnumLine = getModelCommentWithoutLeadingNewline(E)
                         + getKeyword(ptml::CBuilder::Keyword::Enum) + " " + U
                         + " "
                         + ptml::AttributeRegistry::getAttribute<"_PACKED">()
                         + " " + getDefinitionTag(E) + " ";
  *Out << getCommentableTag(std::move(EnumLine), E);

  {
    Scope Scope(*Out);

    using COperator = ptml::CBuilder::Operator;
    for (const auto &Entry : E.Entries()) {
      std::string Result = getModelComment(Entry) + getDefinitionTag(E, Entry)
                           + " " + getOperator(COperator::Assign) + " "
                           + getHex(Entry.Value()) + ',';
      *Out << getCommentableTag(std::move(Result), E, Entry) << '\n';
    }

    if (Configuration.EnablePrintingOfTheMaximumEnumValue) {
      // This ensures the enum is exactly as large as the Underlying type

      auto Prefix = NameBuilder.Configuration.maximumEnumValuePrefix();
      *Out << tokenTag(Prefix.str() + NameBuilder.name(E),
                       ptml::c::tokens::Field)
           << " " + getOperator(COperator::Assign) + " "
           << getHex(MaxBitPatternInEnum) << ",\n";
    }
  }

  *Out << std::move(Suffix) << ";\n";
}

void ptml::ModelCBuilder::printPadding(uint64_t FieldOffset,
                                       uint64_t NextOffset) {
  if (Configuration.EnableExplicitPadding) {
    revng_assert(FieldOffset <= NextOffset);
    if (FieldOffset == NextOffset)
      return; // There is no padding

    *Out << tokenTag("uint8_t", ptml::c::tokens::Type) << " "
         << tokenTag(NameBuilder.paddingFieldName(FieldOffset),
                     ptml::c::tokens::Field)
         << "[" << getNumber(NextOffset - FieldOffset) << "];\n";
  }
}

void ptml::ModelCBuilder::printDefinition(const model::StructDefinition &S,
                                          std::string &&Suffix) {

  std::string StructLine = getModelCommentWithoutLeadingNewline(S)
                           + getKeyword(ptml::CBuilder::Keyword::Struct) + " "
                           + ptml::AttributeRegistry::getAttribute<"_PACKED">()
                           + " ";

  if (S.CanContainCode())
    StructLine += ptml::AttributeRegistry::getAttribute<"_CAN_CONTAIN_CODE">()
                  + " ";

  StructLine += ptml::AttributeRegistry::getAnnotation<"_SIZE">(S.Size()) + " "
                + getDefinitionTag(S) + " ";

  *Out << getCommentableTag(std::move(StructLine), S);

  {
    Scope Scope(*Out, ptml::c::scopes::StructBody);

    size_t PreviousOffset = 0ULL;
    for (const auto &Field : S.Fields()) {
      printPadding(PreviousOffset, Field.Offset());

      auto F = getDefinitionTag(S, Field);
      std::string Result = getModelComment(Field)
                           + getNamedCInstance(*Field.Type(), F) + ';';
      *Out << getCommentableTag(std::move(Result), S, Field) << '\n';

      PreviousOffset = Field.Offset() + Field.Type()->size().value();
    }

    printPadding(PreviousOffset, S.Size());
  }

  *Out << std::move(Suffix) << ";\n";
}

void ptml::ModelCBuilder::printDefinition(const model::UnionDefinition &U,
                                          std::string &&Suffix) {
  std::string UnionLine = getModelCommentWithoutLeadingNewline(U)
                          + getKeyword(ptml::CBuilder::Keyword::Union) + " "
                          + ptml::AttributeRegistry::getAttribute<"_PACKED">()
                          + " " + getDefinitionTag(U) + " ";
  *Out << getCommentableTag(std::move(UnionLine), U);
  {
    Scope Scope(*Out, ptml::c::scopes::UnionBody);
    for (const auto &Field : U.Fields()) {
      auto F = getDefinitionTag(U, Field);
      std::string Result = getModelComment(Field)
                           + getNamedCInstance(*Field.Type(), F) + ';';
      *Out << getCommentableTag(std::move(Result), U, Field) << '\n';
    }
  }

  *Out << std::move(Suffix) << ";\n";
}

using TD = model::TypedefDefinition;
void ptml::ModelCBuilder::printDeclaration(const TD &Typedef) {
  std::string TypedefString;
  if (isDeclarationTheSameAsDefinition(Typedef))
    TypedefString = getModelCommentWithoutLeadingNewline(Typedef);

  auto Type = getDefinitionTag(Typedef);
  TypedefString += getKeyword(ptml::CBuilder::Keyword::Typedef) + " "
                   + getNamedCInstance(*Typedef.UnderlyingType(), Type) + ';';

  *Out << getCommentableTag(std::move(TypedefString), Typedef) << '\n';
}

/// Generate the definition of a new struct type that wraps all the return
/// values of \a F. The name of the struct type is provided by the caller.
using RFT = model::RawFunctionDefinition;
void ptml::ModelCBuilder::printReturnTypeWrapperDefinition(const RFT &F) {

  using abi::FunctionType::Layout;
  auto TheLayout = Layout::make(F);
  using namespace abi::FunctionType::ReturnMethod;
  revng_assert(TheLayout.returnMethod() == RegisterSet);

  std::string StructLine = getWrapperStructComment(F)
                           + getKeyword(ptml::CBuilder::Keyword::Struct) + " "
                           + ptml::AttributeRegistry::getAttribute<"_PACKED">()
                           + " ";

  StructLine += getArtificialStructTag</*IsDefinition*/ true>(F) + " ";

  *Out << getReturnValueTag(std::move(StructLine), F);

  {
    Scope Scope(*Out, ptml::c::scopes::StructBody);
    for (auto &[Index, ReturnValue] : llvm::enumerate(F.ReturnValues())) {
      *Out << getModelComment(ReturnValue);

      auto FieldString = getReturnValueDefinitionTag(F, ReturnValue);
      auto Line = getNamedCInstance(*ReturnValue.Type(), FieldString) + ';';
      *Out << getReturnValueRegisterTag(std::move(Line), F, ReturnValue)
           << '\n';
    }
  }

  *Out << ";\n";
}

/// Generate the definition of a new struct type that wraps all the return
/// values of \a F. The name of the struct type is provided by the caller.
using RFT = model::RawFunctionDefinition;
void ptml::ModelCBuilder::printReturnTypeWrapperDeclaration(const RFT &F) {

  using abi::FunctionType::Layout;
  auto TheLayout = Layout::make(F);
  using namespace abi::FunctionType::ReturnMethod;
  revng_assert(TheLayout.returnMethod() == RegisterSet);

  auto TypeNameReference = getArtificialStructTag</*IsDefinition*/ false>(F);

  *Out << getKeyword(ptml::CBuilder::Keyword::Typedef) << " "
       << getKeyword(ptml::CBuilder::Keyword::Struct) + " "
       << ptml::AttributeRegistry::getAttribute<"_PACKED">() << " "
       << TypeNameReference << " " << TypeNameReference << ";\n";
}

/// Print a typedef for a RawFunctionDefinition, that can be used when you have
/// a variable that is a pointer to a function.
void ptml::ModelCBuilder::printDeclaration(const RFT &F) {
  *Out << getCommentableTag(getModelCommentWithoutLeadingNewline(F)
                              + getKeyword(ptml::CBuilder::Keyword::Typedef)
                              + " ",
                            F);
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionPrototype(F);
  *Out << ";\n";
}

using CFT = model::CABIFunctionDefinition;

/// Print a typedef for a CABI function, that can be used when you have
/// a variable that is a pointer to a function.
void ptml::ModelCBuilder::printDeclaration(const CFT &F) {

  *Out << getCommentableTag(getModelCommentWithoutLeadingNewline(F)
                              + getKeyword(ptml::CBuilder::Keyword::Typedef)
                              + " ",
                            F);

  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionPrototype(F);
  *Out << ";\n";
}

void ptml::ModelCBuilder::printDeclaration(const model::TypeDefinition &T) {
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

void ptml::ModelCBuilder::printDefinition(const model::TypeDefinition &T) {
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

static Logger<> TypePrinterLog{ "type-definition-printer" };

void ptml::ModelCBuilder::printTypeDefinitions() {
  if (not DependencyCache.has_value())
    DependencyCache = DependencyGraph::make(Binary.TypeDefinitions());

  std::set<const TypeDependencyNode *> Defined;
  for (const auto *Root : DependencyCache->nodes()) {
    revng_log(TypePrinterLog, "PostOrder from Root:" << getNodeLabel(Root));

    for (const auto *Node : llvm::post_order_ext(Root, Defined)) {

      LoggerIndent PostOrderIndent{ TypePrinterLog };
      revng_log(TypePrinterLog, "post_order visiting: " << getNodeLabel(Node));

      const model::TypeDefinition *TypeDefinition = Node->T;
      if (Configuration.TypesToOmit.contains(TypeDefinition->key())) {
        revng_log(TypePrinterLog, "Omitted");
        continue;
      }

      if (Node->isArtificial()) {
        auto *RF = llvm::cast<model::RawFunctionDefinition>(TypeDefinition);
        if (Node->isDeclaration()) {
          revng_log(TypePrinterLog, "ArtificialDeclaration");
          printReturnTypeWrapperDeclaration(*RF);
        } else {
          revng_log(TypePrinterLog, "ArtificialDefinition");
          printReturnTypeWrapperDefinition(*RF);
        }
      } else {
        if (Node->isDeclaration()) {
          revng_log(TypePrinterLog, "Declaration");
          printDeclaration(*TypeDefinition);
        } else {
          revng_log(TypePrinterLog, "Definition");
          revng_assert(Node->isDefinition());
          revng_assert(not isDeclarationTheSameAsDefinition(*TypeDefinition));
          printDefinition(*TypeDefinition);
        }
      }

      *Out << "\n";
    }
    revng_log(TypePrinterLog, "PostOrder DONE");
  }
}
