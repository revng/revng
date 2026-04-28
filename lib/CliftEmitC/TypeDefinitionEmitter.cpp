//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/Clift/CliftAttributes.h"
#include "revng/Clift/CliftTypeInterfaces.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/CliftEmitC/TypeDefinitionEmitter.h"
#include "revng/CliftImportModel/CAttributeListBuilder.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/StructField.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/Constants.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

inline Logger TypePrinterLog{ "clift-type-definition-printer" };

void TypeDefinitionEmitter::emitTypeKeyword(clift::DefinedType Type) {
  if (mlir::isa<clift::StructType>(Type))
    Tokens.emitKeyword(ptml::CTokenEmitter::Keyword::Struct);
  else if (mlir::isa<clift::UnionType>(Type))
    Tokens.emitKeyword(ptml::CTokenEmitter::Keyword::Union);
  else if (mlir::isa<clift::EnumType>(Type))
    Tokens.emitKeyword(ptml::CTokenEmitter::Keyword::Enum);
  else
    revng_abort("Declaration typedef of an unknown type was encountered");
}

void TypeDefinitionEmitter::emitDeclarationTypedef(clift::DefinedType Type) {
  Tokens.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  Tokens.emitSpace();

  emitTypeKeyword(Type);

  emitCAttributes(clift::CAttributeListBuilder(Type.getContext())
                    .setOrUpdate<"_PACKED">()
                    .getRaw(),
                  /* SpaceBefore = */ true,
                  /* SpaceAfter = */ true);

  Tokens.emitIdentifier(Type.getName(),
                        Type.getHandle(),
                        chooseEntityKind(Type),
                        ptml::CTokenEmitter::IdentifierKind::Reference);
  Tokens.emitSpace();
  Tokens.emitIdentifier(Type.getName(),
                        Type.getHandle(),
                        chooseEntityKind(Type),
                        ptml::CTokenEmitter::IdentifierKind::Reference);
  Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  Tokens.emitNewline();
}

void TypeDefinitionEmitter::emitTypedefDefinition(clift::TypedefType Typedef) {
  auto Guard = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                  Typedef.getHandle());

  emitDoxygenComment(Typedef);
  Tokens.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  Tokens.emitSpace();

  emitDeclaration(Typedef.getUnderlyingType(),
                  CEmitter::DeclaratorInfo{
                    .Identifier = Typedef.getName(),
                    .Location = Typedef.getHandle(),
                    .CAttributes = {},
                    .Kind = ptml::CTokenEmitter::EntityKind::Typedef });
  Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  Tokens.emitNewline();
}

void TypeDefinitionEmitter::emitFunctionTypedef(clift::FunctionType Function) {
  revng_assert(!Function.getName().empty());

  auto Guard = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                  Function.getHandle());

  emitDoxygenComment(Function);
  Tokens.emitKeyword(ptml::CTokenEmitter::Keyword::Typedef);
  Tokens.emitSpace();

  auto Attrs = clift::CAttributeListBuilder(Function.getContext(),
                                            Function.getCAttributes());
  emitDeclaration(Function,
                  CEmitter::DeclaratorInfo{
                    .Identifier = Function.getName(),
                    .Location = Function.getHandle(),
                    .CAttributes = Attrs.get(),
                    .Kind = ptml::CTokenEmitter::EntityKind::Function,
                    .Parameters = {},
                  });
  Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  Tokens.emitNewline();
}

void TypeDefinitionEmitter::emitTypeDeclaration(clift::DefinedType Type) {
  if (isSeparateDeclarationAllowed(Type)) {
    emitForwardDeclaration(Type);

  } else if (auto Typedef = mlir::dyn_cast<clift::TypedefType>(Type)) {
    emitTypedefDefinition(Typedef);

  } else if (auto Enum = mlir::dyn_cast<clift::EnumType>(Type)) {
    emitEnumDefinition(Enum);

  } else if (auto Function = mlir::dyn_cast<clift::FunctionType>(Type)) {
    emitFunctionTypedef(Function);

  } else {
    Type.dump();
    revng_abort("Unknown defined type.");
  }
}

static std::string paddingFieldName(uint64_t CurrentOffset) {
  // TODO: this discards the prefix configuration option.
  //       We should fix this after the configuration is separate from the
  //       model

  model::CNameBuilder Builder(model::Binary{});
  return Builder.paddingFieldName(CurrentOffset);
}

void TypeDefinitionEmitter::emitPaddingField(clift::ClassType Class,
                                             uint64_t CurrentOffset,
                                             uint64_t NextOffset) {
  revng_assert(CurrentOffset <= NextOffset);
  if (CurrentOffset == NextOffset)
    return; // There is no padding

  auto Char = clift::IntegerType::get(Class.getContext(),
                                      clift::IntegerKind::Unsigned,
                                      /*Size=*/1);
  auto Array = clift::ArrayType::get(Char, NextOffset - CurrentOffset);
  emitDeclaration(Array,
                  DeclaratorInfo{
                    .Identifier = paddingFieldName(CurrentOffset),
                    .Location = {},
                    .CAttributes = {},
                    .Kind = ptml::CTokenEmitter::EntityKind::Field });

  Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  Tokens.emitNewline();
}

void TypeDefinitionEmitter::emitClassDefinition(clift::ClassType Class) {
  {
    auto G = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                Class.getHandle());

    emitDoxygenComment(Class);
    emitTypeKeyword(Class);

    clift::CAttributeListBuilder AttributeBuilder{ Class.getContext(),
                                                   Class.getCAttributes() };
    AttributeBuilder.setOrUpdate<"_PACKED">();

    if (auto S = mlir::dyn_cast<clift::StructType>(Class))
      AttributeBuilder.setOrUpdate<"_SIZE">(S.getSize());

    emitCAttributes(AttributeBuilder.getRaw(),
                    /* SpaceBefore = */ true,
                    /* SpaceAfter = */ true);

    Tokens.emitIdentifier(Class.getName(),
                          Class.getHandle(),
                          chooseEntityKind(Class),
                          ptml::CTokenEmitter::IdentifierKind::Definition);
    Tokens.emitSpace();
  }

  {
    bool IsStruct = mlir::isa<clift::StructType>(Class);

    using ScopeKind = ptml::CTokenEmitter::ScopeKind;
    auto Scope = Tokens.enterScope(IsStruct ? ScopeKind::StructDefinition :
                                              ScopeKind::UnionDefinition,
                                   ptml::CTokenEmitter::Delimiter::Braces);
    Tokens.emitNewline();

    uint64_t PreviousOffset = 0;
    for (const auto &Field : Class.getFields()) {
      if (IsStruct and Configuration.ExplicitPadding)
        emitPaddingField(Class, PreviousOffset, Field.getOffset());

      auto G = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                  Field.getHandle());

      emitDoxygenComment(Field);
      emitDeclaration(Field.getType(),
                      CEmitter::DeclaratorInfo{
                        .Identifier = Field.getName(),
                        .Location = Field.getHandle(),
                        .CAttributes = {},
                        .Kind = ptml::CTokenEmitter::EntityKind::Field,
                      });

      if (IsStruct) {
        // WARNING: this ignores `UnnamedStructFieldPrefix` configuration field,
        //          user might have set!
        //
        // TODO: fix this once the configuration is obtained from the pipe
        //       (new pipeline only).
        model::CNameBuilder UnconfiguredNB(model::Binary{});
        model::StructDefinition FakeStruct; // No model fields are read here.
        model::StructField FakeField(Field.getOffset()); // Only `Offset` read.
        if (not UnconfiguredNB.isAutomaticName(FakeStruct,
                                               FakeField,
                                               Field.getName())) {
          emitCAttributes(clift::CAttributeListBuilder(Class.getContext())
                            .setOrUpdate<"_STARTS_AT">(Field.getOffset())
                            .getRaw(),
                          /* SpaceBefore = */ true,
                          /* SpaceAfter = */ false);
        }
      }

      Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      Tokens.emitNewline();

      PreviousOffset = Field.getOffset()
                       + clift::getObjectSize(Field.getType());
    }
  }

  Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  Tokens.emitNewline();
}

void TypeDefinitionEmitter::emitEnumDefinition(clift::EnumType Enum) {
  {
    auto G = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                Enum.getHandle());

    emitDoxygenComment(Enum);
    Tokens.emitKeyword(ptml::CTokenEmitter::Keyword::Enum);

    clift::ValueType Type = Enum.getUnderlyingType();
    emitCAttributes(clift::CAttributeListBuilder(Enum.getContext())
                      .setOrUpdate<"_ENUM_UNDERLYING">(Type)
                      .setOrUpdate<"_PACKED">()
                      .getRaw(),
                    /* SpaceBefore = */ true,
                    /* SpaceAfter = */ true);

    Tokens.emitIdentifier(Enum.getName(),
                          Enum.getHandle(),
                          ptml::CTokenEmitter::EntityKind::Enum,
                          ptml::CTokenEmitter::IdentifierKind::Definition);
    Tokens.emitSpace();
  }

  {
    using ScopeKind = ptml::CTokenEmitter::ScopeKind;
    auto Scope = Tokens.enterScope(ScopeKind::EnumDefinition,
                                   ptml::CTokenEmitter::Delimiter::Braces);
    Tokens.emitNewline();

    auto PrintEnumEntry =
      [this](llvm::StringRef Name, llvm::StringRef Handle, uint64_t Value) {
        using CTE = ptml::CTokenEmitter;
        Tokens.emitIdentifier(Name,
                              Handle,
                              CTE::EntityKind::Enumerator,
                              CTE::IdentifierKind::Definition);
        Tokens.emitSpace();
        Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Equals);
        Tokens.emitSpace();

        Tokens.emitIntegerLiteral(llvm::APInt{ 64, Value },
                                  ptml::CTokenEmitter::IntegerSuffix{
                                    .Unsigned = true,
                                    .MinimumType = CIntegerKind::Int,
                                  },
                                  16);

        Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Comma);
        Tokens.emitNewline();
      };

    for (const auto &Entry : Enum.getFields()) {
      auto G = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                  Entry.getHandle());

      emitDoxygenComment(Entry);
      PrintEnumEntry(Entry.getName(), Entry.getHandle(), Entry.getRawValue());
    }

    if (Configuration.EmitMaximumEnumValue) {
      // We have to make the enum of the correct size of the underlying type
      auto ByteSize = getObjectSize(Enum);
      revng_assert(ByteSize <= 8);
      size_t MaxValue = llvm::APInt::getAllOnes(8 * ByteSize).getZExtValue();

      // TODO: pull the prefix from the configuration when it's available
      //       without pulling in the model dependency.
      static constexpr llvm::StringRef Prefix = "enum_max_value_";

      namespace ranks = revng::ranks;
      auto EnumLocation = *pipeline::locationFromString(ranks::TypeDefinition,
                                                        Enum.getHandle());
      auto EntryLocation = EnumLocation.extend(ranks::EnumEntry, MaxValue);

      PrintEnumEntry(Prefix.str() + Enum.getName().str(),
                     EntryLocation.toString(),
                     MaxValue);
    }
  }

  Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
  Tokens.emitNewline();

  // Allow using `MyEnum` instead of `enum MyEnum`.
  emitDeclarationTypedef(Enum);
}

void TypeDefinitionEmitter::emitTypeDefinition(clift::DefinedType Type) {
  if (auto Class = mlir::dyn_cast<clift::ClassType>(Type)) {
    emitClassDefinition(Class);
    return;

  } else if (not isSeparateDeclarationAllowed(Type)) {
    emitTypeDeclaration(Type);
    Tokens.emitNewline();
    return;
  }

  Type.dump();
  revng_abort("Unknown defined type.");
}

void TypeDefinitionEmitter::emitTypeTree(const TypeDependencyNode &Root,
                                         NodeSet &Emitted) {
  revng_log(TypePrinterLog,
            "Starting a post order visit from:" << Root.label());

  bool SkipTheRest = false;

  size_t NodesEmittedAlready = Emitted.size();
  for (const auto *Node : llvm::post_order_ext(&Root, Emitted)) {
    LoggerIndent PostOrderIndent{ TypePrinterLog };

    clift::DefinedType Type = Node->T;
    revng_assert(not Type.getHandle().empty());
    if (Type.getHandle() == Configuration.TypeToOmit)
      SkipTheRest = true;

    if (SkipTheRest) {
      revng_log(TypePrinterLog, "skipping (TypeToOmit): " << Node->label());
      continue;
    } else {
      revng_log(TypePrinterLog, "visiting: " << Node->label());
    }

    if (Node->IsDefinition) {
      revng_assert(Node->IsDefinition);
      revng_assert(isSeparateDeclarationAllowed(Type));

      revng_log(TypePrinterLog, "Definition");
      emitTypeDefinition(Type);

    } else {
      revng_log(TypePrinterLog, "Declaration");
      emitTypeDeclaration(Type);
    }
  }

  if (NodesEmittedAlready != Emitted.size())
    Tokens.emitNewline();

  revng_log(TypePrinterLog, "Root is done: " << Root.label());
}
