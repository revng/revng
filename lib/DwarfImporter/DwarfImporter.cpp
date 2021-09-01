/// \file DwarfImporter.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <csignal>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Optional.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/DwarfImporter/DwarfImporter.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/revng.h"

using namespace llvm;
using namespace llvm::dwarf;

static Logger<> DILogger("dwarf-importer");

// WIP: prevent loops

static model::PrimitiveTypeKind::Values
dwarfEncodingToModel(uint32_t Encoding) {
  switch (Encoding) {
  case dwarf::DW_ATE_unsigned_char:
  case dwarf::DW_ATE_unsigned:
  case dwarf::DW_ATE_boolean:
    return model::PrimitiveTypeKind::Unsigned;
  case dwarf::DW_ATE_signed_char:
  case dwarf::DW_ATE_signed:
    return model::PrimitiveTypeKind::Signed;
  case dwarf::DW_ATE_float:
    return model::PrimitiveTypeKind::Float;
  default:
    return model::PrimitiveTypeKind::Invalid;
  }
}

static std::optional<uint64_t>
getUnsignedOrSigned(const DWARFFormValue &Value) {
  auto MaybeUnsigned = Value.getAsUnsignedConstant();
  auto MaybeSigned = Value.getAsSignedConstant();
  if (MaybeUnsigned)
    return *MaybeUnsigned;
  else if (MaybeSigned)
    return *MaybeSigned;
  else
    return {};
}

static std::optional<uint64_t>
getUnsignedOrSigned(const llvm::Optional<DWARFFormValue> &Value) {
  if (not Value)
    return {};
  else
    return getUnsignedOrSigned(*Value);
}

static bool isTrue(const DWARFFormValue &Value) {
  return getUnsignedOrSigned(Value) != 0;
}

template<typename S, typename O, typename... A>
void dumpToStream(S &Stream, const O &Object, A... Args) {
  std::string Buffer;
  {
    llvm::raw_string_ostream WrapperStream(Buffer);
    Object.dump(WrapperStream, Args...);
  }
  Stream << Buffer;
}

static void commentDie(const DWARFDie &Die, const Twine &Reason) {
  if (DILogger.isEnabled()) {
    DILogger << Reason.str();
    dumpToStream(DILogger, Die, 0);
    DILogger << DoLog;
  }
}

static void reportIgnoredDie(const DWARFDie &Die, const Twine &Reason) {
  commentDie(Die, "Ignoring DWARF die: " + Reason);
}

class DwarfToModelConverter {
private:
  DwarfImporter &Importer;
  TupleTree<model::Binary> &Model;
  size_t Index;
  size_t AltIndex;
  DWARFContext &DICtx;
  std::map<size_t, const model::Type *> Placeholders;

public:
  DwarfToModelConverter(DwarfImporter &Importer,
                        DWARFContext &DICtx,
                        size_t Index,
                        size_t AltIndex) :
    Importer(Importer),
    Model(Importer.getModel()),
    Index(Index),
    AltIndex(AltIndex),
    DICtx(DICtx) {}

private:
  const model::QualifiedType &record(const DWARFDie &Die,
                                     const model::TypePath &Path,
                                     bool IsNotPlaceholder) {
    return record(Die, model::QualifiedType{ Path }, IsNotPlaceholder);
  }

  const model::QualifiedType &record(const DWARFDie &Die,
                                     const model::QualifiedType &QT,
                                     bool IsNotPlaceholder) {
    size_t Offset = Die.getOffset();
    revng_assert(QT.UnqualifiedType.isValid());
    if (not IsNotPlaceholder) {
      revng_assert(QT.Qualifiers.size() == 0);
      Placeholders[Offset] = QT.UnqualifiedType.get();
    }

    return Importer.recordType({ Index, Die.getOffset() }, QT);
  }

  enum TypeSearchResult { Invalid, Absent, PlaceholderType, RegularType };

  std::pair<TypeSearchResult, model::QualifiedType *>
  findType(const DWARFDie &Die) {
    return findType(Die.getOffset());
  }

  std::pair<TypeSearchResult, model::QualifiedType *>
  findType(uint64_t Offset) {
    model::QualifiedType *Result = Importer.findType({ Index, Offset });
    TypeSearchResult ResultType = Invalid;
    if (Result == nullptr) {
      ResultType = Absent;
    } else {
      if (Placeholders.count(Offset) != 0)
        ResultType = PlaceholderType;
      else
        ResultType = RegularType;
    }

    return { ResultType, Result };
  }

  model::QualifiedType *findAltType(uint64_t Offset) {
    return Importer.findType({ AltIndex, Offset });
  }

private:
  static bool isType(dwarf::Tag Tag) {
    switch (Tag) {
    case llvm::dwarf::DW_TAG_base_type:
    case llvm::dwarf::DW_TAG_typedef:
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_volatile_type:
    case llvm::dwarf::DW_TAG_structure_type:
    case llvm::dwarf::DW_TAG_union_type:
    case llvm::dwarf::DW_TAG_enumeration_type:
    case llvm::dwarf::DW_TAG_array_type:
    case llvm::dwarf::DW_TAG_const_type:
    case llvm::dwarf::DW_TAG_pointer_type:
    case llvm::dwarf::DW_TAG_subroutine_type:
      return true;
    default:
      return false;
    }
  }

  static bool hasModelIdentity(dwarf::Tag Tag) {
    revng_assert(isType(Tag));
    switch (Tag) {
    case llvm::dwarf::DW_TAG_base_type:
    case llvm::dwarf::DW_TAG_typedef:
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_volatile_type:
    case llvm::dwarf::DW_TAG_structure_type:
    case llvm::dwarf::DW_TAG_union_type:
    case llvm::dwarf::DW_TAG_enumeration_type:
    case llvm::dwarf::DW_TAG_subroutine_type:
      return true;
    case llvm::dwarf::DW_TAG_array_type:
    case llvm::dwarf::DW_TAG_const_type:
    case llvm::dwarf::DW_TAG_pointer_type:
      return false;
    default:
      revng_abort();
    }
  }

  template<typename T>
  void createPlaceholderType(const DWARFDie &Die) {
    record(Die, Model->recordNewType(model::makeType<T>()), false);
  }

  void createType(const DWARFDie &Die) {
    auto Tag = Die.getTag();
    revng_assert(hasModelIdentity(Tag));

    switch (Tag) {
    case llvm::dwarf::DW_TAG_base_type: {
      uint8_t Size = 0;
      model::PrimitiveTypeKind::Values Kind = model::PrimitiveTypeKind::Invalid;

      auto MaybeByteSize = Die.find(DW_AT_byte_size);
      if (MaybeByteSize)
        Size = *MaybeByteSize->getAsUnsignedConstant();

      auto MaybeEncoding = Die.find(DW_AT_encoding);
      if (MaybeEncoding)
        Kind = dwarfEncodingToModel(*MaybeEncoding->getAsUnsignedConstant());

      if (Kind == model::PrimitiveTypeKind::Invalid or Size == 0
          or not isPowerOf2_32(Size)) {
        revng_abort();
      }

      record(Die, Model->getPrimitiveType(Kind, Size), true);
    } break;

    case llvm::dwarf::DW_TAG_subroutine_type:
      record(Die,
             Model->recordNewType(model::makeType<model::CABIFunctionType>()),
             false);
      break;
    case llvm::dwarf::DW_TAG_typedef:
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_volatile_type:
      createPlaceholderType<model::TypedefType>(Die);
      break;
    case llvm::dwarf::DW_TAG_structure_type:
      createPlaceholderType<model::StructType>(Die);
      break;
    case llvm::dwarf::DW_TAG_union_type:
      createPlaceholderType<model::UnionType>(Die);
      break;
    case llvm::dwarf::DW_TAG_enumeration_type:
      createPlaceholderType<model::EnumType>(Die);
      break;

    default:
      revng_abort();
    }
  }

  void handleTypeDeclaration(const DWARFDie &Die) {
    auto Tag = Die.getTag();
    if ((Tag == llvm::dwarf::DW_TAG_structure_type
         or Tag == llvm::dwarf::DW_TAG_union_type
         or Tag == llvm::dwarf::DW_TAG_enumeration_type)) {
      record(Die,
             Model->getPrimitiveType(model::PrimitiveTypeKind::Void, 0),
             true);
    } else {
      reportIgnoredDie(Die,
                       "Unexpected declaration for tag "
                         + llvm::dwarf::TagString(Tag));
    }
  }

  void materializeTypesWithIdentity() {
    for (const auto &CU : DICtx.compile_units()) {
      for (const auto &Entry : CU->dies()) {
        DWARFDie Die = { CU.get(), &Entry };
        auto Tag = Die.getTag();
        if (isType(Tag) and hasModelIdentity(Tag)) {
          auto MaybeDeclaration = Die.find(DW_AT_declaration);
          if (MaybeDeclaration && isTrue(*MaybeDeclaration)) {
            handleTypeDeclaration(Die);
          } else {
            createType(Die);
          }
        }
      }
    }
  }

  static StringRef getName(const DWARFDie &Die) {
    auto MaybeName = Die.find(DW_AT_name);
    if (MaybeName) {
      auto MaybeString = MaybeName->getAsCString();
      if (MaybeString)
        return *MaybeString;
    }

    return {};
  }

  const model::QualifiedType *getType(const DWARFDie &Die) {
    auto MaybeType = Die.find(DW_AT_type);
    if (MaybeType) {
      if (MaybeType->getForm() == llvm::dwarf::DW_FORM_GNU_ref_alt) {
        return findAltType(MaybeType->getRawUValue());
      } else {
        DWARFDie InnerDie = DICtx.getDIEForOffset(*MaybeType->getAsReference());
        return resolveType(InnerDie, false);
      }
    } else {
      return nullptr;
    }
  }

  model::QualifiedType getTypeOrVoid(const DWARFDie &Die) {
    const model::QualifiedType *Result = getType(Die);
    if (Result != nullptr) {
      return *Result;
    } else {
      return { Model->getPrimitiveType(model::PrimitiveTypeKind::Void, 0) };
    }
  }

  // WIP: recursive coroutine
  const model::QualifiedType *
  resolveType(const DWARFDie &Die, bool ResolveIfHasIdentity) {
    auto Tag = Die.getTag();
    auto [MatchType, TypePath] = findType(Die);

    // WIP: drop all assertions
    switch (MatchType) {
    case Absent: {
      // At this stage, all the unqualified types (i.e., those with an identity
      // in the model) should have been materialized.
      // Therefore, here we only deal with DWARF types the model represents as
      // qualifiers.
      revng_assert(not hasModelIdentity(Tag));

      bool HasType = Die.find(DW_AT_type).hasValue();
      model::QualifiedType Type = getTypeOrVoid(Die);

      switch (Tag) {

      case llvm::dwarf::DW_TAG_const_type: {
        // WIP: revng_assert(MaybeType);
        model::Qualifier NewQualifier;
        NewQualifier.Kind = model::QualifierKind::Const;
        Type.Qualifiers.push_back(NewQualifier);
      } break;

      case llvm::dwarf::DW_TAG_array_type: {

        if (not HasType) {
          reportIgnoredDie(Die, "Array does not specify element type");
          return nullptr;
        }

        for (const DWARFDie &ChildDie : Die.children()) {
          if (ChildDie.getTag() == llvm::dwarf::DW_TAG_subrange_type) {
            auto MaybeUpperBound = getUnsignedOrSigned(
              ChildDie.find(DW_AT_upper_bound));
            model::Qualifier NewQualifier;
            NewQualifier.Kind = model::QualifierKind::Array;
            // WIP
            if (MaybeUpperBound) {
              NewQualifier.Size = *MaybeUpperBound + 1;
            } else {
              reportIgnoredDie(Die, "Array upper bound missing or invalid");
              return nullptr;
            }
            Type.Qualifiers.push_back(NewQualifier);
          }
        }
      } break;

      case llvm::dwarf::DW_TAG_pointer_type: {
        model::Qualifier NewQualifier;
        auto MaybeByteSize = Die.find(DW_AT_byte_size);
        // WIP: soft fail if byte size absent (or force to architecture)
        NewQualifier.Kind = model::QualifierKind::Pointer;
        NewQualifier.Size = *MaybeByteSize->getAsUnsignedConstant();
        Type.Qualifiers.push_back(NewQualifier);
      } break;

      default:
        revng_abort();
      }

      return &record(Die, Type, true);
    }
    case PlaceholderType: {
      revng_assert(TypePath != nullptr);
      auto Offset = Die.getOffset();
      revng_assert(Placeholders.count(Offset) != 0);
      // This die is already present in the map. Either it has already been
      // fully imported, or it's a type with an identity on the model.
      // In the latter case, proceed only if explicitly told to do so.
      if (ResolveIfHasIdentity) {
        revng_assert(TypePath->Qualifiers.empty());
        model::Type *T = TypePath->UnqualifiedType.get();

        StringRef Name = getName(Die);

        switch (Tag) {
        case llvm::dwarf::DW_TAG_subroutine_type: {
          auto *FunctionType = cast<model::CABIFunctionType>(T);
          FunctionType->CustomName = Name;
          // WIP
          FunctionType->ABI = model::abi::SystemV_x86_64;
          FunctionType->ReturnType = getTypeOrVoid(Die);

          uint64_t Index = 0;
          for (const DWARFDie &ChildDie : Die.children()) {
            if (ChildDie.getTag() == DW_TAG_formal_parameter) {

              if (Die.getOffset() == 0x0001045b)
                raise(5);

              const model::QualifiedType *ArgumentType = getType(ChildDie);

              if (ArgumentType == nullptr) {
                reportIgnoredDie(Die,
                                 "The type of argument " + Twine(Index + 1)
                                   + " cannot be resolved");
                return nullptr;
              }

              model::Argument &NewArgument = FunctionType->Arguments[Index];
              NewArgument.Type = *ArgumentType;
              Index += 1;
            }
          }

        } break;

        case llvm::dwarf::DW_TAG_typedef:
        case llvm::dwarf::DW_TAG_restrict_type:
        case llvm::dwarf::DW_TAG_volatile_type: {
          model::QualifiedType TargetType = getTypeOrVoid(Die);
          auto *Typedef = cast<model::TypedefType>(T);
          Typedef->CustomName = Name;
          Typedef->UnderlyingType = TargetType;
          revng_assert(Typedef->UnderlyingType.UnqualifiedType.isValid());

        } break;

        case llvm::dwarf::DW_TAG_structure_type: {
          auto MaybeSize = Die.find(DW_AT_byte_size);
          revng_assert(MaybeSize);

          auto *Struct = cast<model::StructType>(T);
          Struct->CustomName = Name;
          Struct->Size = *MaybeSize->getAsUnsignedConstant();

          uint64_t Index = 0;
          for (const DWARFDie &ChildDie : Die.children()) {
            if (ChildDie.getTag() == DW_TAG_member) {

              // Collect offset
              auto MaybeOffset = ChildDie.find(DW_AT_data_member_location);
              revng_assert(MaybeOffset);
              auto Offset = *MaybeOffset->getAsUnsignedConstant();

              if (ChildDie.find(DW_AT_bit_size)) {
                reportIgnoredDie(ChildDie, "Ignoring bitfield in struct");
                continue;
              }

              const model::QualifiedType *MemberType = getType(ChildDie);

              if (MemberType == nullptr) {
                reportIgnoredDie(Die,
                                 "The type of member " + Twine(Index + 1)
                                   + " cannot be resolved");
                return nullptr;
              }

              //  Create new field
              auto &Field = Struct->Fields[Offset];
              Field.CustomName = getName(ChildDie);
              Field.Type = *MemberType;

              ++Index;
            }
          }

          if (Index == 0) {
            reportIgnoredDie(Die, "Struct has no fields");
            return nullptr;
          }

        } break;

        case llvm::dwarf::DW_TAG_union_type: {
          auto *Union = cast<model::UnionType>(T);
          Union->CustomName = Name;

          uint64_t Index = 0;
          for (const DWARFDie &ChildDie : Die.children()) {
            if (ChildDie.getTag() == DW_TAG_member) {
              const model::QualifiedType *MemberType = getType(ChildDie);
              if (MemberType == nullptr) {
                reportIgnoredDie(Die,
                                 "The type of member " + Twine(Index + 1)
                                   + " cannot be resolved");
                return nullptr;
              }

              // Create new field
              auto &Field = Union->Fields[Index];
              Field.CustomName = getName(ChildDie);
              Field.Type = *MemberType;

              // Increment union index
              Index += 1;
            }
          }

          if (Index == 0) {
            reportIgnoredDie(Die, "Union has no fields");
            return nullptr;
          }

        } break;

        case llvm::dwarf::DW_TAG_enumeration_type: {
          auto *Enum = cast<model::EnumType>(T);
          Enum->CustomName = Name;

          const model::QualifiedType *UnderlyingType = getType(Die);
          if (UnderlyingType == nullptr) {
            reportIgnoredDie(Die,
                             "The enum underlying type cannot be resolved");
            return nullptr;
          }

          revng_assert(UnderlyingType->Qualifiers.empty());
          Enum->UnderlyingType = Model->getTypePath(
            UnderlyingType->UnqualifiedType.get());

          uint64_t Index = 0;
          for (const DWARFDie &ChildDie : Die.children()) {
            if (ChildDie.getTag() == DW_TAG_enumerator) {
              // Collect value
              auto MaybeValue = getUnsignedOrSigned(
                ChildDie.find(DW_AT_const_value));
              if (not MaybeValue) {
                reportIgnoredDie(ChildDie,
                                 "Ignoring enum entry " + Twine(Index + 1)
                                   + " without a value");
                return nullptr;
              }

              uint64_t Value = *MaybeValue;

              // Create new entry
              StringRef EntryName = getName(ChildDie);

              // If it's the first time, set CustomName, otherwise, introduce
              // an alias
              auto It = Enum->Entries.find(Value);
              if (It == Enum->Entries.end()) {
                auto &Entry = Enum->Entries[Value];
                Entry.CustomName = EntryName;
              } else {
                It->Aliases.insert(EntryName);
              }

              ++Index;
            }
          }

        } break;

        default:
          revng_abort();
        }

        Placeholders.erase(Offset);
      }
    } break;

    case RegularType:
      revng_assert(TypePath != nullptr);
      break;

    default:
      revng_abort();
    }

    return TypePath;
  }

  void secondPass() {
    for (const auto &CU : DICtx.compile_units()) {
      for (const auto &Entry : CU->dies()) {
        DWARFDie Die = { CU.get(), &Entry };
        if (not isType(Die.getTag()))
          continue;
        resolveType(Die, true);
      }
    }
  }

  std::optional<model::TypePath> getSubprogramPrototype(const DWARFDie &Die) {
    // Create function type
    auto Path = Model->recordNewType(
      model::makeType<model::CABIFunctionType>());
    auto *FunctionType = cast<model::CABIFunctionType>(Path.get());
    FunctionType->CustomName = getName(Die);
    FunctionType->ABI = model::abi::SystemV_x86_64;
    FunctionType->ReturnType = getTypeOrVoid(Die);

    uint64_t Index = 0;
    for (const DWARFDie &ChildDie : Die.children()) {
      if (ChildDie.getTag() == DW_TAG_formal_parameter) {
        const model::QualifiedType *ArgumenType = getType(ChildDie);
        if (ArgumenType == nullptr) {
          reportIgnoredDie(Die,
                           "The type of argument " + Twine(Index + 1)
                             + " cannot be resolved");
          return {};
        }

        model::Argument &NewArgument = FunctionType->Arguments[Index];
        NewArgument.CustomName = getName(ChildDie);
        NewArgument.Type = *ArgumenType;
        Index += 1;
      }
    }

    return Path;
  }

  void thirdPass() {
    for (const auto &CU : DICtx.compile_units()) {
      for (const auto &Entry : CU->dies()) {
        DWARFDie Die = { CU.get(), &Entry };

        if (Die.getTag() != DW_TAG_subprogram)
          continue;

        auto MaybePath = getSubprogramPrototype(Die);
        if (not MaybePath)
          continue;

        auto *FunctionType = cast<model::CABIFunctionType>(MaybePath->get());

        auto Name = FunctionType->CustomName.str();
        if (Name.size() == 0)
          continue;

        // Create actual function
        model::DynamicFunction &Function = Model->DynamicFunctions[Name.str()];
        Function.CustomName = FunctionType->CustomName;
        Function.SymbolName = Name;
        Function.Prototype = *MaybePath;
      }
    }
  }

  void fix() {
    // TODO: collapse uint8_t typedefs into the primitive type

    std::set<std::string> UsedNames;

    for (auto &Type : Model->Types) {
      model::Type *T = Type.get();
      if (isa<model::PrimitiveType>(T)) {
        UsedNames.insert(T->name().str().str());
      }
    }

    for (auto &Type : Model->Types) {
      model::Type *T = Type.get();
      if (isa<model::PrimitiveType>(T))
        continue;

      std::string Name = T->name().str().str();
      while (UsedNames.count(Name) != 0) {
        Name += "_";
      }

      // Rename
      upcast(T, [&Name](auto &Upcasted) {
        using UpcastedType = std::remove_cvref_t<decltype(Upcasted)>;
        if constexpr (not std::is_same_v<model::PrimitiveType, UpcastedType>) {
          Upcasted.CustomName = Name;
        } else {
          revng_abort();
        }
      });

      // Record new name
      UsedNames.insert(Name);
    }
  }

  void fix2() {
    struct TypeNode {
      const model::Type *T;
    };

    using Graph = GenericGraph<ForwardNode<TypeNode>>;

    Graph ReverseDependencyGraph;

    // Create nodes in reverse dependency graph
    std::map<const model::Type *, ForwardNode<TypeNode> *> TypeToNode;
    for (UpcastablePointer<model::Type> &T : Model->Types)
      TypeToNode[T.get()] = ReverseDependencyGraph.addNode(TypeNode{ T.get() });

    auto RegisterDependency = [&](UpcastablePointer<model::Type> &T,
                                  const model::QualifiedType &QT) {
      auto *DependantType = QT.UnqualifiedType.get();
      TypeToNode.at(DependantType)->addSuccessor(TypeToNode.at(T.get()));
    };

    // Populate the graph
    for (UpcastablePointer<model::Type> &T : Model->Types) {
      if (auto *Primitive = dyn_cast<model::PrimitiveType>(T.get())) {
        // Nothing to do here
      } else if (auto *Struct = dyn_cast<model::StructType>(T.get())) {
        for (const model::StructField &Field : Struct->Fields)
          RegisterDependency(T, Field.Type);
      } else if (auto *Union = dyn_cast<model::UnionType>(T.get())) {
        for (const model::UnionField &Field : Union->Fields)
          RegisterDependency(T, Field.Type);
      } else if (auto *Enum = dyn_cast<model::EnumType>(T.get())) {
        RegisterDependency(T, { Enum->UnderlyingType });
      } else if (auto *Typedef = dyn_cast<model::TypedefType>(T.get())) {
        RegisterDependency(T, Typedef->UnderlyingType);
      } else if (auto *RFT = dyn_cast<model::RawFunctionType>(T.get())) {
        for (const model::NamedTypedRegister &Argument : RFT->Arguments)
          RegisterDependency(T, Argument.Type);
        for (const model::TypedRegister &RV : RFT->ReturnValues)
          RegisterDependency(T, RV.Type);
      } else if (auto *CAFT = dyn_cast<model::CABIFunctionType>(T.get())) {
        for (const model::Argument &Argument : CAFT->Arguments)
          RegisterDependency(T, Argument.Type);
        RegisterDependency(T, CAFT->ReturnType);
      } else {
        revng_abort();
      }
    }

    // Prepare for deletion all the nodes reachable from unresolved placeholder
    // nodes
    std::set<const model::Type *> ToDelete;
    for (auto &[Offset, Type] : Placeholders) {
      for (const auto *Node : depth_first(TypeToNode.at(Type))) {
        ToDelete.insert(Node->T);
      }
    }

    // Purge dynamic functions with invalid types
    auto Begin = Model->DynamicFunctions.begin();
    for (auto It = Begin; It != Model->DynamicFunctions.end(); /**/) {
      if (ToDelete.count(It->Prototype.get()) == 0) {
        ++It;
      } else {
        It = Model->DynamicFunctions.erase(It);
      }
    }

    revng_log(DILogger,
              "Purging " << ToDelete.size() << " types due to "
                         << Placeholders.size() << " unresolved types");

    // Purge types depending on unresolved placeholder types
    for (auto It = Model->Types.begin(); It != Model->Types.end();) {
      if (ToDelete.count(It->get()) != 0)
        It = Model->Types.erase(It);
      else
        ++It;
    }

    Placeholders.clear();
  }

  void dedup() {
    // WIP
    // Keep a stack of type pairs that whose comparisons are in progress

    // Compare and recur

    // If you meet a type that is currently being analyzed, ensure it's the same
    // pair
  }

public:
  void run() {
    // WIP: rename
    materializeTypesWithIdentity();
    secondPass();
    thirdPass();
    fix();
    fix2();
    dedup();
    revng_assert(Placeholders.size() == 0);
    Model.serialize(llvm::errs());
    Model->verify(true);
  }
};

template<typename T>
ArrayRef<uint8_t> getSectionsContents(StringRef Name, T &ELF) {
  auto MaybeSections = ELF.sections();
  if (not MaybeSections)
    return {};

  for (const auto &Section : *MaybeSections) {
    auto MaybeName = ELF.getSectionName(Section);
    if (MaybeName) {
      if (MaybeName and *MaybeName == Name) {
        auto MaybeContents = ELF.getSectionContents(Section);
        if (MaybeContents)
          return *MaybeContents;
      }
    }
  }

  return {};
}

static StringRef getAltDebugLinkFileName(object::Binary *B) {
  using namespace llvm::object;

  auto Handler = [&](auto *ELFObject) -> StringRef {
    const auto &ELF = ELFObject->getELFFile();
    ArrayRef<uint8_t> Contents = getSectionsContents(".gnu_debugaltlink", ELF);

    if (Contents.size() == 0)
      return {};

    // TODO: improve accuracy
    // Extract path name and ignore everything after \0
    return StringRef(reinterpret_cast<const char *>(Contents.data()));
  };

  StringRef AltDebugLinkPath;
  if (auto *ELF = dyn_cast<ELF32BEObjectFile>(B)) {
    AltDebugLinkPath = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF64BEObjectFile>(B)) {
    AltDebugLinkPath = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF32LEObjectFile>(B)) {
    AltDebugLinkPath = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF64LEObjectFile>(B)) {
    AltDebugLinkPath = Handler(ELF);
  } else {
    revng_abort();
  }

  return llvm::sys::path::filename(AltDebugLinkPath);
}

static void error(StringRef Prefix, std::error_code EC) {
  if (!EC)
    return;
  std::string Str = Prefix.str();
  Str += ": " + EC.message();
  revng_abort(Str.c_str());
}

void DwarfImporter::import(StringRef FileName) {
  using namespace llvm::object;

  ErrorOr<std::unique_ptr<MemoryBuffer>>
    BuffOrErr = MemoryBuffer::getFileOrSTDIN(FileName);
  error(FileName, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BuffOrErr.get());
  Expected<std::unique_ptr<Binary>> BinOrErr = object::createBinary(*Buffer);
  error(FileName, errorToErrorCode(BinOrErr.takeError()));

  if (auto *ELF = dyn_cast<ObjectFile>(BinOrErr->get())) {
    // Check if we already loaded the alt debug info file
    size_t AltIndex = -1;
    StringRef AltDebugLinkFileName = getAltDebugLinkFileName(ELF);
    if (AltDebugLinkFileName.size() > 0) {
      auto Begin = LoadedFiles.begin();
      auto End = LoadedFiles.end();
      auto It = std::find(Begin, End, AltDebugLinkFileName);
      if (It != End)
        AltIndex = It - Begin;
    }

    auto TheDWARFContext = DWARFContext::create(*ELF);
    DwarfToModelConverter Converter(*this,
                                    *TheDWARFContext,
                                    LoadedFiles.size(),
                                    AltIndex);
    Converter.run();
  }

  LoadedFiles.push_back(sys::path::filename(FileName).str());
}

// WIP:
// TODO: when loading a binary
// 1. Load any available DWARF in the binary itself
// 2. Parse .note.gnu.build-id, .gnu_debugaltlink and .gnu_debuglink
// 3. Load from the following paths:
//    - /usr/lib/debug/.build-id/ab/cdef1234.debug
//    - /usr/bin/ls.debug
//    - /usr/bin/.debug/ls.debug
//    - /usr/lib/debug/usr/bin/ls.debug
//    In turn, parse .gnu_debugaltlink (and .gnu_debuglink?)
// 2. Parse DT_NEEDED
// 3. Look for each library in ld.so.conf directories
// 4. Go to 1
// https://sourceware.org/gdb/onlinedocs/gdb/Separate-Debug-Files.html
