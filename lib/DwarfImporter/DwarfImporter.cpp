/// \file DwarfImporter.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <csignal>
#include <optional>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/DwarfImporter/DwarfImporter.h"
#include "revng/Model/ABI.h"
#include "revng/Model/Processing.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/revng.h"

using namespace llvm;
using namespace llvm::dwarf;

static Logger<> DILogger("dwarf-importer");

template<typename M>
class ScopedSetElement {
private:
  M &Set;
  typename M::value_type ToInsert;

public:
  ScopedSetElement(M &Set, typename M::value_type ToInsert) :
    Set(Set), ToInsert(ToInsert) {}
  ~ScopedSetElement() { Set.erase(ToInsert); }

public:
  bool insert() {
    auto It = Set.find(ToInsert);
    if (It != Set.end()) {
      return false;
    } else {
      Set.insert(It, ToInsert);
      return true;
    }
  }
};

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
getUnsignedOrSigned(const DWARFDie &Die, dwarf::Attribute Attribute) {
  auto Value = Die.find(Attribute);
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
  size_t TypesWithIdentityCount;
  DWARFContext &DICtx;
  model::abi::Values DefaultABI;
  std::map<size_t, const model::Type *> Placeholders;
  std::set<const model::Type *> InvalidPrimitives;
  std::set<const DWARFDie *> InProgressDies;

public:
  DwarfToModelConverter(DwarfImporter &Importer,
                        DWARFContext &DICtx,
                        size_t Index,
                        size_t AltIndex) :
    Importer(Importer),
    Model(Importer.getModel()),
    Index(Index),
    AltIndex(AltIndex),
    DICtx(DICtx) {

    // Detect default ABI from architecture
    // TODO: this needs to be refined
    switch (DICtx.getArch()) {
    case llvm::Triple::x86_64:
      DefaultABI = model::abi::SystemV_x86_64;
      break;
    default:
      DefaultABI = model::abi::Invalid;
    }
  }

private:
  model::abi::Values getABI(CallingConvention CC = DW_CC_normal) const {
    if (CC != DW_CC_normal)
      return model::abi::Invalid;

    return DefaultABI;
  }

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

  const model::QualifiedType *findAltType(uint64_t Offset) {
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
  [[maybe_unused]] T *createPlaceholderType(const DWARFDie &Die) {
    auto NewType = model::makeType<T>();
    T *Result = cast<T>(NewType.get());
    record(Die, Model->recordNewType(std::move(NewType)), false);
    return Result;
  }

  void createInvalidPrimitivePlaceholder(const DWARFDie &Die) {
    InvalidPrimitives.insert(createPlaceholderType<model::TypedefType>(Die));
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

      if (Kind == model::PrimitiveTypeKind::Invalid) {
        reportIgnoredDie(Die, "Unknown primitive type");
        createInvalidPrimitivePlaceholder(Die);
        return;
      }

      if (Size == 0) {
        reportIgnoredDie(Die, "Invalid size for primitive type");
        createInvalidPrimitivePlaceholder(Die);
        return;
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
      reportIgnoredDie(Die, "Unexpected type");
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

    TypesWithIdentityCount = Placeholders.size();
  }

  static std::string getName(const DWARFDie &Die) {
    auto MaybeName = Die.find(DW_AT_name);
    if (MaybeName) {
      auto MaybeString = MaybeName->getAsCString();
      if (MaybeString)
        return *MaybeString;
    }

    return {};
  }

  static model::Identifier getNameAsIdentifier(const DWARFDie &Die) {
    std::string Name = getName(Die);
    if (not Name.empty())
      return model::Identifier::fromString(Name);
    else
      return {};
  }

  RecursiveCoroutine<const model::QualifiedType *>
  getType(const DWARFDie &Die) {
    auto MaybeType = Die.find(DW_AT_type);
    if (MaybeType) {
      if (MaybeType->getForm() == llvm::dwarf::DW_FORM_GNU_ref_alt) {
        rc_return findAltType(MaybeType->getRawUValue());
      } else {
        DWARFDie InnerDie = DICtx.getDIEForOffset(*MaybeType->getAsReference());
        rc_return rc_recur resolveType(InnerDie, false);
      }
    } else {
      rc_return nullptr;
    }
  }

  RecursiveCoroutine<model::QualifiedType> getTypeOrVoid(const DWARFDie &Die) {
    const model::QualifiedType *Result = rc_recur getType(Die);
    if (Result != nullptr) {
      rc_return *Result;
    } else {
      rc_return{ Model->getPrimitiveType(model::PrimitiveTypeKind::Void, 0) };
    }
  }

  RecursiveCoroutine<const model::QualifiedType *>
  resolveTypeWithIdentity(const DWARFDie &Die, model::QualifiedType *TypePath) {
    using namespace model;

    auto Offset = Die.getOffset();
    auto Tag = Die.getTag();

    revng_assert(Placeholders.count(Offset) != 0);
    revng_assert(TypePath->Qualifiers.empty());
    model::Type *T = TypePath->UnqualifiedType.get();

    model::Identifier Name = getNameAsIdentifier(Die);

    if (InvalidPrimitives.count(T) != 0)
      rc_return nullptr;

    switch (Tag) {
    case llvm::dwarf::DW_TAG_subroutine_type: {
      auto *FunctionType = cast<model::CABIFunctionType>(T);
      FunctionType->CustomName = Name;
      FunctionType->ABI = getABI();

      if (FunctionType->ABI == model::abi::Invalid) {
        reportIgnoredDie(Die, "Unknown calling convention");
        rc_return nullptr;
      }

      FunctionType->ReturnType = rc_recur getTypeOrVoid(Die);
      revng_assert(FunctionType->ReturnType.UnqualifiedType.isValid());

      uint64_t Index = 0;
      for (const DWARFDie &ChildDie : Die.children()) {
        if (ChildDie.getTag() == DW_TAG_formal_parameter) {

          const QualifiedType *ArgumentType = rc_recur getType(ChildDie);

          if (ArgumentType == nullptr) {
            reportIgnoredDie(Die,
                             "The type of argument " + Twine(Index + 1)
                               + " cannot be resolved");
            rc_return nullptr;
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
      model::QualifiedType TargetType = rc_recur getTypeOrVoid(Die);
      auto *Typedef = cast<model::TypedefType>(T);
      Typedef->CustomName = Name;
      Typedef->UnderlyingType = TargetType;
      revng_assert(Typedef->UnderlyingType.UnqualifiedType.isValid());

    } break;

    case llvm::dwarf::DW_TAG_structure_type: {
      auto MaybeSize = Die.find(DW_AT_byte_size);

      if (not MaybeSize) {
        reportIgnoredDie(Die, "Struct has no size");
        rc_return nullptr;
      }

      auto *Struct = cast<model::StructType>(T);
      Struct->CustomName = Name;
      Struct->Size = *MaybeSize->getAsUnsignedConstant();

      uint64_t Index = 0;
      for (const DWARFDie &ChildDie : Die.children()) {
        if (ChildDie.getTag() == DW_TAG_member) {

          // Collect offset
          auto MaybeOffset = ChildDie.find(DW_AT_data_member_location);

          if (not MaybeOffset) {
            reportIgnoredDie(ChildDie, "Struct field has no offset");
            continue;
          }

          auto Offset = *MaybeOffset->getAsUnsignedConstant();

          if (ChildDie.find(DW_AT_bit_size)) {
            reportIgnoredDie(ChildDie, "Ignoring bitfield in struct");
            continue;
          }

          const QualifiedType *MemberType = rc_recur getType(ChildDie);

          if (MemberType == nullptr) {
            reportIgnoredDie(Die,
                             "The type of member " + Twine(Index + 1)
                               + " cannot be resolved");
            rc_return nullptr;
          }

          //  Create new field
          auto &Field = Struct->Fields[Offset];
          Field.CustomName = getNameAsIdentifier(ChildDie);
          Field.Type = *MemberType;

          ++Index;
        }
      }

      if (Index == 0) {
        reportIgnoredDie(Die, "Struct has no fields");
        rc_return nullptr;
      }

    } break;

    case llvm::dwarf::DW_TAG_union_type: {
      auto *Union = cast<model::UnionType>(T);
      Union->CustomName = Name;

      uint64_t Index = 0;
      for (const DWARFDie &ChildDie : Die.children()) {
        if (ChildDie.getTag() == DW_TAG_member) {
          const QualifiedType *MemberType = rc_recur getType(ChildDie);
          if (MemberType == nullptr) {
            reportIgnoredDie(Die,
                             "The type of member " + Twine(Index + 1)
                               + " cannot be resolved");
            rc_return nullptr;
          }

          // Create new field
          auto &Field = Union->Fields[Index];
          Field.CustomName = getNameAsIdentifier(ChildDie);
          Field.Type = *MemberType;

          // Increment union index
          Index += 1;
        }
      }

      if (Index == 0) {
        reportIgnoredDie(Die, "Union has no fields");
        rc_return nullptr;
      }

    } break;

    case llvm::dwarf::DW_TAG_enumeration_type: {
      auto *Enum = cast<model::EnumType>(T);
      Enum->CustomName = Name;

      const QualifiedType *QualifiedUnderlyingType = rc_recur getType(Die);
      if (QualifiedUnderlyingType == nullptr) {
        reportIgnoredDie(Die, "The enum underlying type cannot be resolved");
        rc_return nullptr;
      }

      revng_assert(QualifiedUnderlyingType->Qualifiers.empty());
      const model::Type *UnderlyingType = nullptr;
      UnderlyingType = QualifiedUnderlyingType->UnqualifiedType.get();
      Enum->UnderlyingType = Model->getTypePath(UnderlyingType);

      uint64_t Index = 0;
      for (const DWARFDie &ChildDie : Die.children()) {
        if (ChildDie.getTag() == DW_TAG_enumerator) {
          // Collect value
          auto MaybeValue = getUnsignedOrSigned(ChildDie, DW_AT_const_value);
          if (not MaybeValue) {
            reportIgnoredDie(ChildDie,
                             "Ignoring enum entry " + Twine(Index + 1)
                               + " without a value");
            rc_return nullptr;
          }

          uint64_t Value = *MaybeValue;

          // Create new entry
          model::Identifier EntryName = getNameAsIdentifier(ChildDie);

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
      reportIgnoredDie(Die, "Unknown type");
      rc_return nullptr;
    }

    Placeholders.erase(Offset);

    rc_return TypePath;
  }

  RecursiveCoroutine<const model::QualifiedType *>
  resolveType(const DWARFDie &Die, bool ResolveIfHasIdentity) {
    using model::QualifiedType;

    // Ensure there are no loops in the dies we're exploring
    using ScopedSetElement = ScopedSetElement<decltype(InProgressDies)>;
    ScopedSetElement InProgressDie(InProgressDies, &Die);
    if (not InProgressDie.insert()) {
      reportIgnoredDie(Die, "Recursive die");
      rc_return nullptr;
    }

    auto Tag = Die.getTag();
    auto [MatchType, TypePath] = findType(Die);

    switch (MatchType) {
    case Absent: {
      // At this stage, all the unqualified types (i.e., those with an identity
      // in the model) should have been materialized.
      // Therefore, here we only deal with DWARF types the model represents as
      // qualifiers.
      revng_assert(not hasModelIdentity(Tag));

      bool HasType = Die.find(DW_AT_type).hasValue();
      model::QualifiedType Type = rc_recur getTypeOrVoid(Die);

      switch (Tag) {

      case llvm::dwarf::DW_TAG_const_type: {
        model::Qualifier NewQualifier;
        NewQualifier.Kind = model::QualifierKind::Const;
        Type.Qualifiers.push_back(NewQualifier);
      } break;

      case llvm::dwarf::DW_TAG_array_type: {

        if (not HasType) {
          reportIgnoredDie(Die, "Array does not specify element type");
          rc_return nullptr;
        }

        for (const DWARFDie &ChildDie : Die.children()) {
          if (ChildDie.getTag() == llvm::dwarf::DW_TAG_subrange_type) {
            model::Qualifier NewQualifier;
            NewQualifier.Kind = model::QualifierKind::Array;

            auto MaybeUpperBound = getUnsignedOrSigned(ChildDie,
                                                       DW_AT_upper_bound);
            auto MaybeCount = getUnsignedOrSigned(ChildDie, DW_AT_count);

            if (MaybeUpperBound and MaybeCount
                and *MaybeUpperBound != *MaybeCount + 1) {
              reportIgnoredDie(Die, "DW_AT_upper_bound != DW_AT_count + 1");
              rc_return nullptr;
            }

            if (MaybeUpperBound) {
              NewQualifier.Size = *MaybeUpperBound + 1;
            } else if (MaybeCount) {
              NewQualifier.Size = *MaybeCount;
            } else {
              reportIgnoredDie(Die,
                               "Array upper bound/elements count missing or "
                               "invalid");
              rc_return nullptr;
            }

            Type.Qualifiers.push_back(NewQualifier);
          }
        }
      } break;

      case llvm::dwarf::DW_TAG_pointer_type: {
        model::Qualifier NewQualifier;
        auto MaybeByteSize = Die.find(DW_AT_byte_size);

        if (not MaybeByteSize) {
          // TODO: force architecture pointer size
          reportIgnoredDie(Die, "Pointer has no size");
          rc_return nullptr;
        }

        NewQualifier.Kind = model::QualifierKind::Pointer;
        NewQualifier.Size = *MaybeByteSize->getAsUnsignedConstant();
        Type.Qualifiers.push_back(NewQualifier);
      } break;

      default:
        reportIgnoredDie(Die, "Unknown type");
        rc_return nullptr;
      }

      rc_return &record(Die, Type, true);
    }
    case PlaceholderType: {
      if (TypePath == nullptr) {
        reportIgnoredDie(Die, "Couldn't materialize type");
        rc_return nullptr;
      }

      auto Offset = Die.getOffset();
      revng_assert(Placeholders.count(Offset) != 0);
      // This die is already present in the map. Either it has already been
      // fully imported, or it's a type with an identity on the model.
      // In the latter case, proceed only if explicitly told to do so.
      if (ResolveIfHasIdentity)
        rc_recur resolveTypeWithIdentity(Die, TypePath);

    } break;

    case RegularType:
      if (TypePath == nullptr) {
        reportIgnoredDie(Die, "Couldn't materialize type");
        rc_return nullptr;
      }

      break;

    default:
      revng_abort();
    }

    rc_return TypePath;
  }

  void resolveAllTypes() {
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
    using namespace model;

    // Create function type
    UpcastableType NewType = makeType<CABIFunctionType>();
    auto *FunctionType = cast<model::CABIFunctionType>(NewType.get());
    FunctionType->CustomName = getNameAsIdentifier(Die);

    // Detect ABI
    CallingConvention CC = DW_CC_normal;
    auto MaybeCC = getUnsignedOrSigned(Die, DW_AT_calling_convention);
    if (MaybeCC)
      CC = static_cast<CallingConvention>(*MaybeCC);
    FunctionType->ABI = getABI(CC);

    if (FunctionType->ABI == model::abi::Invalid) {
      reportIgnoredDie(Die, "Unknown calling convention");
      return std::nullopt;
    }

    // Arguments
    uint64_t Index = 0;
    for (const DWARFDie &ChildDie : Die.children()) {
      if (ChildDie.getTag() == DW_TAG_formal_parameter) {
        const model::QualifiedType *ArgumenType = getType(ChildDie);
        if (ArgumenType == nullptr) {
          reportIgnoredDie(Die,
                           "The type of argument " + Twine(Index + 1)
                             + " cannot be resolved");
          return std::nullopt;
        }

        model::Argument &NewArgument = FunctionType->Arguments[Index];
        NewArgument.CustomName = getNameAsIdentifier(ChildDie);
        NewArgument.Type = *ArgumenType;
        Index += 1;
      }
    }

    // Return type
    FunctionType->ReturnType = getTypeOrVoid(Die);
    revng_assert(FunctionType->ReturnType.UnqualifiedType.isValid());

    return Model->recordNewType(std::move(NewType));
  }

  void createDynamicFunctions() {
    for (const auto &CU : DICtx.compile_units()) {
      for (const auto &Entry : CU->dies()) {
        DWARFDie Die = { CU.get(), &Entry };

        if (Die.getTag() != DW_TAG_subprogram)
          continue;

        auto MaybePath = getSubprogramPrototype(Die);
        if (not MaybePath) {
          reportIgnoredDie(Die, "Couldn't build subprogram prototype");
          continue;
        }

        std::string SymbolName = getName(Die);
        if (SymbolName.empty()) {
          reportIgnoredDie(Die, "Ignoring unnamed subprogram");
          continue;
        }

        // Get/create dynamic function
        auto &Function = Model->ImportedDynamicFunctions[SymbolName];

        // If a function already has a valid prototype, don't override it
        if (Function.Prototype.isValid())
          continue;

        auto *FunctionType = cast<model::CABIFunctionType>(MaybePath->get());
        Function.Prototype = *MaybePath;
      }
    }
  }

  void dropTypesDependingOnUnresolvedTypes() {
    std::set<const model::Type *> ToDrop;
    for (const auto [_, Type] : Placeholders)
      ToDrop.insert(Type);

    unsigned DroppedTypes = dropTypesDependingOnTypes(Model, ToDrop);

    revng_log(DILogger,
              "Purging " << DroppedTypes << " types (out of "
                         << TypesWithIdentityCount << ") due to "
                         << Placeholders.size() << " unresolved types");

    Placeholders.clear();
  }

public:
  void run() {
    materializeTypesWithIdentity();
    resolveAllTypes();
    createDynamicFunctions();
    dropTypesDependingOnUnresolvedTypes();
    deduplicateEquivalentTypes(Model);
    deduplicateNames(Model);
    revng_assert(Placeholders.size() == 0);
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

static StringRef getAltDebugLinkFileName(const object::Binary *B) {
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

  // TODO: recursively load dependant DWARFs:
  //
  // 1. Load any available DWARF in the binary itself
  // 2. Parse .note.gnu.build-id, .gnu_debugaltlink and .gnu_debuglink
  // 3. Load from the following paths:
  //    * /usr/lib/debug/.build-id/ab/cdef1234.debug
  //    * /usr/bin/ls.debug
  //    * /usr/bin/.debug/ls.debug
  //    * /usr/lib/debug/usr/bin/ls.debug
  //    In turn, parse .gnu_debugaltlink (and .gnu_debuglink?)
  // 2. Parse DT_NEEDED
  // 3. Look for each library in ld.so.conf directories
  // 4. Go to 1
  //
  // Source:
  // https://sourceware.org/gdb/onlinedocs/gdb/Separate-Debug-Files.html

  ErrorOr<std::unique_ptr<MemoryBuffer>>
    BuffOrErr = MemoryBuffer::getFileOrSTDIN(FileName);
  error(FileName, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BuffOrErr.get());
  Expected<std::unique_ptr<Binary>> BinOrErr = object::createBinary(*Buffer);
  error(FileName, errorToErrorCode(BinOrErr.takeError()));
  import(*BinOrErr->get(), FileName);
}

auto zip_pairs(auto &&R) {
  auto BeginIt = R.begin();
  auto EndIt = R.end();

  if (BeginIt == EndIt)
    return zip(make_range(EndIt, EndIt), make_range(EndIt, EndIt));

  auto First = BeginIt;
  auto Second = ++BeginIt;

  if (Second == EndIt)
    return zip(make_range(EndIt, EndIt), make_range(EndIt, EndIt));

  auto End = EndIt;
  auto Last = --EndIt;

  return zip(make_range(First, Last), make_range(Second, End));
}

/// This function considers all symbols with name of type STT_FUNC and clusters
/// them by address/type
static EquivalenceClasses<StringRef>
computeEquivalentSymbols(const llvm::object::ObjectFile &ELF) {
  using namespace llvm::object;

  EquivalenceClasses<StringRef> Result;

  struct SymbolDescriptor {
    uint64_t Address = 0;
    // TODO: one day we will want to consider STT_OBJECT too
    SymbolRef::Type Type = SymbolRef::ST_Unknown;
    /// \note we ignore this field for comparison purposes
    StringRef Name;

    auto key() const { return std::tie(Address, Type); }

    bool operator<(const SymbolDescriptor &Other) const {
      return key() < Other.key();
    }

    bool operator==(const SymbolDescriptor &Other) const {
      return key() == Other.key();
    }
  };

  std::vector<SymbolDescriptor> Symbols;
  for (const object::SymbolRef &Symbol : ELF.symbols()) {
    SymbolDescriptor NewSymbol;

    auto MaybeType = Symbol.getType();
    auto MaybeAddress = Symbol.getAddress();
    auto MaybeName = Symbol.getName();
    auto MaybeFlags = Symbol.getFlags();
    if (not MaybeType or not MaybeAddress or not MaybeName or not MaybeFlags)
      continue;

    // Ignore unnamed and nullptr symbols
    if (MaybeName->size() == 0 or *MaybeAddress == 0)
      continue;

    // Consider only STT_FUNC symbols
    if (*MaybeType != SymbolRef::ST_Function)
      continue;

    // Consider only global symbols
    if (!((*MaybeFlags) & SymbolRef::SF_Global))
      continue;

    Symbols.push_back({ *MaybeAddress, *MaybeType, *MaybeName });
  }

  llvm::sort(Symbols);

  for (const auto &[Previous, Current] : zip_pairs(Symbols))
    if (Previous == Current)
      Result.unionSets(Previous.Name, Current.Name);

  return Result;
}

// TODO: it wuold be beneficial to do this even at other levels
inline void detectAliases(const llvm::object::ObjectFile &ELF,
                          TupleTree<model::Binary> &Model) {
  EquivalenceClasses<StringRef> Aliases = computeEquivalentSymbols(ELF);
  auto &ImportedDynamicFunctions = Model->ImportedDynamicFunctions;

  for (auto AliasesIt = Aliases.begin(), E = Aliases.end(); AliasesIt != E;
       ++AliasesIt) {
    if (AliasesIt->isLeader()) {
      SmallVector<std::string, 4> UnprototypedFunctionsNames;
      model::TypePath Prototype;
      for (auto AliasSetIt = Aliases.member_begin(AliasesIt);
           AliasSetIt != Aliases.member_end();
           ++AliasSetIt) {
        std::string Name = AliasSetIt->str();

        // Create DynamicFunction, it doesn't exist already
        auto It = ImportedDynamicFunctions.find(Name);
        bool Found = It != ImportedDynamicFunctions.end();

        // If DynamicFunction doesn't have a prototype, register it for copying
        // it from the leader.
        // Otherwise, record the type as the leader.
        if (Found and It->Prototype.isValid()) {
          Prototype = It->Prototype;
        } else {
          UnprototypedFunctionsNames.push_back(Name);
        }
      }

      if (Prototype.isValid()) {
        for (const std::string &Name : UnprototypedFunctionsNames) {
          auto It = ImportedDynamicFunctions.find(Name);
          if (It == ImportedDynamicFunctions.end())
            It = ImportedDynamicFunctions.insert({ Name }).first;
          It->Prototype = Prototype;
        }
      }
    }
  }
}

void DwarfImporter::import(const llvm::object::Binary &TheBinary,
                           StringRef FileName) {
  using namespace llvm::object;

  if (auto *ELF = dyn_cast<ObjectFile>(&TheBinary)) {
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

    detectAliases(*ELF, Model);
  }

  LoadedFiles.push_back(sys::path::filename(FileName).str());
}
