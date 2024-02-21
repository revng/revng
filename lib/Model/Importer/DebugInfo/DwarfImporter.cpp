/// \file DwarfImporter.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <csignal>
#include <optional>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Progress.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Model/Argument.h"
#include "revng/Model/CABIFunctionType.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/Pass/AllPasses.h"
#include "revng/Model/Processing.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/ProgramRunner.h"

#include "ImportDebugInfoHelper.h"

using namespace llvm;
using namespace llvm::dwarf;

static Logger<> DILogger("dwarf-importer");
static const std::string GlobalDebugDirectory = "/usr/lib/debug/";

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

static std::optional<uint64_t> getUnsignedOrSigned(const DWARFDie &Die,
                                                   dwarf::Attribute Attribute) {
  auto Value = Die.find(Attribute);
  if (not Value)
    return {};
  else
    return getUnsignedOrSigned(*Value);
}

static std::optional<uint64_t> getAddress(const DWARFFormValue &Value) {
  auto MaybeResult = Value.getAsAddress();
  if (MaybeResult)
    return *MaybeResult;
  else
    return {};
}

static std::optional<uint64_t> getAddress(const DWARFDie &Die) {
  // TODO: Add support for DW_AT_pc, which is DWARF 5 Standard version of the
  // attribute.
  auto Value = Die.find(DW_AT_low_pc);
  if (not Value) {
    auto Ranges = Die.find(DW_AT_ranges);
    if (not Ranges)
      return {};

    auto Offset = *Ranges->getAsSectionOffset();
    auto Range = Die.getDwarfUnit()->findRnglistFromOffset(Offset);
    if (!Range)
      return {};

    // TODO: This is a vector, so we may want to return LowPC from every range
    // we found.
    return Range->begin()->LowPC;
  } else {
    return getAddress(*Value);
  }
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

class DwarfToModelConverter : public BinaryImporterHelper {
private:
  DwarfImporter &Importer;
  TupleTree<model::Binary> &Model;
  size_t Index;
  size_t AltIndex;
  size_t TypesWithIdentityCount;
  DWARFContext &DICtx;
  std::map<size_t, const model::Type *> Placeholders;
  std::set<const model::Type *> InvalidPrimitives;
  std::set<const DWARFDie *> InProgressDies;

public:
  DwarfToModelConverter(DwarfImporter &Importer,
                        DWARFContext &DICtx,
                        size_t Index,
                        size_t AltIndex,
                        uint64_t PreferredBaseAddress) :
    BinaryImporterHelper(Importer.getModel()->Architecture(),
                         PreferredBaseAddress),
    Importer(Importer),
    Model(Importer.getModel()),
    Index(Index),
    AltIndex(AltIndex),
    DICtx(DICtx) {

    Architecture = Model->Architecture();
    BaseAddress = PreferredBaseAddress;

    // Ensure the architecture is consistent.
    auto Arch = model::Architecture::fromLLVMArchitecture(DICtx.getArch());
    if (Model->Architecture() == model::Architecture::Invalid)
      Model->Architecture() = Arch;

    // Detect default ABI from the architecture.
    if (Model->DefaultABI() == model::ABI::Invalid)
      Model->DefaultABI() = model::ABI::getDefault(Model->Architecture());
  }

private:
  model::ABI::Values getABI(CallingConvention CC = DW_CC_normal) const {
    if (CC != DW_CC_normal)
      return model::ABI::Invalid;

    // NOTE: static functions do not always follow the standard calling
    //       convention which is a problem since `CABIFunctionTypes` we generate
    //       for them do not correspond to the real functions, leading to
    //       problems downstream.
    // TODO: investigate.

    return Model->DefaultABI();
  }

  const model::QualifiedType &record(const DWARFDie &Die,
                                     const model::TypePath &Path,
                                     bool IsNotPlaceholder) {
    return record(Die, model::QualifiedType(Path, {}), IsNotPlaceholder);
  }

  const model::QualifiedType &record(const DWARFDie &Die,
                                     const model::QualifiedType &QT,
                                     bool IsNotPlaceholder) {
    uint64_t Offset = Die.getOffset();
    revng_assert(QT.UnqualifiedType().isValid());
    if (not IsNotPlaceholder) {
      revng_assert(QT.Qualifiers().size() == 0);
      Placeholders[Offset] = QT.UnqualifiedType().get();
    }

    return Importer.recordType({ Index, Die.getOffset() }, QT);
  }

  enum TypeSearchResult {
    Invalid,
    Absent,
    PlaceholderType,
    RegularType
  };

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
      if (Placeholders.contains(Offset))
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
    auto [Result, NewTypePath] = Model->makeType<T>();
    record(Die, NewTypePath, false);
    return &Result;
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
      record(Die, Model->makeType<model::CABIFunctionType>().second, false);
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
    SmallVector<llvm::DWARFUnit *, 16> CompileUnits;
    for (const auto &CU : DICtx.compile_units())
      CompileUnits.push_back(CU.get());

    Task T(CompileUnits.size(), "Compile units");
    for (llvm::DWARFUnit *CU : CompileUnits) {
      T.advance("", true);

      SmallVector<llvm::DWARFDebugInfoEntry *, 16> Dies;
      for (llvm::DWARFDebugInfoEntry &Entry : CU->dies())
        Dies.push_back(&Entry);

      for (DWARFDebugInfoEntry *Entry : Dies) {
        DWARFDie Die = { CU, Entry };
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

  static bool isNoReturn(DWARFUnit &CU, const DWARFDie &Die) {
    auto Tag = Die.getTag();
    revng_assert(Tag == DW_TAG_subprogram);

    if (Die.find(DW_AT_noreturn))
      return true;

    // Check if the specification of this subprogram defines it.
    auto SpecificationAttribute = Die.find(DW_AT_specification);
    if (SpecificationAttribute) {
      if (SpecificationAttribute->getAsReference()) {
        auto DieOffset = *(SpecificationAttribute->getAsReference());
        DWARFDie SpecificationDie = CU.getDIEForOffset(DieOffset);
        if (SpecificationDie.find(DW_AT_noreturn))
          return true;
      }
    }

    return false;
  }

  RecursiveCoroutine<const model::QualifiedType *>
  getType(const DWARFDie &Die) {
    if (auto MaybeType = Die.find(DW_AT_type)) {
      if (MaybeType->getForm() == llvm::dwarf::DW_FORM_GNU_ref_alt) {
        rc_return findAltType(MaybeType->getRawUValue());
      } else {
        DWARFDie InnerDie = DICtx.getDIEForOffset(*MaybeType->getAsReference());
        rc_return rc_recur resolveType(InnerDie, false);
      }
    } else if (auto MaybeOrigin = Die.find(DW_AT_abstract_origin)) {
      DWARFDie Origin = DICtx.getDIEForOffset(*MaybeOrigin->getAsReference());
      rc_return rc_recur getType(Origin);
    } else {
      rc_return nullptr;
    }
  }

  RecursiveCoroutine<model::QualifiedType> getTypeOrVoid(const DWARFDie &Die) {
    using namespace model;
    using PTK = model::PrimitiveTypeKind::Values;
    const model::QualifiedType *Result = rc_recur getType(Die);
    if (Result != nullptr) {
      rc_return *Result;
    } else {
      rc_return QualifiedType(Model->getPrimitiveType(PTK::Void, 0), {});
    }
  }

  RecursiveCoroutine<const model::QualifiedType *>
  resolveTypeWithIdentity(const DWARFDie &Die, model::QualifiedType *TypePath) {
    using namespace model;

    auto Offset = Die.getOffset();
    auto Tag = Die.getTag();

    revng_assert(Placeholders.contains(Offset));
    revng_assert(TypePath->Qualifiers().empty());
    model::Type *T = TypePath->UnqualifiedType().get();

    std::string Name = getName(Die);

    if (InvalidPrimitives.contains(T))
      rc_return nullptr;

    switch (Tag) {
    case llvm::dwarf::DW_TAG_subroutine_type: {
      auto *FunctionType = cast<model::CABIFunctionType>(T);
      FunctionType->OriginalName() = Name;
      FunctionType->ABI() = getABI();

      if (FunctionType->ABI() == model::ABI::Invalid) {
        reportIgnoredDie(Die, "Unknown calling convention");
        rc_return nullptr;
      }

      FunctionType->ReturnType() = rc_recur getTypeOrVoid(Die);
      revng_assert(FunctionType->ReturnType().UnqualifiedType().isValid());

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

          // Note: at this stage we don't check the size. If an argument is
          // unsized, the function will be purged later on.

          model::Argument &NewArgument = FunctionType->Arguments()[Index];
          NewArgument.Type() = *ArgumentType;
          Index += 1;
        }
      }

    } break;

    case llvm::dwarf::DW_TAG_typedef:
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_volatile_type: {
      model::QualifiedType TargetType = rc_recur getTypeOrVoid(Die);
      auto *Typedef = cast<model::TypedefType>(T);
      Typedef->OriginalName() = Name;
      Typedef->UnderlyingType() = TargetType;
      revng_assert(Typedef->UnderlyingType().UnqualifiedType().isValid());

    } break;

    case llvm::dwarf::DW_TAG_structure_type: {
      auto MaybeSize = Die.find(DW_AT_byte_size);

      if (not MaybeSize) {
        reportIgnoredDie(Die, "Struct has no size");
        rc_return nullptr;
      }

      auto *Struct = cast<model::StructType>(T);
      Struct->OriginalName() = Name;
      Struct->Size() = *MaybeSize->getAsUnsignedConstant();

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

          // Create new field
          auto &Field = Struct->Fields()[Offset];
          Field.OriginalName() = getName(ChildDie);
          Field.Type() = *MemberType;

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
      Union->OriginalName() = Name;

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
          auto &Field = Union->Fields()[Index];
          Field.OriginalName() = getName(ChildDie);
          Field.Type() = *MemberType;

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
      Enum->OriginalName() = Name;

      const QualifiedType *QualifiedUnderlyingType = rc_recur getType(Die);
      if (QualifiedUnderlyingType == nullptr) {
        reportIgnoredDie(Die, "The enum underlying type cannot be resolved");
        rc_return nullptr;
      }

      revng_assert(QualifiedUnderlyingType->Qualifiers().empty());
      Enum->UnderlyingType() = *QualifiedUnderlyingType;

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
          std::string EntryName = getName(ChildDie);

          // If it's the first time, set OriginalName
          auto It = Enum->Entries().find(Value);
          if (It == Enum->Entries().end()) {
            auto &Entry = Enum->Entries()[Value];
            Entry.OriginalName() = EntryName;
          } else {
            // Ignore aliases
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

      /// \note There could be some TAGs we do not handle/recognize as types.
      if (isType(Tag))
        revng_assert(not hasModelIdentity(Tag));

      bool HasType = Die.find(DW_AT_type).has_value();
      model::QualifiedType Type = rc_recur getTypeOrVoid(Die);

      switch (Tag) {

      case llvm::dwarf::DW_TAG_const_type: {
        model::Qualifier NewQualifier;
        NewQualifier.Kind() = model::QualifierKind::Const;
        Type.Qualifiers().insert(Type.Qualifiers().begin(), NewQualifier);
      } break;

      case llvm::dwarf::DW_TAG_array_type: {

        if (not HasType) {
          reportIgnoredDie(Die, "Array does not specify element type");
          rc_return nullptr;
        }

        for (const DWARFDie &ChildDie : Die.children()) {
          if (ChildDie.getTag() == llvm::dwarf::DW_TAG_subrange_type) {
            model::Qualifier NewQualifier;
            NewQualifier.Kind() = model::QualifierKind::Array;
            NewQualifier.Size() = 0;

            auto MaybeUpperBound = getUnsignedOrSigned(ChildDie,
                                                       DW_AT_upper_bound);
            auto MaybeCount = getUnsignedOrSigned(ChildDie, DW_AT_count);

            if (MaybeUpperBound and MaybeCount
                and *MaybeUpperBound != *MaybeCount + 1) {
              reportIgnoredDie(Die, "DW_AT_upper_bound != DW_AT_count + 1");
              rc_return nullptr;
            }

            if (MaybeUpperBound) {
              NewQualifier.Size() = *MaybeUpperBound + 1;
            } else if (MaybeCount) {
              NewQualifier.Size() = *MaybeCount;
            }

            if (NewQualifier.Size() == 0) {
              reportIgnoredDie(Die,
                               "Array upper bound/elements count missing or "
                               "invalid");
              rc_return nullptr;
            }

            Type.Qualifiers().insert(Type.Qualifiers().begin(), NewQualifier);
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

        NewQualifier.Kind() = model::QualifierKind::Pointer;
        NewQualifier.Size() = *MaybeByteSize->getAsUnsignedConstant();
        Type.Qualifiers().insert(Type.Qualifiers().begin(), NewQualifier);
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
      revng_assert(Placeholders.contains(Offset));
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

    // Detect ABI
    CallingConvention CC = DW_CC_normal;
    auto MaybeCC = getUnsignedOrSigned(Die, DW_AT_calling_convention);
    if (MaybeCC)
      CC = static_cast<CallingConvention>(*MaybeCC);
    FunctionType->ABI() = getABI(CC);

    if (FunctionType->ABI() == model::ABI::Invalid) {
      reportIgnoredDie(Die, "Unknown calling convention");
      return std::nullopt;
    }

    // Arguments
    uint64_t Index = 0;
    for (const DWARFDie &ChildDie : Die.children()) {
      if (ChildDie.getTag() == DW_TAG_formal_parameter) {
        const model::QualifiedType *ArgumentType = getType(ChildDie);
        if (ArgumentType == nullptr) {
          reportIgnoredDie(Die,
                           "The type of argument " + Twine(Index + 1)
                             + " cannot be resolved");
          return std::nullopt;
        }

        // Note: at this stage we don't check the size. If an argument is
        // unsized, the function will be purged later on.

        model::Argument &NewArgument = FunctionType->Arguments()[Index];
        NewArgument.OriginalName() = getName(ChildDie);
        NewArgument.Type() = *ArgumentType;
        Index += 1;
      }
    }

    // Return type
    FunctionType->ReturnType() = getTypeOrVoid(Die);
    revng_assert(FunctionType->ReturnType().UnqualifiedType().isValid());

    return Model->recordNewType(std::move(NewType));
  }

  void createFunctions() {
    for (const auto &CU : DICtx.compile_units()) {
      for (const auto &Entry : CU->dies()) {
        DWARFDie Die = { CU.get(), &Entry };

        if (Die.getTag() != DW_TAG_subprogram)
          continue;

        auto &Functions = Model->ImportedDynamicFunctions();
        auto MaybePath = getSubprogramPrototype(Die);
        std::string SymbolName = getName(Die);

        MetaAddress LowPC;
        if (auto MaybeLowPC = getAddress(Die)) {
          // TODO: do a proper check to see if it's in a valid segment
          if (*MaybeLowPC != 0)
            LowPC = relocate(fromPC(*MaybeLowPC));
        }

        if (LowPC.isValid()) {
          // Get/create the local function
          auto &Function = Model->Functions()[LowPC];

          if (MaybePath && not Function.Prototype().isValid())
            Function.Prototype() = *MaybePath;

          if (SymbolName.size() != 0 and Function.OriginalName().size() == 0)
            Function.OriginalName() = SymbolName;

          Function.ExportedNames().insert(SymbolName);

          if (isNoReturn(*CU.get(), Die))
            Function.Attributes().insert(model::FunctionAttribute::NoReturn);
        } else if (not SymbolName.empty() and Functions.contains(SymbolName)) {
          // It's a dynamic function
          if (not MaybePath) {
            reportIgnoredDie(Die, "Couldn't build subprogram prototype");
            continue;
          }

          // Get/create dynamic function
          auto &DynamicFunction = Model->ImportedDynamicFunctions()[SymbolName];

          // If a function already has a valid prototype, don't override it
          if (DynamicFunction.Prototype().isValid())
            continue;
          DynamicFunction.Prototype() = *MaybePath;
          revng_assert(isa<model::CABIFunctionType>(DynamicFunction.Prototype()
                                                      .get()));

          if (isNoReturn(*CU.get(), Die)) {
            using namespace model;
            DynamicFunction.Attributes().insert(FunctionAttribute::NoReturn);
          }
        } else {
          reportIgnoredDie(Die, "Ignoring subprogram");
        }
      }
    }
  }

  void cleanupTypeSystem() {
    std::set<const model::Type *> ToDrop;

    model::VerifyHelper VH;
    for (auto &Type : Model->Types()) {
      //
      // Drop zero-sized struct/union fields
      //
      if (auto *Struct = dyn_cast<model::StructType>(Type.get())) {
        llvm::erase_if(Struct->Fields(), [&VH](model::StructField &Field) {
          std::optional<uint64_t> MaybeSize = Field.Type().trySize(VH);
          return MaybeSize.value_or(0) == 0;
        });
      } else if (auto *Union = dyn_cast<model::UnionType>(Type.get())) {
        llvm::erase_if(Union->Fields(), [&VH](model::UnionField &Field) {
          std::optional<uint64_t> MaybeSize = Field.Type().trySize(VH);
          return MaybeSize.value_or(0) == 0;
        });
      }

      //
      // Drop an empty enum.
      //
      if (auto *Enum = dyn_cast<model::EnumType>(Type.get())) {
        if (!Enum->Entries().size())
          ToDrop.insert(Type.get());
      }

      //
      // Drop functions with 0-sized arguments.
      //
      if (auto *FunctionType = dyn_cast<model::CABIFunctionType>(Type.get())) {
        for (model::Argument &Argument : FunctionType->Arguments()) {
          std::optional<uint64_t> Size = Argument.Type().trySize(VH);
          if (Size.value_or(0) == 0)
            ToDrop.insert(FunctionType);
        }
      }

      //
      // Collect array whose elements are zero-sized
      //
      for (const model::QualifiedType &QT : Type->edges()) {
        model::TypePath Unqualified = QT.UnqualifiedType();
        if (Unqualified.isValid()) {
          std::optional<uint64_t> Size = Unqualified.get()->trySize(VH);
          if (Size.value_or(0) != 0)
            continue;
        }

        // At this point only invalid types and types with no size remain.
        auto Iterator = revng::find_last_if_not(QT.Qualifiers(),
                                                model::Qualifier::isConst);
        if (Iterator != QT.Qualifiers().rend())
          if (Iterator->Kind() == model::QualifierKind::Array)
            ToDrop.insert(Type.get());
      }
    }

    //
    // Collect unresolved types
    //
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
    Task T(9, "Importing DWARF");
    T.advance("Materialize types with an identity", true);
    materializeTypesWithIdentity();
    T.advance("Resolve types", true);
    resolveAllTypes();
    T.advance("Create model functions", true);
    createFunctions();
    T.advance("Type system cleanup", true);
    cleanupTypeSystem();
    T.advance("Apply fixes to the model", true);
    fixModel(Model);
    T.advance("Deduplicate equivalent types", true);
    deduplicateEquivalentTypes(Model);
    T.advance("Promote OriginalName", true);
    promoteOriginalName(Model);
    T.advance("Purge unnamed unreachable types", true);
    purgeUnnamedAndUnreachableTypes(Model);
    revng_assert(Placeholders.size() == 0);
    T.advance("Verify the model", true);
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
    if (MaybeName and *MaybeName == Name) {
      auto MaybeContents = ELF.getSectionContents(Section);
      if (MaybeContents)
        return *MaybeContents;
    }
  }

  return {};
}

static std::string getBuildID(const object::Binary *B) {
  using namespace llvm::object;

  auto Handler = [&](auto *ELFObject) -> std::string {
    const auto &ELF = ELFObject->getELFFile();
    ArrayRef<uint8_t> Contents = getSectionsContents(".note.gnu.build-id", ELF);
    if (Contents.size() == 0)
      return {};

    std::string StringForBytes;
    raw_string_ostream OutputStream(StringForBytes);
    for (uint8_t Byte : Contents)
      OutputStream << format_hex_no_prefix(Byte, 2);

    // Build ID uses SHA1, so it is 20 bytes long.
    constexpr unsigned SHA1Size = 40;
    return OutputStream.str().substr(OutputStream.str().size() - SHA1Size);
  };

  std::string BuildID;
  if (auto *ELF = dyn_cast<ELF32BEObjectFile>(B)) {
    BuildID = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF64BEObjectFile>(B)) {
    BuildID = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF32LEObjectFile>(B)) {
    BuildID = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF64LEObjectFile>(B)) {
    BuildID = Handler(ELF);
  } else {
    revng_abort();
  }

  return BuildID;
}

static StringRef getDebugFileName(const object::Binary *B) {
  using namespace llvm::object;

  // TODO: Handle Split DWARF/DW_AT_GNU_dwo_name. Part of DWARF 5.

  auto Handler = [&](auto *ELFObject) -> StringRef {
    const auto &ELF = ELFObject->getELFFile();
    ArrayRef<uint8_t> Contents = getSectionsContents(".gnu_debuglink", ELF);
    if (Contents.size() == 0) {
      // If there is no ".gnu_debuglink", try ".gnu_debugaltlink".
      Contents = getSectionsContents(".gnu_debugaltlink", ELF);
    }

    if (Contents.size() == 0) {
      // TODO: Handle .debug_sup, which is DWARF 5 implementation of GNU
      // extension .gnu_debuglink sections.
      return {};
    }

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

static bool fileExists(const Twine &Path) {
  bool Result = sys::fs::exists(Path);

  if (Result) {
    revng_log(DILogger, "The following path does not exist: " << Path.str());
  } else {
    revng_log(DILogger, "Found: " << Path.str());
  }

  return Result;
}

static std::optional<std::string>
findDebugInfoFileByName(StringRef FileName,
                        StringRef DebugFileName,
                        llvm::object::ObjectFile *ELF) {
  // Let's find it in canonical places, where debug info was fetched.
  //  1) Look for a .gnu_debuglink/.gnu_debugaltlink/.debug_sup section.
  //  The .debug file should be in canonical places.
  //  E.g., if the executable is `/usr/bin/ls`, we look for:
  //     - /usr/bin/ls.debug (current dir of exe)
  //     - /usr/bin/.debug/ls.debug
  //     - /usr/lib/debug/usr/bin/ls.debug
  llvm::SmallString<128> ResultPath;
  if (llvm::sys::path::has_parent_path(FileName)) {
    llvm::sys::path::append(ResultPath,
                            llvm::sys::path::parent_path(FileName),
                            DebugFileName);
  } else {
    llvm::sys::path::append(ResultPath, DebugFileName);
  }

  if (fileExists(ResultPath.str())) {
    return std::string(ResultPath.str());
  } else {
    // Try in .debug/ directory.
    ResultPath.clear();
    llvm::sys::path::append(ResultPath,
                            llvm::sys::path::parent_path(FileName),
                            ".debug/",
                            DebugFileName);

    if (fileExists(ResultPath.str())) {
      return std::string(ResultPath.str());
    } else {
      // Try `/usr/lib/debug/usr/bin/ls.debug`-like path.
      ResultPath.clear();
      if (sys::path::is_absolute(FileName)) {
        llvm::sys::path::append(ResultPath,
                                GlobalDebugDirectory,
                                llvm::sys::path::parent_path(FileName),
                                DebugFileName);
      } else {
        // Relative path.
        llvm::SmallString<64> CurrentDirectory;
        auto ErrorCode = llvm::sys::fs::current_path(CurrentDirectory);
        if (!ErrorCode) {
          llvm::sys::path::append(ResultPath,
                                  GlobalDebugDirectory,
                                  CurrentDirectory,
                                  llvm::sys::path::parent_path(FileName),
                                  DebugFileName);
        } else {
          revng_log(DILogger, "Can't get current working path.");
        }
      }

      if (fileExists(ResultPath.str())) {
        return std::string(ResultPath.str());
      } else {
        // Try If build-id is `abcdef1234`, we look for:
        // - /usr/lib/debug/.build-id/ab/cdef1234.debug
        ResultPath.clear();
        auto BuildID = getBuildID(ELF);
        if (BuildID.size()) {
          // First two chars of build-id forms the debug info file directory.
          auto DebugDir = BuildID.substr(0, 2);
          // The rest of build-id forms the debug info file name.
          auto DebugFile = BuildID.substr(BuildID.size() - 38);
          auto DebugFileWithExtension = DebugFile.append(".debug");
          llvm::sys::path::append(ResultPath,
                                  GlobalDebugDirectory,
                                  ".build-id/",
                                  DebugDir,
                                  DebugFileWithExtension);

          if (fileExists(ResultPath.str())) {
            return std::string(ResultPath.str());
          } else {
            // Try in XDG_CACHE_HOME at the end.
            ResultPath.clear();
            auto XDGCacheHome = llvm::sys::Process::GetEnv("XDG_CACHE_HOME");
            SmallString<64> PathHome;
            sys::path::home_directory(PathHome);
            // Default debug directory.
            if (!XDGCacheHome) {
              llvm::sys::path::append(ResultPath,
                                      PathHome,
                                      ".cache/revng/debug-symbols/elf/",
                                      BuildID,
                                      "debug");
            } else {
              llvm::sys::path::append(ResultPath,
                                      *XDGCacheHome,
                                      "revng/debug-symbols/elf/",
                                      BuildID,
                                      "debug");
            }

            if (fileExists(ResultPath.str())) {
              return std::string(ResultPath.str());
            } else {
              revng_log(DILogger, "Can't find " << DebugFileName);
            }
          }
        } else {
          revng_log(DILogger, "Can't parse build-id.");
        }
      }
    }
  }

  // We have not found the debug info file on the device.
  return std::nullopt;
}

void DwarfImporter::import(StringRef FileName, const ImporterOptions &Options) {
  Task T(2, "Import DWARF files");

  using namespace llvm::object;
  ErrorOr<std::unique_ptr<MemoryBuffer>>
    BuffOrErr = MemoryBuffer::getFileOrSTDIN(FileName);
  error(FileName, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BuffOrErr.get());
  Expected<std::unique_ptr<Binary>> BinOrErr = object::createBinary(*Buffer);
  error(FileName, errorToErrorCode(BinOrErr.takeError()));

  // Find Debugging Information.
  // If the file has debug info sections within itself, no need for finding
  // it on the device.
  // TODO: When we add support for Split DWARF, this will need additional
  // improvement.
  auto HasDebugInfo = [](ObjectFile *Object) {
    for (const SectionRef &Section : Object->sections()) {
      StringRef SectionName;
      if (Expected<StringRef> NameOrErr = Section.getName()) {
        SectionName = *NameOrErr;
      } else {
        llvm::consumeError(NameOrErr.takeError());
        continue;
      }

      // TODO: When adding support for Split dwarf, there will be
      // .debug_info.dwo section, so we need to handle it.
      if (SectionName == ".debug_info")
        return true;
    }
    return false;
  };

  auto PerformImport = [this, &T, &Options](StringRef FilePath,
                                            StringRef TheDebugFile) {
    auto ExpectedBinary = object::createBinary(FilePath);
    if (!ExpectedBinary) {
      revng_log(DILogger, "Can't create binary for " << FilePath);
      llvm::consumeError(ExpectedBinary.takeError());
    } else {
      T.advance("Parsing " + llvm::sys::path::filename(TheDebugFile), true);
      import(*ExpectedBinary->getBinary(), TheDebugFile, Options.BaseAddress);
    }
  };

  if (auto *ELF = dyn_cast<ObjectFile>(BinOrErr->get())) {
    if (Options.DebugInfo != DebugInfoLevel::No && !HasDebugInfo(ELF)) {
      // There are no .debug_* sections in the file itself, let's try to find it
      // on the device, otherwise find it on web by using the `fetch-debuginfo`
      // tool.
      auto DebugFile = getDebugFileName(BinOrErr->get());
      if (!DebugFile.size()) {
        revng_log(DILogger, "Can't find file name of the debug file.");
        return;
      }
      auto DebugFilePath = findDebugInfoFileByName(FileName, DebugFile, ELF);
      if (!DebugFilePath) {
        if (!::Runner.isProgramAvailable("revng")) {
          revng_log(DILogger,
                    "Can't find `revng` binary to run `fetch-debuginfo`.");
          return;
        }

        int ExitCode = runFetchDebugInfoWithLevel(FileName);
        if (ExitCode != 0) {
          revng_log(DILogger,
                    "Failed to find debug info with `revng model "
                    "fetch-debuginfo`.");
        } else {
          DebugFilePath = findDebugInfoFileByName(FileName, DebugFile, ELF);
          if (DebugFilePath)
            PerformImport(*DebugFilePath, DebugFile);
        }
      } else {
        PerformImport(*DebugFilePath, DebugFile);
      }
    }
  }

  T.advance("Parsing " + llvm::sys::path::filename(FileName), true);
  import(*BinOrErr->get(), FileName, Options.BaseAddress);
}

auto zipPairs(auto &&R) {
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

  for (const auto &[Previous, Current] : zipPairs(Symbols))
    if (Previous == Current)
      Result.unionSets(Previous.Name, Current.Name);

  return Result;
}

// TODO: it would be beneficial to do this even at other levels
inline void detectAliases(const llvm::object::ObjectFile &ELF,
                          TupleTree<model::Binary> &Model) {
  EquivalenceClasses<StringRef> Aliases = computeEquivalentSymbols(ELF);
  auto &ImportedDynamicFunctions = Model->ImportedDynamicFunctions();
  auto &Functions = Model->Functions();

  std::unordered_map<std::string, model::Function *> FunctionsByName;
  // Map functions by names, so we have faster lookup below.
  for (auto &Function : Functions) {
    if (Function.OriginalName().size()) {
      FunctionsByName[Function.OriginalName()] = &Function;
    }
  }

  for (auto AliasesIt = Aliases.begin(), E = Aliases.end(); AliasesIt != E;
       ++AliasesIt) {
    llvm::SmallVector<std::string, 4> CurrentAliases;
    if (AliasesIt->isLeader()) {
      SmallVector<std::string, 4> UnprototypedFunctionsNames;
      model::TypePath Prototype;
      for (auto AliasSetIt = Aliases.member_begin(AliasesIt);
           AliasSetIt != Aliases.member_end();
           ++AliasSetIt) {
        std::string Name = AliasSetIt->str();
        CurrentAliases.push_back(Name);

        // Create DynamicFunction, if it doesn't exist already
        auto It = ImportedDynamicFunctions.find(Name);
        bool Found = It != ImportedDynamicFunctions.end();

        // If DynamicFunction doesn't have a prototype, register it for copying
        // it from the leader.
        // Otherwise, record the type as the leader.
        if (Found and It->Prototype().isValid()) {
          Prototype = It->Prototype();
        } else {
          UnprototypedFunctionsNames.push_back(Name);
        }
      }

      // Check if we should add an ExportedName for local Functions.
      llvm::SmallVector<std::string, 4> PotentialExportedNamesToBeAdded;
      bool IsLocalFunction = false;
      model::Function *TheFunction = nullptr;
      for (auto &Name : CurrentAliases) {
        auto It = FunctionsByName.find(Name);
        PotentialExportedNamesToBeAdded.push_back(Name);
        if (It != FunctionsByName.end()) {
          // We found a local function.
          // TODO: In some situations OriginalName is not in the ExportedNames?
          // For example in the case of importing `__libc_calloc` from
          // libc.so.6.
          TheFunction = It->second;
        }
      }
      // It is a local function. Populate the ExportedNames.
      if (TheFunction) {
        for (auto &Name : PotentialExportedNamesToBeAdded)
          TheFunction->ExportedNames().insert(Name);
        continue;
      }

      // Consider it as a Dynamic function.
      if (Prototype.isValid()) {
        for (const std::string &Name : UnprototypedFunctionsNames) {
          auto It = ImportedDynamicFunctions.find(Name);
          if (It == ImportedDynamicFunctions.end())
            It = ImportedDynamicFunctions.insert({ Name }).first;

          It->Prototype() = Prototype;
        }
      }
    }
  }
}

void DwarfImporter::import(const llvm::object::Binary &TheBinary,
                           StringRef FileName,
                           uint64_t PreferredBaseAddress) {
  using namespace llvm::object;

  if (auto *ELF = dyn_cast<ELFObjectFileBase>(&TheBinary)) {

    {
      using namespace model::Architecture;
      if (Model->Architecture() == Invalid)
        Model->Architecture() = fromLLVMArchitecture(ELF->getArch());
    }

    if (ELF->getEType() != ELF::ET_DYN)
      PreferredBaseAddress = 0;

    // Check if we already loaded the alt debug info file
    size_t AltIndex = -1;
    // Check if we already loaded the alt debug info file.
    StringRef SeparateDebugFileName = getDebugFileName(ELF);
    if (SeparateDebugFileName.size() > 0) {
      auto Begin = LoadedFiles.begin();
      auto End = LoadedFiles.end();
      auto It = std::find(Begin, End, SeparateDebugFileName);
      if (It != End)
        AltIndex = It - Begin;
    }
    auto TheDWARFContext = DWARFContext::create(*ELF);
    DwarfToModelConverter Converter(*this,
                                    *TheDWARFContext,
                                    LoadedFiles.size(),
                                    AltIndex,
                                    PreferredBaseAddress);
    Converter.run();

    detectAliases(*ELF, Model);
  }

  LoadedFiles.push_back(sys::path::filename(FileName).str());
}
