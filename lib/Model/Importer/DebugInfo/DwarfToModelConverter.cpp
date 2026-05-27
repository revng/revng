//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Progress.h"

#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Model/Pass/DeduplicateEquivalentTypes.h"
#include "revng/Model/Pass/FlattenPrimitiveTypedefs.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/Processing.h"

#include "DwarfToModelConverter.h"
#include "ImportDebugInfoHelper.h"

using namespace llvm;
using namespace llvm::dwarf;

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

// Iterate the children of `Die`, skipping (and logging) invalid ones.
static auto validChildren(const DWARFDie &Die) {
  return llvm::make_filter_range(Die.children(), [&Die](const DWARFDie &Child) {
    if (Child.isValid())
      return true;
    revng_log(DILogger, "Skipping invalid child DIE of " << Die.getOffset());
    return false;
  });
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
    auto MaybeRange = Die.getDwarfUnit()->findRnglistFromOffset(Offset);
    if (auto Error = MaybeRange.takeError()) {
      revng_log(DILogger, "findRnglistFromOffset failed: " << Error);
      consumeError(std::move(Error));
      return {};
    }

    // TODO: This is a vector, so we may want to return LowPC from every range
    // we found.
    return MaybeRange->begin()->LowPC;
  } else {
    return getAddress(*Value);
  }
}

static bool isTrue(const DWARFFormValue &Value) {
  return getUnsignedOrSigned(Value) != 0;
}

static model::PrimitiveKind::Values dwarfEncodingToModel(uint32_t Encoding) {
  switch (Encoding) {
  case dwarf::DW_ATE_unsigned_char:
  case dwarf::DW_ATE_unsigned:
  case dwarf::DW_ATE_boolean:
    return model::PrimitiveKind::Unsigned;
  case dwarf::DW_ATE_signed_char:
  case dwarf::DW_ATE_signed:
    return model::PrimitiveKind::Signed;
  case dwarf::DW_ATE_float:
    return model::PrimitiveKind::Float;
  default:
    return model::PrimitiveKind::Invalid;
  }
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

DwarfToModelConverter::DwarfToModelConverter(DwarfImporter &Importer,
                                             llvm::DWARFContext &Context,
                                             size_t Index,
                                             size_t AltIndex,
                                             uint64_t PreferredBaseAddress) :
  BinaryImporterHelper(Importer.getModel(), PreferredBaseAddress, DILogger),
  Importer(Importer),
  Model(Importer.getModel()),
  Index(Index),
  AltIndex(AltIndex),
  Context(Context) {

  // When we import DWARF, we assume we already have parsed Segments
  processSegments();

  BaseAddress = PreferredBaseAddress;

  // Ensure the architecture is consistent.
  auto Arch = model::Architecture::fromLLVMArchitecture(Context.getArch());
  if (Model->Architecture() == model::Architecture::Invalid)
    Model->Architecture() = Arch;

  // Set default ABI
  if (Model->DefaultABI() == model::ABI::Invalid) {
    if (auto ABI = model::ABI::getDefaultForELF(Model->Architecture())) {
      Model->DefaultABI() = ABI.value();
    } else {
      auto AName = model::Architecture::getName(Model->Architecture()).str();
      revng_abort(("Unsupported architecture for ELF: " + AName).c_str());
    }
  }
}

model::ABI::Values
DwarfToModelConverter::getABI(llvm::dwarf::CallingConvention CC) const {
  if (CC != llvm::dwarf::DW_CC_normal)
    return model::ABI::Invalid;

  // NOTE: static functions do not always follow the standard calling
  //       convention which is a problem since `CABIFunctionTypes` we generate
  //       for them do not correspond to the real functions, leading to
  //       problems downstream.
  // TODO: investigate.

  return Model->DefaultABI();
}

const model::UpcastableType &
DwarfToModelConverter::record(const llvm::DWARFDie &Die,
                              model::UpcastableType &&Type) {
  revng_assert(!Type.isEmpty());
  return Importer.recordType({ Index, Die.getOffset() }, std::move(Type));
}

const model::UpcastableType &
DwarfToModelConverter::recordPlaceholder(const llvm::DWARFDie &Die,
                                         model::UpcastableType &&Type) {
  // `model::UpcastableType::empty()` (as in, no definition) represents
  // a primitive placeholder.
  Placeholders[Die.getOffset()] = Type->tryGetAsDefinition();
  return record(Die, std::move(Type));
}

std::pair<DwarfToModelConverter::TypeSearchResult, model::UpcastableType>
DwarfToModelConverter::findType(const llvm::DWARFDie &Die) {
  auto Found = Importer.findType({ Index, Die.getOffset() });
  if (Found.isEmpty())
    return { TypeSearchResult::Absent, model::UpcastableType::empty() };
  else if (Placeholders.contains(Die.getOffset()))
    return { TypeSearchResult::PlaceholderType, std::move(Found) };
  else
    return { TypeSearchResult::RegularType, std::move(Found) };
}

bool DwarfToModelConverter::isType(llvm::dwarf::Tag Tag) {
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

bool DwarfToModelConverter::hasModelIdentity(llvm::dwarf::Tag Tag) {
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

void DwarfToModelConverter::createInvalidPrimitivePlaceholder(const DWARFDie
                                                                &Die) {
  auto &&[Definition, Type] = Model->makeTypedefDefinition();
  recordPlaceholder(Die, std::move(Type));
  InvalidPrimitives.insert(&Definition);
}

void DwarfToModelConverter::createType(const DWARFDie &Die) {
  auto Tag = Die.getTag();
  revng_assert(hasModelIdentity(Tag));

  switch (Tag) {
  case llvm::dwarf::DW_TAG_base_type: {
    uint8_t Size = 0;
    model::PrimitiveKind::Values Kind = model::PrimitiveKind::Invalid;

    auto MaybeByteSize = Die.find(DW_AT_byte_size);
    if (MaybeByteSize)
      Size = *MaybeByteSize->getAsUnsignedConstant();

    auto MaybeEncoding = Die.find(DW_AT_encoding);
    if (MaybeEncoding)
      Kind = dwarfEncodingToModel(*MaybeEncoding->getAsUnsignedConstant());

    if (Kind == model::PrimitiveKind::Invalid) {
      reportIgnoredDie(Die, "Unknown primitive type");
      createInvalidPrimitivePlaceholder(Die);
      return;
    }

    if (Size == 0) {
      reportIgnoredDie(Die, "Invalid size for primitive type");
      createInvalidPrimitivePlaceholder(Die);
      return;
    }

    if (Kind == model::PrimitiveKind::Float and Size > 8) {
      reportIgnoredDie(Die, "Ignoring floating-point primitives larger than 8");
      createInvalidPrimitivePlaceholder(Die);
      return;
    }

    record(Die, model::PrimitiveType::make(Kind, Size));
  } break;

  case llvm::dwarf::DW_TAG_subroutine_type:
    recordPlaceholder(Die, Model->makeCABIFunctionDefinition().second);
    break;

  case llvm::dwarf::DW_TAG_typedef:
  case llvm::dwarf::DW_TAG_restrict_type:
  case llvm::dwarf::DW_TAG_volatile_type:
    recordPlaceholder(Die, std::move(Model->makeTypedefDefinition().second));
    break;

  case llvm::dwarf::DW_TAG_structure_type:
    recordPlaceholder(Die, std::move(Model->makeStructDefinition().second));
    break;

  case llvm::dwarf::DW_TAG_union_type:
    if (auto MaybeByteSize = Die.find(DW_AT_byte_size);
        MaybeByteSize and not Die.hasChildren()) {
      // Handle small empty unions, usually due to transparent unions
      auto Size = *MaybeByteSize->getAsUnsignedConstant();
      record(Die, model::PrimitiveType::makeGeneric(Size));
    } else {
      // Handle regular unions
      recordPlaceholder(Die, std::move(Model->makeUnionDefinition().second));
    }
    break;

  case llvm::dwarf::DW_TAG_enumeration_type:
    recordPlaceholder(Die, std::move(Model->makeEnumDefinition().second));
    break;

  default:
    reportIgnoredDie(Die, "Unexpected type");
  }
}

void DwarfToModelConverter::handleTypeDeclaration(const DWARFDie &Die) {
  auto Tag = Die.getTag();
  if ((Tag == llvm::dwarf::DW_TAG_structure_type
       or Tag == llvm::dwarf::DW_TAG_union_type
       or Tag == llvm::dwarf::DW_TAG_enumeration_type)) {
    record(Die, model::PrimitiveType::makeVoid());
  } else {
    reportIgnoredDie(Die,
                     "Unexpected declaration for tag "
                       + llvm::dwarf::TagString(Tag));
  }
}

void DwarfToModelConverter::materializeTypesWithIdentity() {
  revng_log(DILogger, "materializeTypesWithIdentity");
  LoggerIndent Indent(DILogger);

  SmallVector<llvm::DWARFUnit *, 16> CompileUnits;
  for (const auto &CU : Context.compile_units())
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

std::string DwarfToModelConverter::getName(const DWARFDie &InitialDie) const {
  DWARFDie Die = InitialDie;
  std::set<uint64_t> Visited;
  while (Die.isValid()) {
    if (auto MaybeName = Die.find(DW_AT_name)) {
      auto MaybeString = MaybeName->getAsCString();
      if (auto Error = MaybeString.takeError()) {
        revng_log(DILogger, "Can't get DIE name: " << Error);
        consumeError(std::move(Error));
        return {};
      }
      return *MaybeString;
    }
    auto MaybeOrigin = Die.find(DW_AT_abstract_origin);
    if (not MaybeOrigin)
      return {};
    if (not Visited.insert(Die.getOffset()).second)
      return {};
    Die = Context.getDIEForOffset(*MaybeOrigin->getAsReference());
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

RecursiveCoroutine<model::UpcastableType>
DwarfToModelConverter::makeType(const DWARFDie &Die) {
  if (auto Type = Die.find(DW_AT_type)) {
    if (Type->getForm() == llvm::dwarf::DW_FORM_GNU_ref_alt) {
      rc_return Importer.findType({ AltIndex, Type->getRawUValue() }).copy();
    } else {
      DWARFDie InnerDie = Context.getDIEForOffset(*Type->getAsReference());
      rc_return rc_recur resolveType(InnerDie, false);
    }
  } else if (auto MaybeOrigin = Die.find(DW_AT_abstract_origin)) {
    DWARFDie Origin = Context.getDIEForOffset(*MaybeOrigin->getAsReference());
    rc_return rc_recur makeType(Origin);
  } else {
    rc_return model::UpcastableType::empty();
  }
}

RecursiveCoroutine<model::UpcastableType>
DwarfToModelConverter::makeTypeOrVoid(const DWARFDie &Die) {
  if (model::UpcastableType Result = rc_recur makeType(Die))
    rc_return Result;
  else
    rc_return model::PrimitiveType::makeVoid();
}

RecursiveCoroutine<void>
DwarfToModelConverter::resolveTypeWithIdentity(const DWARFDie &Die,
                                               model::UpcastableType &Type) {
  using namespace model;

  auto Offset = Die.getOffset();
  auto Tag = Die.getTag();

  revng_assert(Placeholders.contains(Offset));

  std::string Name = getName(Die);

  model::TypeDefinition &Definition = *Type->tryGetAsDefinition();
  revng_log(DILogger, "Handling type " << Definition.ID());
  LoggerIndent Indent(DILogger);

  if (InvalidPrimitives.contains(&Definition)) {
    revng_log(DILogger, "Skipping invalid primitive");
    rc_return;
  }

  switch (Tag) {
  case llvm::dwarf::DW_TAG_subroutine_type: {
    auto &FunctionType = cast<model::CABIFunctionDefinition>(Definition);
    FunctionType.Name() = Name;
    FunctionType.ABI() = getABI();

    if (FunctionType.ABI() == model::ABI::Invalid) {
      reportIgnoredDie(Die, "Unknown calling convention");
      rc_return;
    }

    FunctionType.ReturnType() = rc_recur makeType(Die);

    uint64_t Index = 0;
    for (const DWARFDie &ChildDie : validChildren(Die)) {
      if (ChildDie.getTag() == DW_TAG_formal_parameter) {

        model::UpcastableType ArgumentType = rc_recur makeType(ChildDie);
        if (ArgumentType.isEmpty()) {
          reportIgnoredDie(Die,
                           "The type of argument " + Twine(++Index)
                             + " cannot be resolved");
          rc_return;
        }

        // Note: at this stage we don't check the size. If an argument is
        // unsized, the function will be purged later on.
        FunctionType.addArgument(std::move(ArgumentType));
      }
    }

  } break;

  case llvm::dwarf::DW_TAG_typedef:
  case llvm::dwarf::DW_TAG_restrict_type:
  case llvm::dwarf::DW_TAG_volatile_type: {
    auto &Typedef = cast<model::TypedefDefinition>(Definition);
    Typedef.Name() = Name;
    Typedef.UnderlyingType() = rc_recur makeTypeOrVoid(Die);

  } break;

  case llvm::dwarf::DW_TAG_structure_type: {
    auto MaybeSize = Die.find(DW_AT_byte_size);

    if (not MaybeSize) {
      reportIgnoredDie(Die, "Struct has no size");
      rc_return;
    }

    auto &Struct = cast<model::StructDefinition>(Definition);
    Struct.Name() = Name;
    Struct.Size() = *MaybeSize->getAsUnsignedConstant();

    uint64_t Index = 0;
    for (const DWARFDie &ChildDie : validChildren(Die)) {
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

        model::UpcastableType MemberType = rc_recur makeType(ChildDie);
        if (MemberType.isEmpty()) {
          reportIgnoredDie(Die,
                           "The type of member " + Twine(Index + 1)
                             + " cannot be resolved");
          rc_return;
        }

        // Create new field
        auto &Field = Struct.Fields()[Offset];
        Field.Name() = getName(ChildDie);
        Field.Type() = std::move(MemberType);

        ++Index;
      }
    }

    if (Index == 0) {
      reportIgnoredDie(Die, "Struct has no fields");
      rc_return;
    }

  } break;

  case llvm::dwarf::DW_TAG_union_type: {
    auto &Union = cast<model::UnionDefinition>(Definition);
    Union.Name() = Name;

    for (const DWARFDie &ChildDie : validChildren(Die)) {
      if (ChildDie.getTag() == DW_TAG_member) {
        model::UpcastableType MemberType = rc_recur makeType(ChildDie);
        if (MemberType.isEmpty()) {
          reportIgnoredDie(Die,
                           "The type of one of the fields cannot be "
                           "resolved");
          rc_return;
        }

        // Create new field
        auto &Field = Union.addField(std::move(MemberType));
        Field.Name() = getName(ChildDie);
      }
    }

    if (Union.Fields().empty()) {
      reportIgnoredDie(Die, "Union has no fields");
      rc_return;
    }

  } break;

  case llvm::dwarf::DW_TAG_enumeration_type: {
    auto &Enum = llvm::cast<model::EnumDefinition>(Definition);
    Enum.Name() = Name;

    const model::UpcastableType UnderlyingType = rc_recur makeType(Die);
    if (UnderlyingType.isEmpty()) {
      reportIgnoredDie(Die, "The enum underlying type cannot be resolved");
      rc_return;
    }

    Enum.UnderlyingType() = std::move(UnderlyingType);

    uint64_t Index = 0;
    for (const DWARFDie &ChildDie : validChildren(Die)) {
      if (ChildDie.getTag() == DW_TAG_enumerator) {
        // Collect value
        auto MaybeValue = getUnsignedOrSigned(ChildDie, DW_AT_const_value);
        if (not MaybeValue) {
          reportIgnoredDie(ChildDie,
                           "Ignoring enum entry " + Twine(Index + 1)
                             + " without a value");
          rc_return;
        }

        uint64_t Value = *MaybeValue;

        // Create new entry
        std::string EntryName = getName(ChildDie);

        // If it's the first time, set the name
        auto *It = Enum.Entries().tryGet(Value);
        if (It == nullptr) {
          auto &Entry = Enum.Entries()[Value];
          Entry.Name() = EntryName;
        } else {
          // Ignore aliases
        }

        ++Index;
      }
    }

  } break;

  default:
    reportIgnoredDie(Die, "Unknown type");
    rc_return;
  }

  Placeholders.erase(Offset);

  rc_return;
}

RecursiveCoroutine<model::UpcastableType>
DwarfToModelConverter::resolveType(const DWARFDie &Die,
                                   bool ResolveIfHasIdentity) {
  // Ensure there are no loops in the dies we're exploring
  using ScopedSetElement = ScopedSetElement<decltype(InProgressDies)>;
  ScopedSetElement InProgressDie(InProgressDies, &Die);
  if (not InProgressDie.insert()) {
    reportIgnoredDie(Die, "Recursive die");
    rc_return model::UpcastableType::empty();
  }

  auto Tag = Die.getTag();
  auto &&[SearchResult, Type] = findType(Die);

  switch (SearchResult) {
  case TypeSearchResult::Absent: {
    // At this stage, all the type definitions (as in, types with an identity
    // in the model) should have been materialized.
    // Therefore, here we only deal with DWARF types the model represents as
    // by nesting.

    /// \note There could be some TAGs we do not handle/recognize as types.
    if (isType(Tag))
      revng_assert(not hasModelIdentity(Tag));

    bool HasType = Die.find(DW_AT_type).has_value();
    model::UpcastableType Result = rc_recur makeTypeOrVoid(Die);

    switch (Tag) {

    case llvm::dwarf::DW_TAG_const_type: {
      revng_assert(Result->IsConst() == false);
      Result->IsConst() = true;
    } break;

    case llvm::dwarf::DW_TAG_array_type: {

      if (not HasType) {
        reportIgnoredDie(Die, "Array does not specify element type");
        rc_return model::UpcastableType::empty();
      }

      for (const DWARFDie &ChildDie : validChildren(Die)) {
        if (ChildDie.getTag() == llvm::dwarf::DW_TAG_subrange_type) {
          auto MaybeUpperBound = getUnsignedOrSigned(ChildDie,
                                                     DW_AT_upper_bound);
          auto MaybeCount = getUnsignedOrSigned(ChildDie, DW_AT_count);

          if (MaybeUpperBound and MaybeCount
              and *MaybeUpperBound != *MaybeCount + 1) {
            reportIgnoredDie(Die, "DW_AT_upper_bound != DW_AT_count + 1");
            rc_return model::UpcastableType::empty();
          }

          if (MaybeUpperBound) {
            Result = model::ArrayType::make(std::move(Result),
                                            *MaybeUpperBound + 1);
          } else if (MaybeCount) {
            Result = model::ArrayType::make(std::move(Result), *MaybeCount);
          } else {
            reportIgnoredDie(Die,
                             "Array upper bound/elements count missing or "
                             "invalid");
            rc_return model::UpcastableType::empty();
          }
        }
      }
    } break;

    case llvm::dwarf::DW_TAG_pointer_type: {
      auto MaybeByteSize = Die.find(DW_AT_byte_size);
      if (not MaybeByteSize) {
        // TODO: force architecture pointer size
        reportIgnoredDie(Die, "Pointer has no size");
        rc_return model::UpcastableType::empty();
      }

      uint64_t PointerSize = *MaybeByteSize->getAsUnsignedConstant();
      Result = model::PointerType::make(std::move(Result), PointerSize);
    } break;

    default:
      reportIgnoredDie(Die, "Unknown type");
      rc_return model::UpcastableType::empty();
    }

    rc_return record(Die, std::move(Result)).copy();
  }
  case TypeSearchResult::PlaceholderType: {
    if (Type.isEmpty()) {
      reportIgnoredDie(Die, "Couldn't materialize type");
      rc_return model::UpcastableType::empty();
    }

    revng_assert(Placeholders.contains(Die.getOffset()));
    // This die is already present in the map. Either it has already been
    // fully imported, or it's a type with an identity on the model.
    // In the latter case, proceed only if explicitly told to do so.
    if (ResolveIfHasIdentity)
      rc_recur resolveTypeWithIdentity(Die, Type);

    rc_return std::move(Type);
  }

  case TypeSearchResult::RegularType:
    if (Type.isEmpty()) {
      reportIgnoredDie(Die, "Couldn't materialize type");
      rc_return model::UpcastableType::empty();
    }

    rc_return std::move(Type);

  default:
    revng_abort();
  }
}

void DwarfToModelConverter::resolveAllTypes() {
  revng_log(DILogger, "resolveAllTypes");
  LoggerIndent Indent(DILogger);

  for (const auto &CU : Context.compile_units()) {
    for (const auto &Entry : CU->dies()) {
      DWARFDie Die = { CU.get(), &Entry };
      if (not isType(Die.getTag()))
        continue;
      resolveType(Die, true);
    }
  }
}

model::UpcastableType
DwarfToModelConverter::getSubprogramPrototype(const DWARFDie &InitialDie) {
  using namespace llvm::dwarf;

  DWARFDie Die = InitialDie;

  // Skip over DW_AT_abstract_origin
  std::set<uint64_t> Visited;
  Visited.insert(Die.getOffset());
  while (auto MaybeOrigin = Die.find(DW_AT_abstract_origin)) {
    DWARFDie Origin = Context.getDIEForOffset(*MaybeOrigin->getAsReference());

    if (not Origin.isValid()) {
      reportIgnoredDie(Die, "DW_AT_abstract_origin resolves to an invalid DIE");
      return model::UpcastableType::empty();
    }

    if (Visited.contains(Origin.getOffset())) {
      reportIgnoredDie(Die, "Found a loop in DW_AT_abstract_origin references");
      return model::UpcastableType::empty();
    }

    Die = Origin;
    Visited.insert(Die.getOffset());
  }

  // Create function type
  auto NewType = model::makeTypeDefinition<model::CABIFunctionDefinition>();
  auto &FunctionType = cast<model::CABIFunctionDefinition>(*NewType.get());

  // Detect ABI
  CallingConvention CC = DW_CC_normal;
  auto MaybeCC = getUnsignedOrSigned(Die, DW_AT_calling_convention);
  if (MaybeCC)
    CC = static_cast<CallingConvention>(*MaybeCC);
  FunctionType.ABI() = getABI(CC);

  if (FunctionType.ABI() == model::ABI::Invalid) {
    reportIgnoredDie(Die, "Unknown calling convention");
    return model::UpcastableType::empty();
  }

  // Arguments
  uint64_t Index = 0;
  for (const DWARFDie &ChildDie : validChildren(Die)) {
    if (ChildDie.getTag() == DW_TAG_formal_parameter) {
      model::UpcastableType ArgumentType = makeType(ChildDie);
      if (ArgumentType.isEmpty()) {
        reportIgnoredDie(Die,
                         "The type of argument " + Twine(++Index)
                           + " cannot be resolved");
        return model::UpcastableType::empty();
      }

      // Note: at this stage we don't check the size. If an argument is
      // unsized, the function will be purged later on.
      model::Argument &A = FunctionType.addArgument(std::move(ArgumentType));
      A.Name() = getName(ChildDie);
    }
  }

  bool IsPrototyped = getUnsignedOrSigned(Die, DW_AT_prototyped)
                        .value_or(false);
  bool IsDeclaration = getUnsignedOrSigned(Die, DW_AT_declaration)
                         .value_or(false);
  bool HasType = Die.find(DW_AT_type).has_value();
  bool HasArguments = not FunctionType.Arguments().empty();

  if (IsDeclaration and not HasType and not IsPrototyped and not HasArguments) {
    // Ignore declaration without a prototype
    reportIgnoredDie(Die,
                     "Declaration without any useful prototype information");
    return model::UpcastableType::empty();
  }

  // Return type
  FunctionType.ReturnType() = makeType(Die);

  return Model->recordNewType(std::move(NewType)).second;
}

/// Substitute a glibc IFUNC resolver prototype (no args, returns a pointer
/// to a CABIFunctionDefinition) with the pointee prototype.
static model::UpcastableType
unwrapIfuncResolverPrototype(model::Binary &Binary,
                             const MetaAddress &Address,
                             model::UpcastableType Prototype) {
  auto *Definition = Prototype->tryGetAsDefinition();
  auto *Resolver = dyn_cast_or_null<model::CABIFunctionDefinition>(Definition);
  if (Resolver == nullptr or not Resolver->Arguments().empty()
      or Resolver->ReturnType().isEmpty()
      or not Resolver->ReturnType()->isPointer())
    return Prototype;

  auto *PointeeDefinition = Resolver->ReturnType()
                              ->getPointee()
                              .tryGetAsDefinition();
  if (not isa_and_nonnull<model::CABIFunctionDefinition>(PointeeDefinition))
    return Prototype;

  revng_log(DILogger,
            "Ifunc resolver at " << Address.toString()
                                 << ": substituting resolver prototype "
                                 << Resolver->ID() << " with pointee CABI "
                                 << PointeeDefinition->ID());
  return Binary.makeType(PointeeDefinition->key());
}

void DwarfToModelConverter::createFunctions() {
  revng_log(DILogger, "createFunctions");
  LoggerIndent Indent(DILogger);

  for (const auto &CU : Context.compile_units()) {
    for (const auto &Entry : CU->dies()) {
      DWARFDie Die = { CU.get(), &Entry };

      if (Die.getTag() != DW_TAG_subprogram)
        continue;

      auto &DynamicFunctions = Model->ImportedDynamicFunctions();
      std::string SymbolName = getName(Die);

      MetaAddress LowPC;
      if (auto MaybeLowPC = getAddress(Die)) {
        // TODO: do a proper check to see if it's in a valid segment
        if (*MaybeLowPC != 0) {
          // Relocate the raw DWARF address first, then let matchFunctionEntry
          // pick the code type that matches the model.
          uint64_t Relocated = relocate(*MaybeLowPC).address();
          MetaAddress Match = matchFunctionEntry(Relocated,
                                                 Model->Architecture());
          if (Match.isValid() and Importer.isFunctionAllowed(Match)) {
            LowPC = Match;
          } else {
            revng_log(DILogger,
                      "Ignoring disallowed function at 0x"
                        << llvm::utohexstr(*MaybeLowPC, true) << " with name \""
                        << SymbolName << "\"");
          }
        }
      }

      revng_log(DILogger,
                "Considering function at "
                  << LowPC.toString() << " with name \"" << SymbolName << "\"");
      LoggerIndent Indent(DILogger);

      model::UpcastableType Prototype = getSubprogramPrototype(Die);

      if (LowPC.isValid()) {
        revng_log(DILogger,
                  "Found a subprogram with LowPC "
                    << LowPC.toString() << " and name \"" << SymbolName
                    << "\"");

        // Use the existing model::Function or create a new one at LowPC.
        auto &Function = Model->Functions()[LowPC];

        if (Prototype.isEmpty()) {
          revng_log(DILogger, "Can't get the prototype");
        } else if (not Function.prototype()) {
          // For STT_GNU_IFUNC the DWARF describes the resolver, not the
          // resolved function; unwrap it.
          if (Importer.isIfunc(LowPC))
            Prototype = unwrapIfuncResolverPrototype(*Model,
                                                     LowPC,
                                                     std::move(Prototype));

          revng_log(DILogger,
                    "Assigning prototype "
                      << Prototype->tryGetAsDefinition()->ID());
          Function.Prototype() = std::move(Prototype);
        } else {
          revng_log(DILogger,
                    "Function already has a prototype, not setting it.");
        }

        if (SymbolName.size() != 0) {
          // Note: DWARF support
          if (Function.Name().empty()) {
            Function.Name() = SymbolName;
          } else {
            revng_log(DILogger,
                      "Function already has a name: " << Function.Name()
                                                      << ". Not updating "
                                                         "it.");
          }

          Function.ExportedNames().insert(SymbolName);
        }

        if (isNoReturn(*CU.get(), Die))
          Function.Attributes().insert(model::FunctionAttribute::NoReturn);
      } else if (not SymbolName.empty()
                 and DynamicFunctions.contains(SymbolName)) {
        // It's a dynamic function
        if (Prototype.isEmpty()) {
          reportIgnoredDie(Die, "Couldn't build subprogram prototype");
          continue;
        }

        // Get/create dynamic function
        auto &DynamicFunction = Model->ImportedDynamicFunctions()[SymbolName];

        // If a function already has a valid prototype, don't override it
        if (DynamicFunction.prototype() != nullptr) {
          revng_log(DILogger, "This function already has a prototype");
          continue;
        }

        const auto &Definition = *Prototype->tryGetAsDefinition();
        revng_log(DILogger, "Assigning prototype " << Definition.ID());
        revng_assert(isa<model::CABIFunctionDefinition>(Definition));
        DynamicFunction.Prototype() = std::move(Prototype);

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

void DwarfToModelConverter::purgeUnresolvedPlaceholders() {
  revng_log(DILogger, "purgeUnresolvedPlaceholders");
  LoggerIndent Indent(DILogger);

  std::set<const model::TypeDefinition *> ToDrop;
  for (auto &&[_, Type] : Placeholders)
    ToDrop.insert(Type);

  unsigned DroppedTypes = dropTypesDependingOnDefinitions(Model, ToDrop);
  if (DroppedTypes > 0) {
    // TODO: emit a diagnostic message for the user.
    revng_log(DILogger,
              "Purging " << DroppedTypes << " types (out of "
                         << TypesWithIdentityCount << ") due to "
                         << Placeholders.size() << " unresolved types");
  }

  Placeholders.clear();
}

void DwarfToModelConverter::run() {
  Task T(10, "Importing DWARF");

  T.advance("Materialize types with an identity", true);
  materializeTypesWithIdentity();

  T.advance("Resolve types", true);
  resolveAllTypes();

  T.advance("Create model functions", true);
  createFunctions();

  T.advance("Remove types that depend on unresolved placeholders", true);
  purgeUnresolvedPlaceholders();

  revng_log(DILogger, "Cleaning up and verifying the model");
  LoggerIndent Indent(DILogger);

  T.advance("Remove types that couldn't be imported fully", true);
  purgeInvalidTypes(Model);

  T.advance("Flatten primitive typedefs", true);
  model::flattenPrimitiveTypedefs(Model);

  T.advance("Deduplicate equivalent types", true);
  deduplicateEquivalentTypes(Model);

  T.advance("Deduplicate colliding names", true);
  model::deduplicateCollidingNames(Model);

  T.advance("Purge unnamed unreachable types", true);
  purgeUnnamedAndUnreachableTypes(Model);

  T.advance("Verify the model", true);
  Model->verify(true);
}
