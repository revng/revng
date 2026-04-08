//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/ScopedExchange.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/PointerType.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/TypedefDefinition.h"
#include "revng/Model/UnionDefinition.h"
#include "revng/Support/OverflowSafeInt.h"

#include "PDBTypeResolver.h"

using namespace llvm;
using namespace llvm::codeview;

// Determine the pointer size based on CodeView/PDB data.
static uint32_t getPointerSize(codeview::PointerKind K) {
  switch (K) {
  case codeview::PointerKind::Near64:
    return 8;
  case codeview::PointerKind::Near32:
    return 4;
  default:
    // TODO: Handle all pointer kinds.
    revng_abort();
  }
}

// TODO: This can go into LLVM, but there is an ongoing review that should
// implement this.
static std::optional<uint64_t> getSizeInBytes(TypeIndex TI) {
  if (not TI.isSimple())
    return std::nullopt;
  switch (TI.getSimpleKind()) {
  case SimpleTypeKind::Void:
    return 0;
  case SimpleTypeKind::HResult:
    return 4;
  case SimpleTypeKind::SByte:
  case SimpleTypeKind::Byte:
    return 1;
  case SimpleTypeKind::Int16Short:
  case SimpleTypeKind::UInt16Short:
  case SimpleTypeKind::Int16:
  case SimpleTypeKind::UInt16:
    return 2;
  case SimpleTypeKind::Int32Long:
  case SimpleTypeKind::UInt32Long:
  case SimpleTypeKind::Int32:
  case SimpleTypeKind::UInt32:
    return 4;
  case SimpleTypeKind::Int64Quad:
  case SimpleTypeKind::UInt64Quad:
  case SimpleTypeKind::Int64:
  case SimpleTypeKind::UInt64:
    return 8;
  case SimpleTypeKind::Int128Oct:
  case SimpleTypeKind::UInt128Oct:
  case SimpleTypeKind::Int128:
  case SimpleTypeKind::UInt128:
    return 16;
  case SimpleTypeKind::SignedCharacter:
  case SimpleTypeKind::UnsignedCharacter:
  case SimpleTypeKind::NarrowCharacter:
    return 1;
  case SimpleTypeKind::WideCharacter:
  case SimpleTypeKind::Character16:
    return 2;
  case SimpleTypeKind::Character32:
    return 4;
  case SimpleTypeKind::Float16:
    return 2;
  case SimpleTypeKind::Float32:
    return 4;
  case SimpleTypeKind::Float64:
    return 8;
  case SimpleTypeKind::Float80:
    return 10;
  case SimpleTypeKind::Float128:
    return 16;
  case SimpleTypeKind::Boolean8:
    return 1;
  case SimpleTypeKind::Boolean16:
    return 2;
  case SimpleTypeKind::Boolean32:
    return 4;
  case SimpleTypeKind::Boolean64:
    return 8;
  case SimpleTypeKind::Boolean128:
    return 16;
  default:
    return std::nullopt;
  }
}

static model::PrimitiveKind::Values
codeviewSimpleTypeEncodingToModel(TypeIndex TI) {
  if (not TI.isSimple())
    return model::PrimitiveKind::Invalid;

  switch (TI.getSimpleKind()) {
  case SimpleTypeKind::Void:
    return model::PrimitiveKind::Void;
  case SimpleTypeKind::Boolean8:
  case SimpleTypeKind::Boolean16:
  case SimpleTypeKind::Boolean32:
  case SimpleTypeKind::Boolean64:
  case SimpleTypeKind::Boolean128:
  case SimpleTypeKind::Byte:
  case SimpleTypeKind::UInt16:
  case SimpleTypeKind::UInt32:
  case SimpleTypeKind::UInt64:
  case SimpleTypeKind::UnsignedCharacter:
  case SimpleTypeKind::UInt16Short:
  case SimpleTypeKind::UInt32Long:
  case SimpleTypeKind::UInt64Quad:
  case SimpleTypeKind::UInt128Oct:
  case SimpleTypeKind::UInt128:
    return model::PrimitiveKind::Unsigned;
  case SimpleTypeKind::SignedCharacter:
  case SimpleTypeKind::WideCharacter:
  case SimpleTypeKind::NarrowCharacter:
  case SimpleTypeKind::Character16:
  case SimpleTypeKind::Character32:
  case SimpleTypeKind::Int16:
  case SimpleTypeKind::Int16Short:
  case SimpleTypeKind::SByte:
  case SimpleTypeKind::Int32Long:
  case SimpleTypeKind::Int32:
  case SimpleTypeKind::Int64Quad:
  case SimpleTypeKind::Int64:
  case SimpleTypeKind::Int128Oct:
  case SimpleTypeKind::Int128:
  case SimpleTypeKind::HResult:
    return model::PrimitiveKind::Signed;
  case SimpleTypeKind::Float16:
  case SimpleTypeKind::Float32:
  case SimpleTypeKind::Float64:
  case SimpleTypeKind::Float80:
  case SimpleTypeKind::Float128:
    return model::PrimitiveKind::Float;
  default:
    return model::PrimitiveKind::Invalid;
  }
}

static bool isPointer(TypeIndex TI) {
  if (TI.getSimpleMode() != SimpleTypeMode::Direct) {
    // We have a native pointer.
    switch (TI.getSimpleMode()) {
    case SimpleTypeMode::NearPointer32:
    case SimpleTypeMode::FarPointer32:
    case SimpleTypeMode::NearPointer64:
      return true;
    default:
      return false;
    }
  }

  return false;
}

static bool isTwoBytesLongPointer(TypeIndex TI) {
  if (TI.getSimpleMode() != SimpleTypeMode::Direct) {
    // We have a native pointer.
    switch (TI.getSimpleMode()) {
    case SimpleTypeMode::NearPointer:
    case SimpleTypeMode::FarPointer:
    case SimpleTypeMode::HugePointer:
      return true;
    default:
      return false;
    }
  }
  return false;
}

static bool isSixteenBytesLongPointer(TypeIndex TI) {
  if (TI.getSimpleMode() != SimpleTypeMode::Direct) {
    // We have a native pointer.
    switch (TI.getSimpleMode()) {
    case SimpleTypeMode::NearPointer128:
      return true;
    default:
      return false;
    }
  }
  return false;
}

static std::optional<uint64_t> getPointerSizeFromPDB(TypeIndex TI) {
  if (TI.getSimpleMode() != SimpleTypeMode::Direct) {
    // We have a native pointer.
    switch (TI.getSimpleMode()) {
    case SimpleTypeMode::NearPointer:
    case SimpleTypeMode::FarPointer:
    case SimpleTypeMode::HugePointer:
      return 2;
    case SimpleTypeMode::NearPointer32:
    case SimpleTypeMode::FarPointer32:
      return 4;
    case SimpleTypeMode::NearPointer64:
      return 8;
    case SimpleTypeMode::NearPointer128:
      return 16;
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

const model::UpcastableType *TypeResolver::handleSimpleType(TypeIndex Index) {
  using namespace model;

  revng_log(Log, "handleSimpleType");
  LoggerIndent Indent(Log);

  if (isTwoBytesLongPointer(Index)) {
    // If it is a pointer of size 2, lets create a PointerOrNumber for it.
    constexpr uint64_t MSDOS16Pointer = 2;
    return record(Index,
                  PrimitiveType::makePointerOrNumber(MSDOS16Pointer),
                  true);

  } else if (isSixteenBytesLongPointer(Index)) {
    // If it is a 128-bit long pointer, fail for now. It can be
    // represented as a `struct { pointee; offset; }` since it is how it is
    // implemented in the msvc compiler.
    return fail(Index);
  } else {
    auto PrimitiveKind = codeviewSimpleTypeEncodingToModel(Index);
    auto PrimitiveSize = getSizeInBytes(Index);
    if (not PrimitiveSize.has_value()
        or PrimitiveKind == PrimitiveKind::Invalid) {
      revng_log(Log,
                "Warning: invalid simple type "
                  << toString(Index) << " with simple kind "
                  << static_cast<uint32_t>(Index.getSimpleKind()));
      return fail(Index);
    }

    auto Primitive = PrimitiveType::make(PrimitiveKind, *PrimitiveSize);

    if (isPointer(Index)) {
      auto PointerSize = getPointerSizeFromPDB(Index);
      if (not PointerSize.has_value()) {
        revng_log(Log, "Warning: invalid pointer size " << toString(Index));
        return fail(Index);
      }

      auto Pointer = model::PointerType::make(std::move(Primitive),
                                              *PointerSize);
      return record(Index, std::move(Pointer), true);
    } else {
      // If it is not a pointer `SimpleKind` will be the same as `SimpleType`.
      revng_assert(TypeIndex(Index.getSimpleKind()) == Index);
      return record(Index, std::move(Primitive), true);
    }
  }
}

RecursiveCoroutine<const model::UpcastableType *>
TypeResolver::handle(TypeIndex Index, PointerRecord &Pointer) {
  revng_log(Log, "Handling PointerRecord");
  LoggerIndent Indent(Log);

  auto PointeeIndex = Pointer.getReferentType();

  // Note: we set NeedsSize to false because we don't need the size here,
  // therefore we're OK with non-processed TypeDefinition. This enables
  // recursion via pointers.
  const model::UpcastableType *Pointee = nullptr;
  {
    ScopedExchange<bool> Guard(NeedsSize, false);
    Pointee = rc_recur getTypeForImpl(PointeeIndex);
  }
  revng_assert(Pointee != nullptr);

  if (Pointee->isEmpty()) {
    revng_log(Log, "Pointee type " << toString(PointeeIndex) << " not found");
    rc_return fail(Index);
  }

  using PointerType = model::PointerType;
  auto PointerSize = getPointerSize(Pointer.getPointerKind());
  rc_return record(Index,
                   PointerType::make(Pointee->copy(), PointerSize),
                   true);
}

RecursiveCoroutine<const model::UpcastableType *>
TypeResolver::handle(TypeIndex Index, BitFieldRecord &BitField) {
  revng_log(Log, "Handling BitFieldRecord");
  LoggerIndent Indent(Log);

  // As of now we treat bitfields as their underlying type

  auto UnderlyingIndex = BitField.getType();
  const auto &UnderlyingType = *rc_recur getTypeForImpl(UnderlyingIndex);

  if (UnderlyingType.isEmpty()) {
    revng_log(Log,
              "Warning: underlying type " << toString(UnderlyingIndex)
                                          << " not found");
    rc_return fail(Index);
  }

  // A bitfield's size follows its underlying: only size-available if the
  // underlying was fetched with NeedsSize=true, i.e., the current NeedsSize.
  rc_return record(Index, UnderlyingType.copy(), NeedsSize);
}

RecursiveCoroutine<const model::UpcastableType *>
TypeResolver::handle(TypeIndex Index, ModifierRecord &Modifier) {
  revng_log(Log, "Handling ModifierRecord");
  LoggerIndent Indent(Log);

  const auto &Modified = *rc_recur getTypeForImpl(Modifier.getModifiedType());

  if (Modified.isEmpty()) {
    revng_log(Log, "Warning: modified type not found");
    rc_return fail(Index);
  }

  auto Result = Modified.copy();

  using Modifiers = ModifierOptions;
  if ((Modifier.getModifiers() & Modifiers::Const) != Modifiers::None) {
    Result->IsConst() = true;
  }

  // A modifier wraps the modified type: size is available only when the
  // modified was fetched with NeedsSize=true, i.e., the current NeedsSize.
  rc_return record(Index, std::move(Result), NeedsSize);
}

RecursiveCoroutine<const model::UpcastableType *>
TypeResolver::handle(TypeIndex Index, ArrayRecord &Array) {
  revng_log(Log, "Handling ArrayRecord");
  LoggerIndent Indent(Log);

  // We need the element's size to compute the element count, so force a
  // size-computing resolution of the element regardless of our caller's
  // NeedsSize.
  const model::UpcastableType *ElementType = nullptr;
  {
    ScopedExchange<bool> Guard(NeedsSize, true);
    ElementType = rc_recur getTypeForImpl(Array.getElementType());
  }
  revng_assert(ElementType != nullptr);

  if (ElementType->isEmpty()) {
    revng_log(Log, "Warning: element type not found");
    rc_return fail(Index);
  }

  if (Array.getSize() == 0) {
    revng_log(Log, "Skipping 0-sized array");
    rc_return fail(Index);
  }

  auto MaybeSize = (*ElementType)->size();
  if (not MaybeSize.has_value()) {
    revng_log(Log, "Array of 0-sized elements, skipping");
    rc_return fail(Index);
  }

  const uint64_t ArraySize = Array.getSize() / *MaybeSize;
  rc_return record(Index,
                   model::ArrayType::make(ElementType->copy(), ArraySize),
                   true);
}

RecursiveCoroutine<bool>
TypeResolver::processDefinition(ClassRecord &ClassRecord,
                                model::StructDefinition &Struct) {
  revng_log(Log, "Handling ClassRecord");
  LoggerIndent Indent(Log);

  // TODO: we could process the LF_ONEMETHOD entries and build the virtual table

  // Handle size
  Struct.Name() = ClassRecord.getName();
  Struct.Size() = ClassRecord.getSize();
  if (Struct.Size() == 0) {
    revng_log(Log, "Warning: ignoring 0-sized struct");
    rc_return false;
  }

  // TODO: handle LF_BCLASS by simply adding a field at offset 0

  // Process fields
  auto FieldListIndex = ClassRecord.getFieldList();
  auto MaybeFields = Importer.getDataMemberRecords(FieldListIndex);
  if (not MaybeFields.has_value()) {
    revng_log(Log,
              "The struct has no fields (" << toString(FieldListIndex)
                                           << "), keeping empty struct");
    rc_return true;
  }

  revng_log(Log, "Processing fields");
  LoggerIndent Indent2(Log);

  for (const auto &[Index, Field] : llvm::enumerate(*MaybeFields)) {
    revng_log(Log,
              "Processing field #" << Index << " with type "
                                   << toString(Field.getType()));
    LoggerIndent Indent3(Log);

    // Create new field
    uint64_t Offset = Field.getFieldOffset();
    auto &FieldType = notNull(rc_recur getTypeForImpl(Field.getType()));
    if (FieldType.isEmpty()) {
      revng_log(Log,
                "Field type not found: " << toString(Field.getType())
                                         << ", skipping");
      continue;
    }

    auto MaybeSize = FieldType->size();
    uint64_t Size = MaybeSize.value_or(0);
    if (Size == 0) {
      // Skip 0-sized field.
      revng_log(Log, "Skipping 0-sized field");
      continue;
    }

    OverflowSafeInt<uint64_t> CurrentFieldOffset = Offset;
    CurrentFieldOffset += Size;
    if (not CurrentFieldOffset) {
      revng_log(Log, "Warning: skipping struct field due to overflow.");
      continue;
    }

    if (*CurrentFieldOffset > Struct.Size()) {
      revng_log(Log,
                "Warning: skipping struct field that is outside the struct.");
      continue;
    }

    auto &NewField = Struct.Fields()[Offset];
    NewField.Name() = Field.getName().str();
    NewField.Type() = FieldType.copy();
  }

  rc_return true;
}

RecursiveCoroutine<bool>
TypeResolver::processDefinition(UnionRecord &UnionRecord,
                                model::UnionDefinition &Union) {
  revng_log(Log, "Handling UnionRecord");
  LoggerIndent Indent(Log);

  Union.Name() = UnionRecord.getName();

  // Process fields
  auto MaybeFields = Importer.getDataMemberRecords(UnionRecord.getFieldList());
  if (not MaybeFields.has_value()) {
    revng_log(Log, "Warning: couldn't get fields");
    rc_return false;
  }

  if (MaybeFields->size() == 0) {
    revng_log(Log, "Warning: union has no fields");
    rc_return false;
  }

  for (const auto &[Index, Field] : llvm::enumerate(*MaybeFields)) {
    revng_log(Log, "Processing field " << Index);
    LoggerIndent Indent3(Log);

    // Create new field
    uint64_t Offset = Field.getFieldOffset();
    auto FieldType = rc_recur getTypeForImpl(Field.getType());
    if (FieldType->isEmpty()) {
      revng_log(Log, "Field type not found: " << toString(Field.getType()));
      continue;
    }

    auto MaybeSize = (*FieldType)->size();
    uint64_t Size = MaybeSize.value_or(0);
    if (Size == 0) {
      revng_log(Log, "Warning: skipping 0-sized field");
      continue;
    }

    auto &NewField = Union.addField(FieldType->copy());
    NewField.Name() = Field.getName().str();
  }

  if (Union.size() > 0) {
    rc_return true;
  } else {
    rc_return false;
  }
}

RecursiveCoroutine<bool>
TypeResolver::processDefinition(EnumRecord &EnumRecord,
                                model::EnumDefinition &Enum) {
  revng_log(Log, "Handling EnumRecord");
  LoggerIndent Indent(Log);

  Enum.Name() = EnumRecord.getName();

  const auto &UnderlyingType = rc_recur getTypeForImpl(EnumRecord
                                                         .getUnderlyingType());
  if (UnderlyingType->isEmpty()) {
    revng_log(Log,
              "Underlying type not found: "
                << toString(EnumRecord.getUnderlyingType()));
    Enum.UnderlyingType() = model::PrimitiveType::makeVoid();
    rc_return false;
  }

  Enum.UnderlyingType() = UnderlyingType->copy();

  auto MaybeFields = Importer.getEnumeratorRecords(EnumRecord.getFieldList());
  if (not MaybeFields.has_value()) {
    // TODO: not nice, we're losing the name
    revng_log(Log, "No entries, ignoring");
    rc_return false;
  }

  for (const auto &[Index, Entry] : llvm::enumerate(*MaybeFields)) {
    revng_log(Log, "Processing entry " << Index);
    LoggerIndent Indent3(Log);

    auto &EnumEntry = Enum.Entries()[Entry.getValue().getExtValue()];
    EnumEntry.Name() = Entry.getName().str();
  }

  rc_return true;
}

RecursiveCoroutine<bool>
TypeResolver::processDefinition(ProcedureRecord &ProcedureRecord,
                                model::CABIFunctionDefinition &Prototype) {
  revng_log(Log, "Handling ProcedureRecord");
  LoggerIndent Indent(Log);

  rc_return rc_recur
    processFunctionDefinition(ProcedureRecord.getCallConv(),
                              ProcedureRecord.getReturnType(),
                              ProcedureRecord.getArgumentList(),
                              Prototype);
}

RecursiveCoroutine<bool>
TypeResolver::processDefinition(MemberFunctionRecord &MemberFunctionRecord,
                                model::CABIFunctionDefinition &Prototype) {
  revng_log(Log, "Handling MemberFunctionRecord");
  LoggerIndent Indent(Log);

  // Handle this pointer
  if (MemberFunctionRecord.getThisPointerAdjustment() != 0) {
    revng_log(Log,
              "Warning: multiple virtual inheritance is not supported right "
              "now");
    rc_return false;
  }

  auto ThisTypeIndex = MemberFunctionRecord.getThisType();
  if (not ThisTypeIndex.isNoneType()) {
    const auto &ThisType = *rc_recur getTypeForImpl(ThisTypeIndex);
    if (ThisType.isEmpty()) {
      revng_log(Log, "Warning: couldn't resolve this type");
      rc_return false;
    }

    revng_assert(ThisType->isPointer());
    Prototype.addArgument(ThisType.copy());
  }

  // Delegate all the rest to handleFunction
  rc_return rc_recur
    processFunctionDefinition(MemberFunctionRecord.getCallConv(),
                              MemberFunctionRecord.getReturnType(),
                              MemberFunctionRecord.getArgumentList(),
                              Prototype);
}

static model::ABI::Values getABI(llvm::codeview::CallingConvention CallConv,
                                 model::Architecture::Values Architecture) {
  using namespace llvm::codeview;
  if (Architecture == model::Architecture::x86_64) {
    switch (CallConv) {
    case CallingConvention::NearC:
    case CallingConvention::NearFast:
    case CallingConvention::NearStdCall:
    case CallingConvention::NearSysCall:
    case CallingConvention::ThisCall:
      return model::ABI::Microsoft_x86_64;
    case CallingConvention::NearPascal:
      revng_log(Log, "Pascal is not currently supported");
      return model::ABI::Invalid;
    case CallingConvention::NearVector:
      return model::ABI::Microsoft_x86_64_vectorcall;
    case CallingConvention::ClrCall:
      revng_log(Log, "ClrCall is not currently supported");
      return model::ABI::Invalid;
    default:
      revng_abort();
    }
  } else if (Architecture == model::Architecture::x86) {
    switch (CallConv) {
    case CallingConvention::NearC:
      return model::ABI::Microsoft_x86_cdecl;
    case CallingConvention::NearFast:
      return model::ABI::Microsoft_x86_fastcall;
    case CallingConvention::NearStdCall:
      return model::ABI::Microsoft_x86_stdcall;
    case CallingConvention::NearSysCall:
      return model::ABI::Microsoft_x86_stdcall;
    case CallingConvention::ThisCall:
      return model::ABI::Microsoft_x86_thiscall;
    case CallingConvention::ClrCall:
      revng_log(Log, "ClrCall is not currently supported");
      return model::ABI::Invalid;
    case CallingConvention::NearPascal:
      revng_log(Log, "Pascal is not currently supported");
      return model::ABI::Invalid;
    case CallingConvention::NearVector:
      return model::ABI::Microsoft_x86_vectorcall;
    default:
      revng_log(Log, "Unknown ABI");
      return model::ABI::Invalid;
    }
  } else if (Architecture == model::Architecture::mips
             and CallConv == CallingConvention::MipsCall) {
    return model::ABI::SystemV_MIPS_o32;
  } else if (Architecture == model::Architecture::mipsel
             and CallConv == CallingConvention::MipsCall) {
    return model::ABI::SystemV_MIPSEL_o32;
  } else if (Architecture == model::Architecture::arm
             and CallConv == CallingConvention::ArmCall) {
    return model::ABI::AAPCS;
  } else if (Architecture == model::Architecture::aarch64
             /* and CallConv == CallingConvention::ArmCall
                (I'm seeing CallingConvention::NearC)
             */) {
    return model::ABI::Microsoft_AAPCS64;
  } else {
    return model::ABI::Invalid;
  }
}

RecursiveCoroutine<bool>
TypeResolver::processFunctionDefinition(CallingConvention CallingConvention,
                                        TypeIndex ReturnTypeIndex,
                                        TypeIndex ArgumentListIndex,
                                        model::CABIFunctionDefinition
                                          &Prototype) {
  // Handle ABI
  Prototype.ABI() = getABI(CallingConvention, Architecture);

  // Handle return type
  const auto &ReturnType = *rc_recur getTypeForImpl(ReturnTypeIndex);
  if (ReturnType.isEmpty()) {
    revng_log(Log, "Return type not found: " << toString(ReturnTypeIndex));
    rc_return false;
  }

  if (not ReturnType->isVoidPrimitive())
    Prototype.ReturnType() = ReturnType.copy();

  // Handle arguments
  auto *MaybeArgumentList = Importer
                              .getTypeRecord<ArgListRecord>(ArgumentListIndex);
  if (MaybeArgumentList == nullptr) {
    revng_log(Log,
              "Warning: argument list has an unexpected type: "
                << toString(ArgumentListIndex));
    rc_return false;
  }

  auto ArgumentTypeIndices = MaybeArgumentList->getIndices();
  bool IsVariadic = (ArgumentTypeIndices.size() != 0
                     and ArgumentTypeIndices.back().isNoneType());
  if (IsVariadic) {
    revng_log(Log, "Ignoring variadic function");
    rc_return false;
  }

  for (auto &[ArgumentIndex, ArgumentTypeIndex] :
       llvm::enumerate(ArgumentTypeIndices)) {
    revng_log(Log, "Processing argument #" << ArgumentIndex);
    LoggerIndent Indent(Log);

    const auto &ArgumentType = *rc_recur getTypeForImpl(ArgumentTypeIndex);

    if (ArgumentType.isEmpty()) {
      revng_log(Log, "Warning: couldn't get the argument type");
      rc_return false;
    }

    auto MaybeSize = ArgumentType->size();
    uint64_t Size = MaybeSize.value_or(0);
    if (Size == 0) {
      revng_log(Log, "Warning: 0-sized argument, bailing out");
      rc_return false;
    }

    Prototype.addArgument(ArgumentType.copy());
  }

  rc_return true;
}

RecursiveCoroutine<bool>
TypeResolver::processDefinition(AliasRecord &AliasRecord,
                                model::TypedefDefinition &Typedef) {
  revng_log(Log, "Handling AliasRecord");
  LoggerIndent Indent(Log);

  Typedef.Name() = AliasRecord.Name;

  const auto &UnderlyingType = rc_recur getTypeForImpl(AliasRecord
                                                         .UnderlyingType);
  if (UnderlyingType->isEmpty()) {
    revng_log(Log,
              "Underlying type not found: "
                << toString(AliasRecord.UnderlyingType));
    Typedef.UnderlyingType() = model::PrimitiveType::makeVoid();
    rc_return false;
  }

  Typedef.UnderlyingType() = UnderlyingType->copy();

  rc_return true;
}

RecursiveCoroutine<const model::UpcastableType *>
TypeResolver::getTypeForImpl(TypeIndex Index) {
  using namespace model;

  revng_log(Log, "Resolving type " << toString(Index));
  LoggerIndent Indent(Log);

  // First of all, handle primitive types
  if (Index.isSimple()) {
    // Check cache
    if (auto *Result = Importer.tryGetType(Index)) {
      revng_log(Log, "Simple type " + toString(Index) + " found in cache");
      revng_assert(Result != nullptr);
      rc_return Result;
    }

    rc_return handleSimpleType(Index);
  }

  // Check if we the data for this index
  TypeRecord *TheType = Importer.getTypeRecord(Index);
  if (TheType == nullptr) {
    revng_log(Log, "Warning: couldn't find type " << toString(Index));
    rc_return record(Index, UpcastableType::empty(), true);
  }

  // Resolve non-trivial types

  switch (TheType->Kind) {

  case TypeRecordKind::BitField: {
    rc_return rc_recur handle<BitFieldRecord>(Index, TheType);
  }

  case TypeRecordKind::Pointer: {
    rc_return rc_recur handle<PointerRecord>(Index, TheType);
  }

  case TypeRecordKind::Modifier: {
    rc_return rc_recur handle<ModifierRecord>(Index, TheType);
  }

  case TypeRecordKind::Array: {
    rc_return rc_recur handle<ArrayRecord>(Index, TheType);
  }

  case TypeRecordKind::Class:
  case TypeRecordKind::Struct:
  case TypeRecordKind::Interface: {
    rc_return rc_recur handle<ClassRecord, StructDefinition>(Index, TheType);
  }

  case TypeRecordKind::Union: {
    rc_return rc_recur handle<UnionRecord, UnionDefinition>(Index, TheType);
  }

  case TypeRecordKind::Enum: {
    rc_return rc_recur handle<EnumRecord, EnumDefinition>(Index, TheType);
  }

  case TypeRecordKind::Procedure: {
    rc_return rc_recur handle<ProcedureRecord, CABIFunctionDefinition>(Index,
                                                                       TheType);
  }

  case TypeRecordKind::MemberFunction: {
    rc_return rc_recur
      handle<MemberFunctionRecord, CABIFunctionDefinition>(Index, TheType);
  }

  case TypeRecordKind::Alias: {
    rc_return rc_recur handle<AliasRecord, TypedefDefinition>(Index, TheType);
  }

  case TypeRecordKind::ArgList:
  case TypeRecordKind::FieldList:
    revng_log(Log, "Warning: requesting a type for non-type record, ignoring");
    rc_return fail(Index);

  default:
    revng_log(Log,
              "Warning: ignoring unknown TypeRecord kind: "
                << static_cast<int>(TheType->Kind));
    rc_return fail(Index);
  }

  revng_abort();
}
