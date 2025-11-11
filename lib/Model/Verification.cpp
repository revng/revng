/// \file Verification.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"

#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/Error.h"

#include "NamespaceBuilder.h"

using namespace llvm;

namespace model {

//
// Segments
//

bool Relocation::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Address().isInvalid())
    return VH.fail("Every relocation must have a valid address.", *this);

  if (not model::RelocationType::isValid(Type()))
    return VH.fail("Every relocation must have a valid type.", *this);

  return true;
}

bool Segment::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  using OverflowSafeInt = OverflowSafeInt<uint64_t>;

  if (StartAddress().isInvalid())
    return VH.fail("Every segment must have a valid start address.", *this);

  if (not StartAddress().isGeneric())
    return VH.fail("The segment start address must be generic", *this);

  if (VirtualSize() == 0)
    return VH.fail("The virtual size of a segment must not be 0.", *this);

  if (FileSize() > VirtualSize())
    return VH.fail("FileSize cannot be larger than VirtualSize", *this);

  auto EndOffset = OverflowSafeInt(StartOffset()) + FileSize();
  if (not EndOffset)
    return VH.fail("Computing the segment end offset leads to overflow", *this);

  auto EndAddress = StartAddress() + VirtualSize();
  if (not EndAddress.isValid())
    return VH.fail("Computing the end address leads to overflow", *this);

  for (const auto &CanonicalValue : CanonicalRegisterValues()) {
    if (not model::Register::isValid(CanonicalValue.Register()))
      return VH.fail("Canonical values can only be assigned to valid "
                     "registers.",
                     *this);

    // TODO: check that the register architecture makes sense.
  }

  for (const model::Relocation &Relocation : Relocations())
    if (not Relocation.verify(VH))
      return VH.fail("Invalid relocation", Relocation);

  if (not Type().isEmpty()) {

    if (not Type()->isStruct())
      return VH.fail("Segment's `Type()` must be a struct.", *this);

    if (not Type()->verify(VH))
      return VH.fail("Segment's `Type()` does not verify.", *this);

    const model::StructDefinition &Struct = *type();
    if (VirtualSize() != Struct.Size()) {
      return VH.fail(Twine("Segment's virtual size is not equal to the size of "
                           "its type.\n`VirtualSize`: ")
                       + Twine(VirtualSize())
                       + Twine(" != `Segment.type()->Size()`: ")
                       + Twine(Struct.Size()),
                     *this);
    }

    if (Struct.CanContainCode() != IsExecutable()) {
      if (IsExecutable()) {
        return VH.fail("The StructType representing the type of a executable "
                       "segment has CanContainedCode disabled",
                       *this);
      } else {
        return VH.fail("The StructType representing the type of a "
                       "non-executable segment has CanContainedCode enabled",
                       *this);
      }
    }
  }

  return true;
}

bool BinaryIdentifier::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  auto IsLowerHex = [](const char &C) {
    return ('0' <= C and C <= '9') or ('a' <= C and C <= 'f');
  };

  if (Hash().size() != 64 or not llvm::all_of(Hash(), IsLowerHex))
    return VH.fail("Hash has invalid format. A series of 64 lowercase hex "
                   "digits representing SHA256 is expected.",
                   Hash());

  return true;
}

//
// Functions
//

bool CallSitePrototype::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not CallerBlockAddress().isValid())
    return VH.fail("Every call site must have a caller block address.", *this);

  if (Prototype().isEmpty())
    return VH.fail("Every call site must have a prototype.", *this);

  if (not Prototype()->isPrototype())
    return VH.fail("`Prototype()` must be a prototype.", *this);

  if (not Prototype()->verify(VH))
    return VH.fail();

  for (model::FunctionAttribute::Values Attribute : Attributes())
    if (not model::FunctionAttribute::isValid(Attribute))
      return VH.fail("Every call site attribute must be valid.", *this);

  return true;
}

bool verifyAddressSet(VerifyHelper &VH,
                      const TrackingSortedVector<MetaAddress> &MAs,
                      const auto &ToLog) {
  if (MAs.empty())
    return VH.fail("Empty locations are not allowed.", ToLog);

  std::set<MetaAddress> Deduplicator;
  for (const MetaAddress &Address : MAs) {
    if (Address.isInvalid())
      return VH.fail("Only valid addresses can be a part of a location.",
                     ToLog);

    if (not Deduplicator.insert(Address).second)
      return VH.fail("Duplicated addresses are not allowed as a part of "
                     "a location.",
                     ToLog);
  }

  return true;
}

bool StatementComment::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Body().empty())
    return VH.fail("Comment body must not be empty.", *this);

  return verifyAddressSet(VH, Location(), *this);
}

bool LocalIdentifier::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Name().size() == 0)
    return VH.fail("Every local identifier must have a name.", *this);

  // TODO: drop this once we escape / from locations
  if (StringRef(Name()).contains("/"))
    return VH.fail("\"/\" is not allowed", Name());

  return verifyAddressSet(VH, Location(), *this);
}

bool Function::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not Entry().isValid())
    return VH.fail("Every function must have a valid entry point.", *this);

  if (not Entry().isCode())
    return VH.fail("Function Entry is not a code address", *this);

  if (not Prototype().isEmpty()) {

    if (not Prototype()->isPrototype())
      return VH.fail("`Prototype()` must be a prototype.", *this);

    if (not Prototype()->verify(VH))
      return VH.fail("Function prototype does not verify.", *this);
  }

  if (not StackFrameType().isEmpty()) {

    if (not StackFrameType()->isStruct())
      return VH.fail("`StackFrameType()` must be a struct.", *this);

    if (not StackFrameType()->verify(VH))
      return VH.fail("Stack frame type does not verify.", *this);
  }

  for (auto &CallSitePrototype : CallSitePrototypes())
    if (not CallSitePrototype.verify(VH))
      return VH.fail();

  for (model::FunctionAttribute::Values Attribute : Attributes())
    if (not model::FunctionAttribute::isValid(Attribute))
      return VH.fail("Every function attribute must be valid.", *this);

  for (const auto &[Index, Comment] : llvm::enumerate(Comments())) {
    if (Index != Comment.Index())
      return VH.fail("A function comment has an invalid index", *this);

    if (not Comment.verify())
      return VH.fail();
  }

  {
    std::set<TrackingSortedVector<MetaAddress>> Deduplicator;
    for (const auto &Variable : LocalVariables()) {
      if (not Variable.verify())
        return VH.fail();

      if (!Deduplicator.insert(Variable.Location()).second)
        return VH.fail("Multiple variables with the same address set: '"
                       + addressesToString(Variable.Location()) + "'");
    }
  }

  {
    std::set<TrackingSortedVector<MetaAddress>> Deduplicator;
    for (const auto &GotoLabel : GotoLabels()) {
      if (not GotoLabel.verify())
        return VH.fail();

      if (!Deduplicator.insert(GotoLabel.Location()).second)
        return VH.fail("Multiple goto labels with the same address set: '"
                       + addressesToString(GotoLabel.Location()) + "'");
    }
  }

  return true;
}

bool DynamicFunction::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Name().size() == 0)
    return VH.fail("Every dynamic function must have a name.", *this);

  // TODO: drop this once we escape / from locations
  if (StringRef(Name()).contains("/"))
    return VH.fail("\"/\" is not allowed", Name());

  if (not Prototype().isEmpty()) {
    if (not Prototype()->isPrototype())
      return VH.fail("`Prototype()` type must be a prototype.", *this);

    if (not Prototype()->verify(VH))
      return VH.fail();
  }

  for (auto &Attribute : Attributes()) {
    if (not model::FunctionAttribute::isValid(Attribute))
      return VH.fail("Every dynamic function attribute must be valid.", *this);
    if (Attribute == model::FunctionAttribute::Inline)
      return VH.fail("Dynamic function cannot have Inline attribute", *this);
  }

  return true;
}

//
// Types
//

static constexpr bool isValidPrimitiveSize(PrimitiveKind::Values Kind,
                                           uint8_t Size) {
  constexpr std::array ValidGenericPrimitives{ 1, 2, 4, 8, 16 };
  constexpr std::array ValidFloatPrimitives{ 2, 4, 8, 10, 12, 16 };
  // NOTE: We are supporting floats that are 10 bytes long, since we found such
  //       cases in some PDB files by using VS on Windows platforms. The source
  //       code of those cases could be written in some language other than
  //       C/C++ (probably Swift). We faced some struct fields by using this
  //       (10b long float) type, so by ignoring it we would not have accurate
  //       layout for the structs.

  switch (Kind) {
  case PrimitiveKind::Invalid:
    return false;

  case PrimitiveKind::Void:
    return Size == 0;

  case PrimitiveKind::PointerOrNumber:
  case PrimitiveKind::Number:
  case PrimitiveKind::Unsigned:
  case PrimitiveKind::Signed:
    return std::ranges::binary_search(ValidGenericPrimitives, Size);

  case PrimitiveKind::Float:
    return std::ranges::binary_search(ValidFloatPrimitives, Size);

  case PrimitiveKind::Generic:
    return std::ranges::binary_search(ValidGenericPrimitives, Size)
           || std::ranges::binary_search(ValidFloatPrimitives, Size);

  default:
    revng_abort("Unsupported primitive kind");
  }
}

RecursiveCoroutine<bool> model::Type::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not model::TypeKind::isValid(Kind()))
    rc_return VH.fail("Every type must have a valid kind.");

  bool PointerBeforeDefinition = false;
  const model::Type *Active = this;
  while (Active != nullptr) {
    if (auto *Array = llvm::dyn_cast<model::ArrayType>(Active)) {
      if (Array->ElementCount() == 0)
        rc_return VH.fail("0 element arrays are not supported", *Array);

      if (Array->ElementType().isEmpty()) {
        rc_return VH.fail("Arrays without an element type are not supported",
                          *Array);
      }

      if (!Array->ElementType()->size(VH))
        rc_return VH.fail("Array element type must have a size.", *Array);

      // Because we cannot emit const array in C anyway, we might as well forbid
      // them as early as possible.
      if (Array->IsConst())
        rc_return VH.fail("Arrays must not be const.", *Array);

      Active = Array->ElementType().get();

    } else if (auto *Defined = llvm::dyn_cast<model::DefinedType>(Active)) {
      if (not Defined->Definition().isValid()) {
        rc_return VH.fail("Defined types must contain a valid (non-empty) "
                          "reference",
                          *Defined);
      }

      // Do not recur if this type is a pointer, otherwise we get undesired
      // failures if a type (for example a struct) has a pointer to itself.
      if (PointerBeforeDefinition)
        rc_return true;
      else
        rc_return rc_recur Defined->Definition().get()->verify(VH);

    } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Active)) {
      if (!llvm::isPowerOf2_64(Pointer->PointerSize()))
        rc_return VH.fail("Pointer size is not a power of 2", *Pointer);

      if (Pointer->PointerSize() != 4 && Pointer->PointerSize() != 8) {
        rc_return VH.fail("Only 32-bit and 64-bit pointers are currently "
                          "supported",
                          *Pointer);
      }

      if (Pointer->PointeeType().isEmpty()) {
        rc_return VH.fail("Pointers without an pointee type are not supported. "
                          "Use a `PrimitiveType::makeVoid`, if you want to "
                          "represent `void *`.",
                          *Pointer);
      }

      PointerBeforeDefinition = true;
      Active = Pointer->PointeeType().get();

    } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(Active)) {
      if (not model::PrimitiveKind::isValid(Primitive->PrimitiveKind()))
        rc_return VH.fail("Every primitive must have a valid kind.",
                          *Primitive);

      if (not isValidPrimitiveSize(Primitive->PrimitiveKind(),
                                   Primitive->Size()))
        rc_return VH.fail("Primitive size is not allowed.", *Primitive);

      rc_return true;

    } else {
      rc_return VH.fail("Unsupported type kind.");
    }
  }

  rc_return VH.fail("A required sub-type is missing.");
}

//
// Type definitions
//

bool EnumEntry::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);
  return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const EnumDefinition &T) {
  revng_assert(T.Kind() == model::TypeDefinitionKind::EnumDefinition);

  if (T.Entries().empty())
    rc_return VH.fail("Every enum definition must have at least one entry.");

  if (T.UnderlyingType().isEmpty())
    rc_return VH.fail("Every enum must have an underlying type.", T);

  if (not rc_recur T.UnderlyingType()->verify(VH))
    rc_return VH.fail();

  if (not T.UnderlyingType()->isPrimitive(PrimitiveKind::Signed)
      && not T.UnderlyingType()->isPrimitive(PrimitiveKind::Unsigned)) {
    rc_return VH.fail("UnderlyingType of an enum can only be a Signed or "
                      "Unsigned primitive",
                      T);
  }

  for (auto &Entry : T.Entries()) {

    if (not Entry.verify(VH))
      rc_return VH.fail();

    // TODO: verify Entry.Value is within boundaries
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const TypedefDefinition &T) {
  revng_assert(T.Kind() == TypeDefinitionKind::TypedefDefinition);

  if (T.UnderlyingType().isEmpty())
    rc_return VH.fail("Every typedef definition requires an underlying type.");

  rc_return VH.maybeFail(rc_recur T.UnderlyingType()->verify(VH));
}

RecursiveCoroutine<bool> StructField::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Type().isEmpty())
    rc_return VH.fail("Struct field must have a type.", *this);

  if (not rc_recur Type()->verify(VH))
    rc_return VH.fail();

  // Struct fields cannot be zero-sized
  auto MaybeSize = rc_recur Type()->size(VH);
  if (not MaybeSize)
    rc_return VH.fail("Struct field is zero-sized", Type());

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const StructDefinition &T) {
  revng_assert(T.Kind() == model::TypeDefinitionKind::StructDefinition);

  using namespace llvm;

  if (T.Size() == 0)
    rc_return VH.fail("Struct size must be greater than zero.", T);

  auto FieldIt = T.Fields().begin();
  auto FieldEnd = T.Fields().end();
  for (; FieldIt != FieldEnd; ++FieldIt) {
    auto &Field = *FieldIt;

    if (not rc_recur Field.verify(VH)) {
      rc_return VH.fail("Can't verify type of field at offset "
                          + Twine(Field.Offset()),
                        T);
    }

    uint64_t Size = *rc_recur Field.Type()->size(VH);
    if (Field.Offset() >= T.Size()) {
      rc_return VH.fail("Field at offset " + Twine(Field.Offset())
                          + " is out of struct boundaries (field size: "
                          + Twine(Size) + ", field offset + size: "
                          + Twine(Field.Offset() + Size)
                          + ", struct size: " + Twine(T.Size()) + ")",
                        T);
    }

    auto NextFieldIt = std::next(FieldIt);
    if (NextFieldIt != FieldEnd) {
      // If this field is not the last, check that it does not overlap with the
      // following field.
      if (Field.Offset() + Size > NextFieldIt->Offset()) {
        rc_return VH.fail("Field at offset " + Twine(Field.Offset())
                            + " (with size: " + Twine(Size)
                            + ") overlaps with the field at offset "
                            + Twine(NextFieldIt->Offset()) + " (with size: "
                            + Twine(*rc_recur NextFieldIt->Type()->size(VH))
                            + ")",
                          T);
      }
    } else if (Field.Offset() + Size > T.Size()) {
      // Otherwise, if this field is the last, check that it's not larger than
      // size.
      rc_return VH.fail("Last field ends outside the struct", T);
    }
  }

  rc_return true;
}

RecursiveCoroutine<bool> UnionField::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Type().isEmpty())
    rc_return VH.fail("Union field must have a type.", *this);

  if (not rc_recur Type()->verify(VH))
    rc_return VH.fail();

  // Union fields cannot be zero-sized
  auto MaybeSize = rc_recur Type()->size(VH);
  if (not MaybeSize)
    rc_return VH.fail("Union field is zero-sized", Type());

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const UnionDefinition &T) {
  revng_assert(T.Kind() == model::TypeDefinitionKind::UnionDefinition);

  if (T.Fields().empty())
    rc_return VH.fail("Union must have at least one field.", T);

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Group : llvm::enumerate(T.Fields())) {
    auto &Field = Group.value();
    uint64_t ExpectedIndex = Group.index();

    if (Field.Index() != ExpectedIndex) {
      rc_return VH.fail(Twine("Union type is missing field ")
                          + Twine(ExpectedIndex),
                        T);
    }

    if (not rc_recur Field.verify(VH))
      rc_return VH.fail();
  }

  rc_return true;
}

RecursiveCoroutine<bool> Argument::verify(VerifyHelper &VH) const {
  if (Type().isEmpty())
    rc_return VH.fail("A function argument must have a type", *this);

  if (not rc_recur Type()->verify(VH))
    rc_return VH.fail("A function argument has an invalid type", *this);

  if (not rc_recur Type()->size(VH))
    rc_return VH.fail("A function argument has no size", *this);

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const CABIFunctionDefinition &T) {
  revng_assert(T.Kind() == model::TypeDefinitionKind::CABIFunctionDefinition);

  if (not model::ABI::isValid(T.ABI()))
    rc_return VH.fail("Every C-ABI function must have a valid ABI.", T);

  if (not T.ReturnType().isEmpty()) {
    if (not rc_recur T.ReturnType()->verify(VH))
      rc_return VH.fail();

    if (T.ReturnType()->isVoidPrimitive())
      rc_return VH.fail("`void` return value is not allowed in CABI functions, "
                        "use empty type instead.",
                        T);

    if (T.ReturnType()->isArray())
      rc_return VH.fail("Array return value is not allowed in CABI functions, "
                        "wrap it in a `struct` type instead.",
                        T);

    // Only check for pointers without unwrapping typedefs, because there
    // isn't any valid syntax to return pointers to arrays in C without using
    // typedefs.
    if (const auto *Pointer = dyn_cast<model::PointerType>(&*T.ReturnType())) {
      while (Pointer) {
        const model::Type &Pointee = *Pointer->PointeeType();
        if (isa<model::ArrayType>(Pointee)) {
          rc_return VH.fail("Pointer-to-array return value is not allowed in "
                            "CABI functions, wrap the array in a `struct` "
                            "type instead.",
                            T);
        }
        Pointer = dyn_cast<model::PointerType>(&Pointee);
      }
    }

    if (not rc_recur T.ReturnType()->size(VH))
      rc_return VH.fail("Return value has no size", T);
  }

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Group : llvm::enumerate(T.Arguments())) {
    auto &Argument = Group.value();
    uint64_t ArgPos = Group.index();

    if (Argument.Index() != ArgPos)
      rc_return VH.fail("A function argument has an invalid index", T);

    if (not rc_recur Argument.verify(VH))
      rc_return VH.fail();

    if (Argument.Type()->isArray())
      rc_return VH.fail("Array argument is not allowed in CABI functions, "
                        "wrap it in a `struct` type instead.",
                        Argument);
  }

  rc_return true;
}

RecursiveCoroutine<bool> NamedTypedRegister::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Type().isEmpty())
    rc_return VH.fail("NamedTypedRegister must have a type", *this);

  if (not rc_recur Type()->verify(VH))
    rc_return VH.fail();

  // Ensure the type we're pointing to is a scalar
  if (not Type()->isScalar())
    rc_return VH.fail("Only scalars are allowed in RFTs", Type());

  if (not model::Register::isValid(Location()))
    rc_return VH.fail("NamedTypedRegister must have a location", *this);

  // Zero-sized types are not allowed
  auto MaybeTypeSize = rc_recur Type()->size(VH);
  if (not MaybeTypeSize)
    rc_return VH.fail("Types without size are not allowed in RFTs", Type());

  // Ensure if fits in the corresponding register
  if (not Type()->isFloatPrimitive()) {
    size_t RegisterSize = model::Register::getSize(Location());
    if (*MaybeTypeSize > RegisterSize)
      rc_return VH.fail("Object of " + ::toString(*MaybeTypeSize)
                          + "-byte type does not fit into a "
                          + ::toString(RegisterSize) + "-byte register",
                        Type());
  } else {
    // TODO: handle floating point register sizes properly.
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const RawFunctionDefinition &T) {
  revng_assert(T.Kind() == model::TypeDefinitionKind::RawFunctionDefinition);

  if (not model::Architecture::isValid(T.Architecture()))
    rc_return VH.fail("RFTs must have a valid architecture", T);

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (const NamedTypedRegister &Argument : T.Arguments()) {
    if (not rc_recur Argument.verify(VH))
      rc_return VH.fail();
    if (not isUsedInArchitecture(Argument.Location(), T.Architecture()))
      rc_return VH.fail("Register '" + toString(Argument.Location())
                          + "' must not be an argument of a raw '"
                          + toString(T.Architecture()) + "' function.",
                        T);
  }

  for (const NamedTypedRegister &Returned : T.ReturnValues()) {
    if (not rc_recur Returned.verify(VH))
      rc_return VH.fail();
    if (not isUsedInArchitecture(Returned.Location(), T.Architecture()))
      rc_return VH.fail("Register '" + toString(Returned.Location())
                          + "' must not be returned from a raw '"
                          + toString(T.Architecture()) + "' function.",
                        T);
  }

  for (const Register::Values &Preserved : T.PreservedRegisters()) {
    if (not model::Register::isValid(Preserved))
      rc_return VH.fail("Only valid registers can be preserved by raw "
                        "functions",
                        T);
    if (not isUsedInArchitecture(Preserved, T.Architecture()))
      rc_return VH.fail("Register '" + toString(Preserved)
                          + "' must not be preserved by a raw '"
                          + toString(T.Architecture()) + "' function.",
                        T);
  }

  // TODO: neither arguments nor return values should be preserved.

  auto &StackArgumentsType = T.StackArgumentsType();
  if (not StackArgumentsType.isEmpty()
      and not rc_recur StackArgumentsType->verify(VH))
    rc_return VH.fail();

  rc_return true;
}

RecursiveCoroutine<bool> TypeDefinition::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (VH.isVerified(*this))
    rc_return true;

  // Ensure we have not infinite recursion
  if (VH.isVerificationInProgress(*this))
    rc_return VH.fail();

  VH.verificationInProgress(*this);

  // TODO: make the id of a default constructed type `-1` once we have default
  //       value support in the model.
  if (ID() == size_t(-1))
    rc_return VH.fail("No type definition can have ID of -1u.");

  if (not model::TypeDefinitionKind::isValid(Kind()))
    rc_return VH.fail("Every type definition must have a valid kind.");

  bool Result = false;

  // We could use upcast() but we'd need to work around coroutines.
  if (auto *F = llvm::dyn_cast<model::CABIFunctionDefinition>(this))
    Result = rc_recur verifyImpl(VH, *F);
  else if (auto *F = llvm::dyn_cast<model::RawFunctionDefinition>(this))
    Result = rc_recur verifyImpl(VH, *F);
  else if (auto *E = llvm::dyn_cast<model::EnumDefinition>(this))
    Result = rc_recur verifyImpl(VH, *E);
  else if (auto *T = llvm::dyn_cast<model::TypedefDefinition>(this))
    Result = rc_recur verifyImpl(VH, *T);
  else if (auto *S = llvm::dyn_cast<model::StructDefinition>(this))
    Result = rc_recur verifyImpl(VH, *S);
  else if (auto *U = llvm::dyn_cast<model::UnionDefinition>(this))
    Result = rc_recur verifyImpl(VH, *U);
  else
    revng_abort("Unsupported type definition kind.");

  if (Result) {
    VH.setVerified(*this);
    VH.verificationCompleted(*this);
  }

  rc_return VH.maybeFail(Result);
}

bool Binary::verifyTypeDefinitions(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  for (const model::UpcastableTypeDefinition &Definition : TypeDefinitions()) {
    // All types on their own should verify
    if (not Definition.get()->verify(VH))
      return VH.fail();

    using CFT = model::CABIFunctionDefinition;
    using RFT = model::RawFunctionDefinition;
    if (const auto *T = llvm::dyn_cast<CFT>(Definition.get())) {
      if (getArchitecture(T->ABI()) != Architecture())
        return VH.fail("Function type architecture differs from the binary "
                       "architecture");
    } else if (const auto *T = llvm::dyn_cast<RFT>(Definition.get())) {
      if (T->Architecture() != Architecture())
        return VH.fail("Function type architecture differs from the binary "
                       "architecture");
    }
  }

  return true;
}

//
// Configuration
//

bool Configuration::verify(VerifyHelper &VH) const {
  // TODO: as this helper grows, split it up.

  // These checks are not necessary for now since the can't return an empty
  // string but they will be needed after the have a way to specify the default
  // value of a TTG field (since the default value helpers will go away).
  //
  // As such, let's add them now so that they don't end up forgotten.

  if (Configuration().Naming().UnnamedSegmentPrefix().empty())
    return VH.fail("Segment prefix must not be empty.");

  if (Configuration().Naming().UnnamedFunctionPrefix().empty())
    return VH.fail("Function prefix must not be empty.");

  if (Configuration().Naming().UnnamedDynamicFunctionPrefix().empty())
    return VH.fail("Dynamic function prefix must not be empty.");

  // `UnnamedTypeDefinitionPrefix` can be empty.

  if (Configuration().Naming().UnnamedEnumEntryPrefix().empty())
    return VH.fail("Enum entry prefix must not be empty.");

  if (Configuration().Naming().UnnamedStructFieldPrefix().empty())
    return VH.fail("Struct field prefix must not be empty.");

  if (Configuration().Naming().UnnamedUnionFieldPrefix().empty())
    return VH.fail("Union field prefix must not be empty.");

  if (Configuration().Naming().UnnamedFunctionArgumentPrefix().empty())
    return VH.fail("Argument prefix must not be empty.");

  if (Configuration().Naming().UnnamedFunctionRegisterPrefix().empty())
    return VH.fail("Register prefix must not be empty.");

  if (Configuration().Naming().UnnamedLocalVariablePrefix().empty())
    return VH.fail("Local variable prefix must not be empty.");
  if (Configuration().Naming().UnnamedBreakFromLoopVariablePrefix().empty())
    return VH.fail("\"Break from loop\" variable prefix must not be empty.");
  if (Configuration().Naming().UnnamedGotoLabelPrefix().empty())
    return VH.fail("Goto label prefix must not be empty.");

  if (Configuration().Naming().StructPaddingPrefix().empty())
    return VH.fail("Padding prefix must not be empty.");

  if (Configuration().Naming().OpaqueCSVValuePrefix().empty())
    return VH.fail("Opaque CSV prefix must not be empty.");
  if (Configuration().Naming().MaximumEnumValuePrefix().empty())
    return VH.fail("Maximum enum value prefix must not be empty.");

  if (Configuration().Naming().StackFrameVariableName().empty())
    return VH.fail("Stack frame variable name must not be empty.");
  if (Configuration().Naming().RawStackArgumentName().empty())
    return VH.fail("Raw stack argument name must not be empty.");
  if (Configuration().Naming().LoopStateVariableName().empty())
    return VH.fail("Loop state variable name must not be empty.");

  if (Configuration().Naming().ArtificialReturnValuePrefix().empty())
    return VH.fail("Artificial return value prefix must not be empty.");

  return true;
}

//
// Binary
//

static std::string buildGlobalNamespaceError(const auto &GlobalNamespace) {
  std::string Result;

  for (const auto &[Name, List] : GlobalNamespace) {
    if (List.size() > 1) {
      Result += "- `" + Name.str() + "`:\n";
      for (const auto &[_, Path] : List)
        Result += "    - `" + Path + "`\n";
    }
  }

  return Result;
}

static std::string buildLocalNamespaceError(const auto &Namespaces) {
  std::string Result;

  for (const auto &CurrentNamespace : Namespaces.Local) {
    for (const auto &[Name, List] : CurrentNamespace) {
      const decltype(List) *MaybeGlobalList = nullptr;
      auto Iterator = Namespaces.Global.find(Name);
      if (Iterator != Namespaces.Global.end())
        MaybeGlobalList = &Iterator->second;

      uint64_t TotalEntryCount = List.size();
      if (MaybeGlobalList)
        TotalEntryCount += MaybeGlobalList->size();

      if (TotalEntryCount > 1) {
        Result += "- `" + Name.str() + "`:\n";
        if (MaybeGlobalList)
          for (const auto &[_, Path] : *MaybeGlobalList)
            Result += "    - `" + Path + "`\n";
        for (const auto &[_, Path] : List)
          Result += "    - `" + Path + "`\n";
      }
    }
  }

  return Result;
}

bool Binary::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // Version == 0 is considered an alias to Version == SchemaVersion
  // Any other Version value is rejected; if you need to load such a model, you
  // need to migrate first
  if (Version() != SchemaVersion and Version() != 0)
    return VH.fail("Model version not supported");

  // Build list of executable segments
  SmallVector<const model::Segment *, 4> ExecutableSegments;
  for (const model::Segment &Segment : Segments())
    if (Segment.IsExecutable())
      ExecutableSegments.push_back(&Segment);

  auto IsExecutable = [&ExecutableSegments](const MetaAddress &Address) {
    auto ContainsAddress = [Address](const model::Segment *Segment) -> bool {
      return Segment->hasDataFor(Address);
    };
    return llvm::any_of(ExecutableSegments, ContainsAddress);
  };

  // Verify that there's (0, 1) binaries present
  // TODO: remove this once multi-binary is implemented
  if (Binaries().size() > 1)
    return VH.fail("Binaries must either contain 0 or 1 elements", Binaries());

  // Verify Binaries
  for (const BinaryIdentifier &BI : Binaries()) {
    if (not BI.verify(VH))
      return VH.fail();
  }

  // Verify EntryPoint
  if (EntryPoint().isValid()) {
    if (not EntryPoint().isCode())
      return VH.fail("EntryPoint is not code", EntryPoint());

    if (not IsExecutable(EntryPoint()))
      return VH.fail("Binary entry point not executable", EntryPoint());
  }

  // Verify ExtraCodeAddresses
  for (const MetaAddress &Address : ExtraCodeAddresses()) {
    if (not Address.isValid())
      return VH.fail("Invalid entry in ExtraCodeAddresses", Address);

    if (not Address.isCode())
      return VH.fail("Non-code entry in ExtraCodeAddresses", Address);

    if (not IsExecutable(Address))
      return VH.fail("ExtraCodeAddress entry is not executable", *this);
  }

  // Verify individual functions
  for (const Function &F : Functions()) {
    if (not F.verify(VH))
      return VH.fail();

    if (not IsExecutable(F.Entry()))
      return VH.fail("Function entry not executable", F);
  }

  // Verify DynamicFunctions
  model::CNameBuilder NameBuilder(*this);
  for (const DynamicFunction &DF : ImportedDynamicFunctions())
    if (not DF.verify(VH))
      return VH.fail();

  // Verify Segments
  for (const Segment &S : Segments())
    if (not S.verify(VH))
      return VH.fail();

  // Make sure no segments overlap
  for (const auto &[LHS, RHS] : zip_pairs(Segments())) {
    revng_assert(LHS.StartAddress() <= RHS.StartAddress());
    if (LHS.endAddress() > RHS.StartAddress()) {
      std::string Error = "Overlapping segments:\n" + ::toString(LHS) + "and\n"
                          + ::toString(RHS);
      return VH.fail(Error);
    }
  }

  // Verify the configuration and the type system
  if (not Configuration().verify(VH) or not verifyTypeDefinitions(VH))
    return false;

  // And, finally, ensure there are no colliding names.
  llvm::Expected Namespaces = collectNamespaces(*this);
  if (not Namespaces)
    return VH.fail(revng::unwrapError(Namespaces.takeError()));
  if (auto Err = buildGlobalNamespaceError(Namespaces->Global); !Err.empty())
    return VH.fail("Global namespace collisions were found:\n" + Err);
  if (auto Error = buildLocalNamespaceError(*Namespaces); !Error.empty())
    return VH.fail("Local namespace collisions were found:\n" + Error);

  return true;
}

//
// And the wrappers
//

bool Relocation::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool Relocation::verify() const {
  return verify(false);
}

bool Segment::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool Segment::verify() const {
  return verify(false);
}

bool BinaryIdentifier::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool BinaryIdentifier::verify() const {
  return verify(false);
}

bool CallSitePrototype::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool CallSitePrototype::verify() const {
  return verify(false);
}

bool StatementComment::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool StatementComment::verify() const {
  return verify(false);
}

bool LocalIdentifier::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool LocalIdentifier::verify() const {
  return verify(false);
}

bool Function::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool Function::verify() const {
  return verify(false);
}

bool DynamicFunction::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool DynamicFunction::verify() const {
  return verify(false);
}

bool EnumEntry::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool EnumEntry::verify() const {
  return verify(false);
}

bool StructField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool StructField::verify() const {
  return verify(false);
}

bool UnionField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool UnionField::verify() const {
  return verify(false);
}

bool NamedTypedRegister::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool NamedTypedRegister::verify() const {
  return verify(false);
}

bool Argument::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool Argument::verify() const {
  return verify(false);
}

bool TypeDefinition::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool TypeDefinition::verify() const {
  return verify(false);
}

bool Type::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool Type::verify() const {
  return verify(false);
}

bool Binary::verifyTypeDefinitions(bool Assert) const {
  VerifyHelper VH(Assert);
  return verifyTypeDefinitions(VH);
}
bool Binary::verifyTypeDefinitions() const {
  return verifyTypeDefinitions(false);
}

bool Configuration::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool Configuration::verify() const {
  return verify(false);
}

bool Binary::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool Binary::verify() const {
  return verify(false);
}

} // namespace model
