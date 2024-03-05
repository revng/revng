/// \file Verification.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"

#include "revng/Model/Binary.h"

using namespace llvm;

namespace model {

bool Qualifier::verify() const {
  return verify(false);
}

bool Qualifier::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Qualifier::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  switch (Kind()) {
  case QualifierKind::Invalid:
    return VH.fail("Invalid qualifier found", *this);
  case QualifierKind::Pointer:
    return VH.maybeFail(Size() > 0 and llvm::isPowerOf2_64(Size()),
                        "Pointer qualifier size is not a power of 2",
                        *this);
  case QualifierKind::Const:
    return VH.maybeFail(Size() == 0, "const qualifier has non-0 size", *this);
  case QualifierKind::Array:
    return VH.maybeFail(Size() > 0, "Array qualifier size is 0");
  default:
    revng_abort();
  }

  return VH.fail();
}

static constexpr bool isValidPrimitiveSize(PrimitiveKind::Values PrimKind,
                                           uint8_t BS) {
  switch (PrimKind) {
  case PrimitiveKind::Invalid:
    return false;

  case PrimitiveKind::Void:
    return BS == 0;

  // The ByteSizes allowed for Generic must be a superset of all the other
  // ByteSizes allowed for all other primitive types (except void)
  case PrimitiveKind::Generic:
    return BS == 1 or BS == 2 or BS == 4 or BS == 8 or BS == 10 or BS == 12
           or BS == 16;

  case PrimitiveKind::PointerOrNumber:
  case PrimitiveKind::Number:
  case PrimitiveKind::Unsigned:
  case PrimitiveKind::Signed:
    return BS == 1 or BS == 2 or BS == 4 or BS == 8 or BS == 16;

  // NOTE: We are supporting floats that are 10 bytes long, since we found such
  // cases in some PDB files by using VS on Windows platforms. The source code
  // of those cases could be written in some language other than C/C++ (probably
  // Swift). We faced some struct fields by using this (10b long float) type, so
  // by ignoring it we would not have accurate layout for the structs.
  case PrimitiveKind::Float:
    return BS == 2 or BS == 4 or BS == 8 or BS == 10 or BS == 12 or BS == 16;

  default:
    revng_abort();
  }

  revng_abort();
}

bool EnumEntry::verify() const {
  return verify(false);
}

bool EnumEntry::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool EnumEntry::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);
  return VH.maybeFail(CustomName().verify(VH));
}

static uint64_t makePrimitiveID(PrimitiveKind::Values PrimitiveKind,
                                uint8_t Size) {
  return (static_cast<uint8_t>(PrimitiveKind) << 8) | Size;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const PrimitiveDefinition *T) {
  revng_assert(T->Kind() == TypeDefinitionKind::PrimitiveDefinition);

  if (not T->CustomName().empty() or not T->OriginalName().empty())
    rc_return VH.fail("PrimitiveTypes cannot have OriginalName or CustomName",
                      *T);

  auto ExpectedID = makePrimitiveID(T->PrimitiveKind(), T->Size());
  if (T->ID() != ExpectedID)
    rc_return VH.fail(Twine("Wrong ID for PrimitiveDefinition. Got: ")
                        + Twine(T->ID()) + ". Expected: " + Twine(ExpectedID)
                        + ".",
                      *T);

  if (not isValidPrimitiveSize(T->PrimitiveKind(), T->Size()))
    rc_return VH.fail("Invalid PrimitiveDefinition size: " + Twine(T->Size()),
                      *T);

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const EnumDefinition *T) {
  if (T->Kind() != TypeDefinitionKind::EnumDefinition or T->Entries().empty()
      or not T->CustomName().verify(VH))
    rc_return VH.fail();

  // The underlying type has to be an unqualified primitive type
  if (not rc_recur T->UnderlyingType().verify(VH)
      or not T->UnderlyingType().Qualifiers().empty())
    rc_return VH.fail();

  // We only allow signed/unsigned as underlying type
  if (not T->UnderlyingType().isPrimitive(PrimitiveKind::Signed)
      and not T->UnderlyingType().isPrimitive(PrimitiveKind::Unsigned))
    rc_return VH.fail("UnderlyingType of a EnumDefinition can only be Signed "
                      "or Unsigned",
                      *T);

  for (auto &Entry : T->Entries()) {

    if (not Entry.verify(VH))
      rc_return VH.fail();

    // TODO: verify Entry.Value is within boundaries
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const TypedefDefinition *T) {
  rc_return VH.maybeFail(T->CustomName().verify(VH)
                         and T->Kind() == TypeDefinitionKind::TypedefDefinition
                         and rc_recur T->UnderlyingType().verify(VH));
}

inline RecursiveCoroutine<bool> isScalarImpl(const QualifiedType &QT) {
  for (const Qualifier &Q : QT.Qualifiers()) {
    switch (Q.Kind()) {
    case QualifierKind::Invalid:
      revng_abort();
    case QualifierKind::Pointer:
      rc_return true;
    case QualifierKind::Array:
      rc_return false;
    case QualifierKind::Const:
      break;
    default:
      revng_abort();
    }
  }

  const TypeDefinition *Unqualified = QT.UnqualifiedType().get();
  revng_assert(Unqualified != nullptr);
  if (llvm::isa<model::PrimitiveDefinition>(Unqualified)
      or llvm::isa<model::EnumDefinition>(Unqualified)) {
    rc_return true;
  }

  if (auto *Typedef = llvm::dyn_cast<model::TypedefDefinition>(Unqualified))
    rc_return rc_recur isScalarImpl(Typedef->UnderlyingType());

  rc_return false;
}

bool model::QualifiedType::isScalar() const {
  return isScalarImpl(*this);
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const StructDefinition *T) {

  using namespace llvm;

  revng_assert(T->Kind() == TypeDefinitionKind::StructDefinition);

  if (not T->CustomName().verify(VH))
    rc_return VH.fail("Invalid name", *T);

  if (T->Size() == 0)
    rc_return VH.fail("Struct type has zero size", *T);

  size_t Index = 0;
  llvm::SmallSet<llvm::StringRef, 8> Names;
  auto FieldIt = T->Fields().begin();
  auto FieldEnd = T->Fields().end();
  for (; FieldIt != FieldEnd; ++FieldIt) {
    auto &Field = *FieldIt;

    if (not rc_recur Field.verify(VH)) {
      rc_return VH.fail("Can't verify type of field at offset "
                          + Twine(Field.Offset()),
                        *T);
    }

    if (Field.Offset() >= T->Size()) {
      uint64_t Size = *Field.Type().size();
      rc_return VH.fail("Field at offset " + Twine(Field.Offset())
                          + " is out of struct boundaries (field size: "
                          + Twine(Size) + ", field offset + size: "
                          + Twine(Field.Offset() + Size)
                          + ", struct size: " + Twine(T->Size()) + ")",
                        *T);
    }

    auto MaybeSize = rc_recur Field.Type().size(VH);
    // This is verified AggregateField::verify
    revng_assert(MaybeSize);

    auto FieldEndOffset = Field.Offset() + *MaybeSize;
    auto NextFieldIt = std::next(FieldIt);
    if (NextFieldIt != FieldEnd) {
      // If this field is not the last, check that it does not overlap with the
      // following field.
      if (FieldEndOffset > NextFieldIt->Offset()) {
        rc_return VH.fail("Field at offset " + Twine(Field.Offset())
                            + " (with size: " + Twine(*Field.Type().size())
                            + ") overlaps with the field at offset "
                            + Twine(NextFieldIt->Offset()) + " (with size: "
                            + Twine(*NextFieldIt->Type().size()) + ")",
                          *T);
      }
    } else if (FieldEndOffset > T->Size()) {
      // Otherwise, if this field is the last, check that it's not larger than
      // size.
      rc_return VH.fail("Last field ends outside the struct", *T);
    }

    if (not rc_recur Field.Type().size(VH))
      rc_return VH.fail("Field " + Twine(Index + 1) + " has no size", *T);

    // Verify CustomName for collisions
    if (not Field.CustomName().empty()) {
      if (VH.isGlobalSymbol(Field.CustomName())) {
        rc_return VH.fail("Field \"" + Field.CustomName()
                            + "\" collides with global symbol",
                          *T);
      }

      if (not Names.insert(Field.CustomName()).second)
        rc_return VH.fail("Collision in struct fields names", *T);
    }

    ++Index;
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const UnionDefinition *T) {
  revng_assert(T->Kind() == TypeDefinitionKind::UnionDefinition);

  if (not T->CustomName().verify(VH))
    rc_return VH.fail("Invalid name", *T);

  if (T->Fields().empty())
    rc_return VH.fail("Union type has zero fields", *T);

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Group : llvm::enumerate(T->Fields())) {
    auto &Field = Group.value();
    uint64_t ExpectedIndex = Group.index();

    if (Field.Index() != ExpectedIndex) {
      rc_return VH.fail(Twine("Union type is missing field ")
                          + Twine(ExpectedIndex),
                        *T);
    }

    if (not rc_recur Field.verify(VH))
      rc_return VH.fail();

    auto MaybeSize = rc_recur Field.Type().size(VH);
    // This is verified AggregateField::verify
    revng_assert(MaybeSize);

    if (not rc_recur Field.Type().size(VH)) {
      rc_return VH.fail("Field " + Twine(Field.Index()) + " has no size", *T);
    }

    // Verify CustomName for collisions
    if (not Field.CustomName().empty()) {
      if (VH.isGlobalSymbol(Field.CustomName())) {
        rc_return VH.fail("Field \"" + Field.CustomName()
                            + "\" collides with global symbol",
                          *T);
      }

      if (not Names.insert(Field.CustomName()).second)
        rc_return VH.fail("Collision in union fields names", *T);
    }
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const CABIFunctionDefinition *T) {
  if (not T->CustomName().verify(VH)
      or T->Kind() != TypeDefinitionKind::CABIFunctionDefinition
      or not rc_recur T->ReturnType().verify(VH))
    rc_return VH.fail();

  if (T->ABI() == model::ABI::Invalid)
    rc_return VH.fail("An invalid ABI", *T);

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Group : llvm::enumerate(T->Arguments())) {
    auto &Argument = Group.value();
    uint64_t ArgPos = Group.index();

    if (not Argument.CustomName().verify(VH))
      rc_return VH.fail("An argument has invalid CustomName", *T);

    // Verify CustomName for collisions
    if (not Argument.CustomName().empty()) {
      if (VH.isGlobalSymbol(Argument.CustomName()))
        rc_return VH.fail("Argument name collides with global symbol", *T);

      if (not Names.insert(Argument.CustomName()).second)
        rc_return VH.fail("Collision in argument names", *T);
    }

    if (Argument.Index() != ArgPos)
      rc_return VH.fail("An argument has invalid index", *T);

    if (not rc_recur Argument.Type().verify(VH))
      rc_return VH.fail("An argument has invalid type", *T);

    if (not rc_recur Argument.Type().size(VH))
      rc_return VH.fail("An argument has no size", *T);
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const RawFunctionDefinition *T) {
  const model::Architecture::Values Architecture = T->Architecture();

  if (Architecture == model::Architecture::Invalid)
    rc_return VH.fail();

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (const NamedTypedRegister &Argument : T->Arguments()) {
    if (not rc_recur Argument.verify(VH))
      rc_return VH.fail();
    if (not isUsedInArchitecture(Argument.Location(), Architecture))
      rc_return VH.fail();

    // Verify CustomName for collisions
    if (not Argument.CustomName().empty()) {
      if (VH.isGlobalSymbol(Argument.CustomName()))
        rc_return VH.fail("Argument name collides with global symbol", *T);

      if (not Names.insert(Argument.CustomName()).second)
        rc_return VH.fail("Collision in argument names", *T);
    }
  }

  for (const NamedTypedRegister &Return : T->ReturnValues()) {
    if (not rc_recur Return.verify(VH))
      rc_return VH.fail();
    if (not isUsedInArchitecture(Return.Location(), Architecture))
      rc_return VH.fail();
  }

  for (const Register::Values &Preserved : T->PreservedRegisters()) {
    if (Preserved == Register::Invalid)
      rc_return VH.fail();
    if (not isUsedInArchitecture(Preserved, Architecture))
      rc_return VH.fail();
  }

  auto &StackArgumentsType = T->StackArgumentsType();
  if (not StackArgumentsType.empty()
      and not rc_recur StackArgumentsType.get()->verify(VH))
    rc_return VH.fail();

  rc_return VH.maybeFail(T->CustomName().verify(VH));
}

bool TypeDefinition::verify() const {
  return verify(false);
}

bool TypeDefinition::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> TypeDefinition::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (VH.isVerified(this))
    rc_return true;

  // Ensure we have not infinite recursion
  if (VH.isVerificationInProgress(this))
    rc_return VH.fail();

  VH.verificationInProgress(this);

  if (ID() == 0)
    rc_return VH.fail("A type cannot have ID 0", *this);

  bool Result = false;

  // We could use upcast() but we'd need to workaround coroutines.
  switch (Kind()) {
  case TypeDefinitionKind::PrimitiveDefinition:
    Result = rc_recur verifyImpl(VH, cast<PrimitiveDefinition>(this));
    break;

  case TypeDefinitionKind::EnumDefinition:
    Result = rc_recur verifyImpl(VH, cast<EnumDefinition>(this));
    break;

  case TypeDefinitionKind::TypedefDefinition:
    Result = rc_recur verifyImpl(VH, cast<TypedefDefinition>(this));
    break;

  case TypeDefinitionKind::StructDefinition:
    Result = rc_recur verifyImpl(VH, cast<StructDefinition>(this));
    break;

  case TypeDefinitionKind::UnionDefinition:
    Result = rc_recur verifyImpl(VH, cast<UnionDefinition>(this));
    break;

  case TypeDefinitionKind::CABIFunctionDefinition:
    Result = rc_recur verifyImpl(VH, cast<CABIFunctionDefinition>(this));
    break;

  case TypeDefinitionKind::RawFunctionDefinition:
    Result = rc_recur verifyImpl(VH, cast<RawFunctionDefinition>(this));
    break;

  default: // Do nothing;
    ;
  }

  if (Result) {
    VH.setVerified(this);
    VH.verificationCompleted(this);
  }

  rc_return VH.maybeFail(Result);
}

bool QualifiedType::verify() const {
  return verify(false);
}

bool QualifiedType::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> QualifiedType::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not UnqualifiedType().isValid())
    rc_return VH.fail("Underlying type is invalid", *this);

  // Verify the qualifiers are valid
  for (const auto &Q : Qualifiers())
    if (not Q.verify(VH))
      rc_return VH.fail("Invalid qualifier", Q);

  auto QIt = Qualifiers().begin();
  auto QEnd = Qualifiers().end();
  for (; QIt != QEnd; ++QIt) {
    const auto &Q = *QIt;
    auto NextQIt = std::next(QIt);
    bool HasNext = NextQIt != QEnd;

    // Check that we have not two consecutive const qualifiers
    if (HasNext and Qualifier::isConst(Q) and Qualifier::isConst(*NextQIt))
      rc_return VH.fail("QualifiedType has two consecutive const qualifiers",
                        *this);

    if (Qualifier::isPointer(Q)) {
      // Don't proceed the verification, just make sure the pointer is either
      // 32- or 64-bit
      rc_return VH.maybeFail(Q.Size() == 4 or Q.Size() == 8,
                             "Only 32-bit and 64-bit pointers "
                             "are currently "
                             "supported",
                             *this);

    } else if (Qualifier::isArray(Q)) {
      // Ensure there's at least one element
      if (Q.Size() < 1)
        rc_return VH.fail("Arrays need to have at least an element", *this);

      // Verify element type
      QualifiedType ElementType{ UnqualifiedType(), { NextQIt, QEnd } };
      if (not rc_recur ElementType.verify(VH))
        rc_return VH.fail("Array element invalid", ElementType);

      // Ensure the element type has a size and stop
      auto MaybeSize = rc_recur ElementType.size(VH);
      rc_return VH.maybeFail(MaybeSize.has_value(),
                             "Cannot compute array size",
                             ElementType);
    } else if (Qualifier::isConst(Q)) {
      // const qualifiers must have zero size
      if (Q.Size() != 0)
        rc_return VH.fail("const qualifier has non-0 size");

    } else {
      revng_abort();
    }
  }

  // If we get here, we either have no qualifiers or just const qualifiers:
  // recur on the underlying type
  rc_return VH.maybeFail(rc_recur UnqualifiedType().get()->verify(VH));
}

bool NamedTypedRegister::verify() const {
  return verify(false);
}

bool NamedTypedRegister::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> NamedTypedRegister::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // Ensure the name is valid
  if (not CustomName().verify(VH))
    rc_return VH.fail();

  // Ensure the type we're pointing to is scalar
  if (not Type().isScalar())
    rc_return VH.fail();

  if (Location() == Register::Invalid)
    rc_return VH.fail();

  // Ensure if fits in the corresponding register
  auto MaybeTypeSize = rc_recur Type().size(VH);

  // Zero-sized types are not allowed
  if (not MaybeTypeSize)
    rc_return VH.fail();

  // TODO: handle floating point register sizes properly.
  if (not Type().isFloat()) {
    size_t RegisterSize = model::Register::getSize(Location());
    if (*MaybeTypeSize > RegisterSize)
      rc_return VH.fail();
  }

  rc_return VH.maybeFail(rc_recur Type().verify(VH));
}

bool StructField::verify() const {
  return verify(false);
}

bool StructField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> StructField::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not rc_recur Type().verify(VH))
    rc_return VH.fail("Aggregate field type is not valid");

  // Aggregated fields cannot be zero-sized fields
  auto MaybeSize = rc_recur Type().size(VH);
  if (not MaybeSize)
    rc_return VH.fail("Aggregate field is zero-sized");

  rc_return VH.maybeFail(CustomName().verify(VH));
}

bool UnionField::verify() const {
  return verify(false);
}

bool UnionField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> UnionField::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not rc_recur Type().verify(VH))
    rc_return VH.fail("Aggregate field type is not valid");

  // Aggregated fields cannot be zero-sized fields
  auto MaybeSize = rc_recur Type().size(VH);
  if (not MaybeSize)
    rc_return VH.fail("Aggregate field is zero-sized", Type());

  rc_return VH.maybeFail(CustomName().verify(VH));
}

bool Argument::verify() const {
  return verify(false);
}

bool Argument::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> Argument::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);
  rc_return VH.maybeFail(CustomName().verify(VH)
                         and rc_recur Type().verify(VH));
}

bool Binary::verifyTypeDefinitions() const {
  return verifyTypeDefinitions(false);
}

bool Binary::verifyTypeDefinitions(bool Assert) const {
  VerifyHelper VH(Assert);
  return verifyTypeDefinitions(VH);
}

bool Binary::verifyTypeDefinitions(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // All types on their own should verify
  std::set<Identifier> Names;
  for (const model::UpcastableTypeDefinition &Definition : TypeDefinitions()) {
    // Verify the type
    if (not Definition.get()->verify(VH))
      return VH.fail();

    // Ensure the names are unique
    auto Name = Definition->name();
    if (not Names.insert(Name).second)
      return VH.fail(Twine("Multiple types with the following name: ") + Name);

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

bool Binary::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Binary::verify() const {
  VerifyHelper VH(false);
  return verify(VH);
}

static bool verifyGlobalNamespace(VerifyHelper &VH,
                                  const model::Binary &Model) {

  // Namespacing rules:
  //
  // 1. each struct/union induces a namespace for its field names;
  // 2. each prototype induces a namespace for its arguments (and local
  //    variables, but those are not part of the model yet);
  // 3. the global namespace includes segment names, function names, dynamic
  //    function names, type names and entries of `enum`s;
  //
  // Verify needs to verify that each namespace has no internal clashes.
  // Also, the global namespace clashes with everything.
  for (const Function &F : Model.Functions()) {
    if (not VH.registerGlobalSymbol(F.CustomName(), Model.path(F)))
      return VH.fail("Duplicate name", F);
  }

  // Verify DynamicFunctions
  for (const DynamicFunction &DF : Model.ImportedDynamicFunctions()) {
    if (not VH.registerGlobalSymbol(DF.CustomName(), Model.path(DF)))
      return VH.fail();
  }

  // Verify types and enum entries
  for (const model::UpcastableTypeDefinition &Def : Model.TypeDefinitions()) {
    if (not VH.registerGlobalSymbol(Def->CustomName(), Model.path(*Def)))
      return VH.fail();

    if (auto *Enum = dyn_cast<model::EnumDefinition>(Def.get()))
      for (auto &Entry : Enum->Entries())
        if (not VH.registerGlobalSymbol(Entry.CustomName(),
                                        Model.path(*Enum, Entry)))
          return VH.fail();
  }

  // Verify Segments
  for (const Segment &S : Model.Segments()) {
    if (not VH.registerGlobalSymbol(S.CustomName(), Model.path(S)))
      return VH.fail();
  }

  return true;
}

bool Binary::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // First of all, verify the global namespace: we need to fully populate it
  // before we can verify namespaces with smaller scopes
  if (not verifyGlobalNamespace(VH, *this))
    return VH.fail();

  // Verify individual functions
  for (const Function &F : Functions())
    if (not F.verify(VH))
      return VH.fail();

  // Verify DynamicFunctions
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
      std::string Error = "Overlapping segments:\n" + serializeToString(LHS)
                          + "and\n" + serializeToString(RHS);
      return VH.fail(Error);
    }
  }

  //
  // Verify the type system
  //
  return verifyTypeDefinitions(VH);
}

bool Relocation::verify() const {
  return verify(false);
}

bool Relocation::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Relocation::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Type() == model::RelocationType::Invalid)
    return VH.fail("Invalid relocation", *this);

  return true;
}

bool Segment::verify() const {
  return verify(false);
}

bool Segment::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Segment::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  using OverflowSafeInt = OverflowSafeInt<uint64_t>;

  if (FileSize() > VirtualSize())
    return VH.fail("FileSize cannot be larger than VirtualSize", *this);

  auto EndOffset = OverflowSafeInt(StartOffset()) + FileSize();
  if (not EndOffset)
    return VH.fail("Computing the segment end offset leads to overflow", *this);

  auto EndAddress = StartAddress() + VirtualSize();
  if (not EndAddress.isValid())
    return VH.fail("Computing the end address leads to overflow", *this);

  for (const model::Relocation &Relocation : Relocations()) {
    if (not Relocation.verify(VH))
      return VH.fail("Invalid relocation", Relocation);
  }

  if (not Type().empty()) {

    if (not Type().isValid())
      return VH.fail("Invalid segment type", *this);

    // The segment has a type

    auto *Struct = dyn_cast<model::StructDefinition>(Type().get());
    if (not Struct)
      return VH.fail("The segment type is not a StructDefinition", *this);

    if (VirtualSize() != Struct->Size()) {
      return VH.fail(Twine("The segment's size (VirtualSize) is not equal to "
                           "the size of the segment's type. VirtualSize: ")
                       + Twine(VirtualSize())
                       + Twine(" != Segment->Type()->Size(): ")
                       + Twine(Struct->Size()),
                     *this);
    }

    if (Struct->CanContainCode() != IsExecutable()) {
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

    if (not Type().get()->verify(VH))
      return VH.fail("Segment type does not verify", *this);
  }

  return true;
}

bool Function::verify() const {
  return verify(false);
}

bool Function::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Function::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not Entry().isValid())
    return VH.fail("Invalid Entry", *this);

  if (not Prototype().empty()) {

    if (not Prototype().isValid())
      return VH.fail("Invalid prototype", *this);

    // The function has a prototype

    if (not model::QualifiedType::getFunctionType(Prototype()).has_value()) {
      return VH.fail("The prototype is neither a RawFunctionDefinition nor a "
                     "CABIFunctionDefinition",
                     *this);
    }

    if (not Prototype().get()->verify(VH))
      return VH.fail("Function prototype does not verify", *this);
  }

  if (not StackFrameType().empty()) {

    if (not StackFrameType().isValid())
      return VH.fail("Invalid stack frame type", *this);

    // The stack frame has a type

    if (not isa<model::StructDefinition>(StackFrameType().get()))
      return VH.fail("The stack frame type is not a StructDefinition", *this);

    if (not StackFrameType().get()->verify(VH))
      return VH.fail("Stack frame type does not verify", *this);
  }

  for (auto &CallSitePrototype : CallSitePrototypes())
    if (not CallSitePrototype.verify(VH))
      return VH.fail();

  return true;
}

bool DynamicFunction::verify() const {
  return verify(false);
}

bool DynamicFunction::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool DynamicFunction::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // Ensure we have a name
  if (OriginalName().size() == 0)
    return VH.fail("Dynamic functions must have a OriginalName", *this);

  if (not Prototype().empty() and not Prototype().isValid())
    return VH.fail("Invalid prototype", *this);

  // Prototype is valid
  if (not Prototype().empty()) {
    if (not Prototype().get()->verify(VH))
      return VH.fail();

    if (not model::QualifiedType::getFunctionType(Prototype()).has_value()) {
      return VH.fail("The prototype is neither a RawFunctionDefinition nor a "
                     "CABIFunctionDefinition",
                     *this);
    }
  }

  for (auto &Attribute : Attributes()) {
    if (Attribute == model::FunctionAttribute::Inline) {
      return VH.fail("Dynamic function cannot have Inline attribute", *this);
    }
  }

  return true;
}

bool CallSitePrototype::verify() const {
  return verify(false);
}

bool CallSitePrototype::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool CallSitePrototype::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Prototype().empty() or not Prototype().isValid())
    return VH.fail("Invalid prototype");

  // Prototype is valid
  if (not Prototype().get()->verify(VH))
    return VH.fail();

  if (not model::QualifiedType::getFunctionType(Prototype()).has_value()) {
    return VH.fail("The prototype is neither a RawFunctionDefinition nor a "
                   "CABIFunctionDefinition",
                   *this);
  }

  return true;
}

bool VerifyHelper::isGlobalSymbol(const model::Identifier &Name) const {
  return GlobalSymbols.count(Name) > 0;
}

bool VerifyHelper::registerGlobalSymbol(const model::Identifier &Name,
                                        const std::string &Path) {
  if (Name.empty())
    return true;

  auto It = GlobalSymbols.find(Name);
  if (It == GlobalSymbols.end()) {
    GlobalSymbols.insert({ Name, Path });
    return true;
  } else {
    std::string Message;
    Message += "Duplicate global symbol \"";
    Message += Name.str().str();
    Message += "\":\n\n";

    Message += "  " + It->second + "\n";
    Message += "  " + Path + "\n";
    return fail(Message);
  }
}

} // namespace model
