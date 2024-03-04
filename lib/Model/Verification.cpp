//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"

#include "revng/Model/Binary.h"

using namespace llvm;

namespace model {

//
// Namespacing
//

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

bool model::Binary::verifyGlobalNamespace(VerifyHelper &VH) const {

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
  for (const Function &F : Functions()) {
    if (not VH.registerGlobalSymbol(F.CustomName(), path(F)))
      return VH.fail("Duplicate name", F);
  }

  // Verify DynamicFunctions
  for (const DynamicFunction &DF : ImportedDynamicFunctions()) {
    if (not VH.registerGlobalSymbol(DF.CustomName(), path(DF)))
      return VH.fail();
  }

  // Verify types and enum entries
  for (const model::UpcastableTypeDefinition &Def : TypeDefinitions()) {
    if (not VH.registerGlobalSymbol(Def->CustomName(), path(*Def)))
      return VH.fail();

    if (auto *Enum = dyn_cast<model::EnumDefinition>(Def.get()))
      for (auto &Entry : Enum->Entries())
        if (not VH.registerGlobalSymbol(Entry.CustomName(), path(*Enum, Entry)))
          return VH.fail();
  }

  // Verify Segments
  for (const Segment &S : Segments()) {
    if (not VH.registerGlobalSymbol(S.CustomName(), path(S)))
      return VH.fail();
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
    revng_abort("Unsupported primitive kin");
  }
}

RecursiveCoroutine<bool> model::Type::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  bool PointerBeforeDefinition = false;
  const model::Type *Active = this;
  while (Active != nullptr) {
    if (auto *Array = llvm::dyn_cast<model::ArrayType>(Active)) {
      if (Array->ElementCount() == 0)
        rc_return VH.fail("0 element arrays are not supported", *Array);
      if (Array->ElementType().empty())
        rc_return VH.fail("Arrays without an element type are not supported",
                          *Array);
      if (!Array->ElementType()->size().has_value())
        rc_return VH.fail("Array element type must have a size.", *Array);

      Active = Array->ElementType().get();

    } else if (auto *Defined = llvm::dyn_cast<model::DefinedType>(Active)) {
      if (not Defined->Definition().isValid())
        rc_return VH.fail("Defined types must contain a valid (non-empty) "
                          "reference",
                          *Defined);

      // Do not recur if this type is a pointer, otherwise we get undesired
      // failures if a type (for example a struct) has a pointer to itself.
      if (PointerBeforeDefinition)
        rc_return true;
      else
        rc_return rc_recur Defined->Definition().get()->verify(VH);

    } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(Active)) {
      if (!llvm::isPowerOf2_64(Pointer->PointerSize()))
        rc_return VH.fail("Pointer size is not a power of 2", *Pointer);

      if (Pointer->PointerSize() != 4 && Pointer->PointerSize() != 8)
        rc_return VH.fail("Only 32-bit and 64-bit pointers are currently "
                          "supported",
                          *Pointer);

      if (Pointer->PointeeType().empty())
        rc_return VH.fail("Pointers without an pointee type are not supported. "
                          "Use a `PrimitiveType::makeVoid`, if you want to "
                          "represent `void *`.",
                          *Pointer);

      PointerBeforeDefinition = true;
      Active = Pointer->PointeeType().get();

    } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(Active)) {
      if (not isValidPrimitiveSize(Primitive->PrimitiveKind(),
                                   Primitive->Size()))
        rc_return VH.fail("Primitive is not allowed.", *Primitive);

      rc_return true;

    } else {
      revng_abort("Unsupported type kind.");
    }
  }

  revng_abort("`nullptr` type where there is not supposed to be one.");
}

//
// Functions
//

bool CallSitePrototype::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Prototype().empty())
    return VH.fail("Call sites must have a prototype.");

  if (not Prototype()->isPrototype())
    return VH.fail("`Prototype()` must be a prototype.");

  if (not Prototype()->verify(VH))
    return VH.fail();

  return true;
}

bool Function::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not Entry().isValid())
    return VH.fail("Invalid Entry", *this);

  if (not Prototype().empty()) {

    if (not Prototype()->isPrototype())
      return VH.fail("`Prototype()` must be a prototype.");

    if (not Prototype()->verify(VH))
      return VH.fail("Function prototype does not verify.", *this);
  }

  if (not StackFrameType().empty()) {

    if (not StackFrameType()->isStruct())
      return VH.fail("`StackFrameType()` must be a struct.");

    if (not StackFrameType()->verify(VH))
      return VH.fail("Stack frame type does not verify.", *this);
  }

  for (auto &CallSitePrototype : CallSitePrototypes())
    if (not CallSitePrototype.verify(VH))
      return VH.fail();

  return true;
}

bool DynamicFunction::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // Ensure we have a name
  if (OriginalName().size() == 0)
    return VH.fail("Dynamic functions must have an OriginalName.", *this);

  if (not Prototype().empty()) {
    if (not Prototype()->isPrototype())
      return VH.fail("`Prototype()` type must be a prototype.", *this);

    if (not Prototype()->verify(VH))
      return VH.fail();
  }

  for (auto &Attribute : Attributes()) {
    if (Attribute == model::FunctionAttribute::Inline) {
      return VH.fail("Dynamic function cannot have Inline attribute", *this);
    }
  }

  return true;
}

//
// Segments
//

bool Section::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  auto EndAddress = StartAddress() + Size();
  if (not EndAddress.isValid())
    return VH.fail("Computing the end address leads to overflow");

  return true;
}

bool Relocation::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Type() == model::RelocationType::Invalid)
    return VH.fail("Invalid relocation", *this);

  return true;
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

  for (const model::Section &Section : Sections()) {
    if (not Section.verify(VH))
      return VH.fail("Invalid section", Section);

    if (not contains(Section.StartAddress())
        or (VirtualSize() > 0 and not contains(Section.endAddress() - 1))) {
      return VH.fail("The segment contains a section out of its boundaries",
                     Section);
    }

    if (Section.ContainsCode() and not IsExecutable()) {
      return VH.fail("A Section is marked as containing code but the "
                     "containing segment is not executable",
                     *this);
    }
  }

  for (const model::Relocation &Relocation : Relocations()) {
    if (not Relocation.verify(VH))
      return VH.fail("Invalid relocation", Relocation);
  }

  if (not Type().empty()) {

    if (not Type()->isStruct())
      return VH.fail("Segment's `Type()` must be a struct.");

    if (not Type()->verify(VH))
      return VH.fail("Segment's `Type()` does not verify.", *this);

    if (VirtualSize() != type()->Size()) {
      return VH.fail(Twine("Segment's virtual size is not equal to the size of "
                           "its type.\n`VirtualSize`: ")
                       + Twine(VirtualSize())
                       + Twine(" != `Segment.type()->Size()`: ")
                       + Twine(type()->Size()),
                     *this);
    }
  }

  return true;
}

//
// Type definitions
//

bool EnumEntry::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);
  return VH.maybeFail(CustomName().verify(VH));
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const EnumDefinition &T) {
  if (T.Kind() != TypeDefinitionKind::EnumDefinition or T.Entries().empty()
      or not T.CustomName().verify(VH))
    rc_return VH.fail();

  if (T.UnderlyingType().empty())
    rc_return VH.fail("Enum must have an underlying type.");

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
  rc_return VH.maybeFail(T.CustomName().verify(VH)
                         and T.Kind() == TypeDefinitionKind::TypedefDefinition
                         and not T.UnderlyingType().empty()
                         and rc_recur T.UnderlyingType()->verify(VH));
}

RecursiveCoroutine<bool> StructField::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Type().empty())
    rc_return VH.fail("Struct field must have a type.", *this);

  if (not rc_recur Type()->verify(VH))
    rc_return VH.fail();

  // Struct fields cannot be zero-sized
  auto MaybeSize = rc_recur Type()->size(VH);
  if (not MaybeSize)
    rc_return VH.fail("Struct field is zero-sized", Type());

  rc_return VH.maybeFail(CustomName().verify(VH));
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const StructDefinition &T) {

  using namespace llvm;

  revng_assert(T.Kind() == TypeDefinitionKind::StructDefinition);

  if (not T.CustomName().verify(VH))
    rc_return VH.fail("Invalid name", T);

  if (T.Size() == 0)
    rc_return VH.fail("Struct size must be greater than zero.", T);

  llvm::SmallSet<llvm::StringRef, 8> Names;
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

    // Verify CustomName for collisions
    if (not Field.CustomName().empty()) {
      if (VH.isGlobalSymbol(Field.CustomName())) {
        rc_return VH.fail("Field \"" + Field.CustomName()
                            + "\" collides with global symbol",
                          T);
      }

      if (not Names.insert(Field.CustomName()).second)
        rc_return VH.fail("Collision in struct fields names", T);
    }
  }

  rc_return true;
}

RecursiveCoroutine<bool> UnionField::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Type().empty())
    rc_return VH.fail("Union field must have a type.", *this);

  if (not rc_recur Type()->verify(VH))
    rc_return VH.fail();

  // Union fields cannot be zero-sized
  auto MaybeSize = rc_recur Type()->size(VH);
  if (not MaybeSize)
    rc_return VH.fail("Union field is zero-sized", Type());

  rc_return VH.maybeFail(CustomName().verify(VH));
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const UnionDefinition &T) {
  revng_assert(T.Kind() == TypeDefinitionKind::UnionDefinition);

  if (not T.CustomName().verify(VH))
    rc_return VH.fail("Invalid name", T);

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

    // Verify CustomName for collisions
    if (not Field.CustomName().empty()) {
      if (VH.isGlobalSymbol(Field.CustomName())) {
        rc_return VH.fail("Field \"" + Field.CustomName()
                            + "\" collides with global symbol",
                          T);
      }

      if (not Names.insert(Field.CustomName()).second)
        rc_return VH.fail("Collision in union fields names", T);
    }
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const CABIFunctionDefinition &T) {
  if (not T.CustomName().verify(VH)
      or T.Kind() != TypeDefinitionKind::CABIFunctionDefinition)
    rc_return VH.fail();

  if (not T.ReturnType().empty()) {
    if (not rc_recur T.ReturnType()->verify(VH))
      rc_return VH.fail();

    if (T.ReturnType()->isVoid())
      rc_return VH.fail("`void` return value is not allowed in CABI functions, "
                        "use empty type instead.");

    if (not rc_recur T.ReturnType()->size(VH))
      rc_return VH.fail("Return value has no size", T);
  }

  if (T.ABI() == model::ABI::Invalid)
    rc_return VH.fail("An invalid ABI", T);

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Group : llvm::enumerate(T.Arguments())) {
    auto &Argument = Group.value();
    uint64_t ArgPos = Group.index();

    if (not Argument.CustomName().verify(VH))
      rc_return VH.fail("An argument has invalid CustomName", T);

    // Verify CustomName for collisions
    if (not Argument.CustomName().empty()) {
      if (VH.isGlobalSymbol(Argument.CustomName()))
        rc_return VH.fail("Argument name collides with global symbol", T);

      if (not Names.insert(Argument.CustomName()).second)
        rc_return VH.fail("Collision in argument names", T);
    }

    if (Argument.Index() != ArgPos)
      rc_return VH.fail("An argument has an invalid index", T);

    if (Argument.Type().empty())
      rc_return VH.fail("An argument must have a type", T);

    if (not rc_recur Argument.Type()->verify(VH))
      rc_return VH.fail("An argument has an invalid type", T);

    if (not rc_recur Argument.Type()->size(VH))
      rc_return VH.fail("An argument has no size", T);
  }

  rc_return true;
}

RecursiveCoroutine<bool> NamedTypedRegister::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // Ensure the name is valid
  if (not CustomName().verify(VH))
    rc_return VH.fail();

  if (Type().empty())
    rc_return VH.fail("NamedTypedRegister must have a type", *this);

  if (not rc_recur Type()->verify(VH))
    rc_return VH.fail();

  // Ensure the type we're pointing to is a scalar
  if (not Type()->isScalar())
    rc_return VH.fail();

  if (Location() == Register::Invalid)
    rc_return VH.fail("NamedTypedRegister must have a location", *this);

  // Zero-sized types are not allowed
  auto MaybeTypeSize = rc_recur Type()->size(VH);
  if (not MaybeTypeSize)
    rc_return VH.fail();

  // Ensure if fits in the corresponding register
  if (not Type()->isFloat()) {
    size_t RegisterSize = model::Register::getSize(Location());
    if (*MaybeTypeSize > RegisterSize)
      rc_return VH.fail();
  } else {
    // TODO: handle floating point register sizes properly.
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const RawFunctionDefinition &T) {
  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (const NamedTypedRegister &Argument : T.Arguments()) {
    if (not rc_recur Argument.verify(VH))
      rc_return VH.fail();

    // Verify CustomName for collisions
    if (not Argument.CustomName().empty()) {
      if (VH.isGlobalSymbol(Argument.CustomName()))
        rc_return VH.fail("Argument name collides with global symbol", T);

      if (not Names.insert(Argument.CustomName()).second)
        rc_return VH.fail("Collision in argument names", T);
    }
  }

  for (const NamedTypedRegister &Return : T.ReturnValues())
    if (not rc_recur Return.verify(VH))
      rc_return VH.fail();

  for (const Register::Values &Preserved : T.PreservedRegisters())
    if (Preserved == Register::Invalid)
      rc_return VH.fail();

  // TODO: neither arguments nor return values should be preserved.

  auto &StackArgumentsType = T.StackArgumentsType();
  if (not StackArgumentsType.empty()
      and not rc_recur StackArgumentsType->verify(VH))
    rc_return VH.fail();

  rc_return VH.maybeFail(T.CustomName().verify(VH));
}

RecursiveCoroutine<bool> TypeDefinition::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (VH.isVerified(*this))
    rc_return true;

  // Ensure we have not infinite recursion
  if (VH.isVerificationInProgress(*this))
    rc_return VH.fail();

  VH.verificationInProgress(*this);

  bool Result = false;

  // We could use upcast() but we'd need to workaround coroutines.
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

  std::set<Identifier> Names;
  for (const model::UpcastableTypeDefinition &Definition : TypeDefinitions()) {
    // All types on their own should verify
    if (not Definition.get()->verify(VH))
      return VH.fail();

    // Ensure the names are unique
    auto Name = Definition->name();
    if (not Names.insert(Name).second)
      return VH.fail(Twine("Multiple types with the following name: ") + Name);
  }

  return true;
}

//
// Binary
//

bool Binary::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // First of all, verify the global namespace: we need to fully populate it
  // before we can verify namespaces with smaller scopes
  if (not verifyGlobalNamespace(VH))
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

//
// And the wrappers
//

bool Type::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool CallSitePrototype::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Function::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool DynamicFunction::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Section::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Relocation::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Segment::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool EnumEntry::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool StructField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool UnionField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool NamedTypedRegister::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool TypeDefinition::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Binary::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

} // namespace model
