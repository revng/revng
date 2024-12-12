//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This file contains unit test code shared between model and MLIR type tests.
// The test file including this file must define the static verification
// functions declared in this file.

#include <bit>

#define BOOST_TEST_MODULE ModelType
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/ABI.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Filters.h"

using namespace model;

using llvm::Twine;

using model::PrimitiveKind::Signed;
using model::PrimitiveKind::Void;

static bool verify(const model::TypeDefinition &ModelType,
                   const model::Binary &Binary,
                   bool Assert);
static bool
verify(const model::Type &ModelType, const model::Binary &Binary, bool Assert);
static bool verify(const model::Binary &Tree, bool Assert);
static bool checkSerialization(const TupleTree<model::Binary> &Tree);

BOOST_AUTO_TEST_CASE(PrimitiveTypes) {
  revng_check(verify(*PrimitiveType::makeVoid(), {}, true));
  revng_check(verify(*PrimitiveType::makeUnsigned(1), {}, true));
  revng_check(verify(*PrimitiveType::makeUnsigned(2), {}, true));
  revng_check(verify(*PrimitiveType::makeUnsigned(4), {}, true));
  revng_check(verify(*PrimitiveType::makeUnsigned(8), {}, true));
  revng_check(verify(*PrimitiveType::makeUnsigned(16), {}, true));
  revng_check(verify(*PrimitiveType::makeSigned(1), {}, true));
  revng_check(verify(*PrimitiveType::makeSigned(2), {}, true));
  revng_check(verify(*PrimitiveType::makeSigned(4), {}, true));
  revng_check(verify(*PrimitiveType::makeSigned(8), {}, true));
  revng_check(verify(*PrimitiveType::makeSigned(16), {}, true));
  revng_check(verify(*PrimitiveType::makeFloat(2), {}, true));
  revng_check(verify(*PrimitiveType::makeFloat(4), {}, true));
  revng_check(verify(*PrimitiveType::makeFloat(8), {}, true));
  revng_check(verify(*PrimitiveType::makeFloat(10), {}, true));
  revng_check(verify(*PrimitiveType::makeFloat(12), {}, true));
  revng_check(verify(*PrimitiveType::makeFloat(16), {}, true));
  revng_check(verify(*PrimitiveType::makeGeneric(1), {}, true));
  revng_check(verify(*PrimitiveType::makeGeneric(2), {}, true));
  revng_check(verify(*PrimitiveType::makeGeneric(4), {}, true));
  revng_check(verify(*PrimitiveType::makeGeneric(8), {}, true));
  revng_check(verify(*PrimitiveType::makeGeneric(10), {}, true));
  revng_check(verify(*PrimitiveType::makeGeneric(12), {}, true));
  revng_check(verify(*PrimitiveType::makeGeneric(16), {}, true));
  revng_check(verify(*PrimitiveType::makeNumber(1), {}, true));
  revng_check(verify(*PrimitiveType::makeNumber(2), {}, true));
  revng_check(verify(*PrimitiveType::makeNumber(4), {}, true));
  revng_check(verify(*PrimitiveType::makeNumber(8), {}, true));
  revng_check(verify(*PrimitiveType::makeNumber(16), {}, true));
  revng_check(verify(*PrimitiveType::makePointerOrNumber(1), {}, true));
  revng_check(verify(*PrimitiveType::makePointerOrNumber(2), {}, true));
  revng_check(verify(*PrimitiveType::makePointerOrNumber(4), {}, true));
  revng_check(verify(*PrimitiveType::makePointerOrNumber(8), {}, true));
  revng_check(verify(*PrimitiveType::makePointerOrNumber(16), {}, true));

  for (uint8_t ByteSize = 0; ByteSize < 20; ++ByteSize) {
    auto Unsigned = PrimitiveType::makeUnsigned(ByteSize);
    auto Signed = PrimitiveType::makeSigned(ByteSize);
    auto Number = PrimitiveType::makeNumber(ByteSize);
    auto PointerOrNumber = PrimitiveType::makePointerOrNumber(ByteSize);

    if (std::has_single_bit(ByteSize)) {
      revng_check(verify(*Signed, {}, true));
      auto ExpectedName = ("int" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Signed->toPrimitive().getCName() == ExpectedName);

      revng_check(verify(*Unsigned, {}, true));
      ExpectedName = ("uint" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Unsigned->toPrimitive().getCName() == ExpectedName);

      revng_check(verify(*Number, {}, true));
      ExpectedName = ("number" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Number->toPrimitive().getCName() == ExpectedName);

      revng_check(verify(*PointerOrNumber, {}, true));
      ExpectedName = ("pointer_or_number" + Twine(8 * ByteSize) + "_t").str();
      revng_check(PointerOrNumber->toPrimitive().getCName() == ExpectedName);
    } else {
      revng_check(not verify(*Signed, {}, false));
      revng_check(not verify(*Unsigned, {}, false));
      revng_check(not verify(*Number, {}, false));
      revng_check(not verify(*PointerOrNumber, {}, false));
    }
  }

  for (uint8_t ByteSize = 0; ByteSize < 20; ++ByteSize) {
    auto Float = PrimitiveType::makeFloat(ByteSize);
    auto G = PrimitiveType::makeGeneric(ByteSize);
    if (ByteSize == 2 or ByteSize == 4 or ByteSize == 8 or ByteSize == 10
        or ByteSize == 12 or ByteSize == 16) {
      revng_check(verify(*Float, {}, true));
      auto ExpectedName = ("float" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Float->toPrimitive().getCName() == ExpectedName);

      revng_check(verify(*G, {}, true));
      ExpectedName = ("generic" + Twine(8 * ByteSize) + "_t").str();
      revng_check(G->toPrimitive().getCName() == ExpectedName);
    } else {
      revng_check(not verify(*Float, {}, false));
      if (ByteSize == 1)
        revng_check(verify(*G, {}, true));
      else
        revng_check(not verify(*G, {}, false));
    }
  }

  revng_check(not PrimitiveType::fromCName("foalt32_t"));
  revng_check(not PrimitiveType::fromCName("generic7_t"));
  revng_check(not PrimitiveType::fromCName("generic8"));
  revng_check(not PrimitiveType::fromCName("void1"));
  revng_check(not PrimitiveType::fromCName("void_"));
  revng_check(not PrimitiveType::fromCName("void_t"));
  revng_check(not PrimitiveType::fromCName("void8_t"));

  model::UpcastableType Primitive = PrimitiveType::fromCName("generic8_t");

  revng_check(Primitive->toPrimitive().getCName() == "generic8_t");
  Primitive = PrimitiveType::fromCName("float64_t");
  revng_check(Primitive->toPrimitive().getCName() == "float64_t");
  Primitive = PrimitiveType::fromCName("number128_t");
  revng_check(Primitive->toPrimitive().getCName() == "number128_t");
  Primitive = PrimitiveType::fromCName("void");
  revng_check(Primitive->toPrimitive().getCName() == "void");
}

BOOST_AUTO_TEST_CASE(EnumTypes) {
  TupleTree<model::Binary> T;

  auto &&[EnumDefinition, EnumType] = T->makeEnumDefinition();
  revng_check(T->TypeDefinitions().size() == 1);

  // The enum does not verify unless it has a valid underlying type and at least
  // one enum entry
  revng_check(not verify(EnumDefinition, *T, false));
  revng_check(not verify(*T, false));
  EnumDefinition.UnderlyingType() = model::PrimitiveType::makeSigned(4);
  revng_check(not verify(EnumDefinition, *T, false));
  revng_check(not verify(*T, false));

  // With a valid underlying type and at least one entry we're good, but we
  // have to initialize all the cross references in the tree.
  EnumEntry Entry = EnumEntry{ 0 };
  Entry.Name() = "value0";
  revng_check(Entry.verify(true));

  revng_check(EnumDefinition.Entries().insert(Entry).second);
  revng_check(verify(EnumDefinition, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // We cannot insert other entries with the same value, but we can insert new
  // entries with different values.
  revng_check(EnumDefinition.Entries().size() == 1);
  revng_check(not EnumDefinition.Entries().insert(EnumEntry{ 0 }).second);
  revng_check(EnumDefinition.Entries().size() == 1);
  revng_check(verify(EnumDefinition, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  revng_check(verify(EnumDefinition, *T, true));
  revng_check(verify(*T, true));
  revng_check(EnumDefinition.Entries().insert(EnumEntry{ 1 }).second);
  revng_check(EnumDefinition.Entries().size() == 2);
  revng_check(verify(EnumDefinition, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Inserting two entries with the same name succeeds but does not verify.
  EnumEntry Entry1{ 5 };
  Entry1.Name() = "some_value";
  revng_check(EnumDefinition.Entries().insert(Entry1).second);
  revng_check(EnumDefinition.Entries().size() == 3);
  revng_check(verify(EnumDefinition, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));
  EnumEntry Entry2{ 7 };
  Entry2.Name() = "some_value";
  revng_check(EnumDefinition.Entries().insert(Entry2).second);
  revng_check(EnumDefinition.Entries().size() == 4);
  revng_check(not verify(*T, false));
  // But if we remove the duplicated entry we're good again
  revng_check(EnumDefinition.Entries().erase(7));
  revng_check(EnumDefinition.Entries().size() == 3);
  revng_check(verify(EnumDefinition, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // But if we break the underlying, making it point to a type that does not
  // exist, we're not good anymore
  auto BrokenPath = DefinitionReference::fromString(T.get(),
                                                    "/TypeDefinitions"
                                                    "/42-TypedefDefinition");
  EnumDefinition.UnderlyingType() = model::DefinedType::make(BrokenPath);
  revng_check(not verify(EnumDefinition, *T, false));
  revng_check(not verify(*T, false));

  // Also we set the underlying type to a valid type, but that is not a
  // primitive integer type, we are not good
  EnumDefinition.UnderlyingType() = EnumType.copy();
  revng_check(not verify(EnumDefinition, *T, false));
  revng_check(not verify(*T, false));

  // If we put back the proper underlying type it verifies.
  EnumDefinition.UnderlyingType() = model::PrimitiveType::make(Signed, 4);
  revng_check(verify(EnumDefinition, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // But if we clear the entries it does not verify anymore
  EnumDefinition.Entries().clear();
  revng_check(not verify(EnumDefinition, *T, false));
  revng_check(not verify(*T, false));
}

BOOST_AUTO_TEST_CASE(TypedefTypes) {
  TupleTree<model::Binary> T;

  // Make a typedef of an int32_t
  auto &&[Int32, Int32Type] = T->makeTypeDefinition<TypedefDefinition>();
  revng_check(T->TypeDefinitions().size() == 1);
  Int32.UnderlyingType() = model::PrimitiveType::make(Signed, 4);
  Int32.Name() = "int_32_typedef";
  revng_check(verify(Int32, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Make it const
  Int32.UnderlyingType() = model::PrimitiveType::makeConst(Signed, 4);
  revng_check(verify(Int32, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Make another typedef, this time a pointer.
  auto &&[Pointer, PointerType] = T->makeTypeDefinition<TypedefDefinition>();
  revng_check(T->TypeDefinitions().size() == 2);
  Pointer.UnderlyingType() = model::PointerType::make(std::move(Int32Type), 4);
  Pointer.Name() = "int_32_pointer_typedef";
  revng_check(verify(Pointer, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Make one more, this time an array
  auto &&[Array, ArrayType] = T->makeTypeDefinition<TypedefDefinition>();
  revng_check(T->TypeDefinitions().size() == 3);
  Array.UnderlyingType() = model::ArrayType::make(std::move(PointerType), 25);
  Array.Name() = "int_32_pointer_array_typedef";
  revng_check(verify(Array, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // If the underlying type is the type itself, something is broken
  auto &&[Error, ErrorType] = T->makeTypeDefinition<TypedefDefinition>();
  revng_check(T->TypeDefinitions().size() == 4);
  Error.UnderlyingType() = std::move(ErrorType);
  revng_check(not verify(Error, *T, false));
  revng_check(not verify(*T, false));
}

BOOST_AUTO_TEST_CASE(StructTypes) {
  revng_check(not verify(StructDefinition(), {}, false));

  TupleTree<model::Binary> T;

  // Insert the struct
  auto &&[Struct, StructType] = T->makeStructDefinition();
  revng_check(T->TypeDefinitions().size() == 1);

  // Struct without size does not verify.
  revng_check(not verify(Struct, *T, false));
  revng_check(not verify(*T, false));

  // Let's make it large, so that we can play around with fields.
  Struct.Size() = 1024;
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));

  // Insert field in the struct
  StructField Field0 = StructField{ 0 };
  Field0.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Struct.Fields().insert(Field0).second);
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid
  StructField Field1 = StructField{ 4 };
  Field1.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Struct.Fields().insert(Field1).second);
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Inserting fails if the index is already present
  StructField Field1Bis = StructField{ 4 };
  Field1Bis.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(not Struct.Fields().insert(Field1Bis).second);
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Assigning succeeds if even if an index is already present
  StructField Field1Ter = StructField{ 4 };
  Field1Ter.Type() = model::PrimitiveType::makeSigned(4);
  Field1Ter.Name() = "fld1ter";
  revng_check(not Struct.Fields().insert_or_assign(Field1Ter).second);
  revng_check(Struct.Fields().at(4).Name() == "fld1ter");
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Adding a new field whose position is not consecutive to others builds a
  // struct that is valid
  StructField AnotherField = StructField{ 128 };
  AnotherField.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Struct.Fields().insert(AnotherField).second);
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Adding a new field that overlaps with another is not valid
  StructField Overlap = StructField{ 129 };
  Overlap.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Struct.Fields().insert(Overlap).second);
  revng_check(not verify(Struct, *T, false));
  revng_check(not verify(*T, false));

  // Removing the overlapping field fixes the struct
  revng_check(Struct.Fields().erase(129));
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Erasing a field that's not there fails
  revng_check(not Struct.Fields().erase(129));
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Shrinking the size does not break the struct
  Struct.Size() = 132;
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  for (int I = 0; I < 132; ++I) {
    // But shrinking too much breaks it again
    Struct.Size() = I;
    revng_check(not verify(Struct, *T, false));
    revng_check(not verify(*T, false));
  }

  // Fixing the size fixes the struct
  Struct.Size() = 132;
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Struct without fields are valid as long as their size is not zero
  Struct.Fields().clear();
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  Struct.Size() = 0;
  revng_check(not verify(Struct, *T, false));
  revng_check(not verify(*T, false));

  // Put the size back to a large value for the other tests.
  Struct.Size() = 100;
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));

  // Struct x cannot have a field with type x
  Struct.Fields().clear();
  StructField Same = StructField{ 0 };
  Same.Type() = std::move(StructType);
  revng_check(Struct.Fields().insert(Same).second);
  revng_check(not verify(Struct, *T, false));
  revng_check(not verify(*T, false));

  // But it works, if the field is the pointer to the struct.
  auto &Type = Struct.Fields().at(0).Type();
  Type = model::PointerType::make(std::move(Type), 8);
  revng_check(verify(Struct, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Adding a void field is not valid either
  Struct.Fields().clear();
  StructField VoidField = StructField{ 0 };
  VoidField.Type() = model::PrimitiveType::makeVoid();
  revng_check(Struct.Fields().insert(VoidField).second);
  revng_check(not verify(Struct, *T, false));
  revng_check(not verify(*T, false));
}

BOOST_AUTO_TEST_CASE(UnionTypes) {
  TupleTree<model::Binary> T;

  // Insert the union
  auto &&[Union, UnionType] = T->makeUnionDefinition();
  revng_check(T->TypeDefinitions().size() == 1);
  revng_check(not verify(Union, *T, false));
  revng_check(not verify(*T, false));

  // Insert field in the struct
  UnionField Field0(0);
  Field0.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Union.Fields().insert(Field0).second);
  revng_check(verify(Union, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid
  {
    UnionField Field1(1);
    Field1.Type() = model::PrimitiveType::makeSigned(8);
    Field1.Name() = "fld1";
    auto &&[It, New] = Union.Fields().insert(std::move(Field1));
    revng_check(New);
  }
  revng_check(verify(Union, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  {
    // Assigning another field in a different position with a duplicated name
    // succeeds, but verification of the binary fails.
    UnionField Field1(2);
    Field1.Type() = model::PrimitiveType::makeSigned(4);
    Field1.Name() = "fld1";
    auto &&[It, New] = Union.Fields().insert(std::move(Field1));
    revng_check(New);
    revng_check(Union.Fields().at(It->Index()).Name() == "fld1");
    revng_check(not verify(*T, false));

    // But removing goes back to good again
    revng_check(Union.Fields().erase(It->Index()));
    revng_check(verify(Union, *T, true));
    revng_check(verify(*T, true));
    revng_check(checkSerialization(T));
  }

  // Union without fields are invalid
  Union.Fields().clear();
  revng_check(not verify(Union, *T, false));
  revng_check(not verify(*T, false));

  // Union x cannot have a field with type x
  Union.Fields().clear();
  UnionField Same;
  Same.Type() = std::move(UnionType);
  revng_check(Union.Fields().insert(Same).second);
  revng_check(not verify(Union, *T, false));
  revng_check(not verify(*T, false));

  // But it works, if the field is the pointer to the union.
  auto &Type = Union.Fields().at(0).Type();
  Type = model::PointerType::make(std::move(Type), 8);
  revng_check(verify(Union, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Adding a void field is not valid either
  Union.Fields().clear();
  UnionField VoidField;
  VoidField.Type() = model::PrimitiveType::makeVoid();
  revng_check(Union.Fields().insert(VoidField).second);
  revng_check(not verify(Union, *T, false));
  revng_check(not verify(*T, false));
}

BOOST_AUTO_TEST_CASE(CABIFunctionTypes) {
  TupleTree<model::Binary> T;
  T->Architecture() = Architecture::x86_64;

  // Create a C-like function prototype
  auto &&[Prototype, FunctionType] = T->makeCABIFunctionDefinition();
  Prototype.ABI() = model::ABI::SystemV_x86_64;
  revng_check(T->TypeDefinitions().size() == 1);

  revng_check(Prototype.size() == std::nullopt);
  revng_check(FunctionType->size() == std::nullopt);

  // Insert argument in the function type
  const auto &Arg0 = Prototype.addArgument(model::PrimitiveType::makeSigned(4));

  // Prototype is already valid, since missing return value means void.
  revng_check(verify(Prototype, *T, true));
  revng_check(verify(*T, true));

  // On the other hand explicit `Void` is not allowed.
  Prototype.ReturnType() = model::PrimitiveType::makeVoid();
  revng_check(not verify(Prototype, *T, false));
  revng_check(not verify(*T, false));

  // Setting it to a valid type fixes the prototype.
  Prototype.ReturnType() = model::PrimitiveType::makeSigned(4);
  revng_check(verify(Prototype, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid, and we can have a function type with an
  // argument of the same type of itself.
  Argument Arg1{ 1 };
  Arg1.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Prototype.Arguments().insert(Arg1).second);
  revng_check(verify(Prototype, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Inserting an ArgumentType in a position that is already taken fails
  Argument Arg1Bis{ 1 };
  Arg1Bis.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(not Prototype.Arguments().insert(Arg1Bis).second);
  revng_check(verify(Prototype, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Assigning an ArgumentType in a position that is already taken succeeds
  revng_check(not Prototype.Arguments().insert_or_assign(Arg1Bis).second);
  auto &ArgT = Prototype.Arguments().at(1);
  revng_check(ArgT.Type() == model::PrimitiveType::makeSigned(4));
  revng_check(verify(Prototype, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));

  // Prototypes without arguments are valid
  Prototype.Arguments().clear();
  revng_check(verify(Prototype, *T, true));
  revng_check(verify(*T, true));
  revng_check(checkSerialization(T));
}

BOOST_AUTO_TEST_CASE(RawFunctionTypes) {
  TupleTree<model::Binary> T;
  T->Architecture() = Architecture::x86_64;

  auto &&[Prototype, FunctionType] = T->makeRawFunctionDefinition();
  Prototype.Architecture() = Architecture::x86_64;
  revng_check(Prototype.verify(true));

  //
  // Test non-scalar argument
  //
  {
    model::NamedTypedRegister RAXArgument(model::Register::rax_x86_64);
    model::UpcastableType Primitive32 = model::PrimitiveType::makeGeneric(8);
    RAXArgument.Type() = model::ArrayType::make(std::move(Primitive32), 10);
    revng_check(not RAXArgument.verify(false));
  }

  //
  // Add two arguments
  //
  {
    auto &RDIArg = Prototype.addArgument(model::Register::rdi_x86_64,
                                         model::PrimitiveType::makeGeneric(8));
    revng_check(RDIArg.verify(true));
    revng_check(verify(Prototype, *T, true));
    revng_check(verify(*T, true));

    auto &RSIArg = Prototype.addArgument(model::Register::rsi_x86_64,
                                         model::PrimitiveType::makeGeneric(8));
    revng_check(RSIArg.verify(true));
    RSIArg.Name() = "Second";
    revng_check(RSIArg.verify(true));
    revng_check(verify(Prototype, *T, true));
    revng_check(verify(*T, true));
  }

  // Add a return value
  {
    auto &RV = Prototype.addReturnValue(model::Register::rax_x86_64,
                                        model::PrimitiveType::makeGeneric(8));
    revng_check(RV.verify(true));
    revng_check(verify(Prototype, *T, true));
    revng_check(verify(*T, true));
  }
}

BOOST_AUTO_TEST_CASE(ArraysAndPointers) {
  auto Void = model::PrimitiveType::makeVoid();
  revng_check(verify(*Void, {}, true));
  revng_check(not Void->size().has_value());

  auto Generic64 = model::PrimitiveType::makeGeneric(8);
  revng_check(verify(*Generic64, {}, true));
  revng_check(*Generic64->size() == 8);

  auto VoidPointer = model::PointerType::make(Void.copy(), 4);
  revng_check(verify(*VoidPointer, {}, true));

  auto Void64Pointer = model::PointerType::make(Void.copy(), 8);
  revng_check(verify(*Void64Pointer, {}, true));

  auto DoublePointer = model::PointerType::make(Void64Pointer.copy(), 8);
  revng_check(verify(*DoublePointer, {}, true));

  auto WeirdlySizedPointer = model::PointerType::make(Void.copy(), 7);
  revng_check(not verify(*WeirdlySizedPointer, {}, false));

  auto ConstVoid = model::PrimitiveType::makeConstVoid();
  revng_check(verify(*ConstVoid, {}, true));

  auto ConstPointerToCVoid = model::PointerType::makeConst(ConstVoid.copy(), 8);
  revng_check(verify(*ConstPointerToCVoid, {}, true));

  auto VoidArray = model::ArrayType::make(Void.copy(), 10);
  revng_check(not verify(*VoidArray, {}, false));

  auto VoidPointerArray = model::ArrayType::make(Void64Pointer.copy(), 10);
  revng_check(verify(*VoidPointerArray, {}, true));

  auto ZeroSizedVoidArray = model::ArrayType::make(Void64Pointer.copy(), 0);
  revng_check(not verify(*ZeroSizedVoidArray, {}, false));
}

BOOST_AUTO_TEST_CASE(TypeComparison) {
  // Basic sanity checks
  using PT = model::PrimitiveType;
  revng_check(PT::makeUnsigned(4) == PT::makeUnsigned(4));
  revng_check(PT::makeUnsigned(4) != PT::makeSigned(4));
  revng_check(PT::makeUnsigned(4) != PT::makeUnsigned(8));

  model::Binary Model;

  auto &&[FirstTD, FirstT] = Model.makeTypedefDefinition(PT::makeUnsigned(4));
  revng_check(FirstTD == *FirstT->tryGetAsDefinition());

  auto &&[SecondTD, SecondT] = Model.makeTypedefDefinition(PT::makeUnsigned(4));
  revng_check(SecondTD == *SecondT->tryGetAsDefinition());

  // Ensure different types are different, even if they only differ by the ID.
  revng_check(FirstTD != SecondTD);
  revng_check(FirstT != SecondT);
  revng_check(FirstTD != SecondTD);

  // Two distinct types are the same as long as they match completely.
  revng_check(FirstT == Model.makeType(FirstTD.key()));
  revng_check(SecondT != Model.makeType(FirstTD.key()));

  // Even if they are wrapped in multiple layers.
  auto FirstArray = model::ArrayType::make(Model.makeType(FirstTD.key()), 10);
  auto SecondArray = model::ArrayType::make(Model.makeType(FirstTD.key()), 10);
  revng_check(FirstArray == SecondArray);

  // But not if those layers differ.
  auto FirstPtr = model::PointerType::make(Model.makeType(FirstTD.key()), 8);
  revng_check(FirstArray != FirstPtr);

  // Consequently, maps and sets of types also work.
  std::map<model::UpcastableType, int> Map;
  FirstT->dump();
  SecondT->dump();
  revng_check(FirstT < SecondT);
  revng_check(Map.try_emplace(FirstT, 42).second);
  revng_check(Map.try_emplace(SecondT, 43).second);
  revng_check(!Map.try_emplace(Model.makeType(FirstTD.key())).second);

  revng_check(Map.size() == 2);
  revng_check(Map.begin()->second == 42);
  revng_check(std::next(Map.begin())->second == 43);
}

BOOST_AUTO_TEST_CASE(Filters) {
  model::Binary Model;
  {
    auto &&[Enum, _] = Model.makeEnumDefinition();
    Enum.UnderlyingType() = model::PrimitiveType::makeUnsigned(4);
    Enum.Entries().emplace(1ull);

    (void) Model.makeTypedefDefinition(model::PrimitiveType::makeSigned(4));
    (void) Model.makeTypedefDefinition(model::PrimitiveType::makeFloat(4));
    (void) Model.makeTypedefDefinition(Model.makeStructDefinition(12).second);
    revng_check(Model.TypeDefinitions().size() == 5);
  }

  size_t ScalarCount = 0;
  for (auto &Scalar : Model.TypeDefinitions() | model::filter::Scalar) {
    revng_check(Scalar->isScalar());
    ++ScalarCount;
  }
  revng_check(ScalarCount == 3);

  auto &&[CFT, _] = Model.makeCABIFunctionDefinition();
  for (auto &[Index, Definition] : llvm::enumerate(Model.TypeDefinitions())) {
    if (auto &&Type = Model.makeType(Definition->key());
        Definition->isPrototype()) {
      auto &&Pointer = model::PointerType::make(std::move(Type), 8);
      CFT.Arguments().emplace(Index, std::move(Pointer));
    } else {
      CFT.Arguments().emplace(Index, std::move(Type));
    }
  }
  revng_check(CFT.Arguments().size() == 6);

  ScalarCount = 0;
  for (auto &Argument : CFT.Arguments() | model::filter::Scalar) {
    revng_check(Argument.Type()->isScalar());
    ++ScalarCount;
  }
  revng_check(ScalarCount == 4);

  uint64_t NonScalarCount = 0;
  for (auto &Argument : CFT.Arguments() | model::filter::NonScalar) {
    revng_check(not Argument.Type()->isScalar());
    ++NonScalarCount;
  }
  revng_check(NonScalarCount == 2);
  revng_check(ScalarCount + NonScalarCount == CFT.Arguments().size());
}
