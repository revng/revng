//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <bit>

#define BOOST_TEST_MODULE ModelType
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/ABI.h"
#include "revng/Model/Binary.h"

using namespace model;

using llvm::Twine;

using model::PrimitiveKind::Signed;
using model::PrimitiveKind::Void;

static TupleTree<model::Binary>
serializeDeserialize(const TupleTree<model::Binary> &T) {

  std::string Buffer;
  T.serialize(Buffer);

  auto Deserialized = TupleTree<model::Binary>::deserialize(Buffer);

  std::string OtherBuffer;
  Deserialized->serialize(OtherBuffer);

  return std::move(Deserialized.get());
}

static bool checkSerialization(const TupleTree<model::Binary> &T) {
  revng_check(T->verify(true));
  auto Deserialized = serializeDeserialize(T);
  revng_check(Deserialized->verify(true));
  return T->TypeDefinitions() == Deserialized->TypeDefinitions();
}

BOOST_AUTO_TEST_CASE(PrimitiveTypes) {
  revng_check(PrimitiveType::make(PrimitiveKind::Void, 0)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Unsigned, 1)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Unsigned, 2)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Unsigned, 4)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Unsigned, 8)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Unsigned, 16)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Signed, 1)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Signed, 2)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Signed, 4)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Signed, 8)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Signed, 16)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Float, 2)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Float, 4)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Float, 8)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Float, 10)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Float, 12)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Float, 16)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Generic, 1)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Generic, 2)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Generic, 4)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Generic, 8)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Generic, 10)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Generic, 12)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Generic, 16)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Number, 1)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Number, 2)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Number, 4)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Number, 8)->verify(true));
  revng_check(PrimitiveType::make(PrimitiveKind::Number, 16)->verify(true));
  auto PointerOrNumberKind = PrimitiveKind::PointerOrNumber;
  revng_check(PrimitiveType::make(PointerOrNumberKind, 1)->verify(true));
  revng_check(PrimitiveType::make(PointerOrNumberKind, 2)->verify(true));
  revng_check(PrimitiveType::make(PointerOrNumberKind, 4)->verify(true));
  revng_check(PrimitiveType::make(PointerOrNumberKind, 8)->verify(true));
  revng_check(PrimitiveType::make(PointerOrNumberKind, 16)->verify(true));

  for (uint8_t ByteSize = 0; ByteSize < 20; ++ByteSize) {
    auto Unsigned = PrimitiveType::make(PrimitiveKind::Unsigned, ByteSize);
    auto Signed = PrimitiveType::make(PrimitiveKind::Signed, ByteSize);
    auto Number = PrimitiveType::make(PrimitiveKind::Number, ByteSize);
    auto PointerOrNumber = PrimitiveType::make(PointerOrNumberKind, ByteSize);

    if (std::has_single_bit(ByteSize)) {
      revng_check(Signed->verify(true));
      auto ExpectedName = ("int" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Signed->asPrimitive().getCName() == ExpectedName);

      revng_check(Unsigned->verify(true));
      ExpectedName = ("uint" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Unsigned->asPrimitive().getCName() == ExpectedName);

      revng_check(Number->verify(true));
      ExpectedName = ("number" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Number->asPrimitive().getCName() == ExpectedName);

      revng_check(PointerOrNumber->verify(true));
      ExpectedName = ("pointer_or_number" + Twine(8 * ByteSize) + "_t").str();
      revng_check(PointerOrNumber->asPrimitive().getCName() == ExpectedName);
    } else {
      revng_check(not Signed->verify(false));
      revng_check(not Unsigned->verify(false));
      revng_check(not Number->verify(false));
      revng_check(not PointerOrNumber->verify(false));
    }
  }

  dbg << "1\n";

  for (uint8_t ByteSize = 0; ByteSize < 20; ++ByteSize) {
    auto Fl = PrimitiveType::make(PrimitiveKind::Float, ByteSize);
    auto G = PrimitiveType::make(PrimitiveKind::Generic, ByteSize);
    if (ByteSize == 2 or ByteSize == 4 or ByteSize == 8 or ByteSize == 10
        or ByteSize == 12 or ByteSize == 16) {
      revng_check(Fl->verify(true));
      auto ExpectedName = ("float" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Fl->asPrimitive().getCName() == ExpectedName);

      revng_check(G->verify(true));
      ExpectedName = ("generic" + Twine(8 * ByteSize) + "_t").str();
      revng_check(G->asPrimitive().getCName() == ExpectedName);
    } else {
      revng_check(not Fl->verify(false));
      if (ByteSize == 1)
        revng_check(G->verify(true));
      else
        revng_check(not G->verify(false));
    }
  }

  dbg << "2\n";

  revng_check(not PrimitiveType::fromCName("foalt32_t"));
  revng_check(not PrimitiveType::fromCName("generic7_t"));
  revng_check(not PrimitiveType::fromCName("generic8"));

  dbg << "3\n";

  model::UpcastableType Primitive = PrimitiveType::fromCName("generic8_t");

  dbg << "3.001\n";
  Primitive->asPrimitive();
  dbg << "3.002\n";
  Primitive->asPrimitive().getCName();

  dbg << "3.01\n";
  revng_check(Primitive->asPrimitive().getCName() == "generic8_t");
  dbg << "3.02\n";
  Primitive = PrimitiveType::fromCName("float64_t");
  dbg << "3.03\n";
  revng_check(Primitive->asPrimitive().getCName() == "float64_t");
  dbg << "3.04\n";
  Primitive = PrimitiveType::fromCName("number128_t");
  dbg << "3.05\n";
  revng_check(Primitive->asPrimitive().getCName() == "number128_t");

  dbg << "4\n";
}

BOOST_AUTO_TEST_CASE(EnumTypes) {
  revng_check(not EnumDefinition().verify(false));

  TupleTree<model::Binary> T;

  auto [EnumDefinition, EnumType] = T->makeEnumDefinition();
  revng_check(T->TypeDefinitions().size() == 1);

  // The enum does not verify if we don't define a valid underlying type and
  // at least one enum entry
  EnumDefinition.UnderlyingType() = model::PrimitiveType::make(Signed, 4);
  revng_check(not EnumDefinition.verify(false));
  revng_check(not T->verify(false));

  // With a valid underlying type and at least one entry we're good, but we
  // have to initialize all the cross references in the tree.
  EnumEntry Entry = EnumEntry{ 0 };
  Entry.CustomName() = "value0";
  revng_check(Entry.verify(true));

  revng_check(EnumDefinition.Entries().insert(Entry).second);
  revng_check(EnumDefinition.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // We cannot insert other entries with the same value, but we can insert new
  // entries with different values.
  revng_check(EnumDefinition.Entries().size() == 1);
  revng_check(not EnumDefinition.Entries().insert(EnumEntry{ 0 }).second);
  revng_check(EnumDefinition.Entries().size() == 1);
  revng_check(EnumDefinition.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  revng_check(EnumDefinition.verify(true));
  revng_check(T->verify(true));
  revng_check(EnumDefinition.Entries().insert(EnumEntry{ 1 }).second);
  revng_check(EnumDefinition.Entries().size() == 2);
  revng_check(EnumDefinition.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Inserting two entries with the same name succceds but it's bad.
  EnumEntry Entry1{ 5 };
  Entry1.CustomName() = "some_value";
  revng_check(EnumDefinition.Entries().insert(Entry1).second);
  revng_check(EnumDefinition.Entries().size() == 3);
  revng_check(EnumDefinition.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));
  EnumEntry Entry2{ 7 };
  Entry2.CustomName() = "some_value";
  revng_check(EnumDefinition.Entries().insert(Entry2).second);
  revng_check(EnumDefinition.Entries().size() == 4);
  revng_check(not T->verify(false));
  // But if we remove the duplicated entry we're good again
  revng_check(EnumDefinition.Entries().erase(7));
  revng_check(EnumDefinition.Entries().size() == 3);
  revng_check(EnumDefinition.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // But if we break the underlying, making it point to a type that does not
  // exist, we're not good anymore
  auto BrokenPath = DefinitionReference::fromString(T.get(),
                                                    "/TypeDefinitions/"
                                                    "42-TypedefDefinition");
  EnumDefinition.UnderlyingType() = model::DefinedType::make(BrokenPath);
  revng_check(not EnumDefinition.verify(false));
  revng_check(not T->verify(false));

  // Also we set the underlying type to a valid type, but that is not a
  // primitive integer type, we are not good
  EnumDefinition.UnderlyingType() = EnumType.copy();
  revng_check(not EnumDefinition.verify(false));
  revng_check(not T->verify(false));

  // If we put back the proper underlying type it verifies.
  EnumDefinition.UnderlyingType() = model::PrimitiveType::make(Signed, 4);
  revng_check(EnumDefinition.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // But if we clear the entries it does not verify anymore
  EnumDefinition.Entries().clear();
  revng_check(not EnumDefinition.verify(false));
  revng_check(not T->verify(false));
}

BOOST_AUTO_TEST_CASE(TypedefTypes) {
  TupleTree<model::Binary> T;

  // Make a typedef of an int32_t
  auto [Int32, Int32Type] = T->makeTypeDefinition<TypedefDefinition>();
  revng_check(T->TypeDefinitions().size() == 1);
  Int32.UnderlyingType() = model::PrimitiveType::make(Signed, 4);
  Int32.CustomName() = "int_32_typedef";
  revng_check(Int32.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Make it const
  Int32.UnderlyingType() = model::PrimitiveType::makeConst(Signed, 4);
  revng_check(Int32.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Make another typedef, this time a pointer.
  auto [Pointer, PointerType] = T->makeTypeDefinition<TypedefDefinition>();
  revng_check(T->TypeDefinitions().size() == 2);
  Pointer.UnderlyingType() = model::PointerType::make(std::move(Int32Type), 4);
  Pointer.CustomName() = "int_32_pointer_typedef";
  revng_check(Pointer.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Make one more, this time an array
  auto [Array, ArrayType] = T->makeTypeDefinition<TypedefDefinition>();
  revng_check(T->TypeDefinitions().size() == 3);
  Array.UnderlyingType() = model::ArrayType::make(std::move(PointerType), 25);
  Array.CustomName() = "int_32_pointer_array_typedef";
  revng_check(Array.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // If the underlying type is the type itself, something is broken
  auto [Error, ErrorType] = T->makeTypeDefinition<TypedefDefinition>();
  revng_check(T->TypeDefinitions().size() == 4);
  Error.UnderlyingType() = std::move(ErrorType);
  revng_check(not Error.verify(false));
  revng_check(not T->verify(false));
}

BOOST_AUTO_TEST_CASE(StructTypes) {
  revng_check(not StructDefinition().verify(false));

  TupleTree<model::Binary> T;

  // Insert the struct
  auto [Struct, StructType] = T->makeStructDefinition();
  revng_check(T->TypeDefinitions().size() == 1);

  // Let's make it large, so that we can play around with fields.
  Struct.Size() = 1024;

  // Insert field in the struct
  StructField Field0 = StructField{ 0 };
  Field0.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Struct.Fields().insert(Field0).second);
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid
  StructField Field1 = StructField{ 4 };
  Field1.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Struct.Fields().insert(Field1).second);
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Inserting fails if the index is already present
  StructField Field1Bis = StructField{ 4 };
  Field1Bis.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(not Struct.Fields().insert(Field1Bis).second);
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Assigning succeeds if even if an index is already present
  StructField Field1Ter = StructField{ 4 };
  Field1Ter.Type() = model::PrimitiveType::makeSigned(4);
  Field1Ter.CustomName() = "fld1ter";
  revng_check(not Struct.Fields().insert_or_assign(Field1Ter).second);
  revng_check(Struct.verify(true));
  revng_check(Struct.Fields().at(4).CustomName() == "fld1ter");
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field whose position is not consecutive to others builds a
  // struct that is valid
  StructField AnotherField = StructField{ 128 };
  AnotherField.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Struct.Fields().insert(AnotherField).second);
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field that overlaps with another is not valid
  StructField Overlap = StructField{ 129 };
  Overlap.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Struct.Fields().insert(Overlap).second);
  revng_check(not Struct.verify(false));
  revng_check(not T->verify(false));

  // Removing the overlapping field fixes the struct
  revng_check(Struct.Fields().erase(129));
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Erasing a field that's not there fails
  revng_check(not Struct.Fields().erase(129));
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Shrinking the size does not break the struct
  Struct.Size() = 132;
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  for (int I = 0; I < 132; ++I) {
    // But shrinking too much breaks it again
    Struct.Size() = I;
    revng_check(not Struct.verify(false));
    revng_check(not T->verify(false));
  }

  // Fixing the size fixes the struct
  Struct.Size() = 132;
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Struct without fields are valid as long as their size is not zero
  Struct.Fields().clear();
  revng_check(Struct.verify(false));
  revng_check(T->verify(false));
  Struct.Size() = 0;
  revng_check(not Struct.verify(false));
  revng_check(not T->verify(false));

  // Put the size back to a large value for the other tests.
  Struct.Size() = 100;
  revng_check(Struct.verify(false));
  revng_check(T->verify(false));

  // Struct x cannot have a field with type x
  Struct.Fields().clear();
  StructField Same = StructField{ 0 };
  Same.Type() = std::move(StructType);
  revng_check(Struct.Fields().insert(Same).second);
  revng_check(not Struct.verify(false));
  revng_check(not T->verify(false));

  // But it works, if the field is the pointer to the struct.
  auto &Type = Struct.Fields().at(0).Type();
  Type = model::PointerType::make(std::move(Type), 8);
  revng_check(Struct.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a void field is also not valid
  Struct.Fields().clear();
  StructField VoidField = StructField{ 0 };
  VoidField.Type() = model::PrimitiveType::makeVoid();
  revng_check(Struct.Fields().insert(VoidField).second);
  revng_check(not Struct.verify(false));
  revng_check(not T->verify(false));
}

BOOST_AUTO_TEST_CASE(UnionTypes) {
  revng_check(not UnionDefinition().verify(false));

  TupleTree<model::Binary> T;

  // Insert the union
  auto [Union, UnionType] = T->makeUnionDefinition();
  revng_check(T->TypeDefinitions().size() == 1);

  // Insert field in the struct
  UnionField Field0(0);
  Field0.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Union.Fields().insert(Field0).second);
  revng_check(Union.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid
  {
    UnionField Field1(1);
    Field1.Type() = model::PrimitiveType::makeSigned(8);
    Field1.CustomName() = "fld1";
    const auto [It, New] = Union.Fields().insert(std::move(Field1));
    revng_check(New);
  }
  revng_check(Union.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  {
    // Assigning another field in a different position with a duplicated name
    // succeeds, but verification fails.
    UnionField Field1(2);
    Field1.Type() = model::PrimitiveType::makeSigned(4);
    Field1.CustomName() = "fld1";
    const auto [It, New] = Union.Fields().insert(std::move(Field1));
    revng_check(New);
    revng_check(Union.Fields().at(It->Index()).CustomName() == "fld1");
    revng_check(not Union.verify(false));
    revng_check(not T->verify(false));

    // But removing goes back to good again
    revng_check(Union.Fields().erase(It->Index()));
    revng_check(Union.verify(true));
    revng_check(T->verify(true));
    revng_check(checkSerialization(T));
  }

  // Union without fields are invalid
  Union.Fields().clear();
  revng_check(not Union.verify(false));
  revng_check(not T->verify(false));

  // Union x cannot have a field with type x
  Union.Fields().clear();
  UnionField Same;
  Same.Type() = std::move(UnionType);
  revng_check(Union.Fields().insert(Same).second);
  revng_check(not Union.verify(false));
  revng_check(not T->verify(false));

  // But it works, if the field is the pointer to the union.
  auto &Type = Union.Fields().at(0).Type();
  Type = model::PointerType::make(std::move(Type), 8);
  revng_check(Union.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a void field is not valid
  Union.Fields().clear();
  UnionField VoidField;
  VoidField.Type() = model::PrimitiveType::makeVoid();
  revng_check(Union.Fields().insert(VoidField).second);
  revng_check(not Union.verify(false));
  revng_check(not T->verify(false));
}

BOOST_AUTO_TEST_CASE(CABIFunctionTypes) {
  TupleTree<model::Binary> T;

  // Create a C-like function prototype
  auto [Prototype, FunctionType] = T->makeCABIFunctionDefinition();
  Prototype.ABI() = model::ABI::SystemV_x86_64;
  revng_check(T->TypeDefinitions().size() == 1);

  revng_check(FunctionType->size() == std::nullopt);

  // Insert argument in the function type
  Argument Arg0{ 0 };
  Arg0.Type() = model::PrimitiveType::makeSigned(4);
  const auto &[InsertedArgIt, New] = Prototype.Arguments().insert(Arg0);
  revng_check(InsertedArgIt != Prototype.Arguments().end());
  revng_check(New);

  // Prototype is already valid, since missing return value means void.
  revng_check(Prototype.verify(true));
  revng_check(T->verify(true));

  // On the other hand explicit `Void` is not allowed.
  Prototype.ReturnType() = model::PrimitiveType::makeVoid();
  revng_check(not Prototype.verify(false));
  revng_check(not T->verify(false));

  // Setting it to a valid type fixes the prototype.
  Prototype.ReturnType() = model::PrimitiveType::makeSigned(4);
  revng_check(Prototype.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid, and we can have a function type with an
  // argument of the same type of itself.
  Argument Arg1{ 1 };
  Arg1.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(Prototype.Arguments().insert(Arg1).second);
  revng_check(Prototype.verify(true));
  revng_check(checkSerialization(T));

  // Inserting an ArgumentType in a position that is already taken fails
  Argument Arg1Bis{ 1 };
  Arg1Bis.Type() = model::PrimitiveType::makeSigned(4);
  revng_check(not Prototype.Arguments().insert(Arg1Bis).second);
  revng_check(Prototype.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Assigning an ArgumentType in a position that is already taken succeeds
  revng_check(not Prototype.Arguments().insert_or_assign(Arg1Bis).second);
  revng_check(Prototype.verify(true));
  auto &ArgT = Prototype.Arguments().at(1);
  revng_check(ArgT.Type() == model::PrimitiveType::makeSigned(4));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Prototypes without arguments are valid
  Prototype.Arguments().clear();
  revng_check(Prototype.verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));
}

BOOST_AUTO_TEST_CASE(RawFunctionTypes) {
  TupleTree<model::Binary> T;

  auto [Prototype, FunctionType] = T->makeRawFunctionDefinition();
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
    model::NamedTypedRegister RDIArgument(model::Register::rdi_x86_64);
    RDIArgument.Type() = model::PrimitiveType::makeGeneric(8);
    revng_check(RDIArgument.verify(true));
    Prototype.Arguments().insert(RDIArgument);
    revng_check(Prototype.verify(true));

    model::NamedTypedRegister RSIArgument(model::Register::rsi_x86_64);
    RSIArgument.Type() = model::PrimitiveType::makeGeneric(8);
    RSIArgument.CustomName() = "Second";
    revng_check(RSIArgument.verify(true));
    Prototype.Arguments().insert(RSIArgument);
    revng_check(Prototype.verify(true));
  }

  // Add a return value
  {
    model::NamedTypedRegister RAXReturnValue(model::Register::rax_x86_64);
    RAXReturnValue.Type() = model::PrimitiveType::makeGeneric(8);
    revng_check(RAXReturnValue.verify(true));
    Prototype.ReturnValues().insert(RAXReturnValue);
    revng_check(Prototype.verify(true));
  }
}

BOOST_AUTO_TEST_CASE(ArraysAndPointers) {
  TupleTree<model::Binary> T;

  auto Void = model::PrimitiveType::makeVoid();
  revng_check(Void->verify(true));
  revng_check(not Void->size().has_value());

  auto Generic64 = model::PrimitiveType::makeGeneric(8);
  revng_check(Generic64->verify(true));
  revng_check(*Generic64->size() == 8);

  auto VoidPointer = model::PointerType::make(Void.copy(), 4);
  revng_check(VoidPointer->verify(true));

  auto Void64Pointer = model::PointerType::make(Void.copy(), 8);
  revng_check(Void64Pointer->verify(true));

  auto DoublePointer = model::PointerType::make(Void64Pointer.copy(), 8);
  revng_check(DoublePointer->verify(true));

  auto WeirdlySizedPointer = model::PointerType::make(Void.copy(), 7);
  revng_check(not WeirdlySizedPointer->verify(false));

  auto ConstVoid = model::PrimitiveType::makeConstVoid();
  revng_check(ConstVoid->verify(true));

  auto ConstPointerToCVoid = model::PointerType::makeConst(ConstVoid.copy(), 8);
  revng_check(ConstPointerToCVoid->verify(true));

  auto VoidArray = model::ArrayType::make(Void.copy(), 10);
  revng_check(not VoidArray->verify(false));

  auto VoidPointerArray = model::ArrayType::make(Void64Pointer.copy(), 10);
  revng_check(VoidPointerArray->verify(true));

  auto ZeroSizedVoidArray = model::ArrayType::make(Void64Pointer.copy(), 0);
  revng_check(not ZeroSizedVoidArray->verify(false));
}
