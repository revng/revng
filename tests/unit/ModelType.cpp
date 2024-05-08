//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <bit>

#define BOOST_TEST_MODULE ModelType
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/ABI.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Types.h"

using namespace model;

using llvm::cast;
using llvm::Twine;

using model::PrimitiveTypeKind::Signed;
using model::PrimitiveTypeKind::Void;

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
  return T->Types() == Deserialized->Types();
}

BOOST_AUTO_TEST_CASE(PrimitiveTypes) {
  revng_check(PrimitiveType(PrimitiveTypeKind::Void, 0).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Unsigned, 1).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Unsigned, 2).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Unsigned, 4).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Unsigned, 8).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Unsigned, 16).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Signed, 1).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Signed, 2).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Signed, 4).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Signed, 8).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Signed, 16).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Float, 2).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Float, 4).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Float, 8).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Float, 10).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Float, 12).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Float, 16).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Generic, 1).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Generic, 2).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Generic, 4).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Generic, 8).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Generic, 10).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Generic, 12).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Generic, 16).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Number, 1).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Number, 2).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Number, 4).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Number, 8).verify(true));
  revng_check(PrimitiveType(PrimitiveTypeKind::Number, 16).verify(true));
  auto PointerOrNumberKind = PrimitiveTypeKind::PointerOrNumber;
  revng_check(PrimitiveType(PointerOrNumberKind, 1).verify(true));
  revng_check(PrimitiveType(PointerOrNumberKind, 2).verify(true));
  revng_check(PrimitiveType(PointerOrNumberKind, 4).verify(true));
  revng_check(PrimitiveType(PointerOrNumberKind, 8).verify(true));
  revng_check(PrimitiveType(PointerOrNumberKind, 16).verify(true));

  for (uint8_t ByteSize = 0; ByteSize < 20; ++ByteSize) {

    using namespace std::string_literals;

    auto Unsigned = PrimitiveType(PrimitiveTypeKind::Unsigned, ByteSize);
    auto Signed = PrimitiveType(PrimitiveTypeKind::Signed, ByteSize);
    auto Number = PrimitiveType(PrimitiveTypeKind::Number, ByteSize);
    auto PointerOrNumber = PrimitiveType(PointerOrNumberKind, ByteSize);

    if (std::has_single_bit(ByteSize)) {
      revng_check(Signed.verify(true));
      revng_check(Unsigned.verify(true));
      revng_check(Number.verify(true));
      revng_check(PointerOrNumber.verify(true));
      auto ExpectedName = ("uint" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Unsigned.name() == ExpectedName);
      ExpectedName = ("int" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Signed.name() == ExpectedName);
      ExpectedName = ("number" + Twine(8 * ByteSize) + "_t").str();
      revng_check(Number.name() == ExpectedName);
      ExpectedName = ("pointer_or_number" + Twine(8 * ByteSize) + "_t").str();
      revng_check(PointerOrNumber.name() == ExpectedName);
    } else {
      revng_check(not Signed.verify(false));
      revng_check(not Unsigned.verify(false));
      revng_check(not Number.verify(false));
      revng_check(not PointerOrNumber.verify(false));
    }
  }

  for (uint8_t ByteSize = 0; ByteSize < 20; ++ByteSize) {
    using namespace std::string_literals;

    auto Float = PrimitiveType(PrimitiveTypeKind::Float, ByteSize);
    auto G = PrimitiveType(PrimitiveTypeKind::Generic, ByteSize);
    if (ByteSize == 2 or ByteSize == 4 or ByteSize == 8 or ByteSize == 10
        or ByteSize == 12 or ByteSize == 16) {
      revng_check(Float.verify(true));
      revng_check(Float.name() == ("float" + Twine(8 * ByteSize) + "_t").str());
      revng_check(G.verify(true));
      revng_check(G.name() == ("generic" + Twine(8 * ByteSize) + "_t").str());
    } else {
      revng_check(not Float.verify(false));
      if (ByteSize == 1)
        revng_check(G.verify(true));
      else
        revng_check(not G.verify(false));
    }
  }

  revng_check(not PrimitiveType::fromName("foalt32_t"));
  revng_check(not PrimitiveType::fromName("generic7_t"));
  revng_check(not PrimitiveType::fromName("generic8"));
  revng_check(PrimitiveType::fromName("generic8_t")->name() == "generic8_t");
  revng_check(PrimitiveType::fromName("float64_t")->name() == "float64_t");
  revng_check(PrimitiveType::fromName("number128_t")->name() == "number128_t");
}

BOOST_AUTO_TEST_CASE(EnumTypes) {
  revng_check(not EnumType().verify(false));

  TupleTree<model::Binary> T;

  auto Int32 = T->getPrimitiveType(Signed, 4);

  TypePath EnumPath = T->makeType<EnumType>().second;
  auto *Enum = cast<EnumType>(EnumPath.get());
  revng_check(T->Types().size() == 2);

  // The enum does not verify if we don't define a valid underlying type and
  // at least one enum entry
  auto Int32QT = model::QualifiedType(Int32, {});
  Enum->UnderlyingType() = Int32QT;
  revng_check(not Enum->verify(false));
  revng_check(not T->verify(false));

  // With a valid underlying type and at least one entry we're good, but we
  // have to initialize all the cross references in the tree.
  EnumEntry Entry = EnumEntry{ 0 };
  Entry.CustomName() = "value0";
  revng_check(Entry.verify(true));

  revng_check(Enum->Entries().insert(Entry).second);
  revng_check(Enum->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // We cannot insert other entries with the same value, but we can insert new
  // entries with different values.
  revng_check(Enum->Entries().size() == 1);
  revng_check(not Enum->Entries().insert(EnumEntry{ 0 }).second);
  revng_check(Enum->Entries().size() == 1);
  revng_check(Enum->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  revng_check(Enum->verify(true));
  revng_check(T->verify(true));
  revng_check(Enum->Entries().insert(EnumEntry{ 1 }).second);
  revng_check(Enum->Entries().size() == 2);
  revng_check(Enum->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Inserting two entries with the same name succceds but it's bad.
  EnumEntry Entry1{ 5 };
  Entry1.CustomName() = "some_value";
  revng_check(Enum->Entries().insert(Entry1).second);
  revng_check(Enum->Entries().size() == 3);
  revng_check(Enum->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));
  EnumEntry Entry2{ 7 };
  Entry2.CustomName() = "some_value";
  revng_check(Enum->Entries().insert(Entry2).second);
  revng_check(Enum->Entries().size() == 4);
  revng_check(not T->verify(false));
  // But if we remove the duplicated entry we're good again
  revng_check(Enum->Entries().erase(7));
  revng_check(Enum->Entries().size() == 3);
  revng_check(Enum->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // But if we break the underlying, making it point to a type that does not
  // exist, we're not good anymore
  auto BrokenPath = TypePath::fromString(T.get(), "/Types/42-TypedefType");
  Enum->UnderlyingType() = { BrokenPath, {} };
  revng_check(not Enum->verify(false));
  revng_check(not T->verify(false));

  // Also we set the underlying type to a valid type, but that is not a
  // primitive integer type, we are not good
  auto PathToNonInt = T->getTypePath(Enum);
  Enum->UnderlyingType() = { PathToNonInt, {} };
  revng_check(not Enum->verify(false));
  revng_check(not T->verify(false));

  // If we put back the proper underlying type it verifies.
  Enum->UnderlyingType() = Int32QT;
  revng_check(Enum->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // But if we clear the entries it does not verify anymore
  Enum->Entries().clear();
  revng_check(not Enum->verify(false));
  revng_check(not T->verify(false));
}

BOOST_AUTO_TEST_CASE(TypedefTypes) {
  TupleTree<model::Binary> T;

  auto Int32 = T->getPrimitiveType(Signed, 4);

  // Insert the typedef

  TypePath TypedefPath = T->makeType<TypedefType>().second;
  auto *Typedef = cast<TypedefType>(TypedefPath.get());
  revng_check(T->Types().size() == 2);

  // The pid_t typedef refers to the int32_t
  Typedef->UnderlyingType() = { Int32, {} };
  revng_check(Typedef->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding qualifiers the typedef still verifies
  Typedef->UnderlyingType().Qualifiers().push_back(Qualifier::createConst());
  revng_check(Typedef->verify(true));
  revng_check(T->verify(true));
  Typedef->UnderlyingType().Qualifiers().push_back(Qualifier::createArray(42));
  revng_check(Typedef->verify(true));
  revng_check(T->verify(true));
  Typedef->UnderlyingType().Qualifiers().push_back(Qualifier::createPointer(8));
  revng_check(Typedef->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Removing qualifiers, the typedef still verifies
  Typedef->UnderlyingType().Qualifiers().clear();
  revng_check(Typedef->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // If the underlying type is the type itself something is broken
  Typedef->UnderlyingType().UnqualifiedType() = T->getTypePath(Typedef);
  revng_check(not Typedef->verify(false));
  revng_check(not T->verify(false));
}

BOOST_AUTO_TEST_CASE(StructTypes) {
  revng_check(not StructType().verify(false));

  TupleTree<model::Binary> T;

  auto Int32 = T->getPrimitiveType(Signed, 4);
  auto VoidT = T->getPrimitiveType(Void, 0);

  // Insert the struct
  TypePath StructPath = T->makeType<StructType>().second;
  auto *Struct = cast<StructType>(StructPath.get());
  revng_check(T->Types().size() == 3);

  // Let's make it large, so that we can play around with fields.
  Struct->Size() = 1024;

  // Insert field in the struct
  StructField Field0 = StructField{ 0 };
  Field0.Type() = { Int32, {} };
  revng_check(Struct->Fields().insert(Field0).second);
  revng_check(Struct->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid
  StructField Field1 = StructField{ 4 };
  Field1.Type() = { Int32, {} };
  revng_check(Struct->Fields().insert(Field1).second);
  revng_check(Struct->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Inserting fails if the index is already present
  StructField Field1Bis = StructField{ 4 };
  Field1Bis.Type() = { Int32, {} };
  revng_check(not Struct->Fields().insert(Field1Bis).second);
  revng_check(Struct->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Assigning succeeds if even if an index is already present
  StructField Field1Ter = StructField{ 4 };
  Field1Ter.Type() = { Int32, {} };
  Field1Ter.CustomName() = "fld1ter";
  revng_check(not Struct->Fields().insert_or_assign(Field1Ter).second);
  revng_check(Struct->verify(true));
  revng_check(Struct->Fields().at(4).CustomName() == "fld1ter");
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field whose position is not consecutive to others builds a
  // struct that is valid
  StructField AnotherField = StructField{ 128 };
  AnotherField.Type() = { Int32, {} };
  revng_check(Struct->Fields().insert(AnotherField).second);
  revng_check(Struct->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field that overlaps with another is not valid
  StructField Overlap = StructField{ 129 };
  Overlap.Type() = { Int32, {} };
  revng_check(Struct->Fields().insert(Overlap).second);
  revng_check(not Struct->verify(false));
  revng_check(not T->verify(false));

  // Removing the overlapping field fixes the struct
  revng_check(Struct->Fields().erase(129));
  revng_check(Struct->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Erasing a field that's not there fails
  revng_check(not Struct->Fields().erase(129));
  revng_check(Struct->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Shrinking the size does not break the struct
  Struct->Size() = 132;
  revng_check(Struct->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  for (int I = 0; I < 132; ++I) {
    // But shrinking too much breaks it again
    Struct->Size() = I;
    revng_check(not Struct->verify(false));
    revng_check(not T->verify(false));
  }

  // Fixing the size fixes the struct
  Struct->Size() = 132;
  revng_check(Struct->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Struct without fields are valid as long as their size is not zero
  Struct->Fields().clear();
  revng_check(Struct->verify(false));
  revng_check(T->verify(false));
  Struct->Size() = 0;
  revng_check(not Struct->verify(false));
  revng_check(not T->verify(false));

  // Put the size back to a large value for the other tests.
  Struct->Size() = 100;
  revng_check(Struct->verify(false));
  revng_check(T->verify(false));

  // Struct x cannot have a field with type x
  Struct->Fields().clear();
  StructField Same = StructField{ 0 };
  Same.Type() = { T->getTypePath(Struct), {} };
  revng_check(Struct->Fields().insert(Same).second);
  revng_check(not Struct->verify(false));
  revng_check(not T->verify(false));

  // Adding a void field is not valid
  Struct->Fields().clear();
  StructField VoidField = StructField{ 0 };
  VoidField.Type() = { VoidT, {} };
  revng_check(Struct->Fields().insert(VoidField).second);
  revng_check(not Struct->verify(false));
  revng_check(not T->verify(false));
}

BOOST_AUTO_TEST_CASE(UnionTypes) {
  revng_check(not UnionType().verify(false));

  TupleTree<model::Binary> T;

  auto Int32 = T->getPrimitiveType(Signed, 4);
  auto Int64 = T->getPrimitiveType(Signed, 8);
  auto VoidT = T->getPrimitiveType(Void, 0);

  // Insert the union
  TypePath UnionPath = T->makeType<UnionType>().second;
  auto *Union = cast<UnionType>(UnionPath.get());
  revng_check(T->Types().size() == 4);

  // Insert field in the struct
  UnionField Field0(0);
  Field0.Type() = { Int32, {} };
  revng_check(Union->Fields().insert(Field0).second);
  revng_check(Union->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid
  {
    UnionField Field1(1);
    Field1.Type() = { Int64, {} };
    Field1.CustomName() = "fld1";
    const auto [It, New] = Union->Fields().insert(std::move(Field1));
    revng_check(New);
  }
  revng_check(Union->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  {
    // Assigning another field in a different position with a duplicated name
    // succeeds, but verification fails.
    UnionField Field1(2);
    Field1.Type() = { Int32, {} };
    Field1.CustomName() = "fld1";
    const auto [It, New] = Union->Fields().insert(std::move(Field1));
    revng_check(New);
    revng_check(Union->Fields().at(It->Index()).CustomName() == "fld1");
    revng_check(not Union->verify(false));
    revng_check(not T->verify(false));

    // But removing goes back to good again
    revng_check(Union->Fields().erase(It->Index()));
    revng_check(Union->verify(true));
    revng_check(T->verify(true));
    revng_check(checkSerialization(T));
  }

  // Union without fields are invalid
  Union->Fields().clear();
  revng_check(not Union->verify(false));
  revng_check(not T->verify(false));

  // Union x cannot have a field with type x
  Union->Fields().clear();
  UnionField Same;
  Same.Type() = { T->getTypePath(Union), {} };
  revng_check(Union->Fields().insert(Same).second);
  revng_check(not Union->verify(false));
  revng_check(not T->verify(false));

  // Adding a void field is not valid
  Union->Fields().clear();
  UnionField VoidField;
  VoidField.Type() = { VoidT, {} };
  revng_check(Union->Fields().insert(VoidField).second);
  revng_check(not Union->verify(false));
  revng_check(not T->verify(false));
}

BOOST_AUTO_TEST_CASE(CABIFunctionTypes) {
  TupleTree<model::Binary> T;

  auto Int32 = T->getPrimitiveType(Signed, 4);
  auto VoidT = T->getPrimitiveType(Void, 0);

  // Create a C-like function type
  TypePath FunctionPath = T->makeType<CABIFunctionType>().second;
  auto *FunctionType = cast<CABIFunctionType>(FunctionPath.get());
  FunctionType->ABI() = model::ABI::SystemV_x86_64;
  revng_check(T->Types().size() == 3);

  revng_check(FunctionType->trySize().value() == 0);

  // Insert argument in the function type
  Argument Arg0{ 0 };
  Arg0.Type() = { Int32, {} };
  const auto &[InsertedArgIt, New] = FunctionType->Arguments().insert(Arg0);
  revng_check(InsertedArgIt != FunctionType->Arguments().end());
  revng_check(New);

  // Verification fails due to missing return type
  revng_check(not FunctionType->verify(false));
  revng_check(not T->verify(false));

  QualifiedType RetTy{ Int32, {} };
  FunctionType->ReturnType() = RetTy;
  revng_check(FunctionType->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Adding a new field is valid, and we can have a function type with an
  // argument of the same type of itself.
  Argument Arg1{ 1 };
  Arg1.Type() = { Int32, {} };
  revng_check(FunctionType->Arguments().insert(Arg1).second);
  revng_check(FunctionType->verify(true));
  revng_check(checkSerialization(T));

  // Inserting an ArgumentType in a position that is already taken fails
  Argument Arg1Bis{ 1 };
  Arg1Bis.Type() = { Int32, {} };
  revng_check(not FunctionType->Arguments().insert(Arg1Bis).second);
  revng_check(FunctionType->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // Assigning an ArgumentType in a position that is already taken succeeds
  revng_check(not FunctionType->Arguments().insert_or_assign(Arg1Bis).second);
  revng_check(FunctionType->verify(true));
  auto &ArgT = FunctionType->Arguments().at(1);
  revng_check(ArgT.Type().UnqualifiedType() == Int32);
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));

  // FunctionType without argument are valid
  FunctionType->Arguments().clear();
  revng_check(FunctionType->verify(true));
  revng_check(T->verify(true));
  revng_check(checkSerialization(T));
}

BOOST_AUTO_TEST_CASE(RawFunctionTypes) {
  TupleTree<model::Binary> T;

  auto Primitive64 = T->getPrimitiveType(model::PrimitiveTypeKind::Generic, 4);
  QualifiedType Generic64 = { Primitive64, {} };

  auto RAFPointer = T->makeType<model::RawFunctionType>().second;
  auto *RAF = cast<model::RawFunctionType>(RAFPointer.get());

  revng_check(RAF->verify(true));

  //
  // Test non-scalar argument
  //
  {
    model::TypedRegister RAXArgument(model::Register::rax_x86_64);
    RAXArgument.Type() = { Primitive64, { { QualifierKind::Array, 10 } } };
    revng_check(not RAXArgument.verify(false));
  }

  //
  // Add two arguments
  //
  {
    model::NamedTypedRegister RDIArgument(model::Register::rdi_x86_64);
    RDIArgument.Type() = Generic64;
    revng_check(RDIArgument.verify(true));
    RAF->Arguments().insert(RDIArgument);
    revng_check(RAF->verify(true));

    model::NamedTypedRegister RSIArgument(model::Register::rsi_x86_64);
    RSIArgument.Type() = Generic64;
    RSIArgument.CustomName() = "Second";
    revng_check(RSIArgument.verify(true));
    RAF->Arguments().insert(RSIArgument);
    revng_check(RAF->verify(true));
  }

  // Add a return value
  {
    model::NamedTypedRegister RAXReturnValue(model::Register::rax_x86_64);
    RAXReturnValue.Type() = Generic64;
    revng_check(RAXReturnValue.verify(true));
    RAF->ReturnValues().insert(RAXReturnValue);
    revng_check(RAF->verify(true));
  }
}

BOOST_AUTO_TEST_CASE(QualifiedTypes) {
  TupleTree<model::Binary> T;
  auto Void = T->getPrimitiveType(model::PrimitiveTypeKind::Void, 0);
  auto Generic64 = T->getPrimitiveType(model::PrimitiveTypeKind::Generic, 8);

  revng_check(Void.get()->verify(true));
  revng_check(not Void.get()->size().has_value());

  revng_check(Generic64.get()->verify(true));
  revng_check(*Generic64.get()->size() == 8);

  QualifiedType VoidPointer = { Void,
                                { { model::QualifierKind::Pointer, 4 } } };
  revng_check(VoidPointer.verify(true));

  model::Qualifier Pointer64Qualifier{ model::QualifierKind::Pointer, 8 };
  QualifiedType Generic64Pointer = { Void, { Pointer64Qualifier } };
  revng_check(Generic64Pointer.verify(true));

  QualifiedType DoublePointer = { Void,
                                  { Pointer64Qualifier, Pointer64Qualifier } };
  revng_check(DoublePointer.verify(true));

  QualifiedType WeirdSizedPointer = {
    Void, { { model::QualifierKind::Pointer, 7 } }
  };
  revng_check(not WeirdSizedPointer.verify(false));

  model::Qualifier ConstQualifier{ model::QualifierKind::Const, 0 };
  QualifiedType ConstVoid = { Void, { ConstQualifier } };
  revng_check(ConstVoid.verify(true));

  QualifiedType ConstConstVoid = { Void, { ConstQualifier, ConstQualifier } };
  revng_check(not ConstConstVoid.verify(false));

  QualifiedType ConstPointerConstVoid = {
    Void, { ConstQualifier, Pointer64Qualifier, ConstQualifier }
  };
  revng_check(ConstPointerConstVoid.verify(true));

  model::Qualifier TenElementsArray{ model::QualifierKind::Array, 10 };
  QualifiedType VoidArray = { Void, { TenElementsArray } };
  revng_check(not VoidArray.verify(false));

  QualifiedType VoidPointerArray = { Void,
                                     { TenElementsArray, Pointer64Qualifier } };
  revng_check(VoidPointerArray.verify(true));

  model::Qualifier ZeroElementsArray{ model::QualifierKind::Array, 0 };
  QualifiedType ZeroSizedVoidArray = { Void, { ZeroElementsArray } };
  revng_check(not ZeroSizedVoidArray.verify(false));
}
