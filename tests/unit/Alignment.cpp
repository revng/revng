//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <bit>

#define BOOST_TEST_MODULE TypeAlignment
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ABI/Definition.h"
#include "revng/ADT/Concepts.h"
#include "revng/Model/Binary.h"

static std::string printAlignment(uint64_t Alignment) {
  return Alignment != 0 ? std::to_string(Alignment) : "undefined";
}

struct Expected {
  const abi::Definition &ABI;
  uint64_t Alignment;

  explicit Expected(model::ABI::Values ABIName, uint64_t Alignment) :
    ABI(abi::Definition::get(ABIName)), Alignment(Alignment) {}
};

template<typename... Types>
  requires(same_as<Types, Expected> && ...)
void testAlignment(const model::QualifiedType &Type,
                   const Types &...TestCases) {
  for (auto [ABI, Expected] : std::array{ TestCases... }) {
    std::optional<uint64_t> TestResult = ABI.alignment(Type);
    if (TestResult.value_or(0) != Expected) {
      std::string Error = "Alignment run failed for type:\n"
                          + serializeToString(Type) + "ABI ('"
                          + serializeToString(ABI.ABI())
                          + "') reports the alignment of '"
                          + printAlignment(TestResult.value_or(0)) + "', "
                          + "while the expected value is '"
                          + printAlignment(Expected) + "'.\n";
      revng_abort(Error.c_str());
    }
  }
}

namespace Primitive = model::PrimitiveTypeKind;
static model::QualifiedType
makePrimitive(Primitive::Values Kind, std::size_t Size, model::Binary &Binary) {
  return model::QualifiedType(Binary.getPrimitiveType(Kind, Size), {});
}

BOOST_AUTO_TEST_CASE(GenericPrimitiveTypes) {
  TupleTree<model::Binary> Binary;

  testAlignment(makePrimitive(Primitive::Void, 0, *Binary),
                Expected(model::ABI::AAPCS64, 0),
                Expected(model::ABI::AAPCS, 0),
                Expected(model::ABI::SystemZ_s390x, 0),
                Expected(model::ABI::SystemV_x86, 0));

  testAlignment(makePrimitive(Primitive::Generic, 1, *Binary),
                Expected(model::ABI::AAPCS64, 1),
                Expected(model::ABI::AAPCS, 1),
                Expected(model::ABI::SystemZ_s390x, 1),
                Expected(model::ABI::SystemV_x86, 1));

  testAlignment(makePrimitive(Primitive::Generic, 2, *Binary),
                Expected(model::ABI::AAPCS64, 2),
                Expected(model::ABI::AAPCS, 2),
                Expected(model::ABI::SystemZ_s390x, 2),
                Expected(model::ABI::SystemV_x86, 2));

  testAlignment(makePrimitive(Primitive::Generic, 4, *Binary),
                Expected(model::ABI::AAPCS64, 4),
                Expected(model::ABI::AAPCS, 4),
                Expected(model::ABI::SystemZ_s390x, 4),
                Expected(model::ABI::SystemV_x86, 4));

  testAlignment(makePrimitive(Primitive::Generic, 8, *Binary),
                Expected(model::ABI::AAPCS64, 8),
                Expected(model::ABI::AAPCS, 8),
                Expected(model::ABI::SystemZ_s390x, 8),
                Expected(model::ABI::SystemV_x86, 4));

  testAlignment(makePrimitive(Primitive::Generic, 16, *Binary),
                Expected(model::ABI::AAPCS64, 16),
                Expected(model::ABI::AAPCS, 16),
                Expected(model::ABI::SystemZ_s390x, 8),
                Expected(model::ABI::SystemV_x86, 0));
}

BOOST_AUTO_TEST_CASE(FloatingPointPrimitiveTypes) {
  TupleTree<model::Binary> Binary;

  testAlignment(makePrimitive(Primitive::Float, 2, *Binary),
                Expected(model::ABI::AAPCS64, 2),
                Expected(model::ABI::AAPCS, 2),
                Expected(model::ABI::SystemZ_s390x, 0),
                Expected(model::ABI::SystemV_x86, 0));

  testAlignment(makePrimitive(Primitive::Float, 4, *Binary),
                Expected(model::ABI::AAPCS64, 4),
                Expected(model::ABI::AAPCS, 4),
                Expected(model::ABI::SystemZ_s390x, 4),
                Expected(model::ABI::SystemV_x86, 4));

  testAlignment(makePrimitive(Primitive::Float, 8, *Binary),
                Expected(model::ABI::AAPCS64, 8),
                Expected(model::ABI::AAPCS, 8),
                Expected(model::ABI::SystemZ_s390x, 8),
                Expected(model::ABI::SystemV_x86, 4));

  testAlignment(makePrimitive(Primitive::Float, 16, *Binary),
                Expected(model::ABI::AAPCS64, 16),
                Expected(model::ABI::AAPCS, 0),
                Expected(model::ABI::SystemZ_s390x, 8),
                Expected(model::ABI::SystemV_x86, 16));
}

constexpr std::array TestedABIs{ model::ABI::AAPCS64,
                                 model::ABI::AAPCS,
                                 model::ABI::SystemZ_s390x,
                                 model::ABI::SystemV_x86 };

static void compareTypeAlignments(const abi::Definition &ABI,
                                  const model::QualifiedType &LHS,
                                  const model::QualifiedType &RHS) {
  std::optional<uint64_t> Left = ABI.alignment(LHS);
  std::optional<uint64_t> Right = ABI.alignment(RHS);
  if (Left != Right) {
    std::string Error = "Alignment comparison run failed for types:\n"
                        + serializeToString(LHS) + "and\n"
                        + serializeToString(RHS) + "ABI ('"
                        + serializeToString(ABI.ABI())
                        + "') reports the alignment of '"
                        + printAlignment(Left.value_or(0))
                        + "' for the first one, and '"
                        + printAlignment(Right.value_or(0))
                        + "' for the second one.\n";
    revng_abort(Error.c_str());
  }
}

BOOST_AUTO_TEST_CASE(RemainingPrimitiveTypes) {
  TupleTree<model::Binary> Binary;

  constexpr std::array<model::PrimitiveTypeKind::Values, 5> RemainingTypes{
    model::PrimitiveTypeKind::Unsigned,
    model::PrimitiveTypeKind::Signed,
    model::PrimitiveTypeKind::Number,
    model::PrimitiveTypeKind::PointerOrNumber
  };

  for (model::ABI::Values ABIName : TestedABIs) {
    const abi::Definition &ABI = abi::Definition::get(ABIName);
    for (const auto &PrimitiveKind : RemainingTypes) {
      for (uint64_t Size = 1; Size <= 16; Size *= 2) {
        compareTypeAlignments(ABI,
                              makePrimitive(Primitive::Generic, Size, *Binary),
                              makePrimitive(PrimitiveKind, Size, *Binary));
      }
    }
  }
}

static model::UnionField makeUnionField(uint64_t Index,
                                        model::QualifiedType Type) {
  model::UnionField Result;
  Result.Index() = Index;
  Result.Type() = Type;
  return Result;
}

template<typename... FieldTypes>
  requires(std::is_convertible_v<FieldTypes, model::UnionField> && ...)
static model::QualifiedType
makeUnion(model::Binary &Binary, FieldTypes &&...Fields) {
  auto [Union, Path] = Binary.makeType<model::UnionType>();
  (Union.Fields().emplace(std::forward<FieldTypes>(Fields)), ...);
  return model::QualifiedType(Path, {});
}

BOOST_AUTO_TEST_CASE(UnionTypes) {
  TupleTree<model::Binary> Binary;

  using QT = model::QualifiedType;
  QT Int16 = makePrimitive(model::PrimitiveTypeKind::Signed, 2, *Binary);
  QT Int32 = makePrimitive(model::PrimitiveTypeKind::Signed, 4, *Binary);
  QT Int64 = makePrimitive(model::PrimitiveTypeKind::Signed, 8, *Binary);
  QT Float = makePrimitive(model::PrimitiveTypeKind::Float, 4, *Binary);
  QT Double = makePrimitive(model::PrimitiveTypeKind::Float, 8, *Binary);
  QT LongDouble = makePrimitive(model::PrimitiveTypeKind::Float, 16, *Binary);
  QT WeirdLD = makePrimitive(model::PrimitiveTypeKind::Float, 12, *Binary);

  for (model::ABI::Values ABIName : TestedABIs) {
    const abi::Definition &ABI = abi::Definition::get(ABIName);

    QT SimpleUnion = makeUnion(*Binary,
                               makeUnionField(0, Int32),
                               makeUnionField(1, Int64));
    compareTypeAlignments(ABI, Int64, SimpleUnion);

    QT SmallFloatUnion = makeUnion(*Binary,
                                   makeUnionField(0, Int32),
                                   makeUnionField(1, Float));
    compareTypeAlignments(ABI, Int32, SmallFloatUnion);
    compareTypeAlignments(ABI, Float, SmallFloatUnion);

    QT BigFloatUnion = makeUnion(*Binary,
                                 makeUnionField(0, LongDouble),
                                 makeUnionField(1, Int64));
    compareTypeAlignments(ABI, LongDouble, BigFloatUnion);

    QT WeirdFloatUnion = makeUnion(*Binary,
                                   makeUnionField(0, WeirdLD),
                                   makeUnionField(1, Int32));
    compareTypeAlignments(ABI, WeirdLD, WeirdFloatUnion);

    // Test the case where on top of the float field, there's also another
    // stricter-aligned field, which "eclipses" the float one.
    QT EclipsedFloatUnion = makeUnion(*Binary,
                                      makeUnionField(0, Float),
                                      makeUnionField(1, Int64));
    compareTypeAlignments(ABI, Int64, EclipsedFloatUnion);

    QT NestedUnion = makeUnion(*Binary,
                               makeUnionField(0, SmallFloatUnion),
                               makeUnionField(1, Int16));
    compareTypeAlignments(ABI, Int32, NestedUnion);
    compareTypeAlignments(ABI, Float, NestedUnion);
    compareTypeAlignments(ABI, SmallFloatUnion, NestedUnion);

    QT EclipsedNestedUnion = makeUnion(*Binary,
                                       makeUnionField(0, SmallFloatUnion),
                                       makeUnionField(1, Int64));
    compareTypeAlignments(ABI, Int64, EclipsedNestedUnion);
  }
}

static model::StructField makeStructField(uint64_t Offset,
                                          model::QualifiedType Type) {
  model::StructField Result;
  Result.Offset() = Offset;
  Result.Type() = Type;
  return Result;
}

template<typename... FieldTypes>
  requires(std::is_convertible_v<FieldTypes, model::StructField> && ...)
static model::QualifiedType
makeStruct(model::Binary &Binary, FieldTypes &&...Fields) {
  auto [Struct, Path] = Binary.makeType<model::StructType>();
  (Struct.Fields().emplace(std::forward<FieldTypes>(Fields)), ...);
  return model::QualifiedType(Path, {});
}

BOOST_AUTO_TEST_CASE(StructTypes) {
  TupleTree<model::Binary> Binary;

  using QT = model::QualifiedType;
  QT Int16 = makePrimitive(model::PrimitiveTypeKind::Signed, 2, *Binary);
  QT Int32 = makePrimitive(model::PrimitiveTypeKind::Signed, 4, *Binary);
  QT Int64 = makePrimitive(model::PrimitiveTypeKind::Signed, 8, *Binary);
  QT Float = makePrimitive(model::PrimitiveTypeKind::Float, 4, *Binary);
  QT Double = makePrimitive(model::PrimitiveTypeKind::Float, 8, *Binary);
  QT LongDouble = makePrimitive(model::PrimitiveTypeKind::Float, 16, *Binary);
  QT WeirdLD = makePrimitive(model::PrimitiveTypeKind::Float, 12, *Binary);

  for (model::ABI::Values ABIName : TestedABIs) {
    const abi::Definition &ABI = abi::Definition::get(ABIName);

    QT SimpleStruct = makeStruct(*Binary,
                                 makeStructField(0, Int32),
                                 makeStructField(8, Int64));
    compareTypeAlignments(ABI, Int64, SimpleStruct);

    QT SmallFloatStruct = makeStruct(*Binary,
                                     makeStructField(0, Int32),
                                     makeStructField(4, Float));
    compareTypeAlignments(ABI, Int32, SmallFloatStruct);
    compareTypeAlignments(ABI, Float, SmallFloatStruct);

    QT BigFloatStruct = makeStruct(*Binary,
                                   makeStructField(0, LongDouble),
                                   makeStructField(16, Int64));
    compareTypeAlignments(ABI, LongDouble, BigFloatStruct);

    QT WeirdFloatStruct = makeStruct(*Binary,
                                     makeStructField(0, WeirdLD),
                                     makeStructField(12, Int32));
    compareTypeAlignments(ABI, WeirdLD, WeirdFloatStruct);

    // Test the case where on top of the float field, there's also another
    // stricter-aligned field, which "eclipses" the float one.
    QT EclipsedFloatStruct = makeStruct(*Binary,
                                        makeStructField(0, Float),
                                        makeStructField(8, Int64));
    compareTypeAlignments(ABI, Int64, EclipsedFloatStruct);

    QT NestedStruct = makeStruct(*Binary,
                                 makeStructField(0, SmallFloatStruct),
                                 makeStructField(8, Int16));
    compareTypeAlignments(ABI, Int32, NestedStruct);
    compareTypeAlignments(ABI, Float, NestedStruct);
    compareTypeAlignments(ABI, SmallFloatStruct, NestedStruct);

    QT EclipsedNestedStruct = makeStruct(*Binary,
                                         makeStructField(0, SmallFloatStruct),
                                         makeStructField(8, Int64));
    compareTypeAlignments(ABI, Int64, EclipsedNestedStruct);
  }
}

BOOST_AUTO_TEST_CASE(QualifiedTypes) {
  TupleTree<model::Binary> Binary;

  using QT = model::QualifiedType;
  QT Int32 = makePrimitive(model::PrimitiveTypeKind::Signed, 4, *Binary);
  QT Int64 = makePrimitive(model::PrimitiveTypeKind::Signed, 8, *Binary);
  QT Double = makePrimitive(model::PrimitiveTypeKind::Float, 8, *Binary);

  for (model::ABI::Values ABIName : TestedABIs) {
    const abi::Definition &ABI = abi::Definition::get(ABIName);
    auto Architecture = model::ABI::getArchitecture(ABI.ABI());
    auto PointerQualifier = model::Qualifier::createPointer(Architecture);

    QT IntPointer = Int32;
    IntPointer.Qualifiers().emplace_back(PointerQualifier);
    if (ABI.getPointerSize() == 8)
      compareTypeAlignments(ABI, Int64, IntPointer);
    else
      compareTypeAlignments(ABI, Int32, IntPointer);

    QT IntArray = Int32;
    IntArray.Qualifiers().emplace_back(model::Qualifier::createArray(100));
    compareTypeAlignments(ABI, Int32, IntArray);

    QT ConstInt = Int32;
    ConstInt.Qualifiers().emplace_back(model::Qualifier::createConst());
    compareTypeAlignments(ABI, Int32, ConstInt);

    QT DoublePointer = Double;
    DoublePointer.Qualifiers().emplace_back(PointerQualifier);
    if (ABI.getPointerSize() == 8)
      compareTypeAlignments(ABI, Int64, DoublePointer);
    else
      compareTypeAlignments(ABI, Int32, DoublePointer);

    QT DoubleArray = Double;
    DoubleArray.Qualifiers().emplace_back(model::Qualifier::createArray(100));
    compareTypeAlignments(ABI, Double, DoubleArray);

    QT ConstDouble = Double;
    ConstDouble.Qualifiers().emplace_back(model::Qualifier::createConst());
    compareTypeAlignments(ABI, Double, ConstDouble);
  }
}
