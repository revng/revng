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
void testAlignment(model::UpcastableType &&Type, const Types &...TestCases) {
  for (auto &&[ABI, Expected] : std::array{ TestCases... }) {
    std::optional<uint64_t> TestResult = ABI.alignment(*Type);
    if (TestResult.value_or(0) != Expected) {
      std::string Error = "Alignment run failed for type:\n" + toString(Type)
                          + "ABI (`" + toString(ABI.ABI())
                          + "`) reports the alignment of `"
                          + printAlignment(TestResult.value_or(0)) + "`, "
                          + "while the expected value is `"
                          + printAlignment(Expected) + "`.\n";
      revng_abort(Error.c_str());
    }
  }
}

static bool ABIhasIntsOfSizes(const abi::Definition &ABI,
                              std::initializer_list<uint64_t> Values) {
  return std::ranges::all_of(Values, [&ABI](uint64_t Value) {
    return ABI.ScalarTypes().contains(Value);
  });
}

static bool ABIhasFloatsOfSizes(const abi::Definition &ABI,
                                std::initializer_list<uint64_t> Values) {
  return std::ranges::all_of(Values, [&ABI](uint64_t Value) {
    return ABI.FloatingPointScalarTypes().contains(Value);
  });
}

BOOST_AUTO_TEST_CASE(GenericPrimitiveTypes) {
  TupleTree<model::Binary> Binary;

  testAlignment(model::PrimitiveType::makeVoid(),
                Expected(model::ABI::AAPCS64, 0),
                Expected(model::ABI::AAPCS, 0),
                Expected(model::ABI::SystemZ_s390x, 0),
                Expected(model::ABI::SystemV_x86, 0));

  testAlignment(model::PrimitiveType::makeGeneric(1),
                Expected(model::ABI::AAPCS64, 1),
                Expected(model::ABI::AAPCS, 1),
                Expected(model::ABI::SystemZ_s390x, 1),
                Expected(model::ABI::SystemV_x86, 1));

  testAlignment(model::PrimitiveType::makeGeneric(2),
                Expected(model::ABI::AAPCS64, 2),
                Expected(model::ABI::AAPCS, 2),
                Expected(model::ABI::SystemZ_s390x, 2),
                Expected(model::ABI::SystemV_x86, 2));

  testAlignment(model::PrimitiveType::makeGeneric(4),
                Expected(model::ABI::AAPCS64, 4),
                Expected(model::ABI::AAPCS, 4),
                Expected(model::ABI::SystemZ_s390x, 4),
                Expected(model::ABI::SystemV_x86, 4));

  testAlignment(model::PrimitiveType::makeGeneric(8),
                Expected(model::ABI::AAPCS64, 8),
                Expected(model::ABI::AAPCS, 8),
                Expected(model::ABI::SystemZ_s390x, 8),
                Expected(model::ABI::SystemV_x86, 4));

  testAlignment(model::PrimitiveType::makeGeneric(16),
                Expected(model::ABI::AAPCS64, 16),
                Expected(model::ABI::SystemZ_s390x, 8),
                Expected(model::ABI::SystemV_x86_64, 16));
}

BOOST_AUTO_TEST_CASE(FloatingPointPrimitiveTypes) {
  TupleTree<model::Binary> Binary;

  testAlignment(model::PrimitiveType::makeFloat(2),
                Expected(model::ABI::AAPCS64, 2),
                Expected(model::ABI::AAPCS, 2),
                Expected(model::ABI::SystemV_x86_64, 2));

  testAlignment(model::PrimitiveType::makeFloat(4),
                Expected(model::ABI::AAPCS64, 4),
                Expected(model::ABI::AAPCS, 4),
                Expected(model::ABI::SystemZ_s390x, 4),
                Expected(model::ABI::SystemV_x86, 4));

  testAlignment(model::PrimitiveType::makeFloat(8),
                Expected(model::ABI::AAPCS64, 8),
                Expected(model::ABI::AAPCS, 8),
                Expected(model::ABI::SystemZ_s390x, 8),
                Expected(model::ABI::SystemV_x86, 4));

  testAlignment(model::PrimitiveType::makeFloat(16),
                Expected(model::ABI::AAPCS64, 16),
                Expected(model::ABI::SystemZ_s390x, 8),
                Expected(model::ABI::SystemV_x86, 16),
                Expected(model::ABI::SystemV_x86_64, 16));
}

constexpr std::array TestedABIs{ model::ABI::AAPCS64,
                                 model::ABI::AAPCS,
                                 model::ABI::SystemZ_s390x,
                                 model::ABI::SystemV_x86 };

static void compareTypeAlignments(const abi::Definition &ABI,
                                  const model::UpcastableType &LHS,
                                  const model::UpcastableType &RHS) {
  std::optional<uint64_t> Left = ABI.alignment(*LHS);
  std::optional<uint64_t> Right = ABI.alignment(*RHS);
  if (Left != Right) {
    std::string Error = "Alignment comparison run failed for types:\n"
                        + toString(LHS) + "and\n" + toString(RHS) + "ABI (`"
                        + toString(ABI.ABI()) + "`) reports the alignment of `"
                        + printAlignment(Left.value_or(0))
                        + "` for the first one, and `"
                        + printAlignment(Right.value_or(0))
                        + "` for the second one.\n";
    revng_abort(Error.c_str());
  }
}

BOOST_AUTO_TEST_CASE(RemainingPrimitiveTypes) {
  TupleTree<model::Binary> Binary;

  constexpr std::array<model::PrimitiveKind::Values, 5> RemainingTypes{
    model::PrimitiveKind::Unsigned,
    model::PrimitiveKind::Signed,
    model::PrimitiveKind::Number,
    model::PrimitiveKind::PointerOrNumber
  };

  for (model::ABI::Values ABIName : TestedABIs) {
    const abi::Definition &ABI = abi::Definition::get(ABIName);
    for (const auto &PKind : RemainingTypes)
      for (uint64_t Size = 1; Size <= 16; Size *= 2)
        if (ABIhasIntsOfSizes(ABI, { Size }))
          compareTypeAlignments(ABI,
                                model::PrimitiveType::makeGeneric(Size),
                                model::PrimitiveType::make(PKind, Size));
  }
}

BOOST_AUTO_TEST_CASE(UnionTypes) {
  TupleTree<model::Binary> Binary;

  auto Int16 = model::PrimitiveType::makeSigned(2);
  auto Int32 = model::PrimitiveType::makeSigned(4);
  auto Int64 = model::PrimitiveType::makeSigned(8);

  auto Float = model::PrimitiveType::makeFloat(4);
  auto Double = model::PrimitiveType::makeFloat(8);
  auto LongDouble = model::PrimitiveType::makeFloat(16);
  auto WeirdLD = model::PrimitiveType::makeFloat(12);

  for (model::ABI::Values ABIName : TestedABIs) {
    const abi::Definition &ABI = abi::Definition::get(ABIName);

    auto &&[SimpleDefinition, Simple] = Binary->makeUnionDefinition();
    SimpleDefinition.addField(Int32.copy());
    SimpleDefinition.addField(Int64.copy());
    if (ABIhasIntsOfSizes(ABI, { 4, 8 }))
      compareTypeAlignments(ABI, Int64, Simple);

    auto &&[SmallFloatDefinition, SmallFloat] = Binary->makeUnionDefinition();
    SmallFloatDefinition.addField(Int32.copy());
    SmallFloatDefinition.addField(Float.copy());
    if (ABIhasIntsOfSizes(ABI, { 4 }) && ABIhasFloatsOfSizes(ABI, { 4 })) {
      compareTypeAlignments(ABI, Int32, SmallFloat);
      compareTypeAlignments(ABI, Float, SmallFloat);
    }

    auto &&[BigFloatDefinition, BigFloat] = Binary->makeUnionDefinition();
    BigFloatDefinition.addField(LongDouble.copy());
    BigFloatDefinition.addField(Int64.copy());
    if (ABIhasIntsOfSizes(ABI, { 8 }) && ABIhasFloatsOfSizes(ABI, { 16 }))
      compareTypeAlignments(ABI, LongDouble, BigFloat);

    auto &&[WeirdFloatDefinition, WeirdFloat] = Binary->makeUnionDefinition();
    WeirdFloatDefinition.addField(WeirdLD.copy());
    WeirdFloatDefinition.addField(Int32.copy());
    if (ABIhasIntsOfSizes(ABI, { 4 }) && ABIhasFloatsOfSizes(ABI, { 12 }))
      compareTypeAlignments(ABI, WeirdLD, WeirdFloat);

    // Test the case where on top of the float field, there's also another
    // stricter-aligned field, which "eclipses" the float one.
    auto &&[EclipsedFloatDefinition,
            EclipsedFl] = Binary->makeUnionDefinition();
    EclipsedFloatDefinition.addField(Float.copy());
    EclipsedFloatDefinition.addField(Int64.copy());
    if (ABIhasIntsOfSizes(ABI, { 8 }) && ABIhasFloatsOfSizes(ABI, { 4 }))
      compareTypeAlignments(ABI, Int64, EclipsedFl);

    auto &&[NestedDefinition, Nested] = Binary->makeUnionDefinition();
    NestedDefinition.addField(SmallFloat.copy());
    NestedDefinition.addField(Int16.copy());
    if (ABIhasIntsOfSizes(ABI, { 2, 4 }) && ABIhasFloatsOfSizes(ABI, { 4 })) {
      compareTypeAlignments(ABI, Int32, Nested);
      compareTypeAlignments(ABI, Float, Nested);
      compareTypeAlignments(ABI, SmallFloat, Nested);
    }

    auto &&[EclipsedNestedDefinition,
            EclipsedN] = Binary->makeUnionDefinition();
    EclipsedNestedDefinition.addField(SmallFloat.copy());
    EclipsedNestedDefinition.addField(Int64.copy());
    if (ABIhasIntsOfSizes(ABI, { 4, 8 }) && ABIhasFloatsOfSizes(ABI, { 4 }))
      compareTypeAlignments(ABI, Int64, EclipsedN);
  }
}

BOOST_AUTO_TEST_CASE(StructTypes) {
  TupleTree<model::Binary> Binary;

  auto Int16 = model::PrimitiveType::makeSigned(2);
  auto Int32 = model::PrimitiveType::makeSigned(4);
  auto Int64 = model::PrimitiveType::makeSigned(8);

  auto Float = model::PrimitiveType::makeFloat(4);
  auto Double = model::PrimitiveType::makeFloat(8);
  auto LongDouble = model::PrimitiveType::makeFloat(16);
  auto WeirdLD = model::PrimitiveType::makeFloat(12);

  for (model::ABI::Values ABIName : TestedABIs) {
    const abi::Definition &ABI = abi::Definition::get(ABIName);

    auto &&[SimpleDefinition, Simple] = Binary->makeStructDefinition();
    SimpleDefinition.addField(0, Int32.copy());
    SimpleDefinition.addField(8, Int64.copy());
    if (ABIhasIntsOfSizes(ABI, { 4, 8 }))
      compareTypeAlignments(ABI, Int64, Simple);

    auto &&[SmallFloatDefinition, SmallFloat] = Binary->makeStructDefinition();
    SmallFloatDefinition.addField(0, Int32.copy());
    SmallFloatDefinition.addField(4, Float.copy());
    if (ABIhasIntsOfSizes(ABI, { 4 }) && ABIhasFloatsOfSizes(ABI, { 4 })) {
      compareTypeAlignments(ABI, Int32, SmallFloat);
      compareTypeAlignments(ABI, Float, SmallFloat);
    }

    auto &&[BigFloatDefinition, BigFloat] = Binary->makeStructDefinition();
    BigFloatDefinition.addField(0, LongDouble.copy());
    BigFloatDefinition.addField(16, Int64.copy());
    if (ABIhasIntsOfSizes(ABI, { 8 }) && ABIhasFloatsOfSizes(ABI, { 16 }))
      compareTypeAlignments(ABI, LongDouble, BigFloat);

    auto &&[WeirdFloatDefinition, WeirdFloat] = Binary->makeStructDefinition();
    WeirdFloatDefinition.addField(0, WeirdLD.copy());
    WeirdFloatDefinition.addField(12, Int32.copy());
    if (ABIhasIntsOfSizes(ABI, { 4 }) && ABIhasFloatsOfSizes(ABI, { 12 }))
      compareTypeAlignments(ABI, WeirdLD, WeirdFloat);

    // Test the case where on top of the float field, there's also another
    // stricter-aligned field, which "eclipses" the float one.
    auto &&[EclipsedFloatDefinition,
            EclipsedFl] = Binary->makeStructDefinition();
    EclipsedFloatDefinition.addField(0, Float.copy());
    EclipsedFloatDefinition.addField(8, Int64.copy());
    if (ABIhasIntsOfSizes(ABI, { 8 }) && ABIhasFloatsOfSizes(ABI, { 4 }))
      compareTypeAlignments(ABI, Int64, EclipsedFl);

    auto &&[NestedDefinition, Nested] = Binary->makeStructDefinition();
    NestedDefinition.addField(0, SmallFloat.copy());
    NestedDefinition.addField(8, Int16.copy());
    if (ABIhasIntsOfSizes(ABI, { 2, 4 }) && ABIhasFloatsOfSizes(ABI, { 4 })) {
      compareTypeAlignments(ABI, Int32, Nested);
      compareTypeAlignments(ABI, Float, Nested);
      compareTypeAlignments(ABI, SmallFloat, Nested);
    }

    auto &&[EclipsedNestedDefinition,
            EclipsedN] = Binary->makeStructDefinition();
    EclipsedNestedDefinition.addField(0, SmallFloat.copy());
    EclipsedNestedDefinition.addField(8, Int64.copy());
    if (ABIhasIntsOfSizes(ABI, { 4, 8 }) && ABIhasFloatsOfSizes(ABI, { 4 }))
      compareTypeAlignments(ABI, Int64, EclipsedN);
  }
}

BOOST_AUTO_TEST_CASE(ArraysAndPointers) {
  TupleTree<model::Binary> Binary;

  auto Int32 = model::PrimitiveType::makeSigned(4);
  auto Int64 = model::PrimitiveType::makeSigned(8);
  auto Double = model::PrimitiveType::makeFloat(8);

  for (model::ABI::Values ABIName : TestedABIs) {
    const abi::Definition &ABI = abi::Definition::get(ABIName);

    auto IntPointer = model::PointerType::make(Int32.copy(),
                                               ABI.getPointerSize());
    if (ABI.getPointerSize() == 8)
      compareTypeAlignments(ABI, Int64, IntPointer);
    else
      compareTypeAlignments(ABI, Int32, IntPointer);

    auto IntArray = model::ArrayType::make(Int32.copy(), 100);
    compareTypeAlignments(ABI, Int32, IntArray);

    auto ConstInt = Int32.copy();
    ConstInt->IsConst() = true;
    compareTypeAlignments(ABI, Int32, ConstInt);

    auto DoublePointer = model::PointerType::make(Double.copy(),
                                                  ABI.getPointerSize());
    compareTypeAlignments(ABI, IntPointer, DoublePointer);

    auto DoubleArray = model::ArrayType::make(Double.copy(), 100);
    compareTypeAlignments(ABI, Double, DoubleArray);

    auto ConstDouble = Double.copy();
    ConstInt->IsConst() = true;
    compareTypeAlignments(ABI, Double, ConstDouble);
  }
}
