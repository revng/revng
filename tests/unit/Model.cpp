/// \file Model.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE Model
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/Binary.h"
#include "revng/Model/Pass/AllPasses.h"
#include "revng/Model/Processing.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/Introspection.h"
#include "revng/TupleTree/TupleTreeDiff.h"

using namespace model;

auto ARM1000 = MetaAddress::fromString("0x1000:Code_arm");
auto ARM2000 = MetaAddress::fromString("0x2000:Code_arm");
auto ARM3000 = MetaAddress::fromString("0x3000:Code_arm");

BOOST_AUTO_TEST_CASE(TestIntrospection) {
  using namespace llvm;

  Function TheFunction(MetaAddress::invalid());

  // Use get
  TheFunction.CustomName = "FunctionName";
  revng_check(get<1>(TheFunction) == "FunctionName");

  // Test std::tuple_size
  static_assert(std::tuple_size<Function>::value >= 2);

  // Test TupleLikeTraits
  using TLT = TupleLikeTraits<Function>;
  static_assert(std::is_same_v<std::tuple_element_t<1, Function>,
                               decltype(TheFunction.CustomName)>);
  revng_check(StringRef(TLT::Name) == "model::Function");
  revng_check(StringRef(TLT::FieldsName[1]) == "CustomName");
}

BOOST_AUTO_TEST_CASE(TestPathAccess) {
  Binary TheBinary;
  using FunctionsType = decltype(TheBinary.Functions);
  TupleTreePath Zero;
  Zero.push_back(size_t(0));
  auto *FirstField = getByPath<FunctionsType>(Zero, TheBinary);
  revng_check(FirstField == &TheBinary.Functions);

  auto *FunctionsField = getByPath<FunctionsType>("/Functions", TheBinary);
  revng_check(FunctionsField == &TheBinary.Functions);

  // Test non existing field
  revng_check(getByPath<FunctionsType>("/Function", TheBinary) == nullptr);

  // Test non existing entry in container
  revng_check(getByPath<Function>("/Functions/:Invalid", TheBinary) == nullptr);

  // Test existing entry in container
  Function &F = TheBinary.Functions[MetaAddress::invalid()];
  revng_check(getByPath<Function>("/Functions/:Invalid", TheBinary) == &F);
}

BOOST_AUTO_TEST_CASE(TestCompositeScalar) {
  // MetaAddress pair
  {
    model::Segment::Key BlockKey = { ARM2000, 1000 };
    auto BlockKeyName = getNameFromYAMLScalar(BlockKey);
    revng_check(BlockKeyName == "0x2000:Code_arm-1000");
  }
}

BOOST_AUTO_TEST_CASE(TestStringPathConversion) {
  revng_check(stringAsPath<Binary>("/").value() == TupleTreePath{});
  TupleTreePath Zero;
  Zero.push_back(size_t(0));
  revng_check(stringAsPath<Binary>("/Functions").value() == Zero);

  TupleTreePath InvalidFunctionPath;
  InvalidFunctionPath.push_back(size_t(0));
  InvalidFunctionPath.push_back(MetaAddress::invalid());
  auto MaybeInvalidFunctionPath = stringAsPath<Binary>("/Functions/:Invalid");
  revng_check(MaybeInvalidFunctionPath.value() == InvalidFunctionPath);

  TupleTreePath InvalidFunctionNamePath = InvalidFunctionPath;
  InvalidFunctionNamePath.push_back(size_t(1));
  auto MaybePath = stringAsPath<Binary>("/Functions/:Invalid/CustomName");
  revng_check(MaybePath.value() == InvalidFunctionNamePath);

  auto CheckRoundTrip = [](const char *String) {
    auto Path = stringAsPath<Binary>(String).value();
    auto StringAgain = pathAsString<Binary>(Path);
    revng_check(StringAgain == String);
  };

  CheckRoundTrip("/Functions");
  CheckRoundTrip("/Functions/:Invalid");
  CheckRoundTrip("/Functions/:Invalid/Entry");
  CheckRoundTrip("/Functions/0x1000:Code_arm/Entry");
}

BOOST_AUTO_TEST_CASE(TestPathMatcher) {
  //
  // Single matcher
  //
  {
    auto Matcher = PathMatcher::create<Binary>("/Functions/*/Entry").value();

    auto ARM1000EntryPath = pathAsString<Binary>(Matcher.apply(ARM1000));
    revng_check(ARM1000EntryPath == "/Functions/0x1000:Code_arm/Entry");

    auto MaybeToMatch = stringAsPath<Binary>("/Functions/0x1000:Code_arm/"
                                             "Entry");
    auto MaybeMatch = Matcher.match<MetaAddress>(MaybeToMatch.value());
    revng_check(MaybeMatch);
    revng_check(std::get<0>(*MaybeMatch) == ARM1000);
  }
}

namespace TestTupleTree {
class Element;
class Root;
} // namespace TestTupleTree

class TestTupleTree::Element {
public:
  int Key;
  TupleTreeReference<TestTupleTree::Element, TestTupleTree::Root> Self;
};
INTROSPECTION_NS(TestTupleTree, Element, Key, Self)

template<>
struct KeyedObjectTraits<TestTupleTree::Element> {
  static int key(const TestTupleTree::Element &Obj) { return Obj.Key; }

  static TestTupleTree::Element fromKey(const int &Key) {
    return TestTupleTree::Element{ Key, {} };
  }
};

class TestTupleTree::Root {
public:
  SortedVector<TestTupleTree::Element> Elements;
};

INTROSPECTION_NS(TestTupleTree, Root, Elements)

static_assert(IsTupleLike<TestTupleTree::Root>);

BOOST_AUTO_TEST_CASE(TestTupleTreeReference) {
  using namespace TestTupleTree;

  using Reference = TupleTreeReference<TestTupleTree::Element,
                                       TestTupleTree::Root>;

  TupleTree<Root> TheRoot;
  Element &AnElement = TheRoot->Elements[3];
  AnElement.Self = Reference::fromString(TheRoot.get(), "/Elements/3");

  TheRoot.initializeReferences();

  revng_check(AnElement.Self.get() == &AnElement);
}

template<typename T>
static T *createType(model::Binary &Model) {
  model::TypePath Path = Model.recordNewType(makeType<T>());
  return llvm::cast<T>(Path.get());
}

BOOST_AUTO_TEST_CASE(TestModelDeduplication) {
  TupleTree<model::Binary> Model;
  auto Dedup = [&Model]() {
    int64_t OldTypesCount = Model->Types.size();
    deduplicateEquivalentTypes(Model);
    int64_t NewTypesCount = Model->Types.size();
    return OldTypesCount - NewTypesCount;
  };

  model::TypePath UInt8 = Model->getPrimitiveType(PrimitiveTypeKind::Generic,
                                                  4);

  // Two typedefs
  {
    auto *Typedef1 = createType<TypedefType>(*Model);
    Typedef1->UnderlyingType = { UInt8, {} };

    auto *Typedef2 = createType<TypedefType>(*Model);
    Typedef2->UnderlyingType = { UInt8, {} };

    revng_check(Dedup() == 0);

    Typedef1->OriginalName = "MyUInt8";
    Typedef2->OriginalName = "MyUInt8";

    revng_check(Dedup() == 1);
  }

  // Two structs
  {
    auto *Struct1 = createType<StructType>(*Model);
    Struct1->Fields[0].CustomName = "FirstField";
    Struct1->Fields[0].Type = { UInt8, {} };
    Struct1->OriginalName = "MyStruct";

    auto *Struct2 = createType<StructType>(*Model);
    Struct2->Fields[0].CustomName = "DifferentName";
    Struct2->Fields[0].Type = { UInt8, {} };
    Struct2->OriginalName = "MyStruct";

    revng_check(Dedup() == 0);

    Struct1->Fields[0].CustomName = Struct2->Fields[0].CustomName;

    revng_check(Dedup() == 1);
  }

  // Two pairs of cross-referencing structs
  {
    auto PointerQualifier = Qualifier::createPointer(8);

    auto *Left1 = createType<StructType>(*Model);
    auto *Left2 = createType<StructType>(*Model);

    Left1->Fields[0].Type = { Model->getTypePath(Left2), { PointerQualifier } };
    Left2->Fields[0].Type = { Model->getTypePath(Left1), { PointerQualifier } };

    Left1->OriginalName = "LoopingStructs1";
    Left2->OriginalName = "LoopingStructs2";

    auto *Right1 = createType<StructType>(*Model);
    auto *Right2 = createType<StructType>(*Model);

    Right1->Fields[0].Type = { Model->getTypePath(Right2),
                               { PointerQualifier } };
    Right2->Fields[0].Type = { Model->getTypePath(Right1),
                               { PointerQualifier, PointerQualifier } };

    Right1->OriginalName = "LoopingStructs1";
    Right2->OriginalName = "LoopingStructs2";

    revng_check(Dedup() == 0);

    Right2->Fields[0].Type = { Model->getTypePath(Right1),
                               { PointerQualifier } };

    revng_check(Dedup() == 2);
  }
}

BOOST_AUTO_TEST_CASE(TestTupleTreeDiff) {
  model::Binary Left;
  model::Binary Right;
  diff(Left, Right).dump();
}

static_assert(std::is_default_constructible_v<TupleTree<TestTupleTree::Root>>);
static_assert(not std::is_copy_assignable_v<TupleTree<TestTupleTree::Root>>);
static_assert(not std::is_copy_constructible_v<TupleTree<TestTupleTree::Root>>);
static_assert(std::is_move_assignable_v<TupleTree<TestTupleTree::Root>>);
static_assert(std::is_move_constructible_v<TupleTree<TestTupleTree::Root>>);
