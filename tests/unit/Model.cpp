/// \file Model.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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
#include "revng/TupleTree/DiffError.h"
#include "revng/TupleTree/Introspection.h"
#include "revng/TupleTree/Tracking.h"
#include "revng/TupleTree/TupleTreeDiff.h"
#include "revng/TupleTree/VisitsImpl.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace model;

auto ARM1000 = MetaAddress::fromString("0x1000:Code_arm");
auto ARM2000 = MetaAddress::fromString("0x2000:Code_arm");
auto ARM3000 = MetaAddress::fromString("0x3000:Code_arm");

BOOST_AUTO_TEST_CASE(TestIntrospection) {
  using namespace llvm;

  Function TheFunction(MetaAddress::invalid());

  // Use get
  TheFunction.CustomName() = "FunctionName";
  revng_check(get<1>(TheFunction) == "FunctionName");

  // Test std::tuple_size
  static_assert(std::tuple_size<Function>::value >= 2);

  // Test TupleLikeTraits
  static_assert(TraitedTupleLike<Function>);
  using TLT = TupleLikeTraits<Function>;
  static_assert(std::is_same_v<std::tuple_element_t<1, Function> &,
                               decltype(TheFunction.CustomName())>);
  revng_check(StringRef(TLT::Name) == "Function");
  revng_check(StringRef(TLT::FullName) == "model::Function");
  revng_check(StringRef(TLT::FieldNames[1]) == "CustomName");
}

BOOST_AUTO_TEST_CASE(TestPathAccess) {
  Binary TheBinary;
  using FunctionsType = std::decay_t<decltype(TheBinary.Functions())>;
  TupleTreePath Zero;
  Zero.push_back(size_t(0));
  auto *FirstField = getByPath<FunctionsType>(Zero, TheBinary);
  revng_check(FirstField == &TheBinary.Functions());

  auto *FunctionsField = getByPath<FunctionsType>("/Functions", TheBinary);
  revng_check(FunctionsField == &TheBinary.Functions());

  // Test non existing field
  revng_check(getByPath<FunctionsType>("/Function", TheBinary) == nullptr);

  // Test non existing entry in container
  revng_check(getByPath<Function>("/Functions/:Invalid", TheBinary) == nullptr);

  // Test existing entry in container
  Function &F = TheBinary.Functions()[MetaAddress::invalid()];
  revng_check(getByPath<Function>("/Functions/:Invalid", TheBinary) == &F);

  // Test UpcastablePointer
  auto UInt8Path = TheBinary.getPrimitiveType(PrimitiveTypeKind::Unsigned, 8);
  model::Type *UInt8 = UInt8Path.get();
  using llvm::Twine;
  std::string TypePath = (Twine("/Types/") + Twine(UInt8->ID())
                          + "-PrimitiveType/OriginalName")
                           .str();
  auto *OriginalNamePointer = getByPath<std::string>(TypePath, TheBinary);
  revng_check(OriginalNamePointer == &UInt8->OriginalName());

  TypePath = (Twine("/Types/") + Twine(UInt8->ID()) + Twine("-PrimitiveType"))
               .str();
  revng_check(getByPath<model::Type>(TypePath, TheBinary) == UInt8);
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
  // Test regular matcher
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

  //
  // Test matching through an UpcastablePointer
  //
  {
    auto Matcher = PathMatcher::create<Binary>("/Types/*-RawFunctionType/"
                                               "FinalStackOffset")
                     .value();

    model::Type::Key Key{ 1000, model::TypeKind::RawFunctionType };
    auto Path1000 = pathAsString<Binary>(Matcher.apply(Key));
    revng_check(Path1000 == "/Types/1000-RawFunctionType/FinalStackOffset");

    auto MaybeToMatch = stringAsPath<Binary>(*Path1000);
    revng_check(MaybeToMatch);
    auto MaybeMatch = Matcher.match<model::Type::Key>(MaybeToMatch.value());
    revng_check(MaybeMatch);
    revng_check(std::get<0>(*MaybeMatch) == Key);

    MaybeToMatch = stringAsPath<Binary>("/Types/1000-CABIFunctionType/ID");
    MaybeMatch = Matcher.match<model::Type::Key>(MaybeToMatch.value());
    revng_check(not MaybeMatch);
  }
}

template<typename T>
static T *createType(model::Binary &Model) {
  return &Model.makeType<T>().first;
}

BOOST_AUTO_TEST_CASE(TestModelDeduplication) {
  TupleTree<model::Binary> Model;
  auto Dedup = [&Model]() {
    int64_t OldTypesCount = Model->Types().size();
    deduplicateEquivalentTypes(Model);
    int64_t NewTypesCount = Model->Types().size();
    return OldTypesCount - NewTypesCount;
  };

  model::TypePath UInt8 = Model->getPrimitiveType(PrimitiveTypeKind::Generic,
                                                  4);

  // Two typedefs
  {
    auto *Typedef1 = createType<TypedefType>(*Model);
    Typedef1->UnderlyingType() = { UInt8, {} };

    auto *Typedef2 = createType<TypedefType>(*Model);
    Typedef2->UnderlyingType() = { UInt8, {} };

    revng_check(Dedup() == 0);

    Typedef1->OriginalName() = "MyUInt8";
    Typedef2->OriginalName() = "MyUInt8";

    revng_check(Dedup() == 1);
  }

  // Two structs
  {
    auto *Struct1 = createType<StructType>(*Model);
    Struct1->Fields()[0].CustomName() = "FirstField";
    Struct1->Fields()[0].Type() = { UInt8, {} };
    Struct1->OriginalName() = "MyStruct";

    auto *Struct2 = createType<StructType>(*Model);
    Struct2->Fields()[0].CustomName() = "DifferentName";
    Struct2->Fields()[0].Type() = { UInt8, {} };
    Struct2->OriginalName() = "MyStruct";

    revng_check(Dedup() == 0);

    Struct1->Fields()[0].CustomName() = Struct2->Fields()[0].CustomName();

    revng_check(Dedup() == 1);
  }

  // Two pairs of cross-referencing structs
  {
    auto PointerQualifier = Qualifier::createPointer(8);

    auto *Left1 = createType<StructType>(*Model);
    auto *Left2 = createType<StructType>(*Model);

    Left1->Fields()[0].Type() = { Model->getTypePath(Left2),
                                  { PointerQualifier } };
    Left2->Fields()[0].Type() = { Model->getTypePath(Left1),
                                  { PointerQualifier } };

    Left1->OriginalName() = "LoopingStructs1";
    Left2->OriginalName() = "LoopingStructs2";

    auto *Right1 = createType<StructType>(*Model);
    auto *Right2 = createType<StructType>(*Model);

    Right1->Fields()[0].Type() = { Model->getTypePath(Right2),
                                   { PointerQualifier } };
    Right2->Fields()[0].Type() = { Model->getTypePath(Right1),
                                   { PointerQualifier, PointerQualifier } };

    Right1->OriginalName() = "LoopingStructs1";
    Right2->OriginalName() = "LoopingStructs2";

    revng_check(Dedup() == 0);

    Right2->Fields()[0].Type() = { Model->getTypePath(Right1),
                                   { PointerQualifier } };

    revng_check(Dedup() == 2);
  }
}

BOOST_AUTO_TEST_CASE(TestTupleTreeDiff) {
  model::Binary Left;
  model::Binary Right;
  diff(Left, Right).dump();
}

BOOST_AUTO_TEST_CASE(TestTupleTreeDiffSerialization) {
  model::Binary Left;
  model::Binary Right;
  auto Diff = diff(Left, Right);

  std::string S;
  llvm::raw_string_ostream Stream(S);
  serialize(Stream, Diff);
}

BOOST_AUTO_TEST_CASE(TestTupleTreeDiffDeserialization) {
  model::Binary Empty;
  model::Binary New;

  MetaAddress Address(0x1000, MetaAddressType::Code_aarch64);
  New.ExtraCodeAddresses().insert(Address);

  auto Diff = diff(Empty, New);

  std::string S;
  llvm::raw_string_ostream Stream(S);
  serialize(Stream, Diff);
  Stream.flush();
  llvm::outs() << S << "\n";

  auto Diff2 = llvm::cantFail(deserialize<TupleTreeDiff<model::Binary>>(S));

  std::string S2;
  llvm::raw_string_ostream Stream2(S2);
  serialize(Stream2, Diff2);
  Stream2.flush();

  BOOST_TEST(S == S2);
}

BOOST_AUTO_TEST_CASE(CABIFunctionTypePathShouldParse) {
  const char *Path = "/Types/10000-CABIFunctionType";
  auto MaybeParsed = stringAsPath<model::Binary>(Path);
  BOOST_TEST(MaybeParsed.has_value());
}

BOOST_AUTO_TEST_CASE(CABIFunctionTypeArgumentsPathShouldParse) {
  const char *Path = "/Types/10000-CABIFunctionType/Arguments";
  auto MaybeParsed = stringAsPath<model::Binary>(Path);
  BOOST_TEST(MaybeParsed.has_value());
}

class LocationExample : public revng::LocationBase {
public:
  std::string toString() const final { return "don't care"; };
  ~LocationExample() override = default;
  static std::string getTypeName() { return "LocationExample"; }
};

class DocumentErrorExample
  : public revng::DocumentError<DocumentErrorExample, LocationExample> {
public:
  using DocumentError<DocumentErrorExample, LocationExample>::DocumentError;
  inline static char ID = '0';
  std::string getTypeName() const override { return "Example1"; }
};

class DocumentErrorExample2
  : public revng::DocumentError<DocumentErrorExample2, LocationExample> {
public:
  using DocumentError<DocumentErrorExample2, LocationExample>::DocumentError;
  inline static char ID = '0';
  std::string getTypeName() const override { return "Example2"; }
};

BOOST_AUTO_TEST_CASE(ModelErrors) {
  llvm::Error Error = llvm::make_error<DocumentErrorExample>("something",
                                                             LocationExample());
  BOOST_TEST(Error.isA<DocumentErrorExample>());
  BOOST_TEST(not Error.isA<DocumentErrorExample2>());
  BOOST_TEST(Error.isA<revng::DocumentErrorBase>());
  llvm::consumeError(std::move(Error));
}

BOOST_AUTO_TEST_CASE(CollectReadFieldsShouldCompile) {
  model::Binary Model;
  revng::Tracking::collect(Model);
}

BOOST_AUTO_TEST_CASE(TrackingResetterShouldCompile) {
  model::Binary Model;
  revng::Tracking::clearAndResume(Model);
}

BOOST_AUTO_TEST_CASE(TrackingPushAndPopperShouldCompile) {
  model::Binary Model;
  revng::Tracking::push(Model);
  revng::Tracking::pop(Model);
}

BOOST_AUTO_TEST_CASE(CollectReadFieldsShouldBeEmptyAtFirst) {
  model::Binary Model;
  auto MetaAddress = MetaAddress::fromPC(llvm::Triple::ArchType::x86_64, 0);
  Model.Segments().insert(Segment(MetaAddress, 1000));
  revng::Tracking::clearAndResume(Model);

  auto Collected = revng::Tracking::collect(Model);
  BOOST_TEST(Collected.Read.size() == 0);
}

BOOST_AUTO_TEST_CASE(CollectReadFieldsShouldCollectSegments) {
  model::Binary Model;
  const auto MetaAddress = MetaAddress::fromPC(llvm::Triple::ArchType::x86_64,
                                               0);
  Model.Segments().insert(Segment(MetaAddress, 1000));
  revng::Tracking::clearAndResume(Model);
  const auto &ConstModel = Model;
  ConstModel.Segments().at(Segment::Key(MetaAddress, 1000)).StartAddress();

  auto Collected = revng::Tracking::collect(Model);
  BOOST_TEST(Collected.Read.size() == 2);
  std::vector StringPaths = {
    "/Segments",
    "/Segments/0x0:Code_x86_64-1000",
  };

  std::set<TupleTreePath> Paths;
  for (const auto &Path : StringPaths) {
    Paths.insert(*stringAsPath<model::Binary>(Path));
  }
  BOOST_TEST(Collected.Read == Paths);
}

BOOST_AUTO_TEST_CASE(CollectReadFieldsShouldCollectNotFoundSegments) {
  model::Binary Model;
  const auto MetaAddress = MetaAddress::fromPC(llvm::Triple::ArchType::x86_64,
                                               0);
  revng::Tracking::clearAndResume(Model);
  const auto &ConstModel = Model;
  ConstModel.Segments().tryGet(Segment::Key(MetaAddress, 1000));

  auto Collected = revng::Tracking::collect(Model);
  BOOST_TEST(Collected.Read.size() == 2);
  std::set<TupleTreePath> Paths = {
    *stringAsPath<model::Binary>("/Segments"),
    *stringAsPath<model::Binary>("/Segments/0x0:Code_x86_64-1000"),
  };
  BOOST_TEST(Collected.Read == Paths);
}

BOOST_AUTO_TEST_CASE(CollectReadFieldsShouldCollectAllSegments) {
  model::Binary Model;
  const auto MetaAddress = MetaAddress::fromPC(llvm::Triple::ArchType::x86_64,
                                               0);
  Model.Segments().insert(Segment(MetaAddress, 1000));
  revng::Tracking::clearAndResume(Model);
  const auto &ConstModel = Model;
  ConstModel.Segments().begin();

  auto Collected = revng::Tracking::collect(Model);
  BOOST_TEST(Collected.Read.size() == 1);
  std::set<TupleTreePath> Paths = {
    *stringAsPath<model::Binary>("/Segments"),
  };
  BOOST_TEST(Collected.Read == Paths);
  BOOST_TEST(Collected.ExactVectors == Paths);
}

/// This test asserts that Tracking visits do no inspect inside a vector.
/// We need to find a way to represent non sorted vector, using regular vectors
/// breaks diffs, since they don't have a index to represent a child
BOOST_AUTO_TEST_CASE(QualifiersInsideAVectorAreNotVisited) {
  model::QualifiedType Type;
  const auto MetaAddress = MetaAddress::fromPC(llvm::Triple::ArchType::x86_64,
                                               0);
  Type.Qualifiers().push_back(Qualifier());
  revng::Tracking::clearAndResume(Type);
  const auto &ConstType = Type;
  ConstType.Qualifiers().at(0).Size();

  auto Collected = revng::Tracking::collect(Type);
  BOOST_TEST(Collected.Read.size() == 1);
  std::set<TupleTreePath> Paths = {
    *stringAsPath<model::QualifiedType>("/Qualifiers"),
  };
  BOOST_TEST(Collected.Read == Paths);
}
