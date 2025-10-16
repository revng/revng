/// \file Model.cpp

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
  TheFunction.Name() = "FunctionName";
  revng_check(get<1>(TheFunction) == "FunctionName");

  // Test std::tuple_size
  static_assert(std::tuple_size<Function>::value >= 2);

  // Test TupleLikeTraits
  static_assert(TraitedTupleLike<Function>);
  using TLT = TupleLikeTraits<Function>;
  static_assert(std::is_same_v<std::tuple_element_t<1, Function> &,
                               decltype(TheFunction.Name())>);
  revng_check(StringRef(TLT::Name) == "Function");
  revng_check(StringRef(TLT::FullName) == "model::Function");
  revng_check(StringRef(TLT::FieldNames[1]) == "Name");
}

BOOST_AUTO_TEST_CASE(TestPathAccess) {
  Binary Binary;

  TupleTreePath Zero;
  Zero.push_back(size_t(0));
  auto *FirstField = getByPath<uint64_t>(Zero, Binary);
  revng_check(FirstField == &Binary.Version());

  using FunctionsType = std::decay_t<decltype(Binary.Functions())>;
  auto *FunctionsField = getByPath<FunctionsType>("/Functions", Binary);
  revng_check(FunctionsField == &Binary.Functions());

  // Test non existing field
  revng_check(getByPath<FunctionsType>("/Function", Binary) == nullptr);

  // Test non existing entry in container
  revng_check(getByPath<Function>("/Functions/:Invalid", Binary) == nullptr);

  // Test existing entry in container
  Function &F = Binary.Functions()[MetaAddress::invalid()];
  revng_check(getByPath<Function>("/Functions/:Invalid", Binary) == &F);

  // Test UpcastablePointer
  auto &&[Typedef, TypedefType] = Binary.makeTypedefDefinition();
  Typedef.UnderlyingType() = model::PrimitiveType::make(PrimitiveKind::Unsigned,
                                                        8);

  std::string Path = "/TypeDefinitions/" + toString(Typedef.key())
                     + "/TypedefDefinition::Name";
  auto *OriginalNamePointer = getByPath<std::string>(Path, Binary);
  revng_check(OriginalNamePointer == &Typedef.Name());

  Path = "/TypeDefinitions/" + toString(Typedef.key());
  revng_check(getByPath<model::TypeDefinition>(Path, Binary) == &Typedef);
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
  revng_check(stringAsPath<Binary>("/Version").value() == Zero);

  TupleTreePath InvalidFunctionPath;
  auto FunctionIndex = TupleLikeTraits<model::Binary>::Fields::Functions;
  InvalidFunctionPath.push_back(static_cast<size_t>(FunctionIndex));
  InvalidFunctionPath.push_back(MetaAddress::invalid());
  auto MaybeInvalidFunctionPath = stringAsPath<Binary>("/Functions/:Invalid");
  revng_check(MaybeInvalidFunctionPath.value() == InvalidFunctionPath);

  TupleTreePath InvalidFunctionNamePath = InvalidFunctionPath;
  auto NameIndex = TupleLikeTraits<model::Function>::Fields::Name;
  InvalidFunctionNamePath.push_back(static_cast<size_t>(NameIndex));
  auto MaybePath = stringAsPath<Binary>("/Functions/:Invalid/Name");
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
    auto Matcher = PathMatcher::create<Binary>("/TypeDefinitions"
                                               "/*-RawFunctionDefinition/"
                                               "RawFunctionDefinition::"
                                               "FinalStackOffset")
                     .value();

    model::TypeDefinition::Key Key{
      1000, model::TypeDefinitionKind::RawFunctionDefinition
    };
    auto Path1000 = pathAsString<Binary>(Matcher.apply(Key));
    std::string SerializedPath1000 = "/TypeDefinitions"
                                     "/1000-RawFunctionDefinition/"
                                     "RawFunctionDefinition::FinalStackOffset";
    revng_check(Path1000 == SerializedPath1000);

    auto ToMatch = stringAsPath<Binary>(*Path1000);
    revng_check(ToMatch);
    auto Match = Matcher.match<model::TypeDefinition::Key>(ToMatch.value());
    revng_check(Match);
    revng_check(std::get<0>(*Match) == Key);

    ToMatch = stringAsPath<Binary>("/TypeDefinitions"
                                   "/1000-CABIFunctionDefinition/"
                                   "CABIFunctionDefinition::ID");
    Match = Matcher.match<model::TypeDefinition::Key>(ToMatch.value());
    revng_check(not Match);
  }
}

BOOST_AUTO_TEST_CASE(TestModelDeduplication) {
  TupleTree<model::Binary> Model;
  auto Dedup = [&Model]() {
    int64_t OldTypesCount = Model->TypeDefinitions().size();
    deduplicateEquivalentTypes(Model);
    int64_t NewTypesCount = Model->TypeDefinitions().size();
    return OldTypesCount - NewTypesCount;
  };

  auto UInt32 = model::PrimitiveType::makeGeneric(4);

  // Two typedefs
  {
    auto &Typedef1 = Model->makeTypedefDefinition(UInt32.copy()).first;
    auto &Typedef2 = Model->makeTypedefDefinition(UInt32.copy()).first;

    revng_check(Dedup() == 0);

    Typedef1.Name() = "MyUInt8";
    Typedef2.Name() = "MyUInt8";

    revng_check(Dedup() == 1);
  }

  // Two structs
  {
    auto &Struct1 = Model->makeStructDefinition().first;
    Struct1.Fields()[0].Name() = "FirstField";
    Struct1.Fields()[0].Type() = UInt32.copy();
    Struct1.Name() = "MyStruct";

    auto &Struct2 = Model->makeStructDefinition().first;
    Struct2.Fields()[0].Name() = "DifferentName";
    Struct2.Fields()[0].Type() = UInt32.copy();
    Struct2.Name() = "MyStruct";

    revng_check(Dedup() == 0);

    Struct1.Fields()[0].Name() = Struct2.Fields()[0].Name();

    revng_check(Dedup() == 1);
  }

  // Two pairs of cross-referencing structs
  {
    using Pointer = model::PointerType;

    auto &&[LeftStruct1, LeftType1] = Model->makeStructDefinition();
    auto &&[LeftStruct2, LeftType2] = Model->makeStructDefinition();

    LeftStruct1.Fields()[0].Type() = Pointer::make(std::move(LeftType2), 8);
    LeftStruct2.Fields()[0].Type() = Pointer::make(std::move(LeftType1), 8);

    LeftStruct1.Name() = "LoopingStructs1";
    LeftStruct2.Name() = "LoopingStructs2";

    auto &&[RightStruct1, RightType1] = Model->makeStructDefinition();
    auto &&[RightStruct2, RightType2] = Model->makeStructDefinition();

    RightStruct1.Fields()[0].Type() = Pointer::make(std::move(RightType2), 8);

    auto DoublePtr = Pointer::make(Pointer::make(std::move(RightType1), 8), 8);
    RightStruct2.Fields()[0].Type() = std::move(DoublePtr);

    RightStruct1.Name() = "LoopingStructs1";
    RightStruct2.Name() = "LoopingStructs2";

    revng_check(Dedup() == 0);

    model::UpcastableType &FieldType = RightStruct2.Fields()[0].Type();
    FieldType = std::move(llvm::cast<Pointer>(*FieldType).PointeeType());

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
  std::string S = toString(Diff);

  auto Diff2 = llvm::cantFail(fromString<TupleTreeDiff<model::Binary>>(S));
  std::string S2 = toString(Diff2);

  BOOST_TEST(S == S2);
}

BOOST_AUTO_TEST_CASE(CABIFunctionTypePathShouldParse) {
  const char *Path = "/TypeDefinitions/10000-CABIFunctionDefinition";
  auto MaybeParsed = stringAsPath<model::Binary>(Path);
  BOOST_TEST(MaybeParsed.has_value());
}

BOOST_AUTO_TEST_CASE(CABIFunctionTypeArgumentsPathShouldParse) {
  const char *Path = "/TypeDefinitions/10000-CABIFunctionDefinition/"
                     "CABIFunctionDefinition::Arguments";
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
  BOOST_TEST(Collected.Read.size() == 1);
  std::vector StringPaths = { "/Segments/0x0:Code_x86_64-1000" };

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
  BOOST_TEST(Collected.Read.size() == 1);
  std::set<TupleTreePath> Paths = {
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

  BOOST_TEST(Collected.Read.size() == 0);

  std::set<TupleTreePath> Paths = {
    *stringAsPath<model::Binary>("/Segments"),
  };
  BOOST_TEST(Collected.ExactVectors == Paths);
}
