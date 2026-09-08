//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"

#include "revng/ADT/RecursiveCoroutine-coroutine.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionBundle.h"
#include "revng/Model/Binary.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Pipebox/Helpers.h"
#include "revng/PipeboxCommon/Helpers/PipeRuns/LLVMFunctionMixin.h"
#include "revng/Ranks/Location.h"
#include "revng/SegmentReferences/SegmentUsesEnumerator.h"
#include "revng/SegmentReferences/StringConstants.h"
#include "revng/Support/Unicode.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"
#include "revng/Yield/Generated/ForwardDecls.h"
#include "revng/Yield/Pipes/CollectGlobals.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraph.h"
#include "revng/Yield/Pipes/YieldCallGraphSlice.h"
#include "revng/Yield/SVG.h"

namespace revng::pypeline::piperuns {

void ProcessCallGraph::run() {
  using namespace yield::crossrelations;

  SortedVector<efa::ControlFlowGraph> Metadata;
  for (const ObjectID &Object : Input.objects())
    Metadata.insert(Input.getElement(Object)->MainFunction());

  *Output.getElement(ObjectID()) = CrossRelations(Metadata, Binary);
}

/// How much of a string is kept in the cross relations. The whole text is in
/// the binary, which is where a consumer wanting all of it should read it from.
///
/// This is part of what the schema promises, so `GlobalDescription::Preview`
/// says the same number.
constexpr uint64_t StringPreviewCodePoints = 64;

namespace {

/// Global variables indexed by address, with the functions referencing them.
class GlobalVariables : public LLVMFunctionMixin<GlobalVariables> {
private:
  struct Entry {
    MetaAddress Start;
    MetaAddress End;
    std::string Location;
    std::string Name;
    const model::Type *Type = nullptr;
    llvm::SmallVector<MetaAddress, 2> Users;
  };

private:
  std::vector<Entry> Entries;
  SegmentUsesEnumerator Enumerator;

public:
  GlobalVariables(const model::Binary &Binary, LLVMFunctionContainer &Input) :
    LLVMFunctionMixin(Input),
    Enumerator(Binary, SegmentUsesEnumerator::SegmentAccess::All) {}

public:
  static GlobalVariables fromModel(const model::Binary &Binary,
                                   LLVMFunctionContainer &Input);

public:
  void registerUse(MetaAddress Address, MetaAddress FunctionEntry);

  void runOnLLVMFunction(const model::Function &Function,
                         llvm::Function &LLVMFunction);

  void emit(RawBinaryView &BinaryView,
            yield::crossrelations::CrossRelations &Output);

private:
  Entry *findContaining(MetaAddress Address);

  RecursiveCoroutine<void> collect(const model::StructDefinition &Struct,
                                   const MetaAddress &StartAddress);
};

/// Descend through singleton structs and collect their non-singleton fields.
RecursiveCoroutine<void>
GlobalVariables::collect(const model::StructDefinition &Struct,
                         const MetaAddress &StartAddress) {
  namespace ranks = revng::ranks;

  for (const model::StructField &Field : Struct.Fields()) {
    MetaAddress Start = StartAddress + Field.Offset();
    if (not Start.isValid())
      continue;

    if (const model::StructDefinition *Nested = Field.Type()->getStruct()) {
      if (Nested->IsSingleton()) {
        rc_recur collect(*Nested, Start);
        continue;
      }
    }

    std::optional<uint64_t> Size = Field.Type()->size();
    if (not Size.has_value() or *Size == 0)
      continue;

    MetaAddress End = Start + *Size;
    if (not End.isValid())
      continue;

    Entries.push_back({ Start,
                        End,
                        pipeline::locationString(ranks::StructField,
                                                 Struct.key(),
                                                 Field.key()),
                        Field.Name(),
                        Field.Type().get(),
                        {} });
  }
}

GlobalVariables GlobalVariables::fromModel(const model::Binary &Binary,
                                           LLVMFunctionContainer &Input) {
  GlobalVariables Result(Binary, Input);

  for (const model::Segment &Segment : Binary.Segments()) {
    if (Segment.Type().isEmpty())
      continue;

    Result.collect(*Segment.type(), Segment.StartAddress());
  }

  // Sort so we can perform binary search
  std::ranges::sort(Result.Entries, {}, &Entry::Start);

  return Result;
}

/// Include interior addresses, such as references into the middle of a string.
GlobalVariables::Entry *GlobalVariables::findContaining(MetaAddress Address) {
  auto Iterator = std::ranges::upper_bound(Entries, Address, {}, &Entry::Start);
  if (Iterator == Entries.begin())
    return nullptr;

  // Globals do not overlap, so only the preceding entry can contain Address.
  --Iterator;
  if (Address >= Iterator->End)
    return nullptr;

  return &*Iterator;
}

void GlobalVariables::registerUse(MetaAddress Address,
                                  MetaAddress FunctionEntry) {
  if (Entry *Global = findContaining(Address))
    Global->Users.push_back(FunctionEntry);
}

/// Fill in what a global variable says, when it holds a string.
static void describeString(yield::crossrelations::GlobalDescription &Global,
                           const model::Type &Type,
                           RawBinaryView &BinaryView) {
  using namespace yield::crossrelations;

  unsigned CharSize = getConstCharArrayElementSize(Type);
  if (CharSize == 0)
    return;

  UnicodeCStringView String = readString(BinaryView,
                                         Global.Address(),
                                         Global.Size(),
                                         CharSize);
  if (not String.isValid())
    return;

  switch (String.encoding()) {
  case UnicodeCStringView::Encoding::UTF8:
    Global.Encoding() = StringEncoding::UTF8;
    break;
  case UnicodeCStringView::Encoding::UTF16LE:
    Global.Encoding() = StringEncoding::UTF16LE;
    break;
  case UnicodeCStringView::Encoding::UTF16BE:
    Global.Encoding() = StringEncoding::UTF16BE;
    break;
  case UnicodeCStringView::Encoding::Invalid:
    revng_abort("A valid string cannot have an invalid encoding.");
  }

  Global.CodePoints() = String.codePointCount();
  Global.Preview() = String.truncate(StringPreviewCodePoints);
}

void GlobalVariables::emit(RawBinaryView &BinaryView,
                           yield::crossrelations::CrossRelations &Output) {
  using namespace yield::crossrelations;

  auto Inserter = Output.Globals().batch_insert();
  for (Entry &Global : Entries) {
    GlobalDescription Description;
    Description.Location() = Global.Location;
    Description.Name() = Global.Name;
    Description.Address() = Global.Start;
    Description.Size() = (Global.End - Global.Start).value();
    describeString(Description, *Global.Type, BinaryView);

    // Remove duplicates
    std::ranges::sort(Global.Users);
    Global.Users.erase(std::unique(Global.Users.begin(), Global.Users.end()),
                       Global.Users.end());

    for (MetaAddress Function : Global.Users) {
      auto Location = pipeline::locationString(revng::ranks::Function,
                                               Function);
      Description.Users().insert(std::move(Location));
    }

    Inserter.insert(std::move(Description));
  }
}

void GlobalVariables::runOnLLVMFunction(const model::Function &Function,
                                        llvm::Function &LLVMFunction) {
  // Follow segment getters before filtering uses to the hosting function.
  for (auto &&Use : Enumerator.getUses(*LLVMFunction.getParent())) {
    auto *User = llvm::cast<llvm::Instruction>(Use.TheUse->getUser());
    if (User->getFunction() != &LLVMFunction or not Use.Address.isValid())
      continue;

    registerUse(Use.Address, Function.Entry());
  }
}

} // namespace

void CollectGlobals::run() {
  GlobalVariables Globals = GlobalVariables::fromModel(Binary, Input);
  for (const ObjectID &Object : Input.objects()) {
    const auto &Entry = std::get<MetaAddress>(Object.key());
    Globals.runOnFunction(Binary.Functions().at(Entry));
  }

  Globals.emit(BinaryView, *Output.getElement(ObjectID()).get());
}

void YieldCallGraph::run() {
  using namespace yield::crossrelations;
  const TupleTree<CrossRelations> &Relations = Input.getElement(ObjectID());

  ptml::MarkupBuilder B;
  auto OS = Output.getOStream(ObjectID());
  // Convert the graph to SVG.
  *OS << yield::svg::callGraph(B, *Relations.get(), Binary);
}

void YieldCallGraphSlice::runOnFunction(const model::Function &Function) {
  using namespace yield::crossrelations;
  const TupleTree<CrossRelations> &Relations = Input.getElement(ObjectID());

  // Slice the graph for the current function and convert it to SVG
  auto SlicePoint = pipeline::locationString(revng::ranks::Function,
                                             Function.Entry());

  auto OS = Output.getOStream(ObjectID(Function.Entry()));
  *OS << yield::svg::callGraphSlice(B, SlicePoint, *Relations.get(), Binary);
}

} // namespace revng::pypeline::piperuns
