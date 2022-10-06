/// \file IsolatedFunctionKind.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType.h"
#include "revng/FunctionIsolation/IsolationFunctionKind.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipes/InvalidationHelpers.h"
#include "revng/Pipes/TupleTreeContainer.h"
#include "revng/Support/Assert.h"
#include "revng/Support/MetaAddress.h"
#include "revng/TupleTree/TupleTreeDiff.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"
#include "revng/Yield/Pipes/ProcessCallGraph.h"

using namespace pipeline;
using namespace ::revng::kinds;
using namespace llvm;

static bool
haveFSOAndPreservedRegsChanged(const model::TypePath &OldPrototype,
                               const model::TypePath &NewPrototype) {
  auto Old = abi::FunctionType::Layout::make(OldPrototype);
  auto New = abi::FunctionType::Layout::make(NewPrototype);

  return Old.FinalStackOffset != New.FinalStackOffset
         or Old.CalleeSavedRegisters != New.CalleeSavedRegisters;
}

using IsolatedFK = IsolatedFunctionKind;
void IsolatedFK::getInvalidations(const Context &Ctx,
                                  TargetsList &ToRemove,
                                  const GlobalTupleTreeDiff &Diff,
                                  const pipeline::Global &Before,
                                  const pipeline::Global &After) const {
  using PM = PathMatcher;
  using Binary = model::Binary;
  using FunctionAttribute = model::FunctionAttribute::Values;

  if (not llvm::isa<TupleTreeGlobal<Binary>>(Before))
    return;

  std::set<MetaAddress> Targets;
  const auto &OldModel = cast<TupleTreeGlobal<Binary>>(Before).get();
  const auto &Model = cast<TupleTreeGlobal<Binary>>(After).get();

  // CrossRelations container not found? Take the most conservative approach,
  // namely, to invalidate all targets.
  if (not Ctx.containsReadOnlyContainer(BinaryCrossRelationsRole)) {
    invalidateAllTargetsPerFunctionRank(Model, Targets);
    return;
  }

  static constexpr auto &BCRR = BinaryCrossRelationsRole;
  using CRFileContainer = pipes::CrossRelationsFileContainer;
  const auto &Container = Ctx.getReadOnlyContainer<CRFileContainer>(BCRR);
  if (Container.empty()) {
    invalidateAllTargetsPerFunctionRank(Model, Targets);
    return;
  }

  using CrossRelationTree = TupleTree<yield::crossrelations::CrossRelations>;
  const CrossRelationTree &CrossRelations = Container.get();
  const TupleTreeDiff<Binary> *ModelDiff = Diff.getAs<Binary>();
  if (not ModelDiff)
    return;

  auto FuncAttrMatcher = PM::create<Binary>("/Functions/*/Attributes").value();
  auto DynFuncAttrMatcher = PM::create<Binary>("/ImportedDynamicFunctions/*/"
                                               "Attributes")
                              .value();
  const auto &IDF = Model->ImportedDynamicFunctions();
  const auto &CRGraph = CrossRelations->toCallGraph();

  for (const Change<Binary> &Change : ModelDiff->Changes) {
    revng_assert(not Change.Path.empty());

    // A new `model::Function` has been added or removed?
    if (pathAsString<Binary>(Change.Path) == "/Functions") {
      if (Change.New) {
        // Invalidate all targets
        invalidateAllTargetsPerFunctionRank(Model, Targets);
      } else if (Change.Old) {
        // Invalidate function's callers
        auto Function = std::get<model::Function>(*Change.Old);
        insertCallersAndTransitiveClosureIfInline(CRGraph,
                                                  Model,
                                                  Function.Entry(),
                                                  Targets);
      } else {
        revng_abort();
      }
    }

    // A new `model::DynamicFunction` has been added or removed?
    if (pathAsString<Binary>(Change.Path) == "/ImportedDynamicFunctions") {
      if (Change.New) {
        // Invalidate all targets
        invalidateAllTargetsPerFunctionRank(Model, Targets);
      } else if (Change.Old) {
        // Invalidate model::DynamicFunction's callers
        auto DynamicFunction = std::get<model::DynamicFunction>(*Change.Old);
        insertCallersAndTransitiveClosureIfInline(CRGraph,
                                                  Model,
                                                  DynamicFunction
                                                    .OriginalName(),
                                                  Targets);
      } else {
        revng_abort();
      }
    }

    // Did FinalStackOffset or PreservedRegisters for a `model::Function` have
    // changed?
    // Note that, since matching e.g., FinalStackOffset, for both
    // RawFunctionType and CABIFunctionType is non-trivial, we will not leverage
    // the diff for checking the change related to FSO. Instead, we move to
    // `Layout` by employing the old model, and the new one (the one with the
    // change).
    for (const model::Function &Function : Model->Functions()) {
      MetaAddress Entry = Function.Entry();
      auto It = OldModel->Functions().find(Entry);
      if (It != OldModel->Functions().end())
        if (haveFSOAndPreservedRegsChanged(It->Prototype(),
                                           Function.Prototype())) {
          Targets.insert(Entry);
          insertCallersAndTransitiveClosureIfInline(CRGraph,
                                                    Model,
                                                    Entry,
                                                    Targets);
        }
    }

    // Did FinalStackOffset or PreservedRegisters for a `model::DynamicFunction`
    // have changed?
    for (const model::DynamicFunction &Function : IDF) {
      std::string Name = Function.OriginalName();
      const auto &OldIDF = OldModel->ImportedDynamicFunctions();
      auto It = OldIDF.find(Name);
      if (It != OldIDF.end()) {
        if (haveFSOAndPreservedRegsChanged(getPrototype(*OldModel, *It),
                                           getPrototype(*Model, Function)))
          insertCallersAndTransitiveClosureIfInline(CRGraph,
                                                    Model,
                                                    Name,
                                                    Targets);
      }
    }

    // Did attributes for a `model::Function` have changed?
    auto MaybeFuncAttr = FuncAttrMatcher.match<MetaAddress>(Change.Path);
    if (MaybeFuncAttr) {
      auto Entry = std::get<MetaAddress>(*MaybeFuncAttr);
      bool ToChange = false;

      if (Change.Old.has_value()) {
        auto OldAttribute = std::get<FunctionAttribute>(*Change.Old);
        if (OldAttribute == FunctionAttribute::NoReturn
            || OldAttribute == FunctionAttribute::Inline)
          ToChange = true;
      }

      if (Change.New.has_value()) {
        auto NewAttribute = std::get<FunctionAttribute>(*Change.New);

        // If the attribute of a function changes to `NoReturn` or `Inline`, we
        // need to invalidate its callers (not the function itself as it would
        // not provide any further information to the function itself).
        if (NewAttribute == FunctionAttribute::NoReturn
            || NewAttribute == FunctionAttribute::Inline)
          ToChange = true;
      }

      if (ToChange)
        insertCallersAndTransitiveClosureIfInline(CRGraph,
                                                  Model,
                                                  Entry,
                                                  Targets);
    }

    // Did attributes for a `model::DynamicFunction` have changed?
    auto MaybeDynFuncAttr = DynFuncAttrMatcher.match<std::string>(Change.Path);
    if (MaybeDynFuncAttr) {
      auto Entry = std::get<std::string>(*MaybeDynFuncAttr);
      bool ToChange = false;

      if (Change.Old.has_value()) {
        auto OldAttribute = std::get<FunctionAttribute>(*Change.Old);
        if (OldAttribute == FunctionAttribute::NoReturn)
          ToChange = true;
      }

      if (Change.New.has_value()) {
        auto NewAttribute = std::get<FunctionAttribute>(*Change.New);

        // There is no body for a `model::DynamicFunction`, cannot be inline.
        revng_assert(NewAttribute != FunctionAttribute::Inline);

        if (NewAttribute == FunctionAttribute::NoReturn)
          ToChange = true;
      }

      if (ToChange)
        insertCallers(CrossRelations, Entry, Targets);
    }
  }

  // Fill TargetsList vector
  for (MetaAddress Entry : Targets)
    ToRemove.emplace_back(Target(Entry.toString(), *this));
}
