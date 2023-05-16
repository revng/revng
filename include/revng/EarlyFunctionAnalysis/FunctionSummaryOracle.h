#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "revng/ADT/MutableSet.h"
#include "revng/EarlyFunctionAnalysis/ABIAnalysis.h"
#include "revng/EarlyFunctionAnalysis/BasicBlock.h"
#include "revng/Model/FunctionAttribute.h"

namespace llvm {
class Module;
}

namespace efa {

class FunctionSummaryOracle;

void importModel(llvm::Module &M,
                 GeneratedCodeBasicInfo &GCBI,
                 const model::Binary &Binary,
                 FunctionSummaryOracle &Oracle);

using AttributesSet = MutableSet<model::FunctionAttribute::Values>;

/// A summary of the analysis of a function.
///
/// For each function detected, the following information are included: function
/// attributes (inline, noreturn), which ABI registers are clobbered, its
/// control-flow graph, and an elected stack offset (to tell if the stack
/// pointer is restored at its original position).
struct FunctionSummary {
public:
  AttributesSet Attributes;
  std::set<llvm::GlobalVariable *> ClobberedRegisters;
  // TODO: this field is populated in a different maneer from all the others,
  //       consider changing how this works
  ABIAnalyses::ABIAnalysesResults ABIResults;
  SortedVector<efa::BasicBlock> CFG;
  std::optional<int64_t> ElectedFSO;

public:
  FunctionSummary(MutableSet<model::FunctionAttribute::Values> Attributes,
                  std::set<llvm::GlobalVariable *> ClobberedRegisters,
                  ABIAnalyses::ABIAnalysesResults ABIResults,
                  SortedVector<efa::BasicBlock> CFG,
                  std::optional<int64_t> ElectedFSO) :
    Attributes(Attributes),
    ClobberedRegisters(std::move(ClobberedRegisters)),
    ABIResults(std::move(ABIResults)),
    CFG(std::move(CFG)),
    ElectedFSO(ElectedFSO) {}

  FunctionSummary() = default;
  FunctionSummary(const FunctionSummary &) = delete;
  FunctionSummary &operator=(const FunctionSummary &) = delete;

  FunctionSummary(FunctionSummary &&Other) = default;
  FunctionSummary &operator=(FunctionSummary &&Other) = default;

public:
  void combine(const FunctionSummary &Other) {
    ClobberedRegisters.insert(Other.ClobberedRegisters.begin(),
                              Other.ClobberedRegisters.end());

    if (Other.Attributes.contains(model::FunctionAttribute::NoReturn))
      Attributes.insert(model::FunctionAttribute::NoReturn);
  }

  bool containedOrEqual(const FunctionSummary &Other) const {
    const auto NoReturn = model::FunctionAttribute::NoReturn;

    if (Attributes.contains(NoReturn) && !Other.Attributes.contains(NoReturn))
      return false;

    return std::includes(Other.ClobberedRegisters.begin(),
                         Other.ClobberedRegisters.end(),
                         ClobberedRegisters.begin(),
                         ClobberedRegisters.end());
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << "Dumping summary \n"
           << "  Attributes: [";

    for (model::FunctionAttribute::Values Attribute : Attributes)
      Output << " " << model::FunctionAttribute::getName(Attribute).str();
    Output << " ]\n";

    Output << "  ElectedFSO: " << (ElectedFSO.has_value() ? *ElectedFSO : -1)
           << "\n"
           << "  Clobbered registers: [";

    for (auto *Reg : ClobberedRegisters)
      Output << " " << Reg->getName().str();
    Output << " ]\n";

    Output << "  ABI info: \n";
    ABIResults.dump(Output);
  }
};

/// An oracle providing information about functions.
///
/// This oracle can be populated with analysis results. But even if it has not
/// been popoulated with any result, it will still provide conservative results
/// about the function.
class FunctionSummaryOracle {
private:
  /// Call site-specific overrides
  ///
  /// Key is composed by <FunctionEntryPoint, CallSiteBasicBlockAddress>
  /// Value is composed by <FunctionSummary, IsTailCall>
  using CallSiteDescriptor = std::pair<FunctionSummary, bool>;
  std::map<std::pair<MetaAddress, BasicBlockID>, CallSiteDescriptor> CallSites;

  /// Local functions
  std::map<MetaAddress, FunctionSummary> LocalFunctions;

  /// Dynamic functions
  std::map<std::string, FunctionSummary> DynamicFunctions;

  /// Default
  FunctionSummary Default;

public:
  const FunctionSummary &getDefault() const { return Default; }

  const FunctionSummary &getLocalFunction(MetaAddress PC) const {
    return LocalFunctions.at(PC);
  }

  FunctionSummary &getLocalFunction(MetaAddress PC) {
    return LocalFunctions.at(PC);
  }

  /// \return a description of the call and boolean indicating whether the call
  ///         site is a tail call or not.
  std::pair<const FunctionSummary *, bool>
  getCallSite(MetaAddress Function,
              BasicBlockID CallerBlockAddress,
              MetaAddress CalledLocalFunction,
              llvm::StringRef CalledSymbol) const;

public:
  bool registerCallSite(MetaAddress Function,
                        BasicBlockID CallSite,
                        FunctionSummary &&New,
                        bool IsTailCall);

  bool registerLocalFunction(MetaAddress PC, FunctionSummary &&New);

  bool registerDynamicFunction(llvm::StringRef Name, FunctionSummary &&New);

  void setDefault(FunctionSummary &&Summary) { Default = std::move(Summary); }

  const FunctionSummary &getDynamicFunction(llvm::StringRef Name) const {
    return DynamicFunctions.at(Name.str());
  }

private:
  std::pair<const FunctionSummary *, bool>
  getCallSiteImpl(MetaAddress Function, BasicBlockID CallSite) const {
    auto It = CallSites.find({ Function, CallSite });
    if (It == CallSites.end())
      return { nullptr, false };
    else
      return { &It->second.first, It->second.second };
  }
};

} // namespace efa
