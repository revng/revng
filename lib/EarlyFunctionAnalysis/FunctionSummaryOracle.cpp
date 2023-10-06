/// \file FunctionSummaryOracle.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"

using namespace llvm;

static Logger<> Log("efa-import-model");

namespace efa {

static FunctionSummary
importPrototype(Module &M,
                const llvm::SmallVector<GlobalVariable *, 16> &ABICSVs,
                const AttributesSet &Attributes,
                const model::TypePath &Prototype) {
  using namespace llvm;
  using namespace model;
  using Register = model::Register::Values;
  using State = abi::RegisterState::Values;

  FunctionSummary Summary(Attributes,
                          { ABICSVs.begin(), ABICSVs.end() },
                          ABIAnalyses::ABIAnalysesResults(),
                          {},
                          0);

  for (GlobalVariable *CSV : ABICSVs) {
    Summary.ABIResults.ArgumentsRegisters[CSV] = State::No;
    Summary.ABIResults.FinalReturnValuesRegisters[CSV] = State::No;
  }

  auto Layout = abi::FunctionType::Layout::make(Prototype);

  for (Register ArgumentRegister : Layout.argumentRegisters()) {
    StringRef Name = model::Register::getCSVName(ArgumentRegister);
    if (GlobalVariable *CSV = M.getGlobalVariable(Name, true))
      Summary.ABIResults.ArgumentsRegisters.at(CSV) = State::Yes;
  }

  for (Register ReturnValueRegister : Layout.returnValueRegisters()) {
    StringRef Name = model::Register::getCSVName(ReturnValueRegister);
    if (GlobalVariable *CSV = M.getGlobalVariable(Name, true))
      Summary.ABIResults.FinalReturnValuesRegisters.at(CSV) = State::Yes;
  }

  std::set<llvm::GlobalVariable *> PreservedRegisters;
  for (Register CalleeSavedRegister : Layout.CalleeSavedRegisters) {
    StringRef Name = model::Register::getCSVName(CalleeSavedRegister);
    if (GlobalVariable *CSV = M.getGlobalVariable(Name, true))
      PreservedRegisters.insert(CSV);
  }

  std::erase_if(Summary.ClobberedRegisters,
                [&](const auto &E) { return PreservedRegisters.contains(E); });

  Summary.ElectedFSO = Layout.FinalStackOffset;
  return Summary;
}

std::pair<const FunctionSummary *, bool>
FunctionSummaryOracle::getCallSite(MetaAddress Function,
                                   BasicBlockID CallerBlockAddress,
                                   MetaAddress CalledLocalFunction,
                                   llvm::StringRef CalledSymbol) const {
  auto [Summary, IsTailCall] = getCallSiteImpl(Function, CallerBlockAddress);
  if (Summary != nullptr) {
    return { Summary, IsTailCall };
  } else if (not CalledSymbol.empty()) {
    return { &getDynamicFunction(CalledSymbol), false };
  } else if (CalledLocalFunction.isValid()
             and LocalFunctions.contains(CalledLocalFunction)) {
    return { &getLocalFunction(CalledLocalFunction), false };
  } else {
    return { &getDefault(), false };
  }
}

bool FunctionSummaryOracle::registerCallSite(MetaAddress Function,
                                             BasicBlockID CallSite,
                                             FunctionSummary &&New,
                                             bool IsTailCall) {
  revng_assert(Function.isValid());
  revng_assert(CallSite.isValid());
  std::pair<MetaAddress, BasicBlockID> Key = { Function, CallSite };
  auto It = CallSites.find(Key);
  if (It != CallSites.end()) {
    auto &Recorded = It->second.first;
    bool Changed = not New.containedOrEqual(Recorded);
    New.combine(Recorded);
    Recorded = std::move(New);
    return Changed;
  } else {
    CallSiteDescriptor CSD = { std::move(New), IsTailCall };
    CallSites.emplace(Key, std::move(CSD));
    return true;
  }
}

bool FunctionSummaryOracle::registerLocalFunction(MetaAddress PC,
                                                  FunctionSummary &&New) {
  revng_assert(PC.isValid());

  if (Log.isEnabled()) {
    Log << "registerLocalFunction " << PC.toString() << " with summary:\n";
    New.dump(Log);
    Log << DoLog;
  }

  auto It = LocalFunctions.find(PC);
  bool Changed = It == LocalFunctions.end();
  if (not Changed) {
    auto &Recorded = It->second;
    Changed = not New.containedOrEqual(Recorded);
    New.combine(Recorded);
    Recorded = std::move(New);
  } else {
    LocalFunctions.emplace(PC, std::move(New));
  }

  return Changed;
}

bool FunctionSummaryOracle::registerDynamicFunction(llvm::StringRef Name,
                                                    FunctionSummary &&New) {
  auto It = DynamicFunctions.find(Name.str());
  if (It != DynamicFunctions.end()) {
    auto &Recorded = It->second;
    bool Changed = not New.containedOrEqual(Recorded);
    New.combine(Recorded);
    Recorded = std::move(New);
    return Changed;
  } else {
    DynamicFunctions.emplace(Name, std::move(New));
    return true;
  }
}

void importModel(Module &M,
                 GeneratedCodeBasicInfo &GCBI,
                 const model::Binary &Binary,
                 FunctionSummaryOracle &Oracle) {
  revng_log(Log, "Importing from model");
  LoggerIndent Indent(Log);

  llvm::SmallVector<GlobalVariable *, 16> ABICSVs;
  for (GlobalVariable *CSV : GCBI.abiRegisters())
    if (CSV != nullptr && !(GCBI.isSPReg(CSV)))
      ABICSVs.emplace_back(CSV);

  // Import the default prototype
  revng_assert(Binary.DefaultPrototype().isValid());
  Oracle.setDefault(importPrototype(M, ABICSVs, {}, Binary.DefaultPrototype()));

  std::map<llvm::BasicBlock *, MetaAddress> InlineFunctions;

  // Import existing functions from model
  for (const model::Function &Function : Binary.Functions()) {
    // Import call-site specific information
    for (const model::CallSitePrototype &CallSite :
         Function.CallSitePrototypes()) {

      AttributesSet Attributes;
      for (auto &ToCopy : CallSite.Attributes())
        Attributes.insert(ToCopy);
      Oracle.registerCallSite(Function.Entry(),
                              BasicBlockID(CallSite.CallerBlockAddress()),
                              importPrototype(M,
                                              ABICSVs,
                                              Attributes,
                                              CallSite.prototype()),
                              CallSite.IsTailCall());
    }

    AttributesSet Attributes;
    for (auto &ToCopy : Function.Attributes())
      Attributes.insert(ToCopy);
    auto Summary = importPrototype(M,
                                   ABICSVs,
                                   Attributes,
                                   Function.prototype(Binary));

    // Create function to inline, if necessary
    if (Summary.Attributes.contains(model::FunctionAttribute::Inline))
      InlineFunctions[GCBI.getBlockAt(Function.Entry())] = Function.Entry();

    Oracle.registerLocalFunction(Function.Entry(), std::move(Summary));
  }

  // Register all dynamic symbols
  for (const auto &DynamicFunction : Binary.ImportedDynamicFunctions()) {
    const auto &Prototype = DynamicFunction.prototype(Binary);
    AttributesSet Attributes;
    for (auto &ToCopy : DynamicFunction.Attributes())
      Attributes.insert(ToCopy);
    Oracle.registerDynamicFunction(DynamicFunction.OriginalName(),
                                   importPrototype(M,
                                                   ABICSVs,
                                                   Attributes,
                                                   Prototype));
  }
}

} // namespace efa
