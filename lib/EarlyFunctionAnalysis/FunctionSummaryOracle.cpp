/// \file FunctionSummaryOracle.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/GlobalVariable.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/STLExtras.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"

static Logger<> Log("efa-import-model");

namespace efa {

FunctionSummary
PrototypeImporter::prototype(const AttributesSet &Attributes,
                             const model::DefinitionReference &Prototype) {
  FunctionSummary Summary(Attributes, ABICSVs, {}, {}, {});
  if (Prototype.empty())
    return Summary;

  // Drop known to be preserved registers from `Summary.ClobberedRegisters`.
  auto TransformToCSV = std::views::transform([this](Register Register) {
    return M.getGlobalVariable(model::Register::getCSVName(Register), true);
  });
  auto IgnoreNullptr = std::views::filter([](const auto *Pointer) -> bool {
    return Pointer != nullptr;
  });
  for (auto *CSV : abi::FunctionType::calleeSavedRegisters(Prototype)
                     | TransformToCSV | IgnoreNullptr) {
    Summary.ClobberedRegisters.erase(CSV);
  }

  // Stop importing prototype here, if `ClobberedRegisters` is the only
  // information callee needs.
  if (Level == PrototypeImportLevel::None)
    return Summary;

  Summary.ElectedFSO = abi::FunctionType::finalStackOffset(Prototype);

  // Stop importing prototype here, if callee also needs final stack offset.
  if (Level == PrototypeImportLevel::Partial)
    return Summary;

  for (llvm::GlobalVariable *CSV : ABICSVs) {
    Summary.ABIResults.ArgumentsRegisters.erase(CSV);
    Summary.ABIResults.ReturnValuesRegisters.erase(CSV);
  }

  auto [ArgumentRegisters,
        ReturnValueRegisters] = abi::FunctionType::usedRegisters(Prototype);
  for (Register ArgumentRegister : ArgumentRegisters) {
    llvm::StringRef Name = model::Register::getCSVName(ArgumentRegister);
    if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true))
      Summary.ABIResults.ArgumentsRegisters.insert(CSV);
  }

  for (Register ReturnValueRegister : ReturnValueRegisters) {
    llvm::StringRef Name = model::Register::getCSVName(ReturnValueRegister);
    if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true))
      Summary.ABIResults.ReturnValuesRegisters.insert(CSV);
  }

  // This point is only ever reached on the "full" import (like the one
  // DetectABI does) - the proper function summary in extracted.
  return Summary;
}

std::pair<const FunctionSummary *, bool>
FunctionSummaryOracle::getCallSite(MetaAddress Function,
                                   BasicBlockID CallerBlockAddress,
                                   MetaAddress CalledLocalFunction,
                                   llvm::StringRef CalledSymbol) {
  auto [Summary, IsTailCall] = getExactCallSite(Function, CallerBlockAddress);
  if (Summary != nullptr) {
    return { Summary, IsTailCall };
  } else if (not CalledSymbol.empty()) {
    return { &getDynamicFunction(CalledSymbol), false };
  } else if (CalledLocalFunction.isValid()
             and Binary.Functions().contains(CalledLocalFunction)) {
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

const FunctionSummary &FunctionSummaryOracle::getDefault() {
  if (not Default.has_value())
    setDefault(Importer.prototype({}, Binary.DefaultPrototype()));
  return Default.value();
}

FunctionSummary &FunctionSummaryOracle::getLocalFunction(MetaAddress PC) {
  if (not LocalFunctions.contains(PC)) {
    const model::Function &Function = Binary.Functions().at(PC);
    AttributesSet Attributes;
    for (auto &ToCopy : Function.Attributes())
      Attributes.insert(ToCopy);
    auto Summary = Importer.prototype(Attributes, Function.prototype(Binary));
    registerLocalFunction(Function.Entry(), std::move(Summary));
  }

  return LocalFunctions.at(PC);
}

const FunctionSummary &
FunctionSummaryOracle::getDynamicFunction(llvm::StringRef Name) {
  if (not DynamicFunctions.contains(Name.str())) {
    const auto &DynamicFunction = Binary.ImportedDynamicFunctions()
                                    .at(Name.str());
    const auto &Prototype = DynamicFunction.prototype(Binary);
    AttributesSet Attributes;
    for (auto &ToCopy : DynamicFunction.Attributes())
      Attributes.insert(ToCopy);

    registerDynamicFunction(DynamicFunction.OriginalName(),
                            Importer.prototype(Attributes, Prototype));
  }
  return DynamicFunctions.at(Name.str());
}

std::pair<FunctionSummary *, bool>
FunctionSummaryOracle::getExactCallSite(MetaAddress Entry,
                                        BasicBlockID CallSiteAddress) {
  auto It = CallSites.find({ Entry, CallSiteAddress });
  if (It == CallSites.end()) {
    // Note: in case of absence we're doing the lookup every time, not super
    // efficient.

    const model::Function &Function = Binary.Functions().at(Entry);

    // TODO: should CallSitePrototypes be index by BasicBlockID?
    if (auto *CallSite = Function.CallSitePrototypes()
                           .tryGet(CallSiteAddress.start())) {

      AttributesSet Attributes;
      for (auto &ToCopy : CallSite->Attributes())
        Attributes.insert(ToCopy);
      registerCallSite(Function.Entry(),
                       BasicBlockID(CallSite->CallerBlockAddress()),
                       Importer.prototype(Attributes, CallSite->prototype()),
                       CallSite->IsTailCall());
    }
  }
  It = CallSites.find({ Entry, CallSiteAddress });
  if (It == CallSites.end()) {
    return { nullptr, false };
  } else {
    return { &It->second.first, It->second.second };
  }
}

template<PrototypeImportLevel Level>
FunctionSummaryOracle importImpl(llvm::Module &M,
                                 GeneratedCodeBasicInfo &GCBI,
                                 const model::Binary &Binary) {
  using GV = llvm::GlobalVariable;
  auto RegisterFilter = std::views::filter([SP = GCBI.spReg()](GV *CSV) {
    return CSV != nullptr && CSV != SP;
  });
  PrototypeImporter Importer{
    .Level = Level,
    .M = M,
    .ABICSVs = GCBI.abiRegisters() | RegisterFilter
               | revng::to<std::set<llvm::GlobalVariable *>>()
  };

  return FunctionSummaryOracle(Binary, std::move(Importer));
}

FunctionSummaryOracle
FunctionSummaryOracle::importFullPrototypes(llvm::Module &M,
                                            GeneratedCodeBasicInfo &GCBI,
                                            const model::Binary &Binary) {
  revng_log(Log,
            "Importing from the model, while taking prototypes into the "
            "account");
  return importImpl<PrototypeImportLevel::Full>(M, GCBI, Binary);
}

FunctionSummaryOracle
FunctionSummaryOracle::importBasicPrototypeData(llvm::Module &M,
                                                GeneratedCodeBasicInfo &GCBI,
                                                const model::Binary &Binary) {
  revng_log(Log,
            "Importing from the model, but ignoring some of the prototype "
            "data");
  return importImpl<PrototypeImportLevel::Partial>(M, GCBI, Binary);
}

FunctionSummaryOracle
FunctionSummaryOracle::importWithoutPrototypes(llvm::Module &M,
                                               GeneratedCodeBasicInfo &GCBI,
                                               const model::Binary &Binary) {
  revng_log(Log, "Importing from the model, but ignoring prototypes");
  return importImpl<PrototypeImportLevel::None>(M, GCBI, Binary);
}

} // namespace efa
