/// \file FunctionSummaryOracle.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/STLExtras.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"

static Logger<> Log("efa-import-model");

namespace efa {

enum class PrototypeImportLevel {
  // Do not import any prototype-related information except for
  // the callee saved register list (which is ABI-defined for CFTs).
  None,

  // Only import final stack offset on top of the callee saved register list.
  Partial,

  // Import everything present in the prototype.
  Full
};

template<PrototypeImportLevel Level>
struct PrototypeImporter {
  using Register = model::Register::Values;
  using State = abi::RegisterState::Values;

public:
  llvm::Module &M;
  const std::set<llvm::GlobalVariable *> &ABICSVs;

public:
  FunctionSummary prototype(const AttributesSet &Attributes,
                            const model::TypePath &Prototype) {
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

    auto Preserved = abi::FunctionType::calleeSavedRegisters(Prototype)
                     | TransformToCSV | IgnoreNullptr
                     | revng::to<std::set<llvm::GlobalVariable *>>();
    std::erase_if(Summary.ClobberedRegisters, [&Preserved](const auto &E) {
      return Preserved.contains(E);
    });

    // Stop importing prototype here, if `ClobberedRegisters` is the only
    // information callee needs.
    if constexpr (Level == PrototypeImportLevel::None)
      return Summary;

    Summary.ElectedFSO = abi::FunctionType::finalStackOffset(Prototype);

    // Stop importing prototype here, if callee also needs final stack offset.
    if constexpr (Level == PrototypeImportLevel::None)
      return Summary;

    for (llvm::GlobalVariable *CSV : ABICSVs) {
      Summary.ABIResults.ArgumentsRegisters[CSV] = State::No;
      Summary.ABIResults.FinalReturnValuesRegisters[CSV] = State::No;
    }

    auto [ArgumentRegisters,
          ReturnValueRegisters] = abi::FunctionType::usedRegisters(Prototype);
    for (Register ArgumentRegister : ArgumentRegisters) {
      llvm::StringRef Name = model::Register::getCSVName(ArgumentRegister);
      if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true))
        Summary.ABIResults.ArgumentsRegisters.at(CSV) = State::Yes;
    }

    for (Register ReturnValueRegister : ReturnValueRegisters) {
      llvm::StringRef Name = model::Register::getCSVName(ReturnValueRegister);
      if (llvm::GlobalVariable *CSV = M.getGlobalVariable(Name, true))
        Summary.ABIResults.FinalReturnValuesRegisters.at(CSV) = State::Yes;
    }

    // This point is only ever reached on the "full" import (like the one
    // DetectABI does) - the proper function summary in extracted.
    return Summary;
  }
};

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

template<PrototypeImportLevel Level>
FunctionSummaryOracle importImpl(llvm::Module &M,
                                 GeneratedCodeBasicInfo &GCBI,
                                 const model::Binary &Binary) {
  FunctionSummaryOracle Oracle;

  LoggerIndent Indent(Log);

  using GV = llvm::GlobalVariable;
  auto RegisterFilter = std::views::filter([SP = GCBI.spReg()](GV *CSV) {
    return CSV != nullptr && CSV != SP;
  });
  PrototypeImporter<Level> Importer{
    .M = M,
    .ABICSVs = GCBI.abiRegisters() | RegisterFilter
               | revng::to<std::set<llvm::GlobalVariable *>>()
  };

  // Import the default prototype
  Oracle.setDefault(Importer.prototype({}, Binary.DefaultPrototype()));

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
                              Importer.prototype(Attributes,
                                                 CallSite.prototype()),
                              CallSite.IsTailCall());
    }

    AttributesSet Attributes;
    for (auto &ToCopy : Function.Attributes())
      Attributes.insert(ToCopy);
    auto Summary = Importer.prototype(Attributes, Function.prototype(Binary));

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
                                   Importer.prototype(Attributes, Prototype));
  }

  return Oracle;
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
