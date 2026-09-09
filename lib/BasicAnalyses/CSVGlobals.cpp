//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/CSVGlobals.h"
#include "revng/Model/FunctionTags.h"

using namespace llvm;

CSVGlobals::CSVGlobals(const model::Binary &Binary, llvm::Module &M) {
  using namespace model::Architecture;
  auto Architecture = Binary.Architecture();
  PC = M.getGlobalVariable(getPCCSVName(Architecture), true);
  SP = M.getGlobalVariable(singleCSVName(getStackPointer(Architecture)), true);
  auto ReturnAddressRegister = getReturnAddressRegister(Architecture);
  if (ReturnAddressRegister != model::Register::Invalid)
    RA = M.getGlobalVariable(singleCSVName(ReturnAddressRegister), true);

  for (model::Register::Values Register : registers(Architecture)) {
    for (const model::Register::CSV &CSV : model::Register::getCSVs(Register)) {
      GlobalVariable *Variable = M.getGlobalVariable(CSV.Name, true);
      ABIRegisters.push_back(Variable);
      ABIRegistersSet.insert(Variable);
    }
  }

  for (GlobalVariable &CSV : FunctionTags::CSV.globals(&M))
    CSVs.push_back(&CSV);
}

// TODO: this method should probably be deprecated
/// Check if \p GV is the program counter CSV
bool CSVGlobals::isPCReg(const llvm::GlobalVariable *GV) const {
  revng_assert(PC != nullptr);
  return GV == PC;
}

bool CSVGlobals::isSPReg(const llvm::GlobalVariable *GV) const {
  revng_assert(SP != nullptr);
  return GV == SP;
}

bool CSVGlobals::isSPReg(const llvm::Value *V) const {
  if (auto *GV = llvm::dyn_cast<const llvm::GlobalVariable>(V))
    return isSPReg(GV);
  return false;
}
