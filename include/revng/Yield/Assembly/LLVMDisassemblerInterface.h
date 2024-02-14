#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

#include "revng/Model/DisassemblyConfiguration.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Yield/Instruction.h"

class LLVMDisassemblerInterface {
private:
  std::unique_ptr<llvm::MCSubtargetInfo> SubtargetInformation;
  std::unique_ptr<llvm::MCRegisterInfo> RegisterInformation;
  std::unique_ptr<llvm::MCAsmInfo> AssemblyInformation;
  std::unique_ptr<llvm::MCObjectFileInfo> ObjectFileInformation;
  std::unique_ptr<llvm::MCContext> Context;
  std::unique_ptr<llvm::MCInstrInfo> InstructionInformation;

  std::unique_ptr<llvm::MCDisassembler> Disassembler;
  std::unique_ptr<llvm::MCInstPrinter> Printer;

public:
  explicit LLVMDisassemblerInterface(MetaAddressType::Values AddressType,
                                     const model::DisassemblyConfiguration &);

  struct Disassembled {
    yield::Instruction Instruction;
    bool HasDelaySlot;
    uint64_t Size;
  };
  Disassembled instruction(const MetaAddress &Where,
                           llvm::ArrayRef<uint8_t> RawBytes);

public:
  inline llvm::StringRef getCommentString() const {
    return AssemblyInformation->getCommentString();
  }
  inline llvm::StringRef getLabelSuffix() const {
    return AssemblyInformation->getLabelSuffix();
  }

private:
  std::pair<std::optional<llvm::MCInst>, uint64_t>
  disassemble(const MetaAddress &Address,
              llvm::ArrayRef<uint8_t> RawBytes,
              const llvm::MCDisassembler &Disassembler);
  yield::Instruction parse(const llvm::MCInst &Instruction,
                           const MetaAddress &Address,
                           llvm::MCInstPrinter &Printer,
                           const llvm::MCSubtargetInfo &SI);
};
