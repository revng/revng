/// \file Plain.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FormatVariadic.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"
#include "revng/Yield/Assembly/Assembly.h"
#include "revng/Yield/Plain.h"

namespace yield::plain {

static std::string linkAddress(const MetaAddress &Address) {
  std::string Result = Address.toString();

  constexpr std::array ForbiddenCharacters = { ' ', ':', '!', '#',  '?',
                                               '<', '>', '/', '\\', '{',
                                               '}', '[', ']' };

  for (char &Character : Result)
    if (llvm::find(ForbiddenCharacters, Character) != ForbiddenCharacters.end())
      Character = '_';

  return Result;
}

static std::string link(const MetaAddress &Target,
                        const efa::FunctionMetadata &Metadata,
                        const model::Binary &Binary) {
  if (auto Iterator = Binary.Functions.find(Target);
      Iterator != Binary.Functions.end()) {
    // The target is a function
    return Iterator->name().str().str();
  } else if (auto Iterator = Metadata.ControlFlowGraph.find(Target);
             Iterator != Metadata.ControlFlowGraph.end()) {
    // The target is a basic block

    // TODO: maybe there's something better than the address to put here.
    return "basic_block_at_" + linkAddress(Target);
  } else if (Target.isValid()) {
    // The target is an instruction
    return "instruction_at_" + linkAddress(Target);
  } else {
    // The target is impossible to deduce, it's an indirect call or the like.
    return "(unknown)";
  }
}

static std::string label(const assembly::BasicBlock &BasicBlock,
                         const efa::FunctionMetadata &Metadata,
                         const model::Binary &Binary) {
  if (BasicBlock.CanBeMergedWithPredecessor || BasicBlock.IsAFallthroughTarget)
    return "";

  return link(BasicBlock.Address, Metadata, Binary) + ":\n";
}

static std::string instruction(const assembly::Instruction &Instruction) {
  return Instruction.Text + '\n';
}

static std::string basicBlock(const assembly::BasicBlock &BasicBlock,
                              const efa::FunctionMetadata &Metadata,
                              const model::Binary &Binary) {
  std::string Result;

  Result += label(BasicBlock, Metadata, Binary);
  for (const auto &Instruction : BasicBlock.Instructions)
    Result += instruction(Instruction);

  return Result;
}

std::string assembly(const assembly::BasicBlock &BasicBlock,
                     const efa::FunctionMetadata &Metadata,
                     const model::Binary &Binary) {
  return basicBlock(BasicBlock, Metadata, Binary);
}
std::string assembly(const assembly::Function &Function,
                     const efa::FunctionMetadata &Metadata,
                     const model::Binary &Binary) {
  std::string Result;

  for (const auto &BasicBlock : Function.BasicBlocks)
    Result += basicBlock(BasicBlock, Metadata, Binary);

  return Result;
}

} // namespace yield::plain
