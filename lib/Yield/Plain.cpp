/// \file Plain.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FormatVariadic.h"

#include "revng/Model/Binary.h"
#include "revng/Yield/ControlFlow/FallthroughDetection.h"
#include "revng/Yield/Function.h"
#include "revng/Yield/Plain.h"

static std::string linkAddress(const BasicBlockID &Address) {
  std::string Result = Address.toString();

  constexpr std::array ForbiddenCharacters = { ' ', ':', '!', '#',  '?',
                                               '<', '>', '/', '\\', '{',
                                               '}', '[', ']' };

  for (char &Character : Result)
    if (llvm::find(ForbiddenCharacters, Character) != ForbiddenCharacters.end())
      Character = '_';

  return Result;
}

static std::string deduceName(const BasicBlockID &Target,
                              const yield::Function &Function,
                              const model::Binary &Binary) {
  if (auto *F = yield::tryGetFunction(Binary, Target)) {
    // The target is a function
    return F->name().str().str();
  } else if (auto Iterator = Function.ControlFlowGraph().find(Target);
             Iterator != Function.ControlFlowGraph().end()) {
    // The target is a basic block

    // TODO: maybe there's something better than the address to put here.
    return "basic_block_at_" + linkAddress(Target);
  } else if (Target.isValid()) {
    // The target is an instruction
    return "instruction_at_" + linkAddress(Target);
  } else {
    // The target is impossible to deduce, it's an indirect call or the like.
    return "(error)";
  }
}

static std::string label(const yield::BasicBlock &BasicBlock,
                         const yield::Function &Function,
                         const model::Binary &Binary) {
  std::string Result = deduceName(BasicBlock.ID(), Function, Binary);

  namespace Arch = model::Architecture;
  auto LabelIndicator = Arch::getAssemblyLabelIndicator(Binary.Architecture());
  return (Result += LabelIndicator) += "\n";
}

static std::string instruction(const yield::Instruction &Instruction,
                               const yield::BasicBlock &BasicBlock,
                               const model::Binary &Binary) {
  std::string Result = Instruction.Disassembled();

  namespace A = model::Architecture;
  auto CommentIndicator = A::getAssemblyCommentIndicator(Binary.Architecture());

  if (!Instruction.Error().empty()) {
    Result += ' ';
    Result += CommentIndicator;
    Result += " Error: ";
    Result += Instruction.Error();
  } else if (!Instruction.Comment().empty()) {
    Result += ' ';
    Result += CommentIndicator;
    Result += ' ';
    Result += Instruction.Comment();
  }

  return Result;
}

static std::string basicBlock(const yield::BasicBlock &BasicBlock,
                              const yield::Function &Function,
                              const model::Binary &Binary) {
  std::string Result;

  for (const auto &Instruction : BasicBlock.Instructions())
    Result += instruction(Instruction, BasicBlock, Binary);

  return Result;
}

template<bool ShouldMergeFallthroughTargets>
static std::string labeledBlock(const yield::BasicBlock &FirstBlock,
                                const yield::Function &Function,
                                const model::Binary &Binary) {
  std::string Result;
  Result += label(FirstBlock, Function, Binary);

  if constexpr (ShouldMergeFallthroughTargets == false) {
    Result += basicBlock(FirstBlock, Function, Binary);
  } else {
    auto BasicBlocks = yield::cfg::labeledBlock(FirstBlock, Function, Binary);
    if (BasicBlocks.empty())
      return "";

    for (auto BasicBlock : BasicBlocks)
      Result += basicBlock(*BasicBlock, Function, Binary);
  }

  return Result;
}

std::string yield::plain::functionAssembly(const yield::Function &Function,
                                           const model::Binary &Binary) {
  std::string Result;

  for (const auto &BasicBlock : Function.ControlFlowGraph())
    Result += labeledBlock<true>(BasicBlock, Function, Binary);

  return Result;
}

std::string yield::plain::controlFlowNode(const BasicBlockID &Address,
                                          const yield::Function &Function,
                                          const model::Binary &Binary) {
  auto Iterator = Function.ControlFlowGraph().find(Address);
  revng_assert(Iterator != Function.ControlFlowGraph().end());

  auto Result = labeledBlock<false>(*Iterator, Function, Binary);
  revng_assert(!Result.empty());

  return Result;
}
