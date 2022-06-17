/// \file DisassemblyHelper.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Function.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/Debug.h"
#include "revng/Yield/Assembly/DisassemblyHelper.h"
#include "revng/Yield/Assembly/LLVMDisassemblerInterface.h"

namespace detail {

class DissassemblyHelperImpl
  : public std::map<MetaAddressType::Values, LLVMDisassemblerInterface> {};

} // namespace detail

using DH = DissassemblyHelper;
DH::DissassemblyHelper() :
  Internal{ std::make_unique<detail::DissassemblyHelperImpl>() } {
}
DH::~DissassemblyHelper() {
}

static void analyzeBasicBlocks(yield::Function &Function,
                               const efa::FunctionMetadata &Metadata,
                               const model::Binary &Binary) {
  // Gather all the basic blocks that only have a single predecessor.
  std::map<MetaAddress, std::optional<MetaAddress>> Predecessors;

  for (const efa::BasicBlock &BasicBlock : Metadata.ControlFlowGraph) {
    auto [It, Success] = Predecessors.try_emplace(BasicBlock.Start);
    revng_assert(Success,
                 "Duplicate basic blocks in a `SortedVector`? "
                 "Something is clearly very wrong.");
  }

  // Remove the entry block from the analysis - its label is always required.
  size_t RemovedCount = Predecessors.erase(Function.Entry);
  revng_assert(RemovedCount == 1,
               "No basic block at the function entry address!");

  for (const efa::BasicBlock &BasicBlock : Metadata.ControlFlowGraph) {
    for (const auto &Edge : BasicBlock.Successors) {
      auto [NextBlock, _] = efa::parseSuccessor(*Edge, BasicBlock.End, Binary);
      if (NextBlock.isInvalid()) {
        // Ignore edges with unknown destinations (like indirect jumps).
        continue;
      }

      auto Iterator = Predecessors.find(NextBlock);
      if (Iterator != Predecessors.end()) {
        if (Iterator->second.has_value()) {
          // This basic block already has a predecessor, remove it.
          Predecessors.erase(Iterator);
        } else {
          // First predecessor found - save it.
          Iterator->second = BasicBlock.Start;
        }
      }
    }
  }

  // Save the results of the analysis
  for (auto [CurrentAddress, PredecessorAddress] : Predecessors) {
    if (PredecessorAddress.has_value()) {
      auto Current = Metadata.ControlFlowGraph.find(CurrentAddress);
      revng_assert(Current != Metadata.ControlFlowGraph.end());

      auto Predecessor = Metadata.ControlFlowGraph.find(*PredecessorAddress);
      revng_assert(Predecessor != Metadata.ControlFlowGraph.end());

      auto CurrentBlock = Function.ControlFlowGraph.find(CurrentAddress);
      revng_assert(CurrentBlock != Function.ControlFlowGraph.end());

      if (Predecessor->End == Current->Start)
        CurrentBlock->IsLabelAlwaysRequired = false;
    }
  }
}

yield::Function DH::disassemble(const model::Function &Function,
                                const efa::FunctionMetadata &Metadata,
                                const RawBinaryView &BinaryView,
                                const model::Binary &Binary) {
  auto &Helper = getDisassemblerFor(Function.Entry.type());

  yield::Function ResultFunction;
  ResultFunction.Entry = Function.Entry;
  for (auto BasicBlockInserter = ResultFunction.ControlFlowGraph.batch_insert();
       const efa::BasicBlock &BasicBlock : Metadata.ControlFlowGraph) {
    yield::BasicBlock ResultBasicBlock;
    ResultBasicBlock.Start = BasicBlock.Start;
    ResultBasicBlock.End = BasicBlock.End;
    ResultBasicBlock.Successors = BasicBlock.Successors;
    ResultBasicBlock.IsLabelAlwaysRequired = true;

    namespace Arch = model::Architecture;
    auto Comment = Arch::getAssemblyCommentIndicator(Binary.Architecture);
    revng_assert(Helper.getCommentString() == llvm::StringRef{ Comment });
    auto Label = Arch::getAssemblyLabelIndicator(Binary.Architecture);
    revng_assert(Helper.getLabelSuffix() == llvm::StringRef{ Label });

    auto MaybeBBSize = BasicBlock.End - BasicBlock.Start;
    revng_assert(MaybeBBSize.has_value());

    auto RawBytes = BinaryView.getByAddress(BasicBlock.Start, *MaybeBBSize);
    revng_assert(RawBytes.has_value());

    MetaAddress CurrentAddress = BasicBlock.Start;
    MetaAddress InstructionWithTheDelaySlot = MetaAddress::invalid();
    for (auto InstrInserter = ResultBasicBlock.Instructions.batch_insert();
         CurrentAddress < BasicBlock.End;) {
      auto MaybeInstructionOffset = CurrentAddress - BasicBlock.Start;
      revng_assert(MaybeInstructionOffset.has_value());
      auto InstructionBytes = RawBytes->drop_front(*MaybeInstructionOffset);

      auto [Instruction,
            HasDelaySlot,
            Size] = Helper.instruction(CurrentAddress, InstructionBytes);
      revng_assert(Instruction.Address.isValid());

      if (HasDelaySlot) {
        revng_assert(InstructionWithTheDelaySlot.isInvalid(),
                     "Multiple instructions with delay slots are not allowed "
                     "in the same basic block.");
        InstructionWithTheDelaySlot = Instruction.Address;
      }

      auto MaybeBytes = BinaryView.getByAddress(CurrentAddress, Size);
      revng_assert(MaybeBytes.has_value());
      using ByteContainer = yield::ByteContainer;
      Instruction.RawBytes = ByteContainer(MaybeBytes->begin(),
                                           MaybeBytes->end());

      CurrentAddress += Size;
      revng_assert(CurrentAddress.isValid());
      revng_assert(CurrentAddress <= BasicBlock.End);

      InstrInserter.insert(std::move(Instruction));
    }

    if (InstructionWithTheDelaySlot.isValid()) {
      revng_assert(ResultBasicBlock.Instructions.size() > 1);
      auto Last = std::prev(ResultBasicBlock.Instructions.end());
      revng_assert(InstructionWithTheDelaySlot == std::prev(Last)->Address);

      ResultBasicBlock.HasDelaySlot = true;
    }

    BasicBlockInserter.insert(std::move(ResultBasicBlock));
  }

  analyzeBasicBlocks(ResultFunction, Metadata, Binary);
  return ResultFunction;
}

LLVMDisassemblerInterface &
DH::getDisassemblerFor(MetaAddressType::Values AddressType) {
  revng_assert(Internal != nullptr);

  if (auto It = Internal->find(AddressType); It != Internal->end())
    return It->second;

  using DI = LLVMDisassemblerInterface;
  auto [Result, Success] = Internal->try_emplace(AddressType, DI(AddressType));
  revng_assert(Success);
  return Result->second;
}
