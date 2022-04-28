/// \file DisassemblyHelper.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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

static void analyzeBasicBlocks(assembly::Function &Function,
                               const efa::FunctionMetadata &Metadata) {
  // Gather all the basic blocks that only have a single predecessor.
  std::map<MetaAddress, std::optional<MetaAddress>> Predecessors;

  for (const efa::BasicBlock &BasicBlock : Metadata.ControlFlowGraph) {
    auto [It, Success] = Predecessors.try_emplace(BasicBlock.Start);
    revng_assert(Success,
                 "Duplicate basic blocks in a `SortedVector`? "
                 "Something is clearly very wrong.");
  }

  // Remove the entry block - it should always be present.
  size_t RemovedCount = Predecessors.erase(Function.Address);
  revng_assert(RemovedCount == 1,
               "No basic block at the function entry address!");

  for (const efa::BasicBlock &BasicBlock : Metadata.ControlFlowGraph) {
    for (const auto &Edge : BasicBlock.Successors) {
      if (Edge->Destination.isInvalid()) {
        // Ignore edges with unknown destinations (like indirect calls).
        continue;
      }

      auto Iterator = Predecessors.find(Edge->Destination);
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

      auto R = Function.BasicBlocks.find(CurrentAddress);
      revng_assert(R != Function.BasicBlocks.end());

      if (Predecessor->End == Current->Start) {
        R->IsAFallthroughTarget = true;
        R->CanBeMergedWithPredecessor = (Predecessor->Successors.size() == 1);
      }
    }
  }
}

assembly::Function DH::disassemble(const model::Function &Function,
                                   const efa::FunctionMetadata &Metadata,
                                   const RawBinaryView &BinaryView) {
  auto &Helper = getDisassemblerFor(Function.Entry.type());

  assembly::Function ResultFunction{ .Address = Function.Entry };
  for (auto BasicBlockInserter = ResultFunction.BasicBlocks.batch_insert();
       const efa::BasicBlock &BasicBlock : Metadata.ControlFlowGraph) {
    assembly::BasicBlock ResultBasicBlock{ .Address = BasicBlock.Start };
    ResultBasicBlock.CommentIndicator = Helper.getCommentString();
    ResultBasicBlock.LabelIndicator = Helper.getLabelSuffix();

    auto MaybeBBSize = BasicBlock.End - BasicBlock.Start;
    revng_assert(MaybeBBSize.has_value());

    auto RawBytes = BinaryView.getByAddress(BasicBlock.Start, *MaybeBBSize);
    revng_assert(RawBytes.has_value());

    MetaAddress CurrentAddress = BasicBlock.Start;
    for (auto InstrInserter = ResultBasicBlock.Instructions.batch_insert();
         CurrentAddress < BasicBlock.End;) {
      auto MaybeInstructionOffset = CurrentAddress - BasicBlock.Start;
      revng_assert(MaybeInstructionOffset.has_value());
      auto InstructionBytes = RawBytes->drop_front(*MaybeInstructionOffset);

      auto [Instruction, Size] = Helper.instruction(CurrentAddress,
                                                    InstructionBytes);

      auto MaybeBytes = BinaryView.getByAddress(CurrentAddress, Size);
      revng_assert(MaybeBytes.has_value());
      using ByteContainer = assembly::Instruction::ByteContainer;
      Instruction.Bytes = ByteContainer(MaybeBytes->begin(), MaybeBytes->end());

      CurrentAddress += Size;
      revng_assert(CurrentAddress.isValid());
      revng_assert(CurrentAddress <= BasicBlock.End);

      InstrInserter.insert(std::move(Instruction));
    }

    if (!ResultBasicBlock.Instructions.empty()) {
      if (!BasicBlock.Successors.empty()) {
        auto TargetInserter = ResultBasicBlock.Targets.batch_insert();
        for (auto &SuccessorEdge : BasicBlock.Successors)
          TargetInserter.insert(SuccessorEdge->Destination);
      }
    }

    BasicBlockInserter.insert(std::move(ResultBasicBlock));
  }

  analyzeBasicBlocks(ResultFunction, Metadata);
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
