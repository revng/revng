/// \file
/// \brief  This file handles the translation from QEMU's PTC to LLVM IR.

#include <cstdint>
#include <sstream>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

// LLVM API
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revamb.h"
#include "ptctollvmir.h"
#include "ptcinterface.h"
#include "ptcdump.h"

class SelfDescribingWriter : public llvm::AssemblyAnnotationWriter {
public:
  SelfDescribingWriter(llvm::LLVMContext& Context,
                       llvm::Metadata *Scope,
                       bool DebugInfo) : Context(Context),
                                         Scope(Scope),
                                         DebugInfo(DebugInfo) {
    OriginalInstrMDKind = Context.getMDKindID("oi");
    PTCInstrMDKind = Context.getMDKindID("pi");
    DbgMDKind = Context.getMDKindID("dbg");
  }

  virtual void emitInstructionAnnot(const llvm::Instruction *Instruction,
                                    llvm::formatted_raw_ostream &Output) {

    writeMetadataIfNew(Instruction, OriginalInstrMDKind, Output, "\n\n  ; ");
    writeMetadataIfNew(Instruction, PTCInstrMDKind, Output, "\n  ; ");

    if (DebugInfo) {
      Output.flush();
      auto *Location = llvm::DILocation::get(Context,
                                             Output.getLine(),
                                             Output.getColumn(),
                                             Scope);

      // Sorry Bjarne
      auto *NonConstInstruction = const_cast<llvm::Instruction *>(Instruction);
      NonConstInstruction->setMetadata(DbgMDKind, Location);
    }
  }

private:
  static void writeMetadataIfNew(const llvm::Instruction *Instruction,
                                 unsigned MDKind,
                                 llvm::formatted_raw_ostream &Output,
                                 llvm::StringRef Prefix) {
    llvm::MDString *MD = getMD(Instruction,
                                                MDKind);
    if (MD != nullptr) {
      const llvm::Instruction *PrevInstruction = nullptr;

      if (Instruction != Instruction->getParent()->begin())
        PrevInstruction = Instruction->getPrevNode();

      if (PrevInstruction == nullptr ||
          getMD(PrevInstruction, MDKind) != MD) {
        Output << Prefix << MD->getString();
      }
    }
  }

  static llvm::MDString *getMD(const llvm::Instruction *Instruction,
                               unsigned Kind) {
    assert(Instruction != nullptr);

    llvm::Metadata *MD = Instruction->getMetadata(Kind);

    if (MD == nullptr)
      return nullptr;

    auto Node = llvm::dyn_cast<llvm::MDNode>(MD);

    assert(Node != nullptr);

    const llvm::MDOperand& Operand = Node->getOperand(0);

    llvm::Metadata *MDOperand = Operand.get();

    if (MDOperand == nullptr)
      return nullptr;

    auto *String = llvm::dyn_cast<llvm::MDString>(MDOperand);
    assert(String != nullptr);

    return String;
  }

private:
  llvm::LLVMContext &Context;
  llvm::Metadata *Scope;
  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;
  unsigned DbgMDKind;
  bool DebugInfo;
};

using PTCInstructionListDestructor =
  GenericFunctor<decltype(&ptc_instruction_list_free),
                 &ptc_instruction_list_free>;
using PTCInstructionListPtr = std::unique_ptr<PTCInstructionList,
                                              PTCInstructionListDestructor>;

/// \brief Maintains the list of variables required by PTC.
///
/// It can be queried for a variable, which, if not already existing, will be
/// created on the fly.
class VariableManager {
public:
  VariableManager(llvm::Module& Module,
                  llvm::ArrayRef<llvm::GlobalVariable *> PredefinedGlobals) :
    Module(Module),
    Builder(Module.getContext()) {

    // Store all the predefined globals
    for (llvm::GlobalVariable *Global : PredefinedGlobals)
      Globals[Global->getName()] = Global;
  }
  /// Given a PTC temporary identifier, checks if it already exists in the
  /// generatd LLVM IR, and, if not, it creates it.
  ///
  /// @param TemporaryId the PTC temporary identifier.
  ///
  /// @return an llvm::Value wrapping the request global or local variable.
  llvm::Value* getOrCreate(unsigned int TemporaryId);

  /// Informs the VariableManager that a new function has begun, so it can
  /// discard function- and basic block-level variables.
  ///
  /// @param Delimiter the new point where to insert allocations for local
  /// variables.
  /// @param Instructions the new PTCInstructionList to use from now on.
 void newFunction(llvm::Instruction *Delimiter=nullptr,
                   PTCInstructionList *Instructions=nullptr) {
    LocalTemporaries.clear();
    newBasicBlock(Delimiter, Instructions);
  }

  /// Informs the VariableManager that a new basic block has begun, so it can
  /// discard basic block-level variables.
  ///
  /// @param Delimiter the new point where to insert allocations for local
  /// variables.
  /// @param Instructions the new PTCInstructionList to use from now on.
  void newBasicBlock(llvm::Instruction *Delimiter=nullptr,
                     PTCInstructionList *Instructions=nullptr) {
    Temporaries.clear();
    if (Instructions != nullptr)
      this->Instructions = Instructions;

    if (Delimiter != nullptr)
      Builder.SetInsertPoint(Delimiter);
  }

  void newBasicBlock(llvm::BasicBlock *Delimiter,
                     PTCInstructionList *Instructions=nullptr) {
    Temporaries.clear();
    if (Instructions != nullptr)
      this->Instructions = Instructions;

    if (Delimiter != nullptr)
      Builder.SetInsertPoint(Delimiter);
  }

  static llvm::GlobalVariable*
  createGlobal(llvm::Module& Module,
               llvm::Type *Type,
               llvm::GlobalValue::LinkageTypes Linkage,
               uint64_t Initializer = 0,
               const llvm::Twine& Name = "") {

    return new llvm::GlobalVariable(Module, Type, false, Linkage,
                                    llvm::ConstantInt::get(Type, Initializer),
                                    Name);
  }

private:
  llvm::Module& Module;
  llvm::IRBuilder<> Builder;
  using TemporariesMap = std::map<unsigned int, llvm::AllocaInst *>;
  using GlobalsMap = std::map<std::string, llvm::GlobalVariable *>;
  GlobalsMap Globals;
  TemporariesMap Temporaries;
  TemporariesMap LocalTemporaries;
  PTCInstructionList *Instructions;
  llvm::Instruction *Last;
};

llvm::Value* VariableManager::getOrCreate(unsigned int TemporaryId) {
  assert(Instructions != nullptr);

  PTCTemp *Temporary = ptc_temp_get(Instructions, TemporaryId);
  llvm::Instruction *Result = nullptr;
  // TODO: handle non-32 bit variables
  llvm::Type *VariableType = Builder.getInt32Ty();

  if (ptc_temp_is_global(Instructions, TemporaryId)) {
    auto TemporaryName = llvm::StringRef(Temporary->name).lower();

    GlobalsMap::iterator it = Globals.find(TemporaryName);
    if (it != Globals.end()) {
      return it->second;
    } else {
      llvm::GlobalVariable *NewVariable = nullptr;
      NewVariable = createGlobal(Module,
                                 VariableType,
                                 llvm::GlobalValue::ExternalLinkage,
                                 0,
                                 TemporaryName);
      assert(NewVariable != nullptr);
      Globals[TemporaryName] = NewVariable;

      return NewVariable;
    }

  } else if (Temporary->temp_local) {
    TemporariesMap::iterator it = LocalTemporaries.find(TemporaryId);
    if (it != LocalTemporaries.end()) {
      return it->second;
    } else {
      llvm::AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      LocalTemporaries[TemporaryId] = NewTemporary;
      Result = NewTemporary;
    }
  } else {
    TemporariesMap::iterator it = Temporaries.find(TemporaryId);
    if (it != Temporaries.end()) {
      return it->second;
    } else {
      llvm::AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      Temporaries[TemporaryId] = NewTemporary;
      Result = NewTemporary;
    }
  }

  Last = Result;

  return Result;
}

/// Converts a PTC condition into an LLVM predicate
///
/// @param Condition the input PTC condition.
///
/// @return the corresponding LLVM predicate.
static llvm::CmpInst::Predicate conditionToPredicate(PTCCondition Condition) {
  switch (Condition) {
  case PTC_COND_NEVER:
    // TODO: this is probably wrong
    return llvm::CmpInst::FCMP_FALSE;
  case PTC_COND_ALWAYS:
    // TODO: this is probably wrong
    return llvm::CmpInst::FCMP_TRUE;
  case PTC_COND_EQ:
    return llvm::CmpInst::ICMP_EQ;
  case PTC_COND_NE:
    return llvm::CmpInst::ICMP_NE;
  case PTC_COND_LT:
    return llvm::CmpInst::ICMP_SLT;
  case PTC_COND_GE:
    return llvm::CmpInst::ICMP_SGE;
  case PTC_COND_LE:
    return llvm::CmpInst::ICMP_SLE;
  case PTC_COND_GT:
    return llvm::CmpInst::ICMP_SGT;
  case PTC_COND_LTU:
    return llvm::CmpInst::ICMP_ULT;
  case PTC_COND_GEU:
    return llvm::CmpInst::ICMP_UGE;
  case PTC_COND_LEU:
    return llvm::CmpInst::ICMP_ULE;
  case PTC_COND_GTU:
    return llvm::CmpInst::ICMP_UGT;
  default:
    llvm_unreachable("Unknown comparison operator");
  }
}

/// Obtains the LLVM binary operation corresponding to the specified PTC opcode.
///
/// @param Opcode the PTC opcode.
///
/// @return the LLVM binary operation matching opcode.
static llvm::Instruction::BinaryOps opcodeToBinaryOp(PTCOpcode Opcode) {
  switch (Opcode) {
  case PTC_INSTRUCTION_op_add_i32:
  case PTC_INSTRUCTION_op_add_i64:
  case PTC_INSTRUCTION_op_add2_i32:
  case PTC_INSTRUCTION_op_add2_i64:
    return llvm::Instruction::Add;
  case PTC_INSTRUCTION_op_sub_i32:
  case PTC_INSTRUCTION_op_sub_i64:
  case PTC_INSTRUCTION_op_sub2_i32:
  case PTC_INSTRUCTION_op_sub2_i64:
    return llvm::Instruction::Sub;
  case PTC_INSTRUCTION_op_mul_i32:
  case PTC_INSTRUCTION_op_mul_i64:
    return llvm::Instruction::Mul;
  case PTC_INSTRUCTION_op_div_i32:
  case PTC_INSTRUCTION_op_div_i64:
    return llvm::Instruction::SDiv;
  case PTC_INSTRUCTION_op_divu_i32:
  case PTC_INSTRUCTION_op_divu_i64:
    return llvm::Instruction::UDiv;
  case PTC_INSTRUCTION_op_rem_i32:
  case PTC_INSTRUCTION_op_rem_i64:
    return llvm::Instruction::SRem;
  case PTC_INSTRUCTION_op_remu_i32:
  case PTC_INSTRUCTION_op_remu_i64:
    return llvm::Instruction::URem;
  case PTC_INSTRUCTION_op_and_i32:
  case PTC_INSTRUCTION_op_and_i64:
    return llvm::Instruction::And;
  case PTC_INSTRUCTION_op_or_i32:
  case PTC_INSTRUCTION_op_or_i64:
    return llvm::Instruction::Or;
  case PTC_INSTRUCTION_op_xor_i32:
  case PTC_INSTRUCTION_op_xor_i64:
    return llvm::Instruction::Xor;
  case PTC_INSTRUCTION_op_shl_i32:
  case PTC_INSTRUCTION_op_shl_i64:
    return llvm::Instruction::Shl;
  case PTC_INSTRUCTION_op_shr_i32:
  case PTC_INSTRUCTION_op_shr_i64:
    return llvm::Instruction::LShr;
  case PTC_INSTRUCTION_op_sar_i32:
  case PTC_INSTRUCTION_op_sar_i64:
    return llvm::Instruction::AShr;
  default:
    llvm_unreachable("PTC opcode is not a binary operator");
  }
}

/// Returns the maximum value which can be represented with the specified number
/// of bits.
static uint64_t getMaxValue(unsigned Bits) {
  if (Bits == 32)
    return 0xffffffff;
  else if (Bits == 64)
    return 0xffffffffffffffff;
  else
    llvm_unreachable("Not the number of bits in a integer type");
}

/// Maps an opcode the corresponding input and output register size.
///
/// @return the size, in bits, of the registers used by the opcode.
static unsigned getRegisterSize(unsigned Opcode) {
  switch (Opcode) {
  case PTC_INSTRUCTION_op_add2_i32:
  case PTC_INSTRUCTION_op_add_i32:
  case PTC_INSTRUCTION_op_andc_i32:
  case PTC_INSTRUCTION_op_and_i32:
  case PTC_INSTRUCTION_op_brcond2_i32:
  case PTC_INSTRUCTION_op_brcond_i32:
  case PTC_INSTRUCTION_op_bswap16_i32:
  case PTC_INSTRUCTION_op_bswap32_i32:
  case PTC_INSTRUCTION_op_deposit_i32:
  case PTC_INSTRUCTION_op_div2_i32:
  case PTC_INSTRUCTION_op_div_i32:
  case PTC_INSTRUCTION_op_divu2_i32:
  case PTC_INSTRUCTION_op_divu_i32:
  case PTC_INSTRUCTION_op_eqv_i32:
  case PTC_INSTRUCTION_op_ext16s_i32:
  case PTC_INSTRUCTION_op_ext16u_i32:
  case PTC_INSTRUCTION_op_ext8s_i32:
  case PTC_INSTRUCTION_op_ext8u_i32:
  case PTC_INSTRUCTION_op_ld16s_i32:
  case PTC_INSTRUCTION_op_ld16u_i32:
  case PTC_INSTRUCTION_op_ld8s_i32:
  case PTC_INSTRUCTION_op_ld8u_i32:
  case PTC_INSTRUCTION_op_ld_i32:
  case PTC_INSTRUCTION_op_movcond_i32:
  case PTC_INSTRUCTION_op_mov_i32:
  case PTC_INSTRUCTION_op_movi_i32:
  case PTC_INSTRUCTION_op_mul_i32:
  case PTC_INSTRUCTION_op_muls2_i32:
  case PTC_INSTRUCTION_op_mulsh_i32:
  case PTC_INSTRUCTION_op_mulu2_i32:
  case PTC_INSTRUCTION_op_muluh_i32:
  case PTC_INSTRUCTION_op_nand_i32:
  case PTC_INSTRUCTION_op_neg_i32:
  case PTC_INSTRUCTION_op_nor_i32:
  case PTC_INSTRUCTION_op_not_i32:
  case PTC_INSTRUCTION_op_orc_i32:
  case PTC_INSTRUCTION_op_or_i32:
  case PTC_INSTRUCTION_op_qemu_ld_i32:
  case PTC_INSTRUCTION_op_qemu_st_i32:
  case PTC_INSTRUCTION_op_rem_i32:
  case PTC_INSTRUCTION_op_remu_i32:
  case PTC_INSTRUCTION_op_rotl_i32:
  case PTC_INSTRUCTION_op_rotr_i32:
  case PTC_INSTRUCTION_op_sar_i32:
  case PTC_INSTRUCTION_op_setcond2_i32:
  case PTC_INSTRUCTION_op_setcond_i32:
  case PTC_INSTRUCTION_op_shl_i32:
  case PTC_INSTRUCTION_op_shr_i32:
  case PTC_INSTRUCTION_op_st16_i32:
  case PTC_INSTRUCTION_op_st8_i32:
  case PTC_INSTRUCTION_op_st_i32:
  case PTC_INSTRUCTION_op_sub2_i32:
  case PTC_INSTRUCTION_op_sub_i32:
  case PTC_INSTRUCTION_op_trunc_shr_i32:
  case PTC_INSTRUCTION_op_xor_i32:
    return 32;
  case PTC_INSTRUCTION_op_add2_i64:
  case PTC_INSTRUCTION_op_add_i64:
  case PTC_INSTRUCTION_op_andc_i64:
  case PTC_INSTRUCTION_op_and_i64:
  case PTC_INSTRUCTION_op_brcond_i64:
  case PTC_INSTRUCTION_op_bswap16_i64:
  case PTC_INSTRUCTION_op_bswap32_i64:
  case PTC_INSTRUCTION_op_bswap64_i64:
  case PTC_INSTRUCTION_op_deposit_i64:
  case PTC_INSTRUCTION_op_div2_i64:
  case PTC_INSTRUCTION_op_div_i64:
  case PTC_INSTRUCTION_op_divu2_i64:
  case PTC_INSTRUCTION_op_divu_i64:
  case PTC_INSTRUCTION_op_eqv_i64:
  case PTC_INSTRUCTION_op_ext16s_i64:
  case PTC_INSTRUCTION_op_ext16u_i64:
  case PTC_INSTRUCTION_op_ext32s_i64:
  case PTC_INSTRUCTION_op_ext32u_i64:
  case PTC_INSTRUCTION_op_ext8s_i64:
  case PTC_INSTRUCTION_op_ext8u_i64:
  case PTC_INSTRUCTION_op_ld16s_i64:
  case PTC_INSTRUCTION_op_ld16u_i64:
  case PTC_INSTRUCTION_op_ld32s_i64:
  case PTC_INSTRUCTION_op_ld32u_i64:
  case PTC_INSTRUCTION_op_ld8s_i64:
  case PTC_INSTRUCTION_op_ld8u_i64:
  case PTC_INSTRUCTION_op_ld_i64:
  case PTC_INSTRUCTION_op_movcond_i64:
  case PTC_INSTRUCTION_op_mov_i64:
  case PTC_INSTRUCTION_op_movi_i64:
  case PTC_INSTRUCTION_op_mul_i64:
  case PTC_INSTRUCTION_op_muls2_i64:
  case PTC_INSTRUCTION_op_mulsh_i64:
  case PTC_INSTRUCTION_op_mulu2_i64:
  case PTC_INSTRUCTION_op_muluh_i64:
  case PTC_INSTRUCTION_op_nand_i64:
  case PTC_INSTRUCTION_op_neg_i64:
  case PTC_INSTRUCTION_op_nor_i64:
  case PTC_INSTRUCTION_op_not_i64:
  case PTC_INSTRUCTION_op_orc_i64:
  case PTC_INSTRUCTION_op_or_i64:
  case PTC_INSTRUCTION_op_qemu_ld_i64:
  case PTC_INSTRUCTION_op_qemu_st_i64:
  case PTC_INSTRUCTION_op_rem_i64:
  case PTC_INSTRUCTION_op_remu_i64:
  case PTC_INSTRUCTION_op_rotl_i64:
  case PTC_INSTRUCTION_op_rotr_i64:
  case PTC_INSTRUCTION_op_sar_i64:
  case PTC_INSTRUCTION_op_setcond_i64:
  case PTC_INSTRUCTION_op_shl_i64:
  case PTC_INSTRUCTION_op_shr_i64:
  case PTC_INSTRUCTION_op_st16_i64:
  case PTC_INSTRUCTION_op_st32_i64:
  case PTC_INSTRUCTION_op_st8_i64:
  case PTC_INSTRUCTION_op_st_i64:
  case PTC_INSTRUCTION_op_sub2_i64:
  case PTC_INSTRUCTION_op_sub_i64:
  case PTC_INSTRUCTION_op_xor_i64:
    return 64;
  case PTC_INSTRUCTION_op_br:
  case PTC_INSTRUCTION_op_call:
  case PTC_INSTRUCTION_op_debug_insn_start:
  case PTC_INSTRUCTION_op_discard:
  case PTC_INSTRUCTION_op_exit_tb:
  case PTC_INSTRUCTION_op_goto_tb:
  case PTC_INSTRUCTION_op_set_label:
    return 0;
  default:
    llvm_unreachable("Unexpected opcode");
    break;
  }
}

/// Create a compare instruction given a comparison operator and the operands
///
/// @param Builder the builder to use to create the instruction.
/// @param RawCondition the PTC condition.
/// @param FirstOperand the first operand of the comparison.
/// @param SecondOperand the second operand of the comparison.
///
/// @return a compare instruction.
template<typename T>
static llvm::Value *CreateICmp(T& Builder,
                               uint64_t RawCondition,
                               llvm::Value *FirstOperand,
                               llvm::Value *SecondOperand) {
  PTCCondition Condition = static_cast<PTCCondition>(RawCondition);
  return Builder.CreateICmp(conditionToPredicate(Condition),
                            FirstOperand,
                            SecondOperand);
}

class JumpTargetManager {
public:
  using BlockWithAddress = std::pair<uint64_t, llvm::BasicBlock *>;
  static const BlockWithAddress NoMoreTargets;

public:
  JumpTargetManager(llvm::LLVMContext& Context,
                    llvm::Value *PCReg,
                    llvm::Function *Function) :
    Context(Context),
    Function(Function),
    OriginalInstructionAddresses(),
    JumpTargets(),
    PCReg(PCReg) { }

  /// Handle a new program counter. We might already have a basic block for that
  /// program counter, or we could even have a translation for it. Return one
  /// of these, if appropriate.
  ///
  /// @param PC the new program counter.
  /// @param ShouldContinue an out parameter indicating whether the returned
  ///        basic block was just a placeholder or actually contains a
  ///        translation.
  ///
  /// @return the basic block to use from now on, or null if the program counter
  ///         is not associated to a basic block.
  llvm::BasicBlock *newPC(uint64_t PC, bool& ShouldContinue) {
    // Did we already meet this PC?
    auto It = JumpTargets.find(PC);
    if (It != JumpTargets.end()) {
      // If it was planned to explore it in the future, just to do it now
      for (auto It = Unexplored.begin(); It != Unexplored.end(); It++) {
        if (It->first == PC) {
          Unexplored.erase(It, It + 1);
          ShouldContinue = true;
          assert(It->second->empty());
          return It->second;
        }
      }

      // It wasn't planned to visit it, so we've already been there, just jump
      // there
      assert(!It->second->empty());
      ShouldContinue = false;
      return It->second;
    }

    // We don't know anything about this PC
    return nullptr;
  }

  /// Save the PC-Instruction association for future use (jump target)
  void registerInstruction(uint64_t PC, llvm::Instruction *Instruction) {
    // Never save twice a PC
    assert(OriginalInstructionAddresses.find(PC) ==
           OriginalInstructionAddresses.end());
    OriginalInstructionAddresses[PC] = Instruction;
  }

  /// Save the PC-BasicBlock association for futur use (jump target)
  void registerBlock(uint64_t PC, llvm::BasicBlock *Block) {
    // If we already met it, it must point to the same block
    auto It = JumpTargets.find(PC);
    assert(It == JumpTargets.end() || It->second == Block);
    if (It->second != Block)
      JumpTargets[PC] = Block;
  }

  /// Look for all the stores targeting the program counter and add a branch
  /// there as appropriate.
  void translateMovePC(uint64_t BasePC) {
    for (llvm::Use& PCUse : PCReg->uses()) {
      // TODO: what to do in case of read of the PC?
      // Is the PC the store destination?
      if (PCUse.getOperandNo() == 1) {
        if (auto Jump = llvm::dyn_cast<llvm::StoreInst>(PCUse.getUser())) {
          llvm::Value *Destination = Jump->getValueOperand();

          // Is desintation a constant?
          if (auto Address = llvm::dyn_cast<llvm::ConstantInt>(Destination)) {
            // Compute the actual PC
            uint64_t TargetPC = BasePC + Address->getSExtValue();

            // Get or create the block for this PC and branch there
            llvm::BasicBlock *TargetBlock = getBlockAt(TargetPC);
            llvm::Instruction *Branch = llvm::BranchInst::Create(TargetBlock);

            // Cleanup of what's afterwards (only a unconditional jump is
            // allowed)
            llvm::BasicBlock::iterator I = Jump;
            llvm::BasicBlock::iterator BlockEnd = Jump->getParent()->end();
            if (++I != BlockEnd)
              purgeBranch(I);

            Branch->insertAfter(Jump);
            Jump->eraseFromParent();
          } else {
            // TODO: very strong assumption here
            // Destination is not a constant, assume it's a return
            llvm::ReturnInst::Create(Context, nullptr, Jump);

            // Cleanup everything it's aftewards
            llvm::BasicBlock *Parent = Jump->getParent();
            llvm::Instruction *ToDelete = &*(--Parent->end());
            while (ToDelete != Jump) {
              if (auto DeadBranch = llvm::dyn_cast<llvm::BranchInst>(ToDelete))
                purgeBranch(DeadBranch);
              else
                ToDelete->eraseFromParent();

              ToDelete = &*(--Parent->end());
            }

            // Remove the store to PC
            Jump->eraseFromParent();
          }
        } else
          llvm_unreachable("Unknown instruction using the PC");
      } else
        llvm_unreachable("Unhandled usage of the PC");
    }
  }

  /// Pop from the list of program counters to explore
  ///
  /// @return a pair containing the PC and the initial block to use, or
  ///         JumpTarget::NoMoreTargets if we're done.
  BlockWithAddress peekJumpTarget() {
    if (Unexplored.empty())
      return NoMoreTargets;
    else {
      BlockWithAddress Result = Unexplored.back();
      Unexplored.pop_back();
      return Result;
    }
  }

private:
  /// Get or create a block for the given PC
  llvm::BasicBlock *getBlockAt(uint64_t PC) {
    // Do we already have a BasicBlock for this PC?
    BlockMap::iterator TargetIt = JumpTargets.find(PC);
    if (TargetIt != JumpTargets.end()) {
      // Case 1: there's already a BasicBlock for that address, return it
      return TargetIt->second;
    }

    // Did we already meet this PC (i.e. to we know what's the associated
    // instruction)?
    llvm::BasicBlock *NewBlock = nullptr;
    InstructionMap::iterator InstrIt = OriginalInstructionAddresses.find(PC);
    if (InstrIt != OriginalInstructionAddresses.end()) {
      // Case 2: the address has already been met, but needs to be promoted to
      //         BasicBlock level.
      llvm::BasicBlock *ContainingBlock = InstrIt->second->getParent();
      if (InstrIt->second == &*ContainingBlock->begin())
        NewBlock = ContainingBlock;
      else {
        assert(InstrIt->second != nullptr &&
               InstrIt->second != ContainingBlock->end());
        // Split the block in the appropriate position. Note that
        // OriginalInstructionAddresses stores a reference to the last generated
        // instruction for the previous instruction.
        llvm::Instruction *Next = InstrIt->second->getNextNode();
        NewBlock = ContainingBlock->splitBasicBlock(Next);
      }
    } else {
      // Case 3: the address has never been met, create a temporary one,
      // register it for future exploration and return it
      NewBlock = llvm::BasicBlock::Create(Context, "", Function);
      Unexplored.push_back(BlockWithAddress(PC, NewBlock));
    }

    // Associate the PC with the chosen basic block
    JumpTargets[PC] = NewBlock;
    return NewBlock;
  }

  /// Helper function to destroy an unconditional branch and, in case, the
  /// target basic block, if it doesn't have any predecessors left.
  void purgeBranch(llvm::BasicBlock::iterator I) {
    auto *DeadBranch = llvm::dyn_cast<llvm::BranchInst>(I);
    // We allow only an unconditional branch and nothing else
    assert(DeadBranch != nullptr &&
           DeadBranch->isUnconditional() &&
           ++I == DeadBranch->getParent()->end());

    // Obtain the target of the dead branch
    llvm::BasicBlock *DeadBranchTarget = DeadBranch->getSuccessor(0);

    // Destroy the dead branch
    DeadBranch->eraseFromParent();

    // Check if someone else was jumping there and then destroy
    if (llvm::pred_empty(DeadBranchTarget))
      DeadBranchTarget->eraseFromParent();
  }

private:
  using BlockMap = std::map<uint64_t, llvm::BasicBlock *>;
  using InstructionMap = std::map<uint64_t, llvm::Instruction *>;

  llvm::LLVMContext& Context;
  llvm::Function* Function;
  /// Holds the association between a PC and the last generated instruction for
  /// the previous instruction.
  InstructionMap OriginalInstructionAddresses;
  /// Holds the association between a PC and a BasicBlock.
  BlockMap JumpTargets;
  /// Queue of program counters we still have to translate.
  std::vector<BlockWithAddress> Unexplored;
  llvm::Value *PCReg;
};

const JumpTargetManager::BlockWithAddress JumpTargetManager::NoMoreTargets =
  JumpTargetManager::BlockWithAddress(0, nullptr);

int Translate(std::string OutputPath,
              llvm::ArrayRef<uint8_t> Code,
              DebugInfoType DebugInfo,
              std::string DebugPath) {
  const uint8_t *CodePointer = Code.data();
  const uint8_t *CodeEnd = CodePointer + Code.size();

  // TODO: move me somewhere where it makes sense
  Architecture SourceArchitecture;
  Architecture TargetArchitecture;

  llvm::LLVMContext& Context = llvm::getGlobalContext();
  unsigned OriginalInstrMDKind = Context.getMDKindID("oi");
  unsigned PTCInstrMDKind = Context.getMDKindID("pi");
  unsigned DbgMDKind = Context.getMDKindID("dbg");
  llvm::IRBuilder<> Builder(Context);
  std::unique_ptr<llvm::Module> Module(new llvm::Module("top", Context));

  // Debugging information
  if (DebugPath.empty()) {
   if (DebugInfo == DebugInfoType::PTC)
     DebugPath = OutputPath + ".ptc";
   else if (DebugInfo == DebugInfoType::OriginalAssembly)
     DebugPath = OutputPath + ".S";
   else if (DebugInfo == DebugInfoType::LLVMIR)
     DebugPath = OutputPath;
  }


  llvm::DIBuilder Dbg(*Module);
  llvm::DICompileUnit *DbgCompileUnit = nullptr;

  if (DebugInfo != DebugInfoType::None) {
    DbgCompileUnit = Dbg.createCompileUnit(llvm::dwarf::DW_LANG_C,
                                           DebugPath,
                                           "",
                                           "revamb",
                                           false,
                                           "",
                                           0 /* Runtime version */);

    // Add the current debug info version into the module.
    Module->addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                          llvm::DEBUG_METADATA_VERSION);
    Module->addModuleFlag(llvm::Module::Warning, "Dwarf Version", 2);
  }

  // Create main function
  llvm::FunctionType *MainType = nullptr;
  MainType = llvm::FunctionType::get(Builder.getVoidTy(), false);
  llvm::Function *MainFunction = nullptr;
  MainFunction = llvm::Function::Create(MainType,
                                        llvm::Function::ExternalLinkage,
                                        "root",
                                        Module.get());

  llvm::DISubprogram *DbgMain = nullptr;
  if (DebugInfo != DebugInfoType::None) {
    llvm::DISubroutineType *EmptyType = nullptr;
    EmptyType = Dbg.createSubroutineType(DbgCompileUnit->getFile(),
                                         Dbg.getOrCreateTypeArray({}));

    DbgMain = Dbg.createFunction(DbgCompileUnit, /* Scope */
                                 "root", /* Name */
                                 llvm::StringRef(), /* Linkage name */
                                 DbgCompileUnit->getFile(), /* DIFile */
                                 1, /* Line */
                                 EmptyType, /* Subroutine type */
                                 false, /* isLocalToUnit */
                                 true, /* isDefinition */
                                 1, /* ScopeLine */
                                 llvm::DINode::FlagPrototyped, /* Flags */
                                 false, /* isOptimized */
                                 MainFunction /* Function */);
  }

  // Create the first basic block and create a placeholder for variable
  // allocations
  llvm::BasicBlock *Entry = llvm::BasicBlock::Create(Context,
                                                     "entrypoint",
                                                     MainFunction);
  Builder.SetInsertPoint(Entry);
  llvm::Instruction *Delimiter = Builder.CreateUnreachable();

  // Create register needed for managing the control flow
  llvm::GlobalVariable *PCReg = nullptr;
  PCReg = VariableManager::createGlobal(*Module,
                                        Builder.getInt32Ty(),
                                        llvm::GlobalValue::ExternalLinkage,
                                        0,
                                        "pc");

  // Instantiate helpers
  VariableManager Variables(*Module, { PCReg });
  JumpTargetManager JumpTargets(Context, PCReg, MainFunction);

  while (Entry != nullptr) {
    Builder.SetInsertPoint(Entry);

    std::map<std::string, llvm::BasicBlock *> LabeledBasicBlocks;

    // TODO: rename this type
    PTCInstructionListPtr InstructionList(new PTCInstructionList);
    size_t ConsumedSize = 0;

    ConsumedSize = ptc.translate(CodePointer,
                                 CodeEnd - CodePointer,
                                 InstructionList.get());

    Variables.newFunction(Delimiter, InstructionList.get());
    unsigned j = 0;

    while (j < InstructionList->instruction_count &&
           InstructionList->instructions[j].opc !=
           PTC_INSTRUCTION_op_debug_insn_start) {
      j++;
    }

    assert(j < InstructionList->instruction_count);

    llvm::MDNode* MDOriginalInstr = nullptr;


    bool StopTranslation = false;
    for (; j < InstructionList->instruction_count && !StopTranslation; j++) {
      PTCInstruction Instruction = InstructionList->instructions[j];
      PTCOpcode Opcode = Instruction.opc;

      if (Opcode == PTC_INSTRUCTION_op_call) {
        dumpTranslation(std::cerr, InstructionList.get());
        return EXIT_FAILURE;
      }

      std::vector<llvm::BasicBlock *> Blocks { Builder.GetInsertBlock() };

      std::vector<llvm::Value *> OutArguments;
      std::vector<llvm::Value *> InArguments;
      std::vector<uint64_t> ConstArguments;

      // Create or get variables
      llvm::Value *LoadInstruction = nullptr;
      unsigned InArgumentsCount = 0;
      InArgumentsCount = ptc_instruction_in_arg_count(&ptc, &Instruction);
      for (unsigned i = 0; i < InArgumentsCount; i++) {
        unsigned TemporaryId = ptc_instruction_in_arg(&ptc, &Instruction, i);
        LoadInstruction = Builder.CreateLoad(Variables.getOrCreate(TemporaryId));
        InArguments.push_back(LoadInstruction);
      }

      // Collect constant parameters
      unsigned ConstArgumentsCount = 0;
      ConstArgumentsCount = ptc_instruction_const_arg_count(&ptc, &Instruction);
      for (unsigned i = 0; i < ConstArgumentsCount; i++) {
        uint64_t Argument = ptc_instruction_const_arg(&ptc, &Instruction, i);
        ConstArguments.push_back(Argument);
      }

      unsigned RegisterSize = getRegisterSize(Opcode);
      llvm::Type *RegisterType = nullptr;
      if (RegisterSize == 32)
        RegisterType = Builder.getInt32Ty();
      else if (RegisterSize == 64)
        RegisterType = Builder.getInt64Ty();
      else if (RegisterSize != 0)
        llvm_unreachable("Unexpected register size");

      switch (Opcode) {
      case PTC_INSTRUCTION_op_movi_i32:
      case PTC_INSTRUCTION_op_movi_i64:
        OutArguments.push_back(llvm::ConstantInt::get(RegisterType,
                                                      ConstArguments[0]));
        break;
      case PTC_INSTRUCTION_op_discard:
        // Nothing to do here
        break;
      case PTC_INSTRUCTION_op_mov_i32:
      case PTC_INSTRUCTION_op_mov_i64:
        OutArguments.push_back(InArguments[0]);
        break;
      case PTC_INSTRUCTION_op_setcond_i32:
      case PTC_INSTRUCTION_op_setcond_i64:
        {
          llvm::Value *Compare = CreateICmp(Builder,
                                            ConstArguments[0],
                                            InArguments[0],
                                            InArguments[1]);
          // TODO: convert single-bit registers to i1
          llvm::Value *Result = Builder.CreateZExt(Compare, RegisterType);
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_movcond_i32: // Resist the fallthrough temptation
      case PTC_INSTRUCTION_op_movcond_i64:
        {
          llvm::Value *Compare = CreateICmp(Builder,
                                            ConstArguments[0],
                                            InArguments[0],
                                            InArguments[1]);
          llvm::Value *Select = Builder.CreateSelect(Compare,
                                                     InArguments[2],
                                                     InArguments[3]);
          OutArguments.push_back(Select);
        }
        break;
      case PTC_INSTRUCTION_op_qemu_ld_i32:
      case PTC_INSTRUCTION_op_qemu_ld_i64:
      case PTC_INSTRUCTION_op_qemu_st_i32:
      case PTC_INSTRUCTION_op_qemu_st_i64:
        {
          PTCLoadStoreArg MemoryAccess;
          MemoryAccess = ptc.parse_load_store_arg(ConstArguments[0]);

          // What are we supposed to do in this case?
          assert(MemoryAccess.access_type != PTC_MEMORY_ACCESS_UNKNOWN);

          unsigned AccessAlignment = 0;
          if (MemoryAccess.access_type == PTC_MEMORY_ACCESS_UNALIGNED)
            AccessAlignment = 1;
          else
            AccessAlignment = SourceArchitecture.defaultAlignment();

          // Load size
          llvm::IntegerType *MemoryType = nullptr;
          switch (ptc_get_memory_access_size(MemoryAccess.type)) {
          case PTC_MO_8:
            MemoryType = Builder.getInt8Ty();
            break;
          case PTC_MO_16:
            MemoryType = Builder.getInt16Ty();
            break;
          case PTC_MO_32:
            MemoryType = Builder.getInt32Ty();
            break;
          case PTC_MO_64:
            MemoryType = Builder.getInt64Ty();
            break;
          default:
            llvm_unreachable("Unexpected load size");
          }

          bool SignExtend = ptc_is_sign_extended_load(MemoryAccess.type);

          // TODO: handle 64 on 32
          // TODO: handle endianess mismatch
          assert(SourceArchitecture.endianess() ==
                 TargetArchitecture.endianess() &&
                 "Different endianess between the source and the target is not "
                 "supported yet");

          llvm::Value *Pointer = nullptr;
          if (Opcode == PTC_INSTRUCTION_op_qemu_ld_i32 ||
              Opcode == PTC_INSTRUCTION_op_qemu_ld_i64) {

            Pointer = Builder.CreateIntToPtr(InArguments[0],
                                             MemoryType->getPointerTo());
            llvm::Value *Load = Builder.CreateAlignedLoad(Pointer,
                                                          AccessAlignment);

            llvm::Value *Result = nullptr;
            if (SignExtend)
              Result = Builder.CreateSExt(Load, RegisterType);
            else
              Result = Builder.CreateZExt(Load, RegisterType);

            OutArguments.push_back(Result);

          } else if (Opcode == PTC_INSTRUCTION_op_qemu_st_i32 ||
                     Opcode == PTC_INSTRUCTION_op_qemu_st_i64) {

            Pointer = Builder.CreateIntToPtr(InArguments[1],
                                             MemoryType->getPointerTo());
            llvm::Value *Value = Builder.CreateTrunc(InArguments[0], MemoryType);
            Builder.CreateAlignedStore(Value, Pointer, AccessAlignment);

          } else
            llvm_unreachable("Unknown load type");

          break;
        }
      case PTC_INSTRUCTION_op_ld8u_i32:
      case PTC_INSTRUCTION_op_ld8s_i32:
      case PTC_INSTRUCTION_op_ld16u_i32:
      case PTC_INSTRUCTION_op_ld16s_i32:
      case PTC_INSTRUCTION_op_ld_i32:
      case PTC_INSTRUCTION_op_ld8u_i64:
      case PTC_INSTRUCTION_op_ld8s_i64:
      case PTC_INSTRUCTION_op_ld16u_i64:
      case PTC_INSTRUCTION_op_ld16s_i64:
      case PTC_INSTRUCTION_op_ld32u_i64:
      case PTC_INSTRUCTION_op_ld32s_i64:
      case PTC_INSTRUCTION_op_ld_i64:
      case PTC_INSTRUCTION_op_st8_i32:
      case PTC_INSTRUCTION_op_st16_i32:
      case PTC_INSTRUCTION_op_st_i32:
      case PTC_INSTRUCTION_op_st8_i64:
      case PTC_INSTRUCTION_op_st16_i64:
      case PTC_INSTRUCTION_op_st32_i64:
      case PTC_INSTRUCTION_op_st_i64:
        // We shouldn't have these instructions
        continue;
      case PTC_INSTRUCTION_op_add_i32:
      case PTC_INSTRUCTION_op_sub_i32:
      case PTC_INSTRUCTION_op_mul_i32:
      case PTC_INSTRUCTION_op_div_i32:
      case PTC_INSTRUCTION_op_divu_i32:
      case PTC_INSTRUCTION_op_rem_i32:
      case PTC_INSTRUCTION_op_remu_i32:
      case PTC_INSTRUCTION_op_and_i32:
      case PTC_INSTRUCTION_op_or_i32:
      case PTC_INSTRUCTION_op_xor_i32:
      case PTC_INSTRUCTION_op_shl_i32:
      case PTC_INSTRUCTION_op_shr_i32:
      case PTC_INSTRUCTION_op_sar_i32:
      case PTC_INSTRUCTION_op_add_i64:
      case PTC_INSTRUCTION_op_sub_i64:
      case PTC_INSTRUCTION_op_mul_i64:
      case PTC_INSTRUCTION_op_div_i64:
      case PTC_INSTRUCTION_op_divu_i64:
      case PTC_INSTRUCTION_op_rem_i64:
      case PTC_INSTRUCTION_op_remu_i64:
      case PTC_INSTRUCTION_op_and_i64:
      case PTC_INSTRUCTION_op_or_i64:
      case PTC_INSTRUCTION_op_xor_i64:
      case PTC_INSTRUCTION_op_shl_i64:
      case PTC_INSTRUCTION_op_shr_i64:
      case PTC_INSTRUCTION_op_sar_i64:
        {
          // TODO: assert on sizes?
          llvm::Instruction::BinaryOps BinaryOp = opcodeToBinaryOp(Opcode);
          llvm::Value *Operation = Builder.CreateBinOp(BinaryOp,
                                                       InArguments[0],
                                                       InArguments[1]);
          OutArguments.push_back(Operation);
          break;
        }
      case PTC_INSTRUCTION_op_div2_i32:
      case PTC_INSTRUCTION_op_divu2_i32:
      case PTC_INSTRUCTION_op_div2_i64:
      case PTC_INSTRUCTION_op_divu2_i64:
        {
          llvm::Instruction::BinaryOps DivisionOp, RemainderOp;

          if (Opcode == PTC_INSTRUCTION_op_div2_i32 ||
              Opcode == PTC_INSTRUCTION_op_div2_i64) {
            DivisionOp = llvm::Instruction::SDiv;
            RemainderOp = llvm::Instruction::SRem;
          } else if (Opcode == PTC_INSTRUCTION_op_div2_i32 ||
                     Opcode == PTC_INSTRUCTION_op_div2_i64) {
            DivisionOp = llvm::Instruction::UDiv;
            RemainderOp = llvm::Instruction::URem;
          } else
            llvm_unreachable("Unknown operation type");

          // TODO: we're ignoring InArguments[1], which is the MSB
          // TODO: assert on sizes?
          llvm::Value *Division = Builder.CreateBinOp(DivisionOp,
                                                      InArguments[0],
                                                      InArguments[2]);
          llvm::Value *Remainder = Builder.CreateBinOp(RemainderOp,
                                                       InArguments[0],
                                                       InArguments[2]);
          OutArguments.push_back(Division);
          OutArguments.push_back(Remainder);
          break;
        }
      case PTC_INSTRUCTION_op_rotr_i32:
      case PTC_INSTRUCTION_op_rotr_i64:
      case PTC_INSTRUCTION_op_rotl_i32:
      case PTC_INSTRUCTION_op_rotl_i64:
        {
          llvm::Value *Bits = llvm::ConstantInt::get(RegisterType, RegisterSize);

          llvm::Instruction::BinaryOps FirstShiftOp, SecondShiftOp;
          if (Opcode == PTC_INSTRUCTION_op_rotl_i32 ||
              Opcode == PTC_INSTRUCTION_op_rotl_i64) {
            FirstShiftOp = llvm::Instruction::LShr;
            SecondShiftOp = llvm::Instruction::Shl;
          } else if (Opcode == PTC_INSTRUCTION_op_rotr_i32 ||
                     Opcode == PTC_INSTRUCTION_op_rotr_i64) {
            FirstShiftOp = llvm::Instruction::Shl;
            SecondShiftOp = llvm::Instruction::LShr;
          } else
            llvm_unreachable("Unexpected opcode");

          llvm::Value *FirstShift = Builder.CreateBinOp(FirstShiftOp,
                                                        InArguments[0],
                                                        InArguments[1]);
          llvm::Value *SecondShiftAmount = Builder.CreateSub(Bits,
                                                             InArguments[1]);
          llvm::Value *SecondShift = Builder.CreateBinOp(SecondShiftOp,
                                                         InArguments[0],
                                                         SecondShiftAmount);
          llvm::Value *Result = Builder.CreateOr(FirstShift, SecondShift);

          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_deposit_i32:
      case PTC_INSTRUCTION_op_deposit_i64:
        {
          unsigned Position = ConstArguments[0];
          if (Position == RegisterSize) {
            OutArguments.push_back(InArguments[0]);
            break;
          }

          unsigned Length = ConstArguments[1];
          uint64_t Bits = 0;

          // Thou shall not << 32
          if (Length == RegisterSize)
            Bits = getMaxValue(RegisterSize);
          else
            Bits = (1 << Length) - 1;

          // result = (t1 & ~(bits << position)) | ((t2 & bits) << position)
          uint64_t BaseMask = ~(Bits << Position);
          llvm::Value *MaskedBase = Builder.CreateAnd(InArguments[0], BaseMask);
          llvm::Value *Deposit = Builder.CreateAnd(InArguments[1], Bits);
          llvm::Value *ShiftedDeposit = Builder.CreateShl(Deposit, Position);
          llvm::Value *Result = Builder.CreateOr(MaskedBase, ShiftedDeposit);

          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_ext8s_i32:
      case PTC_INSTRUCTION_op_ext16s_i32:
      case PTC_INSTRUCTION_op_ext8u_i32:
      case PTC_INSTRUCTION_op_ext16u_i32:
      case PTC_INSTRUCTION_op_ext8s_i64:
      case PTC_INSTRUCTION_op_ext16s_i64:
      case PTC_INSTRUCTION_op_ext32s_i64:
      case PTC_INSTRUCTION_op_ext8u_i64:
      case PTC_INSTRUCTION_op_ext16u_i64:
      case PTC_INSTRUCTION_op_ext32u_i64:
        {
          llvm::Type *SourceType = nullptr;
          switch (Opcode) {
          case PTC_INSTRUCTION_op_ext8s_i32:
          case PTC_INSTRUCTION_op_ext8u_i32:
          case PTC_INSTRUCTION_op_ext8s_i64:
          case PTC_INSTRUCTION_op_ext8u_i64:
            SourceType = Builder.getInt8Ty();
            break;
          case PTC_INSTRUCTION_op_ext16s_i32:
          case PTC_INSTRUCTION_op_ext16u_i32:
          case PTC_INSTRUCTION_op_ext16s_i64:
          case PTC_INSTRUCTION_op_ext16u_i64:
            SourceType = Builder.getInt16Ty();
            break;
          case PTC_INSTRUCTION_op_ext32s_i64:
          case PTC_INSTRUCTION_op_ext32u_i64:
            SourceType = Builder.getInt32Ty();
            break;
          default:
            llvm_unreachable("Unexpected opcode");
          }

          llvm::Value *Truncated = Builder.CreateTrunc(InArguments[0],
                                                       SourceType);

          llvm::Value *Result = nullptr;
          switch (Opcode) {
          case PTC_INSTRUCTION_op_ext8s_i32:
          case PTC_INSTRUCTION_op_ext8s_i64:
          case PTC_INSTRUCTION_op_ext16s_i32:
          case PTC_INSTRUCTION_op_ext16s_i64:
          case PTC_INSTRUCTION_op_ext32s_i64:
            Result = Builder.CreateSExt(Truncated, RegisterType);
            break;
          case PTC_INSTRUCTION_op_ext8u_i32:
          case PTC_INSTRUCTION_op_ext8u_i64:
          case PTC_INSTRUCTION_op_ext16u_i32:
          case PTC_INSTRUCTION_op_ext16u_i64:
          case PTC_INSTRUCTION_op_ext32u_i64:
            Result = Builder.CreateZExt(Truncated, RegisterType);
            break;
          default:
            llvm_unreachable("Unexpected opcode");
          }

          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_not_i32:
      case PTC_INSTRUCTION_op_not_i64:
        {
          llvm::Value *Result = Builder.CreateXor(InArguments[0],
                                                  getMaxValue(RegisterSize));
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_neg_i32:
      case PTC_INSTRUCTION_op_neg_i64:
        {
          llvm::Value *Zero = llvm::ConstantInt::get(RegisterType, 0);
          llvm::Value *Result = Builder.CreateSub(Zero, InArguments[0]);
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_andc_i32:
      case PTC_INSTRUCTION_op_andc_i64:
      case PTC_INSTRUCTION_op_orc_i32:
      case PTC_INSTRUCTION_op_orc_i64:
      case PTC_INSTRUCTION_op_eqv_i32:
      case PTC_INSTRUCTION_op_eqv_i64:
        {
          llvm::Instruction::BinaryOps ExternalOp;
          switch (Opcode) {
          case PTC_INSTRUCTION_op_andc_i32:
          case PTC_INSTRUCTION_op_andc_i64:
            ExternalOp = llvm::Instruction::And;
            break;
          case PTC_INSTRUCTION_op_orc_i32:
          case PTC_INSTRUCTION_op_orc_i64:
            ExternalOp = llvm::Instruction::Or;
            break;
          case PTC_INSTRUCTION_op_eqv_i32:
          case PTC_INSTRUCTION_op_eqv_i64:
            ExternalOp = llvm::Instruction::Xor;
            break;
          default:
            llvm_unreachable("Unexpected opcode");
          }

          llvm::Value *Negate = Builder.CreateXor(InArguments[1],
                                                  getMaxValue(RegisterSize));
          llvm::Value *Result = Builder.CreateBinOp(ExternalOp,
                                                    InArguments[0],
                                                    Negate);
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_nand_i32:
      case PTC_INSTRUCTION_op_nand_i64:
        {
          llvm::Value *AndValue = Builder.CreateAnd(InArguments[0],
                                                    InArguments[1]);
          llvm::Value *Result = Builder.CreateXor(AndValue,
                                                  getMaxValue(RegisterSize));
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_nor_i32:
      case PTC_INSTRUCTION_op_nor_i64:
        {
          llvm::Value *OrValue = Builder.CreateOr(InArguments[0],
                                                  InArguments[1]);
          llvm::Value *Result = Builder.CreateXor(OrValue,
                                                  getMaxValue(RegisterSize));
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_bswap16_i32:
      case PTC_INSTRUCTION_op_bswap32_i32:
      case PTC_INSTRUCTION_op_bswap16_i64:
      case PTC_INSTRUCTION_op_bswap32_i64:
      case PTC_INSTRUCTION_op_bswap64_i64:
        {
          llvm::Type *SwapType = nullptr;
          switch (Opcode) {
          case PTC_INSTRUCTION_op_bswap16_i32:
          case PTC_INSTRUCTION_op_bswap16_i64:
            SwapType = Builder.getInt16Ty();
          case PTC_INSTRUCTION_op_bswap32_i32:
          case PTC_INSTRUCTION_op_bswap32_i64:
            SwapType = Builder.getInt32Ty();
          case PTC_INSTRUCTION_op_bswap64_i64:
            SwapType = Builder.getInt64Ty();
          default:
            llvm_unreachable("Unexpected opcode");
          }

          llvm::Value *Truncated = Builder.CreateTrunc(InArguments[0], SwapType);

          std::vector<llvm::Type *> BSwapParameters { RegisterType };
          llvm::Function *BSwapFunction = nullptr;
          BSwapFunction = llvm::Intrinsic::getDeclaration(Module.get(),
                                                          llvm::Intrinsic::bswap,
                                                          BSwapParameters);
          llvm::Value *Swapped = Builder.CreateCall(BSwapFunction, Truncated);

          llvm::Value *Result = Builder.CreateZExt(Swapped, RegisterType);
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_set_label:
        {
          unsigned LabelId = ptc.get_arg_label_id(ConstArguments[0]);
          std::string Label = "L" + std::to_string(LabelId);

          llvm::BasicBlock *Fallthrough = nullptr;
          auto ExistingBasicBlock = LabeledBasicBlocks.find(Label);

          if (ExistingBasicBlock == LabeledBasicBlocks.end()) {
            Fallthrough = llvm::BasicBlock::Create(Context, Label, MainFunction);
            LabeledBasicBlocks[Label] = Fallthrough;
          } else {
            // A basic block with that label already exist
            Fallthrough = LabeledBasicBlocks[Label];

            // Ensure it's empty
            assert(Fallthrough->begin() == Fallthrough->end());

            // Move it to the bottom
            Fallthrough->removeFromParent();
            MainFunction->getBasicBlockList().push_back(Fallthrough);
          }

          Blocks.push_back(Fallthrough);
          Builder.CreateBr(Fallthrough);
          Builder.SetInsertPoint(Fallthrough);
          Variables.newBasicBlock();
          break;
        }
      case PTC_INSTRUCTION_op_br:
      case PTC_INSTRUCTION_op_brcond_i32:
      case PTC_INSTRUCTION_op_brcond2_i32:
      case PTC_INSTRUCTION_op_brcond_i64:
        {
          // We take the last constant arguments, which is the LabelId both in
          // conditional and unconditional jumps
          unsigned LabelId = ptc.get_arg_label_id(ConstArguments.back());
          std::string Label = "L" + std::to_string(LabelId);

          llvm::BasicBlock *Fallthrough = nullptr;
          Fallthrough = llvm::BasicBlock::Create(Context, "", MainFunction);

          // Look for a matching label
          llvm::BasicBlock *Target = nullptr;
          auto ExistingBasicBlock = LabeledBasicBlocks.find(Label);

          // No matching label, create a temporary block
          if (ExistingBasicBlock == LabeledBasicBlocks.end()) {
            Target = llvm::BasicBlock::Create(Context,
                                              Label,
                                              MainFunction);
            LabeledBasicBlocks[Label] = Target;
          } else
            Target = LabeledBasicBlocks[Label];

          if (Opcode == PTC_INSTRUCTION_op_br) {
            // Unconditional jump
            Builder.CreateBr(Target);
          } else if (Opcode == PTC_INSTRUCTION_op_brcond_i32 ||
                     Opcode == PTC_INSTRUCTION_op_brcond_i64) {
            // Conditional jump
            llvm::Value *Compare = CreateICmp(Builder,
                                              ConstArguments[0],
                                              InArguments[0],
                                              InArguments[1]);
            Builder.CreateCondBr(Compare, Target, Fallthrough);
          } else
            llvm_unreachable("Unhandled opcode");

          Blocks.push_back(Fallthrough);
          Builder.SetInsertPoint(Fallthrough);
          Variables.newBasicBlock();
          break;
        }
      case PTC_INSTRUCTION_op_debug_insn_start:
        {
          // A new original instruction, let's create a new metadata node
          // referencing it for all the next instructions to come
          uint64_t PC = ConstArguments[0];

          // TODO: replace using a field in Architecture
          if (ConstArguments.size() > 1)
            PC |= ConstArguments[1] << 32;

          std::stringstream OriginalStringStream;
          disassembleOriginal(OriginalStringStream, PC);
          std::string OriginalString = OriginalStringStream.str();
          llvm::MDString *MDOriginalString = nullptr;
          MDOriginalString = llvm::MDString::get(Context, OriginalString);
          MDOriginalInstr = llvm::MDNode::getDistinct(Context, MDOriginalString);

          if (PC != 0) {
            // Check if this PC already has a block and use it
            // TODO: rename me
            uint64_t RealPC = CodePointer - Code.data() + PC;
            bool ShouldContinue;
            llvm::BasicBlock *DivergeTo = JumpTargets.newPC(RealPC,
                                                            ShouldContinue);
            if (DivergeTo != nullptr) {
              Builder.CreateBr(DivergeTo);

              if (ShouldContinue) {
                // The block is empty, let's fill it
                Blocks.push_back(DivergeTo);
                Builder.SetInsertPoint(DivergeTo);
                Variables.newBasicBlock(DivergeTo);
              } else {
                // The block already contains translated code, early exit
                StopTranslation = true;
                break;
              }
            }

            // Inform the JumpTargetManager about the new PC we met
            llvm::BasicBlock::iterator CurrentIt = Builder.GetInsertPoint();
            if (CurrentIt == Builder.GetInsertBlock()->begin())
              JumpTargets.registerBlock(RealPC, Builder.GetInsertBlock());
            else
              JumpTargets.registerInstruction(RealPC, &*--CurrentIt);
          }

          break;
        }
      case PTC_INSTRUCTION_op_call:
        // TODO: implement call to helpers
        llvm_unreachable("Call to helpers not implemented");
      case PTC_INSTRUCTION_op_exit_tb:
      case PTC_INSTRUCTION_op_goto_tb:
        // Nothing to do here
        continue;
        break;
      case PTC_INSTRUCTION_op_add2_i32:
      case PTC_INSTRUCTION_op_sub2_i32:
      case PTC_INSTRUCTION_op_add2_i64:
      case PTC_INSTRUCTION_op_sub2_i64:
        {
          llvm::Value *FirstOperandLow = nullptr;
          llvm::Value *FirstOperandHigh = nullptr;
          llvm::Value *SecondOperandLow = nullptr;
          llvm::Value *SecondOperandHigh = nullptr;

          llvm::IntegerType *DestinationType = nullptr;
          DestinationType = Builder.getIntNTy(RegisterSize * 2);

          FirstOperandLow = Builder.CreateZExt(InArguments[0], DestinationType);
          FirstOperandHigh = Builder.CreateZExt(InArguments[1], DestinationType);
          SecondOperandLow = Builder.CreateZExt(InArguments[2], DestinationType);
          SecondOperandHigh = Builder.CreateZExt(InArguments[3],
                                                 DestinationType);

          FirstOperandHigh = Builder.CreateShl(FirstOperandHigh, RegisterSize);
          SecondOperandHigh = Builder.CreateShl(SecondOperandHigh, RegisterSize);

          llvm::Value *FirstOperand = Builder.CreateOr(FirstOperandHigh,
                                                       FirstOperandLow);
          llvm::Value *SecondOperand = Builder.CreateOr(SecondOperandHigh,
                                                        SecondOperandLow);

          llvm::Instruction::BinaryOps BinaryOp = opcodeToBinaryOp(Opcode);

          llvm::Value *Result = Builder.CreateBinOp(BinaryOp,
                                                    FirstOperand,
                                                    SecondOperand);

          llvm::Value *ResultLow = Builder.CreateTrunc(Result, RegisterType);
          llvm::Value *ShiftedResult = Builder.CreateLShr(Result, RegisterSize);
          llvm::Value *ResultHigh = Builder.CreateTrunc(ShiftedResult,
                                                        RegisterType);

          OutArguments.push_back(ResultLow);
          OutArguments.push_back(ResultHigh);

          break;
        }
      case PTC_INSTRUCTION_op_mulu2_i32:
      case PTC_INSTRUCTION_op_muls2_i32:
      case PTC_INSTRUCTION_op_mulu2_i64:
      case PTC_INSTRUCTION_op_muls2_i64:
      case PTC_INSTRUCTION_op_muluh_i32:
      case PTC_INSTRUCTION_op_mulsh_i32:
      case PTC_INSTRUCTION_op_muluh_i64:
      case PTC_INSTRUCTION_op_mulsh_i64:

      case PTC_INSTRUCTION_op_setcond2_i32:

      case PTC_INSTRUCTION_op_trunc_shr_i32:
        llvm_unreachable("Instruction not implemented");
      default:
        llvm_unreachable("Unknown opcode");
      }

      // Save the results in the output arguments
      unsigned OutArgumentsCount = ptc_instruction_out_arg_count(&ptc,
                                                                 &Instruction);
      assert(OutArgumentsCount == OutArguments.size());
      for (unsigned i = 0; i < OutArgumentsCount; i++) {
        unsigned TemporaryId = ptc_instruction_out_arg(&ptc, &Instruction, i);
        Builder.CreateStore(OutArguments[i], Variables.getOrCreate(TemporaryId));
      }

      // Create a new metadata referencing the PTC instruction we have just
      // translated
      std::stringstream PTCStringStream;
      dumpInstruction(PTCStringStream, InstructionList.get(), j);
      std::string PTCString = PTCStringStream.str() + "\n";
      llvm::MDString *MDPTCString = llvm::MDString::get(Context, PTCString);
      llvm::MDNode* MDPTCInstr = llvm::MDNode::getDistinct(Context, MDPTCString);

      // Set metadata for all the new instructions
      for (llvm::BasicBlock *Block : Blocks) {
        llvm::BasicBlock::iterator I = Block->end();
        while (I != Block->begin() && !(--I)->hasMetadata()) {
          I->setMetadata(OriginalInstrMDKind, MDOriginalInstr);
          I->setMetadata(PTCInstrMDKind, MDPTCInstr);
        }
      }

    } // End loop over instructions

    // Replace stores to PC with branches
    // TODO: constant propagation here
    JumpTargets.translateMovePC(CodePointer - Code.data());

    // Obtain a new program counter to translate
    uint64_t NewPC = 0;
    (void) ConsumedSize;
    std::tie(NewPC, Entry) = JumpTargets.peekJumpTarget();
    CodePointer = Code.data() + NewPC;
  } // End translations loop

  Delimiter->eraseFromParent();

  if (DebugInfo == DebugInfoType::PTC
      || DebugInfo == DebugInfoType::OriginalAssembly) {
    std::map<llvm::MDNode *, llvm::MDNode *> DbgMapping;
    unsigned LineIndex = 1;
    unsigned MetadataKind = DebugInfo == DebugInfoType::PTC ?
      PTCInstrMDKind : OriginalInstrMDKind;

    std::fstream Source(DebugPath, std::fstream::out);
    for (llvm::BasicBlock& Block : *MainFunction) {
      for (llvm::Instruction& Instruction : Block) {
        llvm::MDNode *MD = Instruction.getMetadata(MetadataKind);
        if (MD != nullptr) {
          auto MappingIt = DbgMapping.find(MD);
          if (false || MappingIt != DbgMapping.end()) {
            Instruction.setMetadata(DbgMDKind, MappingIt->second);
          } else {
            auto Body = llvm::dyn_cast<llvm::MDString>(MD->getOperand(0).get());
            std::string BodyString = Body->getString().str();
            Source << BodyString;

            auto *DbgLocation = llvm::DILocation::get(Context, LineIndex, 0,
                                                      DbgMain);
            DbgMapping[MD] = DbgLocation;
            Instruction.setMetadata(DbgMDKind, DbgLocation);
            LineIndex += std::count(BodyString.begin(), BodyString.end(), '\n');
          }
        }
      }
    }

  } else if (DebugInfo == DebugInfoType::LLVMIR) {
    llvm::raw_null_ostream Stream;
    SelfDescribingWriter Annotator(Context, DbgMain, true /* DebugInfo */);
    Module->print(Stream, &Annotator);
  }

  Dbg.finalize();

  {
    std::ofstream Output(OutputPath);
    llvm::raw_os_ostream Stream(Output);
    SelfDescribingWriter Annotator(Context, DbgMain, false /* DebugInfo */);
    Module->print(Stream, &Annotator);
  }

  if (DebugInfo == DebugInfoType::LLVMIR && DebugPath != OutputPath) {
    std::ifstream Source(OutputPath, std::ios::binary);
    std::ofstream Destination(DebugPath, std::ios::binary);

    Destination << Source.rdbuf();
  }

  return EXIT_SUCCESS;
}
