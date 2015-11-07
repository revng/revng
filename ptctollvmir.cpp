/// \file
/// \brief  This file handles the translation from QEMU's PTC to LLVM IR.

#include <cstdint>
#include <sstream>
#include <vector>
#include <fstream>
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

#include "ptctollvmir.h"
#include "ptcinterface.h"
#include "ptcdump.h"
#include "rai.h"
#include "range.h"
#include "transformadapter.h"

using namespace llvm;

/// Boring code to get the text of the metadata with the specified kind
/// associated to the given instruction
static MDString *getMD(const Instruction *Instruction,
                       unsigned Kind) {
  assert(Instruction != nullptr);

  Metadata *MD = Instruction->getMetadata(Kind);

  if (MD == nullptr)
    return nullptr;

  auto Node = dyn_cast<MDNode>(MD);

  assert(Node != nullptr);

  const MDOperand& Operand = Node->getOperand(0);

  Metadata *MDOperand = Operand.get();

  if (MDOperand == nullptr)
    return nullptr;

  auto *String = dyn_cast<MDString>(MDOperand);
  assert(String != nullptr);

  return String;
}

/// AssemblyAnnotationWriter implementation inserting in the generated LLVM IR
/// comments containing the original assembly and the PTC. It can also decorate
/// the IR with debug information (i.e. DILocations) refered to the generated
/// LLVM IR itself.
class DebugAnnotationWriter : public AssemblyAnnotationWriter {
public:
  DebugAnnotationWriter(LLVMContext& Context,
                        Metadata *Scope,
                        bool DebugInfo) : Context(Context),
                                          Scope(Scope),
                                          DebugInfo(DebugInfo) {
    OriginalInstrMDKind = Context.getMDKindID("oi");
    PTCInstrMDKind = Context.getMDKindID("pi");
    DbgMDKind = Context.getMDKindID("dbg");
  }

  virtual void emitInstructionAnnot(const Instruction *TheInstruction,
                                    formatted_raw_ostream &Output) {

    writeMetadataIfNew(TheInstruction, OriginalInstrMDKind, Output, "\n\n  ; ");
    writeMetadataIfNew(TheInstruction, PTCInstrMDKind, Output, "\n  ; ");

    if (DebugInfo) {
      // If DebugInfo is activated the generated LLVM IR textual representation
      // will contain some reference to dangling pointers. So ignore the output
      // stream if you're using the annotator to generate debug info about the
      // IR itself.
      assert(Scope != nullptr);

      // Flushing is required to have correct line and column numbers
      Output.flush();
      auto *Location = DILocation::get(Context,
                                       Output.getLine(),
                                       Output.getColumn(),
                                       Scope);

      // Sorry Bjarne
      auto *NonConstInstruction = const_cast<Instruction *>(TheInstruction);
      NonConstInstruction->setMetadata(DbgMDKind, Location);
    }
  }

private:
  /// Writes the text contained in the metadata with the specified kind ID to
  /// the output stream, unless that metadata is exactly the same as in the
  /// previous instruction.
  static void writeMetadataIfNew(const Instruction *TheInstruction,
                                 unsigned MDKind,
                                 formatted_raw_ostream &Output,
                                 StringRef Prefix) {
    MDString *MD = getMD(TheInstruction, MDKind);
    if (MD != nullptr) {
      const Instruction *PrevInstruction = nullptr;

      if (TheInstruction != TheInstruction->getParent()->begin())
        PrevInstruction = TheInstruction->getPrevNode();

      if (PrevInstruction == nullptr || getMD(PrevInstruction, MDKind) != MD)
        Output << Prefix << MD->getString();

    }
  }

private:
  LLVMContext &Context;
  Metadata *Scope;
  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;
  unsigned DbgMDKind;
  bool DebugInfo;
};

/// \brief Maintains the list of variables required by PTC.
///
/// It can be queried for a variable, which, if not already existing, will be
/// created on the fly.
class VariableManager {
public:
  VariableManager(Module& TheModule,
                  ArrayRef<GlobalVariable *> PredefinedGlobals) :
    TheModule(TheModule),
    Builder(TheModule.getContext()) {

    // Store all the predefined globals
    for (GlobalVariable *Global : PredefinedGlobals)
      Globals[Global->getName()] = Global;
  }
  /// Given a PTC temporary identifier, checks if it already exists in the
  /// generatd LLVM IR, and, if not, it creates it.
  ///
  /// \param TemporaryId the PTC temporary identifier.
  ///
  /// \return an Value wrapping the request global or local variable.
  Value* getOrCreate(unsigned int TemporaryId);

  /// Informs the VariableManager that a new function has begun, so it can
  /// discard function- and basic block-level variables.
  ///
  /// \param Delimiter the new point where to insert allocations for local
  /// variables.
  /// \param Instructions the new PTCInstructionList to use from now on.
  void newFunction(Instruction *Delimiter=nullptr,
                   PTCInstructionList *Instructions=nullptr) {
    LocalTemporaries.clear();
    newBasicBlock(Delimiter, Instructions);
  }

  /// Informs the VariableManager that a new basic block has begun, so it can
  /// discard basic block-level variables.
  ///
  /// \param Delimiter the new point where to insert allocations for local
  /// variables.
  /// \param Instructions the new PTCInstructionList to use from now on.
  void newBasicBlock(Instruction *Delimiter=nullptr,
                     PTCInstructionList *Instructions=nullptr) {
    Temporaries.clear();
    if (Instructions != nullptr)
      this->Instructions = Instructions;

    if (Delimiter != nullptr)
      Builder.SetInsertPoint(Delimiter);
  }

  void newBasicBlock(BasicBlock *Delimiter,
                     PTCInstructionList *Instructions=nullptr) {
    Temporaries.clear();
    if (Instructions != nullptr)
      this->Instructions = Instructions;

    if (Delimiter != nullptr)
      Builder.SetInsertPoint(Delimiter);
  }

  static GlobalVariable*
  createGlobal(Module& TheModule,
               Type *Type,
               GlobalValue::LinkageTypes Linkage,
               uint64_t Initializer = 0,
               const Twine& Name = "") {

    return new GlobalVariable(TheModule, Type, false, Linkage,
                              ConstantInt::get(Type, Initializer),
                              Name);
  }

private:
  Module& TheModule;
  IRBuilder<> Builder;
  using TemporariesMap = std::map<unsigned int, AllocaInst *>;
  using GlobalsMap = std::map<std::string, GlobalVariable *>;
  GlobalsMap Globals;
  TemporariesMap Temporaries;
  TemporariesMap LocalTemporaries;
  PTCInstructionList *Instructions;
  Instruction *Last;
};

Value* VariableManager::getOrCreate(unsigned int TemporaryId) {
  assert(Instructions != nullptr);

  PTCTemp *Temporary = ptc_temp_get(Instructions, TemporaryId);
  Type *VariableType = Temporary->type == PTC_TYPE_I32 ?
    Builder.getInt32Ty() : Builder.getInt64Ty();

  if (ptc_temp_is_global(Instructions, TemporaryId)) {
    auto TemporaryName = StringRef(Temporary->name).lower();

    GlobalsMap::iterator it = Globals.find(TemporaryName);
    if (it != Globals.end()) {
      return it->second;
    } else {
      GlobalVariable *NewVariable = createGlobal(TheModule,
                                                 VariableType,
                                                 GlobalValue::ExternalLinkage,
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
      AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      LocalTemporaries[TemporaryId] = NewTemporary;
      Result = NewTemporary;
    }
  } else {
    TemporariesMap::iterator it = Temporaries.find(TemporaryId);
    if (it != Temporaries.end()) {
      return it->second;
    } else {
      AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      Temporaries[TemporaryId] = NewTemporary;
      Result = NewTemporary;
    }
  }

  Last = Result;

  return Result;
}

/// Converts a PTC condition into an LLVM predicate
///
/// \param Condition the input PTC condition.
///
/// \return the corresponding LLVM predicate.
static CmpInst::Predicate conditionToPredicate(PTCCondition Condition) {
  switch (Condition) {
  case PTC_COND_NEVER:
    // TODO: this is probably wrong
    return CmpInst::FCMP_FALSE;
  case PTC_COND_ALWAYS:
    // TODO: this is probably wrong
    return CmpInst::FCMP_TRUE;
  case PTC_COND_EQ:
    return CmpInst::ICMP_EQ;
  case PTC_COND_NE:
    return CmpInst::ICMP_NE;
  case PTC_COND_LT:
    return CmpInst::ICMP_SLT;
  case PTC_COND_GE:
    return CmpInst::ICMP_SGE;
  case PTC_COND_LE:
    return CmpInst::ICMP_SLE;
  case PTC_COND_GT:
    return CmpInst::ICMP_SGT;
  case PTC_COND_LTU:
    return CmpInst::ICMP_ULT;
  case PTC_COND_GEU:
    return CmpInst::ICMP_UGE;
  case PTC_COND_LEU:
    return CmpInst::ICMP_ULE;
  case PTC_COND_GTU:
    return CmpInst::ICMP_UGT;
  default:
    llvm_unreachable("Unknown comparison operator");
  }
}

/// Obtains the LLVM binary operation corresponding to the specified PTC opcode.
///
/// \param Opcode the PTC opcode.
///
/// \return the LLVM binary operation matching opcode.
static Instruction::BinaryOps opcodeToBinaryOp(PTCOpcode Opcode) {
  switch (Opcode) {
  case PTC_INSTRUCTION_op_add_i32:
  case PTC_INSTRUCTION_op_add_i64:
  case PTC_INSTRUCTION_op_add2_i32:
  case PTC_INSTRUCTION_op_add2_i64:
    return Instruction::Add;
  case PTC_INSTRUCTION_op_sub_i32:
  case PTC_INSTRUCTION_op_sub_i64:
  case PTC_INSTRUCTION_op_sub2_i32:
  case PTC_INSTRUCTION_op_sub2_i64:
    return Instruction::Sub;
  case PTC_INSTRUCTION_op_mul_i32:
  case PTC_INSTRUCTION_op_mul_i64:
    return Instruction::Mul;
  case PTC_INSTRUCTION_op_div_i32:
  case PTC_INSTRUCTION_op_div_i64:
    return Instruction::SDiv;
  case PTC_INSTRUCTION_op_divu_i32:
  case PTC_INSTRUCTION_op_divu_i64:
    return Instruction::UDiv;
  case PTC_INSTRUCTION_op_rem_i32:
  case PTC_INSTRUCTION_op_rem_i64:
    return Instruction::SRem;
  case PTC_INSTRUCTION_op_remu_i32:
  case PTC_INSTRUCTION_op_remu_i64:
    return Instruction::URem;
  case PTC_INSTRUCTION_op_and_i32:
  case PTC_INSTRUCTION_op_and_i64:
    return Instruction::And;
  case PTC_INSTRUCTION_op_or_i32:
  case PTC_INSTRUCTION_op_or_i64:
    return Instruction::Or;
  case PTC_INSTRUCTION_op_xor_i32:
  case PTC_INSTRUCTION_op_xor_i64:
    return Instruction::Xor;
  case PTC_INSTRUCTION_op_shl_i32:
  case PTC_INSTRUCTION_op_shl_i64:
    return Instruction::Shl;
  case PTC_INSTRUCTION_op_shr_i32:
  case PTC_INSTRUCTION_op_shr_i64:
    return Instruction::LShr;
  case PTC_INSTRUCTION_op_sar_i32:
  case PTC_INSTRUCTION_op_sar_i64:
    return Instruction::AShr;
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
/// \return the size, in bits, of the registers used by the opcode.
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
/// \param Builder the builder to use to create the instruction.
/// \param RawCondition the PTC condition.
/// \param FirstOperand the first operand of the comparison.
/// \param SecondOperand the second operand of the comparison.
///
/// \return a compare instruction.
template<typename T>
static Value *CreateICmp(T& Builder,
                         uint64_t RawCondition,
                         Value *FirstOperand,
                         Value *SecondOperand) {
  PTCCondition Condition = static_cast<PTCCondition>(RawCondition);
  return Builder.CreateICmp(conditionToPredicate(Condition),
                            FirstOperand,
                            SecondOperand);
}

class JumpTargetManager {
public:
  using BlockWithAddress = std::pair<uint64_t, BasicBlock *>;
  static const BlockWithAddress NoMoreTargets;

public:
  JumpTargetManager(LLVMContext& Context,
                    Value *PCReg,
                    Function *TheFunction) :
    Context(Context),
    TheFunction(TheFunction),
    OriginalInstructionAddresses(),
    JumpTargets(),
    PCReg(PCReg) { }

  /// Handle a new program counter. We might already have a basic block for that
  /// program counter, or we could even have a translation for it. Return one
  /// of these, if appropriate.
  ///
  /// \param PC the new program counter.
  /// \param ShouldContinue an out parameter indicating whether the returned
  ///        basic block was just a placeholder or actually contains a
  ///        translation.
  ///
  /// \return the basic block to use from now on, or null if the program counter
  ///         is not associated to a basic block.
  BasicBlock *newPC(uint64_t PC, bool& ShouldContinue) {
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
  void registerInstruction(uint64_t PC, Instruction *Instruction) {
    // Never save twice a PC
    assert(OriginalInstructionAddresses.find(PC) ==
           OriginalInstructionAddresses.end());
    OriginalInstructionAddresses[PC] = Instruction;
  }

  /// Save the PC-BasicBlock association for futur use (jump target)
  void registerBlock(uint64_t PC, BasicBlock *Block) {
    // If we already met it, it must point to the same block
    auto It = JumpTargets.find(PC);
    assert(It == JumpTargets.end() || It->second == Block);
    if (It->second != Block)
      JumpTargets[PC] = Block;
  }

  /// Look for all the stores targeting the program counter and add a branch
  /// there as appropriate.
  void translateMovePC(uint64_t BasePC) {
    for (Use& PCUse : PCReg->uses()) {
      // TODO: what to do in case of read of the PC?
      // Is the PC the store destination?
      if (PCUse.getOperandNo() == 1) {
        if (auto Jump = dyn_cast<StoreInst>(PCUse.getUser())) {
          Value *Destination = Jump->getValueOperand();

          // Is desintation a constant?
          if (auto Address = dyn_cast<ConstantInt>(Destination)) {
            // Compute the actual PC
            uint64_t TargetPC = BasePC + Address->getSExtValue();

            // Get or create the block for this PC and branch there
            BasicBlock *TargetBlock = getBlockAt(TargetPC);
            Instruction *Branch = BranchInst::Create(TargetBlock);

            // Cleanup of what's afterwards (only a unconditional jump is
            // allowed)
            BasicBlock::iterator I = Jump;
            BasicBlock::iterator BlockEnd = Jump->getParent()->end();
            if (++I != BlockEnd)
              purgeBranch(I);

            Branch->insertAfter(Jump);
            Jump->eraseFromParent();
          } else {
            // TODO: very strong assumption here
            // Destination is not a constant, assume it's a return
            ReturnInst::Create(Context, nullptr, Jump);

            // Cleanup everything it's aftewards
            BasicBlock *Parent = Jump->getParent();
            Instruction *ToDelete = &*(--Parent->end());
            while (ToDelete != Jump) {
              if (auto DeadBranch = dyn_cast<BranchInst>(ToDelete))
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
  /// \return a pair containing the PC and the initial block to use, or
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
  BasicBlock *getBlockAt(uint64_t PC) {
    // Do we already have a BasicBlock for this PC?
    BlockMap::iterator TargetIt = JumpTargets.find(PC);
    if (TargetIt != JumpTargets.end()) {
      // Case 1: there's already a BasicBlock for that address, return it
      return TargetIt->second;
    }

    // Did we already meet this PC (i.e. to we know what's the associated
    // instruction)?
    BasicBlock *NewBlock = nullptr;
    InstructionMap::iterator InstrIt = OriginalInstructionAddresses.find(PC);
    if (InstrIt != OriginalInstructionAddresses.end()) {
      // Case 2: the address has already been met, but needs to be promoted to
      //         BasicBlock level.
      BasicBlock *ContainingBlock = InstrIt->second->getParent();
      if (InstrIt->second == &*ContainingBlock->begin())
        NewBlock = ContainingBlock;
      else {
        assert(InstrIt->second != nullptr &&
               InstrIt->second != ContainingBlock->end());
        // Split the block in the appropriate position. Note that
        // OriginalInstructionAddresses stores a reference to the last generated
        // instruction for the previous instruction.
        Instruction *Next = InstrIt->second->getNextNode();
        NewBlock = ContainingBlock->splitBasicBlock(Next);
      }
    } else {
      // Case 3: the address has never been met, create a temporary one,
      // register it for future exploration and return it
      NewBlock = BasicBlock::Create(Context, "", TheFunction);
      Unexplored.push_back(BlockWithAddress(PC, NewBlock));
    }

    // Associate the PC with the chosen basic block
    JumpTargets[PC] = NewBlock;
    return NewBlock;
  }

  /// Helper function to destroy an unconditional branch and, in case, the
  /// target basic block, if it doesn't have any predecessors left.
  void purgeBranch(BasicBlock::iterator I) {
    auto *DeadBranch = dyn_cast<BranchInst>(I);
    // We allow only an unconditional branch and nothing else
    assert(DeadBranch != nullptr &&
           DeadBranch->isUnconditional() &&
           ++I == DeadBranch->getParent()->end());

    // Obtain the target of the dead branch
    BasicBlock *DeadBranchTarget = DeadBranch->getSuccessor(0);

    // Destroy the dead branch
    DeadBranch->eraseFromParent();

    // Check if someone else was jumping there and then destroy
    if (pred_empty(DeadBranchTarget))
      DeadBranchTarget->eraseFromParent();
  }

private:
  using BlockMap = std::map<uint64_t, BasicBlock *>;
  using InstructionMap = std::map<uint64_t, Instruction *>;

  LLVMContext& Context;
  Function* TheFunction;
  /// Holds the association between a PC and the last generated instruction for
  /// the previous instruction.
  InstructionMap OriginalInstructionAddresses;
  /// Holds the association between a PC and a BasicBlock.
  BlockMap JumpTargets;
  /// Queue of program counters we still have to translate.
  std::vector<BlockWithAddress> Unexplored;
  Value *PCReg;
};

const JumpTargetManager::BlockWithAddress JumpTargetManager::NoMoreTargets =
  JumpTargetManager::BlockWithAddress(0, nullptr);

/// Handle all the debug-related operations of code generation
class DebugHelper {
public:
  DebugHelper(std::string OutputPath,
              std::string DebugPath,
              Module *TheModule,
              DebugInfoType Type) : OutputPath(OutputPath),
                                    DebugPath(DebugPath),
                                    Builder(*TheModule),
                                    Type(Type),
                                    TheModule(TheModule)
  {
    OriginalInstrMDKind = TheModule->getContext().getMDKindID("oi");
    PTCInstrMDKind = TheModule->getContext().getMDKindID("pi");
    DbgMDKind = TheModule->getContext().getMDKindID("dbg");

    // Generate automatically the name of the source file for debugging
    if (DebugPath.empty()) {
      if (Type == DebugInfoType::PTC)
        DebugPath = OutputPath + ".ptc";
      else if (Type == DebugInfoType::OriginalAssembly)
        DebugPath = OutputPath + ".S";
      else if (Type == DebugInfoType::LLVMIR)
        DebugPath = OutputPath;
    }

    if (Type != DebugInfoType::None) {
      CompileUnit = Builder.createCompileUnit(dwarf::DW_LANG_C,
                                              DebugPath,
                                              "",
                                              "revamb",
                                              false,
                                              "",
                                              0 /* Runtime version */);

      // Add the current debug info version into the module.
      TheModule->addModuleFlag(Module::Warning, "Debug Info Version",
                               DEBUG_METADATA_VERSION);
      TheModule->addModuleFlag(Module::Warning, "Dwarf Version", 2);
    }
  }

  /// \brief Handle a new function
  ///
  /// Generates the debug information for the given function and caches it for
  /// future use.
  void newFunction(Function *Function) {
    if (Type != DebugInfoType::None) {
      DISubroutineType *EmptyType = nullptr;
      EmptyType = Builder.createSubroutineType(CompileUnit->getFile(),
                                               Builder.getOrCreateTypeArray({}));

      CurrentFunction = Function;
      CurrentSubprogram = Builder.createFunction(CompileUnit, /* Scope */
                                                 Function->getName(),
                                                 StringRef(), /* Linkage name */
                                                 CompileUnit->getFile(),
                                                 1, /* Line */
                                                 EmptyType, /* Subroutine type */
                                                 false, /* isLocalToUnit */
                                                 true, /* isDefinition */
                                                 1, /* ScopeLine */
                                                 DINode::FlagPrototyped,
                                                 false, /* isOptimized */
                                                 CurrentFunction /* Function */);
    }
  }

  /// Decorates the current function with the request debug info
  void generateDebugInfo() {
    switch (Type) {
    case DebugInfoType::PTC:
    case DebugInfoType::OriginalAssembly:
      {
        assert(CurrentSubprogram != nullptr && CurrentFunction != nullptr);

        // Generate the source file and the debugging information in tandem

        unsigned LineIndex = 1;
        unsigned MetadataKind = Type == DebugInfoType::PTC ?
          PTCInstrMDKind : OriginalInstrMDKind;

        std::ofstream Source(DebugPath);
        for (BasicBlock& Block : *CurrentFunction) {
          for (Instruction& Instruction : Block) {
            MDString *Body = getMD(&Instruction, MetadataKind);
            std::string BodyString = Body->getString().str();

            Source << BodyString;

            auto *Location = DILocation::get(TheModule->getContext(),
                                             LineIndex,
                                             0,
                                             CurrentSubprogram);
            Instruction.setMetadata(DbgMDKind, Location);
            LineIndex += std::count(BodyString.begin(),
                                    BodyString.end(),
                                    '\n');
          }
        }
        break;
      }
    case DebugInfoType::LLVMIR:
      {
        // Use the annotator to obtain line and column of the textual LLVM IR
        // for each instruction. Discard the output since it will contain
        // errors, regenerating it later will give a correct result.
        raw_null_ostream NullStream;
        TheModule->print(NullStream, annotator(true /* DebugInfo */));

        std::ofstream Output(DebugPath);
        raw_os_ostream Stream(Output);
        TheModule->print(Stream, annotator(false));

        break;
      }
    default:
      break;
    }

    Builder.finalize();
  }

  /// Create a new AssemblyAnnotationWriter
  ///
  /// \param DebugInfo whether to decorate the IR with debug information or not
  DebugAnnotationWriter *annotator(bool DebugInfo) {
    Annotator.reset(new DebugAnnotationWriter(TheModule->getContext(),
                                              CurrentSubprogram,
                                              DebugInfo));
    return Annotator.get();
  }

  /// Copy the debug file to the output path, if they are the same
  bool copySource() {
    // If debug info refer to LLVM IR, just copy the output file
    if (Type == DebugInfoType::LLVMIR && DebugPath != OutputPath) {
      std::ifstream Source(DebugPath, std::ios::binary);
      std::ofstream Destination(OutputPath, std::ios::binary);

      Destination << Source.rdbuf();

      return true;
    }

    return false;
  }

private:
  std::string& OutputPath;
  std::string DebugPath;
  DIBuilder Builder;
  DebugInfoType Type;
  Module *TheModule;
  DICompileUnit *CompileUnit;
  DISubprogram *CurrentSubprogram;
  Function *CurrentFunction;
  std::unique_ptr<DebugAnnotationWriter> Annotator;

  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;
  unsigned DbgMDKind;
};

// Outline the destructor for the sake of privacy in the header
CodeGenerator::~CodeGenerator() = default;

CodeGenerator::CodeGenerator(Architecture& Source,
                             Architecture& Target,
                             std::string OutputPath,
                             DebugInfoType DebugInfo,
                             std::string DebugPath) :
  SourceArchitecture(Source),
  TargetArchitecture(Target),
  Context(getGlobalContext()),
  TheModule((new Module("top", Context))),
  Debug(new DebugHelper(OutputPath, DebugPath, TheModule.get(), DebugInfo)),
  OutputPath(OutputPath)
{
  OriginalInstrMDKind = Context.getMDKindID("oi");
  PTCInstrMDKind = Context.getMDKindID("pi");
  DbgMDKind = Context.getMDKindID("dbg");
}

int CodeGenerator::translate(ArrayRef<uint8_t> Code,
                             std::string Name) {
  const uint8_t *CodePointer = Code.data();
  const uint8_t *CodeEnd = CodePointer + Code.size();

  IRBuilder<> Builder(Context);

  // Create main function
  FunctionType *MainType = nullptr;
  MainType = FunctionType::get(Builder.getVoidTy(), false);
  Function *MainFunction = nullptr;
  MainFunction = Function::Create(MainType,
                                  Function::ExternalLinkage,
                                  Name,
                                  TheModule.get());

  Debug->newFunction(MainFunction);

  // Create the first basic block and create a placeholder for variable
  // allocations
  BasicBlock *Entry = BasicBlock::Create(Context,
                                         "entrypoint",
                                         MainFunction);
  Builder.SetInsertPoint(Entry);
  Instruction *Delimiter = Builder.CreateUnreachable();

  // Create register needed for managing the control flow
  GlobalVariable *PCReg = nullptr;
  PCReg = VariableManager::createGlobal(*TheModule,
                                        Builder.getInt32Ty(),
                                        GlobalValue::ExternalLinkage,
                                        0, /* Initial value */
                                        "pc");

  // Instantiate helpers
  VariableManager Variables(*TheModule, { PCReg });
  JumpTargetManager JumpTargets(Context, PCReg, MainFunction);

  while (Entry != nullptr) {
    Builder.SetInsertPoint(Entry);

    std::map<std::string, BasicBlock *> LabeledBasicBlocks;

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

    MDNode* MDOriginalInstr = nullptr;

    bool StopTranslation = false;
    for (; j < InstructionList->instruction_count && !StopTranslation; j++) {
      PTCInstruction Instruction = InstructionList->instructions[j];
      PTCOpcode Opcode = Instruction.opc;

      if (Opcode == PTC_INSTRUCTION_op_call) {
        dumpTranslation(std::cerr, InstructionList.get());
        return EXIT_FAILURE;
      }

      std::vector<BasicBlock *> Blocks { Builder.GetInsertBlock() };

      std::vector<Value *> OutArguments;
      std::vector<Value *> InArguments;
      std::vector<uint64_t> ConstArguments;

      // Create or get variables
      Value *LoadInstruction = nullptr;
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
      Type *RegisterType = nullptr;
      if (RegisterSize == 32)
        RegisterType = Builder.getInt32Ty();
      else if (RegisterSize == 64)
        RegisterType = Builder.getInt64Ty();
      else if (RegisterSize != 0)
        llvm_unreachable("Unexpected register size");

      switch (Opcode) {
      case PTC_INSTRUCTION_op_movi_i32:
      case PTC_INSTRUCTION_op_movi_i64:
        OutArguments.push_back(ConstantInt::get(RegisterType,
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
          Value *Compare = CreateICmp(Builder,
                                      ConstArguments[0],
                                      InArguments[0],
                                      InArguments[1]);
          // TODO: convert single-bit registers to i1
          Value *Result = Builder.CreateZExt(Compare, RegisterType);
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_movcond_i32: // Resist the fallthrough temptation
      case PTC_INSTRUCTION_op_movcond_i64:
        {
          Value *Compare = CreateICmp(Builder,
                                      ConstArguments[0],
                                      InArguments[0],
                                      InArguments[1]);
          Value *Select = Builder.CreateSelect(Compare,
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
          IntegerType *MemoryType = nullptr;
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

          Value *Pointer = nullptr;
          if (Opcode == PTC_INSTRUCTION_op_qemu_ld_i32 ||
              Opcode == PTC_INSTRUCTION_op_qemu_ld_i64) {

            Pointer = Builder.CreateIntToPtr(InArguments[0],
                                             MemoryType->getPointerTo());
            Value *Load = Builder.CreateAlignedLoad(Pointer,
                                                    AccessAlignment);

            Value *Result = nullptr;
            if (SignExtend)
              Result = Builder.CreateSExt(Load, RegisterType);
            else
              Result = Builder.CreateZExt(Load, RegisterType);

            OutArguments.push_back(Result);

          } else if (Opcode == PTC_INSTRUCTION_op_qemu_st_i32 ||
                     Opcode == PTC_INSTRUCTION_op_qemu_st_i64) {

            Pointer = Builder.CreateIntToPtr(InArguments[1],
                                             MemoryType->getPointerTo());
            Value *Value = Builder.CreateTrunc(InArguments[0], MemoryType);
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
          Instruction::BinaryOps BinaryOp = opcodeToBinaryOp(Opcode);
          Value *Operation = Builder.CreateBinOp(BinaryOp,
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
          Instruction::BinaryOps DivisionOp, RemainderOp;

          if (Opcode == PTC_INSTRUCTION_op_div2_i32 ||
              Opcode == PTC_INSTRUCTION_op_div2_i64) {
            DivisionOp = Instruction::SDiv;
            RemainderOp = Instruction::SRem;
          } else if (Opcode == PTC_INSTRUCTION_op_div2_i32 ||
                     Opcode == PTC_INSTRUCTION_op_div2_i64) {
            DivisionOp = Instruction::UDiv;
            RemainderOp = Instruction::URem;
          } else
            llvm_unreachable("Unknown operation type");

          // TODO: we're ignoring InArguments[1], which is the MSB
          // TODO: assert on sizes?
          Value *Division = Builder.CreateBinOp(DivisionOp,
                                                InArguments[0],
                                                InArguments[2]);
          Value *Remainder = Builder.CreateBinOp(RemainderOp,
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
          Value *Bits = ConstantInt::get(RegisterType, RegisterSize);

          Instruction::BinaryOps FirstShiftOp, SecondShiftOp;
          if (Opcode == PTC_INSTRUCTION_op_rotl_i32 ||
              Opcode == PTC_INSTRUCTION_op_rotl_i64) {
            FirstShiftOp = Instruction::LShr;
            SecondShiftOp = Instruction::Shl;
          } else if (Opcode == PTC_INSTRUCTION_op_rotr_i32 ||
                     Opcode == PTC_INSTRUCTION_op_rotr_i64) {
            FirstShiftOp = Instruction::Shl;
            SecondShiftOp = Instruction::LShr;
          } else
            llvm_unreachable("Unexpected opcode");

          Value *FirstShift = Builder.CreateBinOp(FirstShiftOp,
                                                  InArguments[0],
                                                  InArguments[1]);
          Value *SecondShiftAmount = Builder.CreateSub(Bits,
                                                       InArguments[1]);
          Value *SecondShift = Builder.CreateBinOp(SecondShiftOp,
                                                   InArguments[0],
                                                   SecondShiftAmount);
          Value *Result = Builder.CreateOr(FirstShift, SecondShift);

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
          Value *MaskedBase = Builder.CreateAnd(InArguments[0], BaseMask);
          Value *Deposit = Builder.CreateAnd(InArguments[1], Bits);
          Value *ShiftedDeposit = Builder.CreateShl(Deposit, Position);
          Value *Result = Builder.CreateOr(MaskedBase, ShiftedDeposit);

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
          Type *SourceType = nullptr;
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

          Value *Truncated = Builder.CreateTrunc(InArguments[0],
                                                 SourceType);

          Value *Result = nullptr;
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
          Value *Result = Builder.CreateXor(InArguments[0],
                                            getMaxValue(RegisterSize));
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_neg_i32:
      case PTC_INSTRUCTION_op_neg_i64:
        {
          Value *Zero = ConstantInt::get(RegisterType, 0);
          Value *Result = Builder.CreateSub(Zero, InArguments[0]);
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
          Instruction::BinaryOps ExternalOp;
          switch (Opcode) {
          case PTC_INSTRUCTION_op_andc_i32:
          case PTC_INSTRUCTION_op_andc_i64:
            ExternalOp = Instruction::And;
            break;
          case PTC_INSTRUCTION_op_orc_i32:
          case PTC_INSTRUCTION_op_orc_i64:
            ExternalOp = Instruction::Or;
            break;
          case PTC_INSTRUCTION_op_eqv_i32:
          case PTC_INSTRUCTION_op_eqv_i64:
            ExternalOp = Instruction::Xor;
            break;
          default:
            llvm_unreachable("Unexpected opcode");
          }

          Value *Negate = Builder.CreateXor(InArguments[1],
                                            getMaxValue(RegisterSize));
          Value *Result = Builder.CreateBinOp(ExternalOp,
                                              InArguments[0],
                                              Negate);
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_nand_i32:
      case PTC_INSTRUCTION_op_nand_i64:
        {
          Value *AndValue = Builder.CreateAnd(InArguments[0],
                                              InArguments[1]);
          Value *Result = Builder.CreateXor(AndValue,
                                            getMaxValue(RegisterSize));
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_nor_i32:
      case PTC_INSTRUCTION_op_nor_i64:
        {
          Value *OrValue = Builder.CreateOr(InArguments[0],
                                            InArguments[1]);
          Value *Result = Builder.CreateXor(OrValue,
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
          Type *SwapType = nullptr;
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

          Value *Truncated = Builder.CreateTrunc(InArguments[0], SwapType);

          std::vector<Type *> BSwapParameters { RegisterType };
          Function *BSwapFunction = Intrinsic::getDeclaration(TheModule.get(),
                                                              Intrinsic::bswap,
                                                              BSwapParameters);
          Value *Swapped = Builder.CreateCall(BSwapFunction, Truncated);

          Value *Result = Builder.CreateZExt(Swapped, RegisterType);
          OutArguments.push_back(Result);
          break;
        }
      case PTC_INSTRUCTION_op_set_label:
        {
          unsigned LabelId = ptc.get_arg_label_id(ConstArguments[0]);
          std::string Label = "L" + std::to_string(LabelId);

          BasicBlock *Fallthrough = nullptr;
          auto ExistingBasicBlock = LabeledBasicBlocks.find(Label);

          if (ExistingBasicBlock == LabeledBasicBlocks.end()) {
            Fallthrough = BasicBlock::Create(Context, Label, MainFunction);
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

          BasicBlock *Fallthrough = BasicBlock::Create(Context,
                                                       "",
                                                       MainFunction);

          // Look for a matching label
          BasicBlock *Target = nullptr;
          auto ExistingBasicBlock = LabeledBasicBlocks.find(Label);

          // No matching label, create a temporary block
          if (ExistingBasicBlock == LabeledBasicBlocks.end()) {
            Target = BasicBlock::Create(Context,
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
            Value *Compare = CreateICmp(Builder,
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
          MDString *MDOriginalString = MDString::get(Context, OriginalString);
          MDOriginalInstr = MDNode::getDistinct(Context, MDOriginalString);

          if (PC != 0) {
            // Check if this PC already has a block and use it
            // TODO: rename me
            uint64_t RealPC = CodePointer - Code.data() + PC;
            bool ShouldContinue;
            BasicBlock *DivergeTo = JumpTargets.newPC(RealPC,
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
            BasicBlock::iterator CurrentIt = Builder.GetInsertPoint();
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
          Value *FirstOperandLow = nullptr;
          Value *FirstOperandHigh = nullptr;
          Value *SecondOperandLow = nullptr;
          Value *SecondOperandHigh = nullptr;

          IntegerType *DestinationType = Builder.getIntNTy(RegisterSize * 2);

          FirstOperandLow = Builder.CreateZExt(InArguments[0], DestinationType);
          FirstOperandHigh = Builder.CreateZExt(InArguments[1], DestinationType);
          SecondOperandLow = Builder.CreateZExt(InArguments[2], DestinationType);
          SecondOperandHigh = Builder.CreateZExt(InArguments[3],
                                                 DestinationType);

          FirstOperandHigh = Builder.CreateShl(FirstOperandHigh, RegisterSize);
          SecondOperandHigh = Builder.CreateShl(SecondOperandHigh, RegisterSize);

          Value *FirstOperand = Builder.CreateOr(FirstOperandHigh,
                                                 FirstOperandLow);
          Value *SecondOperand = Builder.CreateOr(SecondOperandHigh,
                                                  SecondOperandLow);

          Instruction::BinaryOps BinaryOp = opcodeToBinaryOp(Opcode);

          Value *Result = Builder.CreateBinOp(BinaryOp,
                                              FirstOperand,
                                              SecondOperand);

          Value *ResultLow = Builder.CreateTrunc(Result, RegisterType);
          Value *ShiftedResult = Builder.CreateLShr(Result, RegisterSize);
          Value *ResultHigh = Builder.CreateTrunc(ShiftedResult,
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
      MDString *MDPTCString = MDString::get(Context, PTCString);
      MDNode* MDPTCInstr = MDNode::getDistinct(Context, MDPTCString);

      // Set metadata for all the new instructions
      for (BasicBlock *Block : Blocks) {
        BasicBlock::iterator I = Block->end();
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

  Debug->generateDebugInfo();

  return EXIT_SUCCESS;
}

void CodeGenerator::serialize() {
  // Ask the debug handler if it already has a good copy of the IR, if not dump
  // it
  if (!Debug->copySource()) {
    std::ofstream Output(OutputPath);
    raw_os_ostream Stream(Output);
    TheModule->print(Stream, Debug->annotator(false));
  }
}
