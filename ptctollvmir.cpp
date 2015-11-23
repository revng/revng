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
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Dominators.h"

#include "ptctollvmir.h"
#include "ptcinterface.h"
#include "ptcdump.h"
#include "rai.h"
#include "range.h"
#include "transformadapter.h"

using namespace llvm;

static uint64_t getConst(Value *Constant) {
  return cast<ConstantInt>(Constant)->getLimitedValue();
}

/// Helper function to destroy an unconditional branch and, in case, the
/// target basic block, if it doesn't have any predecessors left.
static void purgeBranch(BasicBlock::iterator I) {
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

namespace PTC {

template<bool C>
class InstructionImpl;

enum ArgumentType {
  In,
  Out,
  Const
};

template<ArgumentType Type, bool IsCall>
class InstructionArgumentsIterator :
  public RandomAccessIterator<uint64_t,
                              InstructionArgumentsIterator<Type, IsCall>,
                              false> {
public:
  using base = RandomAccessIterator<uint64_t,
                                    InstructionArgumentsIterator,
                                    false>;

  InstructionArgumentsIterator&
  operator=(const InstructionArgumentsIterator& r) {
    base::operator=(r);
    TheInstruction = r.TheInstruction;
    return *this;
  }


  InstructionArgumentsIterator(const InstructionArgumentsIterator& r) :
    base(r),
    TheInstruction(r.TheInstruction)
  { }

  InstructionArgumentsIterator(const InstructionArgumentsIterator& r,
                               unsigned Index) :
    base(Index),
    TheInstruction(r.TheInstruction)
  { }

  InstructionArgumentsIterator(PTCInstruction *TheInstruction, unsigned Index) :
    base(Index),
    TheInstruction(TheInstruction)
  { }

  bool isCompatible(const InstructionArgumentsIterator& r) const {
    return TheInstruction == r.TheInstruction;
  }

public:
  uint64_t get(unsigned Index) const;

private:
  PTCInstruction *TheInstruction;
};

template<>
inline uint64_t
InstructionArgumentsIterator<In, true>::get(unsigned Index) const {
  return ptc_call_instruction_in_arg(&ptc, TheInstruction, Index);
}

template<>
inline uint64_t
InstructionArgumentsIterator<Const, true>::get(unsigned Index) const {
  return ptc_call_instruction_const_arg(&ptc, TheInstruction, Index);
}

template<>
inline uint64_t
InstructionArgumentsIterator<Out, true>::get(unsigned Index) const {
  return ptc_call_instruction_out_arg(&ptc, TheInstruction, Index);
}

template<>
inline uint64_t
InstructionArgumentsIterator<In, false>::get(unsigned Index) const {
  return ptc_instruction_in_arg(&ptc, TheInstruction, Index);
}

template<>
inline uint64_t
InstructionArgumentsIterator<Const, false>::get(unsigned Index) const {
  return ptc_instruction_const_arg(&ptc, TheInstruction, Index);
}

template<>
inline uint64_t
InstructionArgumentsIterator<Out, false>::get(unsigned Index) const {
  return ptc_instruction_out_arg(&ptc, TheInstruction, Index);
}

template<bool IsCall>
class InstructionImpl {
private:
  template<ArgumentType Type>
  using arguments = InstructionArgumentsIterator<Type, IsCall>;
public:
  InstructionImpl(PTCInstruction *TheInstruction) :
    TheInstruction(TheInstruction),
    InArguments(arguments<In>(TheInstruction, 0),
                arguments<In>(TheInstruction, inArgCount())),
    ConstArguments(arguments<Const>(TheInstruction, 0),
                   arguments<Const>(TheInstruction, constArgCount())),
    OutArguments(arguments<Out>(TheInstruction, 0),
                 arguments<Out>(TheInstruction, outArgCount()))
  { }

  PTCOpcode opcode() const {
    return TheInstruction->opc;
  }

  std::string helperName() const {
    assert(IsCall);
    PTCHelperDef *Helper = ptc_find_helper(&ptc, ConstArguments[0]);
    assert(Helper != nullptr && Helper->name != nullptr);
    return std::string(Helper->name);
  }

private:
  PTCInstruction* TheInstruction;

public:
  const Range<InstructionArgumentsIterator<In, IsCall>> InArguments;
  const Range<InstructionArgumentsIterator<Const, IsCall>> ConstArguments;
  const Range<InstructionArgumentsIterator<Out, IsCall>> OutArguments;

private:
  unsigned inArgCount() const;
  unsigned constArgCount() const;
  unsigned outArgCount() const;
};

using Instruction = InstructionImpl<false>;
using CallInstruction = InstructionImpl<true>;

template<>
inline unsigned CallInstruction::inArgCount() const {
  return ptc_call_instruction_in_arg_count(&ptc, TheInstruction);
}

template<>
inline unsigned Instruction::inArgCount() const {
  return ptc_instruction_in_arg_count(&ptc, TheInstruction);
}

template<>
inline unsigned CallInstruction::constArgCount() const {
  return ptc_call_instruction_const_arg_count(&ptc, TheInstruction);
}

template<>
inline unsigned Instruction::constArgCount() const {
  return ptc_instruction_const_arg_count(&ptc, TheInstruction);
}

template<>
inline unsigned CallInstruction::outArgCount() const {
  return ptc_call_instruction_out_arg_count(&ptc, TheInstruction);
}

template<>
inline unsigned Instruction::outArgCount() const {
  return ptc_instruction_out_arg_count(&ptc, TheInstruction);
}

}

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
                                       Output.getLine() + 1,
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
                  StructType *CPUStateType,
                  const DataLayout *HelpersModuleLayout) :
    TheModule(TheModule),
    Builder(TheModule.getContext()),
    CPUStateType(CPUStateType),
    HelpersModuleLayout(HelpersModuleLayout) {
  }
  /// Given a PTC temporary identifier, checks if it already exists in the
  /// generatd LLVM IR, and, if not, it creates it.
  ///
  /// \param TemporaryId the PTC temporary identifier.
  ///
  /// \return an Value wrapping the request global or local variable.
  // TODO: rename to getByTemporaryId
  Value *getOrCreate(unsigned int TemporaryId);

  GlobalVariable *getByCPUStateOffset(intptr_t Offset, std::string Name);

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

private:
  Module& TheModule;
  IRBuilder<> Builder;
  using TemporariesMap = std::map<unsigned int, AllocaInst *>;
  using GlobalsMap = std::map<intptr_t, GlobalVariable *>;
  GlobalsMap CPUStateGlobals;
  GlobalsMap OtherGlobals;
  TemporariesMap Temporaries;
  TemporariesMap LocalTemporaries;
  PTCInstructionList *Instructions;

  StructType *CPUStateType;
  const DataLayout *HelpersModuleLayout;
};

static Type *getTypeAtOffset(const DataLayout *TheLayout,
                             StructType *TheStruct,
                             intptr_t Offset) {
  const StructLayout *Layout = TheLayout->getStructLayout(TheStruct);
  unsigned FieldIndex = Layout->getElementContainingOffset(Offset);
  uint64_t FieldOffset = Layout->getElementOffset(FieldIndex);

  Type *VariableType = TheStruct->getTypeAtIndex(FieldIndex);

  if (VariableType->isIntegerTy())
    return VariableType;
  else if (VariableType->isArrayTy())
    return VariableType->getArrayElementType();
  else if (VariableType->isStructTy())
    return getTypeAtOffset(TheLayout,
                           dyn_cast<StructType>(VariableType),
                           Offset - FieldOffset);
  else
    llvm_unreachable("Unexpected data type");
}

GlobalVariable* VariableManager::getByCPUStateOffset(intptr_t Offset,
                                                     std::string Name="") {

  GlobalsMap::iterator it = CPUStateGlobals.find(Offset);
  if (it != CPUStateGlobals.end()) {
    // TODO: handle renaming
    return it->second;
  } else {
    Type *VariableType = getTypeAtOffset(HelpersModuleLayout,
                                         CPUStateType,
                                         Offset);

    if (Name.size() == 0) {
      std::stringstream NameStream;
      NameStream << "state_0x" << std::hex << Offset;
      Name = NameStream.str();
    }

    auto *NewVariable = new GlobalVariable(TheModule,
                                           VariableType,
                                           false,
                                           GlobalValue::ExternalLinkage,
                                           ConstantInt::get(VariableType, 0),
                                           Name);
    assert(NewVariable != nullptr);
    CPUStateGlobals[Offset] = NewVariable;

    return NewVariable;
  }

}

Value* VariableManager::getOrCreate(unsigned int TemporaryId) {
  assert(Instructions != nullptr);

  PTCTemp *Temporary = ptc_temp_get(Instructions, TemporaryId);
  Type *VariableType = Temporary->type == PTC_TYPE_I32 ?
    Builder.getInt32Ty() : Builder.getInt64Ty();

  if (ptc_temp_is_global(Instructions, TemporaryId)) {
    // Basically we use fixed_reg to detect "env"
    if (Temporary->fixed_reg == 0) {
      return getByCPUStateOffset(Temporary->mem_offset,
                                 StringRef(Temporary->name));
    } else {
      GlobalsMap::iterator it = OtherGlobals.find(TemporaryId);
      if (it != OtherGlobals.end()) {
        return it->second;
      } else {
        GlobalVariable *Result = new GlobalVariable(TheModule,
                                          VariableType,
                                          false,
                                          GlobalValue::ExternalLinkage,
                                          ConstantInt::get(VariableType, 0),
                                          StringRef(Temporary->name));
        OtherGlobals[TemporaryId] = Result;
        return Result;
      }
    }
  } else if (Temporary->temp_local) {
    TemporariesMap::iterator it = LocalTemporaries.find(TemporaryId);
    if (it != LocalTemporaries.end()) {
      return it->second;
    } else {
      AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      LocalTemporaries[TemporaryId] = NewTemporary;
      return NewTemporary;
    }
  } else {
    TemporariesMap::iterator it = Temporaries.find(TemporaryId);
    if (it != Temporaries.end()) {
      return it->second;
    } else {
      AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      Temporaries[TemporaryId] = NewTemporary;
      return NewTemporary;
    }
  }
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
  JumpTargetManager(Module& TheModule,
                    Value *PCReg,
                    Function *TheFunction) :
    TheModule(TheModule),
    Context(TheModule.getContext()),
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

  void translateIndirectJumps() {
    BasicBlock *Dispatcher = createDispatcher(TheFunction, PCReg, true);

    for (Use& PCUse : PCReg->uses()) {
      if (PCUse.getOperandNo() == 1) {
        if (auto Jump = dyn_cast<StoreInst>(PCUse.getUser())) {
          BasicBlock::iterator It(Jump);
          auto *Branch = BranchInst::Create(Dispatcher, ++It);

          // Cleanup everything it's aftewards
          BasicBlock *Parent = Jump->getParent();
          Instruction *ToDelete = &*(--Parent->end());
          while (ToDelete != Branch) {
            if (auto DeadBranch = dyn_cast<BranchInst>(ToDelete))
              purgeBranch(DeadBranch);
            else
              ToDelete->eraseFromParent();

            ToDelete = &*(--Parent->end());
          }
        }
      }
    }
  }

  Value *PC() {
    return PCReg;
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

  /// Get or create a block for the given PC
  BasicBlock *getBlockAt(uint64_t PC) {
    // Do we already have a BasicBlock for this PC?
    BlockMap::iterator TargetIt = JumpTargets.find(PC);
    if (TargetIt != JumpTargets.end()) {
      // Case 1: there's already a BasicBlock for that address, return it
      return TargetIt->second;
    }

    // Did we already meet this PC (i.e. do we know what's the associated
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

private:
  // TODO: instead of a gigantic switch case we could map the original memory
  //       area and write the address of the translated basic block at the jump
  //       target
  BasicBlock *createDispatcher(Function *OutputFunction,
                               Value *SwitchOnPtr,
                               bool JumpDirectly) {
    IRBuilder<> Builder(Context);

    // Create the first block of the function
    BasicBlock *Entry = BasicBlock::Create(Context, "", OutputFunction);

    // The default case of the switch statement it's an unhandled cases
    auto *Default = BasicBlock::Create(Context, "", OutputFunction);
    Builder.SetInsertPoint(Default);
    Builder.CreateUnreachable();

    // Switch on the first argument of the function
    Builder.SetInsertPoint(Entry);
    Value *SwitchOn = Builder.CreateLoad(SwitchOnPtr);
    SwitchInst *Switch = Builder.CreateSwitch(SwitchOn, Default);
    auto *SwitchOnType = cast<IntegerType>(SwitchOn->getType());

    {
      // We consider a jump to NULL as a program end
      auto *NullBlock = BasicBlock::Create(Context, "", OutputFunction);
      Switch->addCase(ConstantInt::get(SwitchOnType, 0), NullBlock);
      Builder.SetInsertPoint(NullBlock);
      Builder.CreateRetVoid();
    }

    // Create a case for each jump target we saw so far
    for (auto& Pair : JumpTargets) {
      // Create a case for the address associated to the current block
      auto *Block = BasicBlock::Create(Context, "", OutputFunction);
      Switch->addCase(ConstantInt::get(SwitchOnType, Pair.first), Block);

      Builder.SetInsertPoint(Block);
      if (JumpDirectly) {
        // Assume we're injecting the switch case directly into the function
        // the blocks are in, so we can jump to the target block directly
        assert(Pair.second->getParent() == OutputFunction);
        Builder.CreateBr(Pair.second);
      } else {
        // Return the address of the current block
        Builder.CreateRet(BlockAddress::get(OutputFunction, Pair.second));
      }
    }

    return Entry;
  }

private:
  using BlockMap = std::map<uint64_t, BasicBlock *>;
  using InstructionMap = std::map<uint64_t, Instruction *>;

  Module &TheModule;
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

class TranslateDirectBranchesPass : public FunctionPass {
public:
  static char ID;

  TranslateDirectBranchesPass() : FunctionPass(ID),
                                  JTM(nullptr),
                                  NewPCMarker(nullptr) { }

  TranslateDirectBranchesPass(JumpTargetManager *JTM,
                              Function *NewPCMarker) :
    FunctionPass(ID),
    JTM(JTM),
    NewPCMarker(NewPCMarker) { }

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<DominatorTreeWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    LLVMContext &Context = F.getParent()->getContext();

    for (Use& PCUse : JTM->PC()->uses()) {
      // TODO: what to do in case of read of the PC?
      // Is the PC the store destination?
      if (PCUse.getOperandNo() == 1) {
        if (auto Jump = dyn_cast<StoreInst>(PCUse.getUser())) {
          Value *Destination = Jump->getValueOperand();

          // Is destination a constant?
          if (auto Address = dyn_cast<ConstantInt>(Destination)) {
            // If necessary notify the about the existence of the basic block
            // coming after this jump
            // TODO: handle delay slots
            BasicBlock *FakeFallthrough = JTM->getBlockAt(getNextPC(Jump));

            // Compute the actual PC and get the associated BasicBlock
            uint64_t TargetPC = Address->getSExtValue();
            BasicBlock *TargetBlock = JTM->getBlockAt(TargetPC);

            // Use a conditional branch here, even if the condition is always
            // true. This way the "fallthrough" basic block is always reachable
            // and the dominator tree computation works properly even if the
            // dispatcher switch has not been emitted yet
            Instruction *Branch = BranchInst::Create(TargetBlock,
                                                     FakeFallthrough,
                                                     ConstantInt::getTrue(Context));

            // Cleanup of what's afterwards (only a unconditional jump is
            // allowed)
            BasicBlock::iterator I = Jump;
            BasicBlock::iterator BlockEnd = Jump->getParent()->end();
            if (++I != BlockEnd)
              purgeBranch(I);

            Branch->insertAfter(Jump);
            Jump->eraseFromParent();
          }
        } else
          llvm_unreachable("Unknown instruction using the PC");
      } else
        llvm_unreachable("Unhandled usage of the PC");
    }

    return true;
  }

private:
  uint64_t getNextPC(Instruction *TheInstruction) {
    DominatorTree& DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();

    BasicBlock *Block = TheInstruction->getParent();
    BasicBlock::iterator It(TheInstruction);

    while (true) {
      BasicBlock::iterator Begin(Block->begin());

      // Go back towards the beginning of the basic block looking for a call to
      // NewPCMarker
      CallInst *Marker = nullptr;
      for (; It != Begin; It--)
        if ((Marker = dyn_cast<CallInst>(&*It)))
          if (Marker->getCalledFunction() == NewPCMarker) {
            uint64_t PC = getConst(Marker->getArgOperand(0));
            uint64_t Size = getConst(Marker->getArgOperand(1));
            assert(Size != 0);
            return PC + Size;
          }

      auto *Node = DT.getNode(Block);
      assert(Node != nullptr);

      Block = Node->getIDom()->getBlock();
      It = Block->end();
    }

    llvm_unreachable("Can't find the PC marker");
  }

private:
  Value *PCReg;
  JumpTargetManager *JTM;
  Function *NewPCMarker;
};

char TranslateDirectBranchesPass::ID = 0;
static RegisterPass<TranslateDirectBranchesPass> X("hello", "Hello World Pass", false, false);

/// Handle all the debug-related operations of code generation
class DebugHelper {
public:
  DebugHelper(std::string Output,
              std::string Debug,
              Module *TheModule,
              DebugInfoType Type) :
    OutputPath(Output),
    DebugPath(Debug),
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
      EmptyType = Builder.createSubroutineType(Builder.getOrCreateTypeArray({}));

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

        MDString *Last = nullptr;
        std::ofstream Source(DebugPath);
        for (BasicBlock& Block : *CurrentFunction) {
          for (Instruction& Instruction : Block) {
            MDString *Body = getMD(&Instruction, MetadataKind);

            if (Body != nullptr && Last != Body) {
              Last = Body;
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
        }

        Builder.finalize();
        break;
      }
    case DebugInfoType::LLVMIR:
      {
        // Use the annotator to obtain line and column of the textual LLVM IR
        // for each instruction. Discard the output since it will contain
        // errors, regenerating it later will give a correct result.
        Builder.finalize();

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

  }

  void print(std::ostream& Output, bool DebugInfo) {
    raw_os_ostream OutputStream(Output);
    TheModule->print(OutputStream, annotator(DebugInfo));
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
  /// Create a new AssemblyAnnotationWriter
  ///
  /// \param DebugInfo whether to decorate the IR with debug information or not
  DebugAnnotationWriter *annotator(bool DebugInfo) {
    Annotator.reset(new DebugAnnotationWriter(TheModule->getContext(),
                                              CurrentSubprogram,
                                              DebugInfo));
    return Annotator.get();
  }

private:
  std::string OutputPath;
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

static bool startsWith(std::string String, std::string Prefix) {
  return String.substr(0, Prefix.size()) == Prefix;
}

CodeGenerator::CodeGenerator(Architecture& Source,
                             Architecture& Target,
                             std::string Output,
                             std::string Helpers,
                             DebugInfoType DebugInfo,
                             std::string Debug) :
  SourceArchitecture(Source),
  TargetArchitecture(Target),
  Context(getGlobalContext()),
  TheModule((new Module("top", Context))),
  OutputPath(Output),
  Debug(new DebugHelper(Output, Debug, TheModule.get(), DebugInfo)),
  CPUStateType(nullptr),
  HelpersModuleLayout(nullptr)
{
  OriginalInstrMDKind = Context.getMDKindID("oi");
  PTCInstrMDKind = Context.getMDKindID("pi");
  DbgMDKind = Context.getMDKindID("dbg");

  SMDiagnostic Errors;
  HelpersModule = parseIRFile(Helpers, Errors, Context);

  using ElectionMap = std::map<StructType *, unsigned>;
  using ElectionMapElement = std::pair<StructType * const, unsigned>;
  ElectionMap EnvElection;
  const std::string HelperPrefix = "helper_";
  for (Function& HelperFunction : *HelpersModule) {
    if (startsWith(HelperFunction.getName(), HelperPrefix)
        && HelperFunction.getFunctionType()->getNumParams() > 1) {

      for (Type *Candidate : HelperFunction.getFunctionType()->params()) {
        if (Candidate->isPointerTy()) {
          auto *PointeeType = Candidate->getPointerElementType();
          auto *EnvType = dyn_cast<StructType>(PointeeType);
          // Ensure it is a struct and not a union
          if (EnvType != nullptr && EnvType->getNumElements() > 1) {

            auto It = EnvElection.find(EnvType);
            if (It != EnvElection.end())
              EnvElection[EnvType]++;
            else
              EnvElection[EnvType] = 1;
          }
        }
      }
    }
  }

  assert(EnvElection.size() > 0);

  CPUStateType = std::max_element(EnvElection.begin(),
                                  EnvElection.end(),
                                  [] (ElectionMapElement& It1,
                                      ElectionMapElement& It2) {
                                    return It1.second < It2.second;
                                  })->first;

  HelpersModuleLayout = &HelpersModule->getDataLayout();
}

class InstructionTranslator {
public:
  InstructionTranslator(IRBuilder<>& Builder,
                        VariableManager& Variables,
                        JumpTargetManager& JumpTargets,
                        std::map<std::string, BasicBlock *>& LabeledBasicBlocks,
                        std::vector<BasicBlock *> Blocks,
                        Module& TheModule,
                        Function *TheFunction,
                        Architecture& SourceArchitecture,
                        Architecture& TargetArchitecture) :
    Builder(Builder),
    Variables(Variables),
    JumpTargets(JumpTargets),
    LabeledBasicBlocks(LabeledBasicBlocks),
    Blocks(Blocks),
    TheModule(TheModule),
    TheFunction(TheFunction),
    SourceArchitecture(SourceArchitecture),
    TargetArchitecture(TargetArchitecture),
    NewPCMarker(nullptr),
    LastMarker(nullptr) {

    auto &Context = TheModule.getContext();
    NewPCMarker = Function::Create(FunctionType::get(Type::getVoidTy(Context),
                                                     {
                                                       Type::getInt64Ty(Context),
                                                       Type::getInt64Ty(Context)
                                                     },
                                                     false),
                                   GlobalValue::ExternalLinkage,
                                   "newpc",
                                   &TheModule);
  }

  TranslateDirectBranchesPass *createTranslateDirectBranchesPass() {
    return new TranslateDirectBranchesPass(&JumpTargets, NewPCMarker);
  }

  std::pair<bool, MDNode *> newInstruction(PTCInstruction *Instr,
                                           bool IsFirst);
  void translate(PTCInstruction *Instr);
  void translateCall(PTCInstruction *Instr);

  void removeNewPCMarkers() {

    for (User *Call : NewPCMarker->users())
      if (cast<Instruction>(Call)->getParent() != nullptr)
        cast<Instruction>(Call)->eraseFromParent();

    TheModule.dump();

    NewPCMarker->eraseFromParent();
  }

  void closeLastInstruction(uint64_t PC) {
    assert(LastMarker != nullptr);

    auto *Operand = cast<ConstantInt>(LastMarker->getArgOperand(0));
    uint64_t StartPC = Operand->getLimitedValue();

    assert(PC > StartPC);
    LastMarker->setArgOperand(1, Builder.getInt64(PC - StartPC));

    LastMarker = nullptr;
  }

private:
  std::vector<Value *> translateOpcode(PTCOpcode Opcode,
                                       std::vector<uint64_t> ConstArguments,
                                       std::vector<Value *> InArguments);
private:
  IRBuilder<>& Builder;
  VariableManager& Variables;
  JumpTargetManager& JumpTargets;
  std::map<std::string, BasicBlock *>& LabeledBasicBlocks;
  std::vector<BasicBlock *> Blocks;
  Module& TheModule;

  Function *TheFunction;

  Architecture& SourceArchitecture;
  Architecture& TargetArchitecture;

  Function *NewPCMarker;
  CallInst *LastMarker;
};

std::pair<bool, MDNode *>
InstructionTranslator::newInstruction(PTCInstruction *Instr,
                                      bool IsFirst) {
  const PTC::Instruction TheInstruction(Instr);
  // A new original instruction, let's create a new metadata node
  // referencing it for all the next instructions to come
  uint64_t PC = TheInstruction.ConstArguments[0];

  // TODO: replace using a field in Architecture
  if (TheInstruction.ConstArguments.size() > 1)
    PC |= TheInstruction.ConstArguments[1] << 32;

  std::stringstream OriginalStringStream;
  disassembleOriginal(OriginalStringStream, PC);
  std::string OriginalString = OriginalStringStream.str();
  LLVMContext& Context = TheModule.getContext();
  MDString *MDOriginalString = MDString::get(Context, OriginalString);
  MDNode *MDOriginalInstr = MDNode::getDistinct(Context, MDOriginalString);

  if (!IsFirst) {
    // Check if this PC already has a block and use it
    bool ShouldContinue;
    BasicBlock *DivergeTo = JumpTargets.newPC(PC, ShouldContinue);
    if (DivergeTo != nullptr) {
      Builder.CreateBr(DivergeTo);

      if (ShouldContinue) {
        // The block is empty, let's fill it
        Blocks.push_back(DivergeTo);
        Builder.SetInsertPoint(DivergeTo);
        Variables.newBasicBlock();
      } else {
        // The block contains already translated code, early exit
        return { true, MDOriginalInstr };
      }
    }
  }

  if (LastMarker != nullptr)
    closeLastInstruction(PC);
  LastMarker = Builder.CreateCall(NewPCMarker,
                                  { Builder.getInt64(PC), Builder.getInt64(0) });

  if (!IsFirst) {
    // Inform the JumpTargetManager about the new PC we met
    BasicBlock::iterator CurrentIt = Builder.GetInsertPoint();
    if (CurrentIt == Builder.GetInsertBlock()->begin())
      JumpTargets.registerBlock(PC, Builder.GetInsertBlock());
    else
      JumpTargets.registerInstruction(PC, LastMarker);
  }

  return { false, MDOriginalInstr };
}

void InstructionTranslator::translateCall(PTCInstruction *Instr) {
  const PTC::CallInstruction TheCall(Instr);

  auto LoadArgs = [this] (uint64_t TemporaryId) -> Value * {
    return Builder.CreateLoad(Variables.getOrCreate(TemporaryId));
  };

  auto GetValueType = [] (Value *Argument) { return Argument->getType(); };

  std::vector<Value *> InArgs = (TheCall.InArguments | LoadArgs).toVector();
  std::vector<Type *> InArgsType = (InArgs | GetValueType).toVector();

  // TODO: handle multiple return arguments
  assert(TheCall.OutArguments.size() <= 1);

  Value *ResultDestination = nullptr;
  Type *ResultType = nullptr;

  if (TheCall.OutArguments.size() != 0) {
    ResultDestination = Variables.getOrCreate(TheCall.OutArguments[0]);
    ResultType = ResultDestination->getType()->getPointerElementType();
  } else {
    ResultType = Builder.getVoidTy();
  }

  auto *CalleeType = FunctionType::get(ResultType,
                                       ArrayRef<Type *>(InArgsType),
                                       false);

  std::string HelperName = "helper_" + TheCall.helperName();
  Constant *FunctionDeclaration = TheModule.getOrInsertFunction(HelperName,
                                                                CalleeType);
  Value *Result = Builder.CreateCall(FunctionDeclaration, InArgs);

  if (TheCall.OutArguments.size() != 0)
    Builder.CreateStore(Result, ResultDestination);
}

void InstructionTranslator::translate(PTCInstruction *Instr) {
  const PTC::Instruction TheInstruction(Instr);

  auto LoadArgs = [this] (uint64_t TemporaryId) -> Value * {
    return Builder.CreateLoad(Variables.getOrCreate(TemporaryId));
  };

  auto ConstArgs = TheInstruction.ConstArguments;
  auto InArgs = TheInstruction.InArguments | LoadArgs;

  std::vector<Value *> Result = translateOpcode(TheInstruction.opcode(),
                                                ConstArgs.toVector(),
                                                InArgs.toVector());

  assert(Result.size() == (size_t) TheInstruction.OutArguments.size());
  // TODO: use ZipIterator here
  for (unsigned I = 0; I < Result.size(); I++)
    Builder.CreateStore(Result[I],
                        Variables.getOrCreate(TheInstruction.OutArguments[I]));
}

std::vector<Value *>
InstructionTranslator::translateOpcode(PTCOpcode Opcode,
                                       std::vector<uint64_t> ConstArguments,
                                       std::vector<Value *> InArguments) {
  LLVMContext& Context = TheModule.getContext();
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
    return { ConstantInt::get(RegisterType, ConstArguments[0]) };
  case PTC_INSTRUCTION_op_discard:
    // Let's overwrite the discarded temporary with a 0
    return { ConstantInt::get(RegisterType, 0) };
  case PTC_INSTRUCTION_op_mov_i32:
  case PTC_INSTRUCTION_op_mov_i64:
    return { Builder.CreateTrunc(InArguments[0], RegisterType) };
  case PTC_INSTRUCTION_op_setcond_i32:
  case PTC_INSTRUCTION_op_setcond_i64:
    {
      Value *Compare = CreateICmp(Builder,
                                  ConstArguments[0],
                                  InArguments[0],
                                  InArguments[1]);
      // TODO: convert single-bit registers to i1
      return { Builder.CreateZExt(Compare, RegisterType) };
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
      return { Select };
    }
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

      // // TODO: handle 64 on 32
      // // TODO: handle endianess mismatch
      // assert(SourceArchitecture.endianess() ==
      //        TargetArchitecture.endianess() &&
      //        "Different endianess between the source and the target is not "
      //        "supported yet");

      Value *Pointer = nullptr;
      if (Opcode == PTC_INSTRUCTION_op_qemu_ld_i32 ||
          Opcode == PTC_INSTRUCTION_op_qemu_ld_i64) {

        Pointer = Builder.CreateIntToPtr(InArguments[0],
                                         MemoryType->getPointerTo());
        Value *Load = Builder.CreateAlignedLoad(Pointer,
                                                AccessAlignment);

        if (SignExtend)
          return { Builder.CreateSExt(Load, RegisterType) };
        else
          return { Builder.CreateZExt(Load, RegisterType) };

      } else if (Opcode == PTC_INSTRUCTION_op_qemu_st_i32 ||
                 Opcode == PTC_INSTRUCTION_op_qemu_st_i64) {

        Pointer = Builder.CreateIntToPtr(InArguments[1],
                                         MemoryType->getPointerTo());
        Value *Value = Builder.CreateTrunc(InArguments[0], MemoryType);
        Builder.CreateAlignedStore(Value, Pointer, AccessAlignment);

        return { };
      } else
        llvm_unreachable("Unknown load type");
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
    // TODO: handle this
    return { ConstantInt::get(RegisterType, 0) };
  case PTC_INSTRUCTION_op_st8_i32:
  case PTC_INSTRUCTION_op_st16_i32:
  case PTC_INSTRUCTION_op_st_i32:
  case PTC_INSTRUCTION_op_st8_i64:
  case PTC_INSTRUCTION_op_st16_i64:
  case PTC_INSTRUCTION_op_st32_i64:
  case PTC_INSTRUCTION_op_st_i64:
    {
      Value *Base = dyn_cast<LoadInst>(InArguments[1])->getPointerOperand();
      assert(Base->getName() == "env");
      Value *Target = Variables.getByCPUStateOffset(ConstArguments[0]);
      Builder.CreateStore(InArguments[0], Target);
      return { };
    }
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
      return { Operation };
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
      return { Division, Remainder };
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

      return { Builder.CreateOr(FirstShift, SecondShift) };
    }
  case PTC_INSTRUCTION_op_deposit_i32:
  case PTC_INSTRUCTION_op_deposit_i64:
    {
      unsigned Position = ConstArguments[0];
      if (Position == RegisterSize)
        return { InArguments[0] };

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

      return { Result };
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

      switch (Opcode) {
      case PTC_INSTRUCTION_op_ext8s_i32:
      case PTC_INSTRUCTION_op_ext8s_i64:
      case PTC_INSTRUCTION_op_ext16s_i32:
      case PTC_INSTRUCTION_op_ext16s_i64:
      case PTC_INSTRUCTION_op_ext32s_i64:
        return { Builder.CreateSExt(Truncated, RegisterType) };
      case PTC_INSTRUCTION_op_ext8u_i32:
      case PTC_INSTRUCTION_op_ext8u_i64:
      case PTC_INSTRUCTION_op_ext16u_i32:
      case PTC_INSTRUCTION_op_ext16u_i64:
      case PTC_INSTRUCTION_op_ext32u_i64:
        return { Builder.CreateZExt(Truncated, RegisterType) };
      default:
        llvm_unreachable("Unexpected opcode");
      }
    }
  case PTC_INSTRUCTION_op_not_i32:
  case PTC_INSTRUCTION_op_not_i64:
    return { Builder.CreateXor(InArguments[0], getMaxValue(RegisterSize)) };
  case PTC_INSTRUCTION_op_neg_i32:
  case PTC_INSTRUCTION_op_neg_i64:
    return {
      Builder.CreateSub(ConstantInt::get(RegisterType, 0),
                        InArguments[0])
        };
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
      return { Result };
    }
  case PTC_INSTRUCTION_op_nand_i32:
  case PTC_INSTRUCTION_op_nand_i64:
    {
      Value *AndValue = Builder.CreateAnd(InArguments[0],
                                          InArguments[1]);
      Value *Result = Builder.CreateXor(AndValue,
                                        getMaxValue(RegisterSize));
      return { Result };
    }
  case PTC_INSTRUCTION_op_nor_i32:
  case PTC_INSTRUCTION_op_nor_i64:
    {
      Value *OrValue = Builder.CreateOr(InArguments[0],
                                        InArguments[1]);
      Value *Result = Builder.CreateXor(OrValue,
                                        getMaxValue(RegisterSize));
      return { Result };
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
      Function *BSwapFunction = Intrinsic::getDeclaration(&TheModule,
                                                          Intrinsic::bswap,
                                                          BSwapParameters);
      Value *Swapped = Builder.CreateCall(BSwapFunction, Truncated);

      return { Builder.CreateZExt(Swapped, RegisterType) };
    }
  case PTC_INSTRUCTION_op_set_label:
    {
      unsigned LabelId = ptc.get_arg_label_id(ConstArguments[0]);
      std::string Label = "L" + std::to_string(LabelId);

      BasicBlock *Fallthrough = nullptr;
      auto ExistingBasicBlock = LabeledBasicBlocks.find(Label);

      if (ExistingBasicBlock == LabeledBasicBlocks.end()) {
        Fallthrough = BasicBlock::Create(Context, Label, TheFunction);
        LabeledBasicBlocks[Label] = Fallthrough;
      } else {
        // A basic block with that label already exist
        Fallthrough = LabeledBasicBlocks[Label];

        // Ensure it's empty
        assert(Fallthrough->begin() == Fallthrough->end());

        // Move it to the bottom
        Fallthrough->removeFromParent();
        TheFunction->getBasicBlockList().push_back(Fallthrough);
      }

      Builder.CreateBr(Fallthrough);

      Blocks.push_back(Fallthrough);
      Builder.SetInsertPoint(Fallthrough);
      Variables.newBasicBlock();

      return { };
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
                                                   TheFunction);

      // Look for a matching label
      BasicBlock *Target = nullptr;
      auto ExistingBasicBlock = LabeledBasicBlocks.find(Label);

      // No matching label, create a temporary block
      if (ExistingBasicBlock == LabeledBasicBlocks.end()) {
        Target = BasicBlock::Create(Context,
                                    Label,
                                    TheFunction);
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

      return { };
    }
  case PTC_INSTRUCTION_op_call:
    // TODO: implement call to helpers
    llvm_unreachable("Call to helpers not implemented");
  case PTC_INSTRUCTION_op_exit_tb:
  case PTC_INSTRUCTION_op_goto_tb:
    // Nothing to do here
    return { };
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

      FirstOperandLow = Builder.CreateSExt(InArguments[0], DestinationType);
      FirstOperandHigh = Builder.CreateSExt(InArguments[1], DestinationType);
      SecondOperandLow = Builder.CreateSExt(InArguments[2], DestinationType);
      SecondOperandHigh = Builder.CreateSExt(InArguments[3],
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

      return { ResultLow, ResultHigh };
    }
  case PTC_INSTRUCTION_op_mulu2_i32:
  case PTC_INSTRUCTION_op_mulu2_i64:
  case PTC_INSTRUCTION_op_muls2_i32:
  case PTC_INSTRUCTION_op_muls2_i64:
    {
      IntegerType *DestinationType = Builder.getIntNTy(RegisterSize * 2);

      Value *FirstOperand = nullptr;
      Value *SecondOperand = nullptr;

      if (Opcode == PTC_INSTRUCTION_op_muls2_i32
          || Opcode == PTC_INSTRUCTION_op_muls2_i64) {
        FirstOperand = Builder.CreateZExt(InArguments[0], DestinationType);
        SecondOperand = Builder.CreateZExt(InArguments[1], DestinationType);
      } else if (Opcode == PTC_INSTRUCTION_op_muls2_i32
                 || Opcode == PTC_INSTRUCTION_op_muls2_i64) {
        FirstOperand = Builder.CreateSExt(InArguments[0], DestinationType);
        SecondOperand = Builder.CreateSExt(InArguments[1], DestinationType);
      } else
        llvm_unreachable("Unexpected opcode");

      Value *Result = Builder.CreateMul(FirstOperand, SecondOperand);

      Value *ResultLow = Builder.CreateTrunc(Result, RegisterType);
      Value *ShiftedResult = Builder.CreateLShr(Result, RegisterSize);
      Value *ResultHigh = Builder.CreateTrunc(ShiftedResult,
                                              RegisterType);

      return { ResultLow, ResultHigh };
    }
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
}

void CodeGenerator::translate(size_t LoadAddress,
                              ArrayRef<uint8_t> Code,
                              size_t VirtualAddress,
                              std::string Name) {
  const uint8_t *CodePointer = Code.data();
  const uint8_t *CodeEnd = CodePointer + Code.size();

  IRBuilder<> Builder(Context);

  // Create main function
  auto *MainType  = FunctionType::get(Builder.getVoidTy(), false);
  auto *MainFunction = Function::Create(MainType,
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

  // Instantiate helpers
  VariableManager Variables(*TheModule,
                            CPUStateType,
                            HelpersModuleLayout);

  GlobalVariable *PCReg = Variables.getByCPUStateOffset(ptc.get_pc(), "pc");

  JumpTargetManager JumpTargets(*TheModule, PCReg, MainFunction);
  std::map<std::string, BasicBlock *> LabeledBasicBlocks;
  std::vector<BasicBlock *> Blocks;

  InstructionTranslator Translator(Builder,
                                   Variables,
                                   JumpTargets,
                                   LabeledBasicBlocks,
                                   Blocks,
                                   *TheModule,
                                   MainFunction,
                                   SourceArchitecture,
                                   TargetArchitecture);

  ptc.mmap(LoadAddress, Code.data(), Code.size());

  while (Entry != nullptr) {
    Builder.SetInsertPoint(Entry);

    LabeledBasicBlocks.clear();

    // TODO: rename this type
    PTCInstructionListPtr InstructionList(new PTCInstructionList);
    size_t ConsumedSize = 0;

    assert(CodeEnd > CodePointer);

    ConsumedSize = ptc.translate(VirtualAddress,
                                 InstructionList.get());

    dumpTranslation(std::cerr, InstructionList.get());

    Variables.newFunction(Delimiter, InstructionList.get());
    unsigned j = 0;

    // Skip everything is before the first PTC_INSTRUCTION_op_debug_insn_start
    while (j < InstructionList->instruction_count &&
           InstructionList->instructions[j].opc !=
           PTC_INSTRUCTION_op_debug_insn_start) {
      j++;
    }

    assert(j < InstructionList->instruction_count);

    MDNode* MDOriginalInstr = nullptr;
    bool StopTranslation = false;

    // Handle the first PTC_INSTRUCTION_op_debug_insn_start
    {
      PTCInstruction *Instruction = &InstructionList->instructions[j];
      auto Result = Translator.newInstruction(Instruction, true);
      std::tie(StopTranslation, MDOriginalInstr) = Result;
      j++;
    }

    for (; j < InstructionList->instruction_count && !StopTranslation; j++) {
      PTCInstruction Instruction = InstructionList->instructions[j];
      PTCOpcode Opcode = Instruction.opc;

      Blocks.clear();
      Blocks.push_back(Builder.GetInsertBlock());

      switch(Opcode) {
      case PTC_INSTRUCTION_op_discard:
        // Instructions we don't even consider
        break;
      case PTC_INSTRUCTION_op_debug_insn_start:
        {
          std::tie(StopTranslation,
                   MDOriginalInstr) = Translator.newInstruction(&Instruction,
                                                                false);
          break;
        }
      case PTC_INSTRUCTION_op_call:
        Translator.translateCall(&Instruction);
        break;
      default:
        Translator.translate(&Instruction);
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

    Translator.closeLastInstruction(VirtualAddress + ConsumedSize);

    // Before looking for writes to the PC, give a shot of SROA
    legacy::PassManager PM;
    PM.add(createSROAPass());
    PM.add(Translator.createTranslateDirectBranchesPass());
    PM.run(*TheModule);

    // Obtain a new program counter to translate
    uint64_t NewPC = 0;
    std::tie(NewPC, Entry) = JumpTargets.peekJumpTarget();
    VirtualAddress = NewPC;
    CodePointer = Code.data() + (NewPC - LoadAddress);
  } // End translations loop

  Delimiter->eraseFromParent();

  JumpTargets.translateIndirectJumps();

  Translator.removeNewPCMarkers();

  Debug->generateDebugInfo();

}

void CodeGenerator::serialize() {
  // Ask the debug handler if it already has a good copy of the IR, if not dump
  // it
  if (!Debug->copySource()) {
    std::ofstream Output(OutputPath);
    Debug->print(Output, false);
  }
}
