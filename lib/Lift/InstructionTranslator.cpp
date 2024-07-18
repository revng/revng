/// \file InstructionTranslator.cpp
/// \brief This file implements the logic to translate a libtcg instruction in
///        to LLVM IR.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <fstream>
#include <queue>
#include <set>
#include <sstream>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include "revng/Lift/Lift.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/RandomAccessIterator.h"
#include "revng/Support/Range.h"

#include "InstructionTranslator.h"
#include "qemu/libtcg/libtcg.h"
#include "VariableManager.h"

using namespace llvm;

static cl::opt<bool> RecordASM("record-asm",
                               cl::desc("create metadata for assembly"),
                               cl::cat(MainCategory));

using IT = InstructionTranslator;

static uint64_t pc(LibTcgInstruction *Instr) {
  revng_assert(Instr->opcode == LIBTCG_op_insn_start);
  uint64_t PC = Instr->constant_args[0].constant;
  if (Instr->nb_cargs > 1)
    PC |= Instr->constant_args[1].constant << 32;
  return PC;
}

/// Converts a libtcg condition into an LLVM predicate
///
/// \param Condition the input libtcg condition.
///
/// \return the corresponding LLVM predicate.

static CmpInst::Predicate conditionToPredicate(LibTcgCond Condition) {
  switch (Condition) {
  case LIBTCG_COND_NEVER:
    // TODO(anjo): Double check this
    // TODO: this is probably wrong
    return CmpInst::FCMP_FALSE;
  case LIBTCG_COND_ALWAYS:
    // TODO(anjo): Double check this
    // TODO: this is probably wrong
    return CmpInst::FCMP_TRUE;
  case LIBTCG_COND_EQ:
    return CmpInst::ICMP_EQ;
  case LIBTCG_COND_NE:
    return CmpInst::ICMP_NE;
  case LIBTCG_COND_LT:
    return CmpInst::ICMP_SLT;
  case LIBTCG_COND_GE:
    return CmpInst::ICMP_SGE;
  case LIBTCG_COND_LE:
    return CmpInst::ICMP_SLE;
  case LIBTCG_COND_GT:
    return CmpInst::ICMP_SGT;
  case LIBTCG_COND_LTU:
    return CmpInst::ICMP_ULT;
  case LIBTCG_COND_GEU:
    return CmpInst::ICMP_UGE;
  case LIBTCG_COND_LEU:
    return CmpInst::ICMP_ULE;
  case LIBTCG_COND_GTU:
    return CmpInst::ICMP_UGT;
  default:
    revng_unreachable("Unknown libtcg condition");
  }
}

/// Obtains the LLVM binary operation corresponding to the specified libtcg
/// opcode.
///
/// \param Opcode the libtcg opcode.
///
/// \return the LLVM binary operation matching opcode.
static Instruction::BinaryOps opcodeToBinaryOp(LibTcgOpcode Opcode) {
  switch (Opcode) {
  case LIBTCG_op_add_i32:
  case LIBTCG_op_add_i64:
  case LIBTCG_op_add2_i32:
  case LIBTCG_op_add2_i64:
    return Instruction::Add;
  case LIBTCG_op_sub_i32:
  case LIBTCG_op_sub_i64:
  case LIBTCG_op_sub2_i32:
  case LIBTCG_op_sub2_i64:
    return Instruction::Sub;
  case LIBTCG_op_mul_i32:
  case LIBTCG_op_mul_i64:
    return Instruction::Mul;
  case LIBTCG_op_div_i32:
  case LIBTCG_op_div_i64:
    return Instruction::SDiv;
  case LIBTCG_op_divu_i32:
  case LIBTCG_op_divu_i64:
    return Instruction::UDiv;
  case LIBTCG_op_rem_i32:
  case LIBTCG_op_rem_i64:
    return Instruction::SRem;
  case LIBTCG_op_remu_i32:
  case LIBTCG_op_remu_i64:
    return Instruction::URem;
  case LIBTCG_op_and_i32:
  case LIBTCG_op_and_i64:
    return Instruction::And;
  case LIBTCG_op_or_i32:
  case LIBTCG_op_or_i64:
    return Instruction::Or;
  case LIBTCG_op_xor_i32:
  case LIBTCG_op_xor_i64:
    return Instruction::Xor;
  case LIBTCG_op_shl_i32:
  case LIBTCG_op_shl_i64:
    return Instruction::Shl;
  case LIBTCG_op_shr_i32:
  case LIBTCG_op_shr_i64:
    return Instruction::LShr;
  case LIBTCG_op_sar_i32:
  case LIBTCG_op_sar_i64:
    return Instruction::AShr;
  default:
    revng_unreachable("libtcg opcode is not a binary operator");
  }
}

/// Returns the maximum value which can be represented with the specified number
/// of bits.
// TODO(anjo): This can be done better
static uint64_t getMaxValue(unsigned Bits) {
  if (Bits == 32)
    return 0xffffffff;
  else if (Bits == 64)
    return 0xffffffffffffffff;
  else
    revng_unreachable("Not the number of bits in an integer type");
}

/// Maps an opcode the corresponding input and output register size.
///
/// \return the size, in bits, of the registers used by the opcode.
static unsigned getRegisterSize(const LibTcgInterface &LibTcg, LibTcgOpcode Opcode) {
  switch (Opcode) {
  case LIBTCG_op_add2_i32:
  case LIBTCG_op_add_i32:
  case LIBTCG_op_andc_i32:
  case LIBTCG_op_and_i32:
  case LIBTCG_op_brcond2_i32:
  case LIBTCG_op_brcond_i32:
  case LIBTCG_op_bswap16_i32:
  case LIBTCG_op_bswap32_i32:
  case LIBTCG_op_deposit_i32:
  case LIBTCG_op_div2_i32:
  case LIBTCG_op_div_i32:
  case LIBTCG_op_divu2_i32:
  case LIBTCG_op_divu_i32:
  case LIBTCG_op_eqv_i32:
  case LIBTCG_op_ext16s_i32:
  case LIBTCG_op_ext16u_i32:
  case LIBTCG_op_ext8s_i32:
  case LIBTCG_op_ext8u_i32:
  case LIBTCG_op_extrl_i64_i32:
  case LIBTCG_op_extrh_i64_i32:
  case LIBTCG_op_ld16s_i32:
  case LIBTCG_op_ld16u_i32:
  case LIBTCG_op_ld8s_i32:
  case LIBTCG_op_ld8u_i32:
  case LIBTCG_op_ld_i32:
  case LIBTCG_op_movcond_i32:
  case LIBTCG_op_mov_i32:
  case LIBTCG_op_mul_i32:
  case LIBTCG_op_muls2_i32:
  case LIBTCG_op_mulsh_i32:
  case LIBTCG_op_mulu2_i32:
  case LIBTCG_op_muluh_i32:
  case LIBTCG_op_nand_i32:
  case LIBTCG_op_neg_i32:
  case LIBTCG_op_nor_i32:
  case LIBTCG_op_not_i32:
  case LIBTCG_op_orc_i32:
  case LIBTCG_op_or_i32:
  case LIBTCG_op_qemu_ld_a32_i32:
  case LIBTCG_op_qemu_ld_a64_i32:
  case LIBTCG_op_qemu_st_a32_i32:
  case LIBTCG_op_qemu_st_a64_i32:
  case LIBTCG_op_rem_i32:
  case LIBTCG_op_remu_i32:
  case LIBTCG_op_rotl_i32:
  case LIBTCG_op_rotr_i32:
  case LIBTCG_op_sar_i32:
  case LIBTCG_op_setcond2_i32:
  case LIBTCG_op_setcond_i32:
  case LIBTCG_op_shl_i32:
  case LIBTCG_op_shr_i32:
  case LIBTCG_op_st16_i32:
  case LIBTCG_op_st8_i32:
  case LIBTCG_op_st_i32:
  case LIBTCG_op_sub2_i32:
  case LIBTCG_op_sub_i32:
  case LIBTCG_op_xor_i32:
  case LIBTCG_op_extract_i32:
  case LIBTCG_op_sextract_i32:
  case LIBTCG_op_extract2_i32:
  case LIBTCG_op_clz_i32:
    return 32;
  case LIBTCG_op_add2_i64:
  case LIBTCG_op_add_i64:
  case LIBTCG_op_andc_i64:
  case LIBTCG_op_and_i64:
  case LIBTCG_op_brcond_i64:
  case LIBTCG_op_bswap16_i64:
  case LIBTCG_op_bswap32_i64:
  case LIBTCG_op_bswap64_i64:
  case LIBTCG_op_deposit_i64:
  case LIBTCG_op_div2_i64:
  case LIBTCG_op_div_i64:
  case LIBTCG_op_divu2_i64:
  case LIBTCG_op_divu_i64:
  case LIBTCG_op_eqv_i64:
  case LIBTCG_op_ext16s_i64:
  case LIBTCG_op_ext16u_i64:
  case LIBTCG_op_ext_i32_i64:
  case LIBTCG_op_extu_i32_i64:
  case LIBTCG_op_ext32s_i64:
  case LIBTCG_op_ext32u_i64:
  case LIBTCG_op_ext8s_i64:
  case LIBTCG_op_ext8u_i64:
  case LIBTCG_op_ld16s_i64:
  case LIBTCG_op_ld16u_i64:
  case LIBTCG_op_ld32s_i64:
  case LIBTCG_op_ld32u_i64:
  case LIBTCG_op_ld8s_i64:
  case LIBTCG_op_ld8u_i64:
  case LIBTCG_op_ld_i64:
  case LIBTCG_op_movcond_i64:
  case LIBTCG_op_mov_i64:
  case LIBTCG_op_mul_i64:
  case LIBTCG_op_muls2_i64:
  case LIBTCG_op_mulsh_i64:
  case LIBTCG_op_mulu2_i64:
  case LIBTCG_op_muluh_i64:
  case LIBTCG_op_nand_i64:
  case LIBTCG_op_neg_i64:
  case LIBTCG_op_nor_i64:
  case LIBTCG_op_not_i64:
  case LIBTCG_op_orc_i64:
  case LIBTCG_op_or_i64:
  case LIBTCG_op_qemu_ld_a32_i64:
  case LIBTCG_op_qemu_ld_a64_i64:
  case LIBTCG_op_qemu_st_a32_i64:
  case LIBTCG_op_qemu_st_a64_i64:
  case LIBTCG_op_rem_i64:
  case LIBTCG_op_remu_i64:
  case LIBTCG_op_rotl_i64:
  case LIBTCG_op_rotr_i64:
  case LIBTCG_op_sar_i64:
  case LIBTCG_op_setcond_i64:
  case LIBTCG_op_shl_i64:
  case LIBTCG_op_shr_i64:
  case LIBTCG_op_st16_i64:
  case LIBTCG_op_st32_i64:
  case LIBTCG_op_st8_i64:
  case LIBTCG_op_st_i64:
  case LIBTCG_op_sub2_i64:
  case LIBTCG_op_sub_i64:
  case LIBTCG_op_xor_i64:
  case LIBTCG_op_clz_i64:
  case LIBTCG_op_extract2_i64:
    return 64;
  case LIBTCG_op_br:
  case LIBTCG_op_call:
  case LIBTCG_op_insn_start:
  case LIBTCG_op_discard:
  case LIBTCG_op_exit_tb:
  case LIBTCG_op_goto_tb:
  case LIBTCG_op_goto_ptr:
  case LIBTCG_op_set_label:
    return 0;
  default: {
    // For debugging purposes printing the actual opcode
    // really helps.
    std::stringstream ErrSS;
    ErrSS << "Unexpected libtcg opcode ["
          << Opcode
          << "]: "
          << LibTcg.get_instruction_name(Opcode);
    revng_unreachable(ErrSS.str().c_str());
  }
  }
}

static Value *genDeposit(IRBuilder<> &Builder,
                         unsigned RegisterSize,
                         Value *Into, Value *From,
                         Value *Offset, Value *Length) {
    revng_assert(isa<ConstantInt>(Offset));
    uint64_t ConstOffset = cast<ConstantInt>(Offset)->getLimitedValue();
    if (ConstOffset == RegisterSize)
      return Into;

    revng_assert(isa<ConstantInt>(Length));
    uint64_t ConstLength = cast<ConstantInt>(Length)->getLimitedValue();

    uint64_t Bits = 0;
    // Thou shall not << 32
    if (ConstLength == RegisterSize)
      Bits = getMaxValue(RegisterSize);
    else
      Bits = (1 << ConstLength) - 1;

    // result = (t1 & ~(bits << position)) | ((t2 & bits) << position)
    uint64_t BaseMask = ~(Bits << ConstOffset);
    Value *MaskedBase = Builder.CreateAnd(Into, BaseMask);
    Value *Deposit = Builder.CreateAnd(From, Bits);
    Value *ShiftedDeposit = Builder.CreateShl(Deposit, ConstOffset);
    Value *Result = Builder.CreateOr(MaskedBase, ShiftedDeposit);

    return Result;
}


/// Create a compare instruction given a comparison operator and the operands
///
/// \param Builder the builder to use to create the instruction.
/// \param Condition the libtcg condition.
/// \param FirstOperand the first operand of the comparison.
/// \param SecondOperand the second operand of the comparison.
///
/// \return a compare instruction.
template<typename T>
static Value *createICmp(T &Builder,
                         LibTcgCond Condition,
                         Value *FirstOperand,
                         Value *SecondOperand) {
  return Builder.CreateICmp(conditionToPredicate(Condition),
                            FirstOperand,
                            SecondOperand);
}

using LBM = IT::LabeledBlocksMap;
IT::InstructionTranslator(const LibTcgInterface &LibTcg,
                          IRBuilder<> &Builder,
                          VariableManager &Variables,
                          JumpTargetManager &JumpTargets,
                          std::vector<BasicBlock *> Blocks,
                          bool EndianessMismatch,
                          ProgramCounterHandler *PCH) :
  LibTcg(LibTcg),
  Builder(Builder),
  Variables(Variables),
  JumpTargets(JumpTargets),
  Blocks(Blocks),
  TheModule(*Builder.GetInsertBlock()->getParent()->getParent()),
  TheFunction(Builder.GetInsertBlock()->getParent()),
  EndianessMismatch(EndianessMismatch),
  NewPCMarker(nullptr),
  LastPC(MetaAddress::invalid()),
  PCH(PCH) {

  auto &Context = TheModule.getContext();
  using FT = FunctionType;
  // The newpc function call takes the following parameters:
  //
  // * BasicBlockID of the instruction in string form
  // * instruction size
  // * isJT (-1: unknown, 0: no, 1: yes)
  // * inlining index
  // * pointer to the disassembled instruction
  // * all the local variables used by this instruction
  auto *NewPCMarkerTy = FT::get(Type::getVoidTy(Context),
                                { Type::getInt8PtrTy(Context),
                                  Type::getInt64Ty(Context),
                                  Type::getInt32Ty(Context),
                                  Type::getInt32Ty(Context),
                                  Type::getInt8PtrTy(Context),
                                  Type::getInt8PtrTy(Context) },
                                true);
  NewPCMarker = Function::Create(NewPCMarkerTy,
                                 GlobalValue::ExternalLinkage,
                                 "newpc",
                                 &TheModule);
  FunctionTags::Marker.addTo(NewPCMarker);
  NewPCMarker->addFnAttr(Attribute::WillReturn);
  NewPCMarker->addFnAttr(Attribute::NoUnwind);
  NewPCMarker->addFnAttr(Attribute::NoMerge);
}

void IT::finalizeNewPCMarkers() {
  size_t FixedArgCount = NewPCMarker->arg_size();

  llvm::SmallVector<CallInst *, 4> CallsToRemove;

  for (User *U : NewPCMarker->users()) {
    auto *Call = cast<CallInst>(U);

    // Report the instruction on the coverage CSV
    using namespace NewPCArguments;
    MetaAddress PC = addressFromNewPC(Call);
    uint64_t Size = getLimitedValue(Call->getArgOperand(InstructionSize));
    bool IsJT = JumpTargets.isJumpTarget(PC);

    // We already finished discovering new code to translate, so we can remove
    // the references to local variables as argument of the calls to newpc and
    // create room for more optimizations.
    if (Call->arg_size() != FixedArgCount) {
      SmallVector<Value *, 8> Args;
      auto *AI = Call->arg_begin();
      for (size_t Idx = 0; Idx < FixedArgCount; ++Idx, ++AI)
        Args.emplace_back(*AI);

      auto *NewCall = CallInst::Create(NewPCMarker, Args, "", Call);
      NewCall->setCallingConv(Call->getCallingConv());
      NewCall->setDebugLoc(Call->getDebugLoc());
      NewCall->copyMetadata(*Call);
      // Note: we intentionally do not copy attributes. We do not expect to have
      //       any and removing those on extra arguments leads to a mysterious
      //       failure in verify "Attribute after last parameter".

      revng_assert(Call->use_empty());
      CallsToRemove.push_back(Call);
    }
  }

  for (auto *Call : CallsToRemove)
    eraseFromParent(Call);
}

SmallSet<unsigned, 1> IT::preprocess(LibTcgInstructionList InstructionList) {
  SmallSet<unsigned, 1> Result;

  for (unsigned I = 0; I < InstructionList.instruction_count; ++I) {
    LibTcgInstruction &Instruction = InstructionList.list[I];
    switch (Instruction.opcode) {
    case LIBTCG_op_mov_i32:
    case LIBTCG_op_mov_i64:
      break;
    default:
      continue;
    }

    LibTcgArgument Argument = Instruction.output_args[0];
    revng_assert(Argument.kind == LIBTCG_ARG_TEMP);
    LibTcgTemp *Temp = Argument.temp;

    if (Temp->kind != LIBTCG_TEMP_GLOBAL)
      continue;

    if (strcmp("btarget", Temp->name) != 0)
      continue;

    for (unsigned J = I + 1; J < InstructionList.instruction_count; ++J) {
      LibTcgOpcode Opcode = InstructionList.list[J].opcode;
      if (Opcode == LIBTCG_op_insn_start)
        Result.insert(J);
    }

    break;
  }

  return Result;
}

CallInst *IT::emitNewPCCall(IRBuilder<> &Builder,
                            MetaAddress PC,
                            uint64_t Size,
                            Value *String) const {
  PointerType *Int8PtrTy = getStringPtrType(TheModule.getContext());
  auto *Int8NullPtr = ConstantPointerNull::get(Int8PtrTy);
  std::vector<Value *> Args = { BasicBlockID(PC).toValue(&TheModule),
                                Builder.getInt64(Size),
                                Builder.getInt32(-1),
                                Builder.getInt32(0),
                                String != nullptr ? String : Int8NullPtr,
                                Int8NullPtr };

  // Insert a call to NewPCMarker capturing all the local temporaries
  // This prevents SROA from transforming them in SSA values, which is bad
  // in case we have to split a basic block
  for (AllocaInst *Local : Variables.locals())
    Args.push_back(Local);

  return Builder.CreateCall(NewPCMarker, Args);
}

std::tuple<IT::TranslationResult, MDNode *, MetaAddress, MetaAddress>
IT::newInstruction(LibTcgInstruction *Instr,
                   LibTcgInstruction *Next,
                   MetaAddress StartPC,
                   MetaAddress EndPC,
                   bool IsFirst) {
  using R = std::tuple<TranslationResult, MDNode *, MetaAddress, MetaAddress>;
  revng_assert(Instr != nullptr);

  LLVMContext &Context = TheModule.getContext();

  // A new original instruction, let's create a new metadata node
  // referencing it for all the next instructions to come
  MetaAddress PC = StartPC.replaceAddress(pc(Instr));

  // Prevent translation of non-executable code
  if (not JumpTargets.isExecutableAddress(PC))
    return R{ Abort, nullptr, MetaAddress::invalid(), MetaAddress::invalid() };

  // Compute NextPC
  MetaAddress NextPC = MetaAddress::invalid();
  if (Next != nullptr)
    NextPC = StartPC.replaceAddress(pc(Next));
  else
    NextPC = EndPC;

  PointerType *Int8PtrTy = getStringPtrType(Context);

  MDNode *MDOriginalInstr = nullptr;
  Constant *String = nullptr;
  if (RecordASM) {
    std::stringstream OriginalStringStream;
    revng_assert(NextPC - PC);
    // TODO(anjo): make up to date
    //disassemble(OriginalStringStream, PC, *(NextPC - PC));
    std::string OriginalString = OriginalStringStream.str();

    // We don't deduplicate this string since performing a lookup each time is
    // increasingly expensive and we should have relatively few collisions
    std::string AddressName = JumpTargets.nameForAddress(PC);
    String = getUniqueString(&TheModule, OriginalString);

    auto *MDOriginalString = ConstantAsMetadata::get(String);
    auto *MDPC = ConstantAsMetadata::get(PC.toValue(&TheModule));
    MDOriginalInstr = MDNode::get(Context, { MDOriginalString, MDPC });
  } else {
    String = ConstantPointerNull::get(Int8PtrTy);
  }

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
      } else {
        // The block contains already translated code, early exit
        return R{ Stop, MDOriginalInstr, PC, NextPC };
      }
    }
  }

  Variables.newBasicBlock();

  revng_assert(NextPC - PC);
  auto *Call = emitNewPCCall(Builder, PC, *(NextPC - PC), String);

  if (!IsFirst) {
    // Inform the JumpTargetManager about the new PC we met
    BasicBlock::iterator CurrentIt = Builder.GetInsertPoint();
    if (CurrentIt == Builder.GetInsertBlock()->begin())
      revng_assert(JumpTargets.getBlockAt(PC) == Builder.GetInsertBlock());
    else
      JumpTargets.registerInstruction(PC, Call);
  }

  return R{ Success, MDOriginalInstr, PC, NextPC };
}

IT::TranslationResult IT::translateCall(LibTcgInstruction *Instr) {
  std::vector<Value *> InArgs;

  for (uint8_t I = 0; I < Instr->nb_iargs; ++I) {
    auto *Load = Variables.load(Builder, &Instr->input_args[I]);
    if (Load == nullptr)
      return Abort;
    InArgs.push_back(Load);
  }

  const auto GetValueType = [](Value *Argument) { return Argument->getType(); };
  auto ValueTypes = llvm::map_range(InArgs, GetValueType);
  std::vector<Type *> InArgsType(ValueTypes.begin(), ValueTypes.end());

  // TODO: handle multiple return arguments
  revng_assert(Instr->nb_oargs <= 1);

  Value *ResultDestination = nullptr;
  Type *ResultType = nullptr;

  if (Instr->nb_oargs != 0) {
    ResultDestination = Variables.getOrCreate(&Instr->output_args[0]);
    if (ResultDestination == nullptr)
      return Abort;
    ResultType = getVariableType(ResultDestination);
  } else {
    ResultType = Builder.getVoidTy();
  }

  auto *CalleeType = FunctionType::get(ResultType,
                                       ArrayRef<Type *>(InArgsType),
                                       false);

  LibTcgHelperInfo Info = LibTcg.get_helper_info(Instr);
  std::string HelperName = "helper_" + std::string(Info.func_name);
  FunctionCallee FDecl = TheModule.getOrInsertFunction(HelperName, CalleeType);

  FunctionTags::Helper.addTo(cast<Function>(skipCasts(FDecl.getCallee())));

  CallInst *Result = Builder.CreateCall(FDecl, InArgs);

  if (Instr->nb_oargs != 0) {
    Builder.CreateStore(Result, ResultDestination);
  }

  if (Info.func_flags & LIBTCG_CALL_NO_RETURN) {
    handleExitTB();
  }

  return Success;
}

IT::TranslationResult
IT::translate(LibTcgInstruction *Instr, MetaAddress PC, MetaAddress NextPC) {
  std::vector<Value *> InArgs;
  for (unsigned I = 0; I < Instr->nb_iargs; ++I) {
    auto *Load = Variables.load(Builder, &Instr->input_args[I]);
    if (Load == nullptr)
      return Abort;
    InArgs.push_back(Load);
  }

  // TODO(anjo): Remove this, constant args to eg add are not passed as const but const temps
  std::vector<LibTcgArgument> ConstArgs;
  {
    unsigned RegisterSize = getRegisterSize(LibTcg, Instr->opcode);
    Type *RegisterType = nullptr;
    if (RegisterSize == 32)
      RegisterType = Builder.getInt32Ty();
    else if (RegisterSize == 64 or RegisterSize == 0)
      RegisterType = Builder.getInt64Ty();
    else if (RegisterSize != 0)
      revng_unreachable("Unexpected register size");

    for (unsigned I = 0; I < Instr->nb_cargs; ++I) {
      if (Instr->constant_args[I].kind == LIBTCG_ARG_CONSTANT) {
        InArgs.push_back(ConstantInt::get(RegisterType, Instr->constant_args[I].constant));
      } else {
        ConstArgs.push_back(Instr->constant_args[I]);
      }
    }
  }

  //errs() << "TRANSLATING: " << LibTcg.get_instruction_name(Instr->opcode) << "\n";

  LastPC = PC;
  auto Result = translateOpcode(Instr->opcode,
                                ConstArgs,
                                InArgs);

  // Check if there was an error while translating the instruction
  if (!Result)
    return Abort;

  revng_assert(Result->size() == Instr->nb_oargs);

  // TODO: use ZipIterator here
  for (unsigned I = 0; I < Result->size(); I++) {
    auto *Destination = Variables.getOrCreate(&Instr->output_args[I]);
    if (Destination == nullptr)
      return Abort;

    auto *Store = Builder.CreateStore(Result.get()[I], Destination);

    if (PCH->affectsPC(Store)) {
      // This is a PC-related store
      PCH->handleStore(Builder, Store);
    } else {
      // If we're writing somewhere an immediate, register it for exploration
      if (auto *Constant = dyn_cast<ConstantInt>(Store->getValueOperand())) {
        MetaAddress Address = JumpTargets.fromPC(Constant->getLimitedValue());
        if (Address.isValid() and PC != Address and JumpTargets.isPC(Address)
            and not JumpTargets.hasJT(Address)) {
          JumpTargets.registerSimpleLiteral(Address);
        }
      }
    }
  }

  return Success;
}

void IT::registerDirectJumps() {

  for (BasicBlock *ExitBB : ExitBlocks) {
    auto [Result, NextPC] = PCH->getUniqueJumpTarget(ExitBB);
    if (Result == NextJumpTarget::Unique and JumpTargets.isPC(NextPC)
        and not JumpTargets.hasJT(NextPC)) {
      JumpTargets.registerJT(NextPC, JTReason::DirectJump);
    }
  }

  ExitBlocks.clear();
}

ErrorOr<std::vector<Value *>>
IT::translateOpcode(LibTcgOpcode Opcode,
                    std::vector<LibTcgArgument> ConstArguments,
                    std::vector<Value *> InArguments) {
  LLVMContext &Context = TheModule.getContext();
  unsigned RegisterSize = getRegisterSize(LibTcg, Opcode);
  Type *RegisterType = nullptr;
  if (RegisterSize == 32)
    RegisterType = Builder.getInt32Ty();
  else if (RegisterSize == 64)
    RegisterType = Builder.getInt64Ty();
  else if (RegisterSize != 0)
    revng_unreachable("Unexpected register size");

  using v = std::vector<Value *>;
  switch (Opcode) {
  // TODO(anjo): movi are now just normal mov
  //case LIBTCG_op_movi_i32:
  //case LIBTCG_op_movi_i64:
  //  return v{ ConstantInt::get(RegisterType, ConstArguments[0]) };
  case LIBTCG_op_discard:
    // Let's overwrite the discarded temporary with a 0
    return v{ ConstantInt::get(RegisterType, 0) };
  case LIBTCG_op_mov_i32:
  case LIBTCG_op_mov_i64:
    if (auto Constant = dyn_cast<ConstantInt>(InArguments[0])) {
      return v{ Constant };
    } else {
      return v{ Builder.CreateTrunc(InArguments[0], RegisterType) };
    }
  case LIBTCG_op_setcond_i32:
  case LIBTCG_op_setcond_i64: {
    revng_assert(ConstArguments[0].kind == LIBTCG_ARG_COND);
    Value *Compare = createICmp(Builder,
                                ConstArguments[0].cond,
                                InArguments[0],
                                InArguments[1]);
    // TODO: convert single-bit registers to i1
    return v{ Builder.CreateZExt(Compare, RegisterType) };
  }
  case LIBTCG_op_movcond_i32: // Resist the fallthrough temptation
  case LIBTCG_op_movcond_i64: {
    revng_assert(ConstArguments[0].kind == LIBTCG_ARG_COND);
    Value *Compare = createICmp(Builder,
                                ConstArguments[0].cond,
                                InArguments[0],
                                InArguments[1]);
    Value *Select = Builder.CreateSelect(Compare,
                                         InArguments[2],
                                         InArguments[3]);
    return v{ Select };
  }
  case LIBTCG_op_qemu_ld_a32_i32:
  case LIBTCG_op_qemu_ld_a64_i32:
  case LIBTCG_op_qemu_ld_a32_i64:
  case LIBTCG_op_qemu_ld_a64_i64:
  case LIBTCG_op_qemu_st_a32_i32:
  case LIBTCG_op_qemu_st_a64_i32:
  case LIBTCG_op_qemu_st_a32_i64:
  case LIBTCG_op_qemu_st_a64_i64: {
    revng_assert(ConstArguments[0].kind == LIBTCG_ARG_MEM_OP_INDEX);
    LibTcgMemOp MemoryOp = ConstArguments[0].mem_op_index.op;

    unsigned Alignment = 1;

    // Load size
    IntegerType *MemoryType = nullptr;
    auto MemoryOpSize = static_cast<LibTcgMemOp>(MemoryOp & LIBTCG_MO_SIZE);
    switch (MemoryOpSize) {
    case LIBTCG_MO_8:
      MemoryType = Builder.getInt8Ty();
      break;
    case LIBTCG_MO_16:
      MemoryType = Builder.getInt16Ty();
      break;
    case LIBTCG_MO_32:
      MemoryType = Builder.getInt32Ty();
      break;
    case LIBTCG_MO_64:
      MemoryType = Builder.getInt64Ty();
      break;
    default:
      revng_unreachable("Unexpected load size");
    }

    // If necessary, handle endianness mismatch
    // TODO: it might be a bit overkill, but it be nice to make this function
    //       template-parametric w.r.t. endianness mismatch
    Function *BSwapFunction = nullptr;
    if (MemoryType != Builder.getInt8Ty() and EndianessMismatch)
      BSwapFunction = Intrinsic::getDeclaration(&TheModule,
                                                Intrinsic::bswap,
                                                { MemoryType });

    // Is the memory op a sign extended load?
    bool SignExtend = (MemoryOp & LIBTCG_MO_SIGN) != 0;

    Value *Pointer = nullptr;
    if (Opcode == LIBTCG_op_qemu_ld_a32_i32 or
        Opcode == LIBTCG_op_qemu_ld_a64_i32 or
        Opcode == LIBTCG_op_qemu_ld_a32_i64 or
        Opcode == LIBTCG_op_qemu_ld_a64_i64) {

      Pointer = Builder.CreateIntToPtr(InArguments[0],
                                       MemoryType->getPointerTo());
      auto *Load = Builder.CreateAlignedLoad(MemoryType,
                                             Pointer,
                                             MaybeAlign(Alignment));
      Value *Loaded = Load;

      if (BSwapFunction != nullptr)
        Loaded = Builder.CreateCall(BSwapFunction, Load);

      if (SignExtend)
        return v{ Builder.CreateSExt(Loaded, RegisterType) };
      else
        return v{ Builder.CreateZExt(Loaded, RegisterType) };

    } else if (Opcode == LIBTCG_op_qemu_st_a32_i32 or
               Opcode == LIBTCG_op_qemu_st_a64_i32 or
               Opcode == LIBTCG_op_qemu_st_a32_i64 or
               Opcode == LIBTCG_op_qemu_st_a64_i64) {

      Pointer = Builder.CreateIntToPtr(InArguments[1],
                                       MemoryType->getPointerTo());
      Value *Value = Builder.CreateTrunc(InArguments[0], MemoryType);

      if (BSwapFunction != nullptr)
        Value = Builder.CreateCall(BSwapFunction, Value);

      Builder.CreateAlignedStore(Value, Pointer, MaybeAlign(Alignment));

      return v{};
    } else {
      revng_unreachable("Unknown load type");
    }
  }
  case LIBTCG_op_ld8u_i32:
  case LIBTCG_op_ld8s_i32:
  case LIBTCG_op_ld16u_i32:
  case LIBTCG_op_ld16s_i32:
  case LIBTCG_op_ld_i32:
  case LIBTCG_op_ld8u_i64:
  case LIBTCG_op_ld8s_i64:
  case LIBTCG_op_ld16u_i64:
  case LIBTCG_op_ld16s_i64:
  case LIBTCG_op_ld32u_i64:
  case LIBTCG_op_ld32s_i64:
  case LIBTCG_op_ld_i64: {
    Value *Base = dyn_cast<LoadInst>(InArguments[0])->getPointerOperand();
    if (Base == nullptr or !Variables.isEnv(Base)) {
      // TODO: emit warning
      return std::errc::invalid_argument;
    }

    bool Signed;
    switch (Opcode) {
    case LIBTCG_op_ld_i32:
    case LIBTCG_op_ld_i64:

    case LIBTCG_op_ld8u_i32:
    case LIBTCG_op_ld16u_i32:
    case LIBTCG_op_ld8u_i64:
    case LIBTCG_op_ld16u_i64:
    case LIBTCG_op_ld32u_i64:
      Signed = false;
      break;
    case LIBTCG_op_ld8s_i32:
    case LIBTCG_op_ld16s_i32:
    case LIBTCG_op_ld8s_i64:
    case LIBTCG_op_ld16s_i64:
    case LIBTCG_op_ld32s_i64:
      Signed = true;
      break;
    default:
      revng_unreachable("Unexpected opcode");
    }

    unsigned LoadSize;
    switch (Opcode) {
    case LIBTCG_op_ld8u_i32:
    case LIBTCG_op_ld8s_i32:
    case LIBTCG_op_ld8u_i64:
    case LIBTCG_op_ld8s_i64:
      LoadSize = 1;
      break;
    case LIBTCG_op_ld16u_i32:
    case LIBTCG_op_ld16s_i32:
    case LIBTCG_op_ld16u_i64:
    case LIBTCG_op_ld16s_i64:
      LoadSize = 2;
      break;
    case LIBTCG_op_ld_i32:
    case LIBTCG_op_ld32u_i64:
    case LIBTCG_op_ld32s_i64:
      LoadSize = 4;
      break;
    case LIBTCG_op_ld_i64:
      LoadSize = 8;
      break;
    default:
      revng_unreachable("Unexpected opcode");
    }

    revng_assert(isa<ConstantInt>(InArguments[1]));
    Value *Result = Variables.loadFromEnvOffset(Builder,
                                                LoadSize,
                                                cast<ConstantInt>(InArguments[1])->getLimitedValue());
    revng_assert(Result != nullptr);

    // Zero/sign extend in the target dimension
    if (Signed)
      return v{ Builder.CreateSExt(Result, RegisterType) };
    else
      return v{ Builder.CreateZExt(Result, RegisterType) };
  }
  case LIBTCG_op_st8_i32:
  case LIBTCG_op_st16_i32:
  case LIBTCG_op_st_i32:
  case LIBTCG_op_st8_i64:
  case LIBTCG_op_st16_i64:
  case LIBTCG_op_st32_i64:
  case LIBTCG_op_st_i64: {
    unsigned StoreSize;
    switch (Opcode) {
    case LIBTCG_op_st8_i32:
    case LIBTCG_op_st8_i64:
      StoreSize = 1;
      break;
    case LIBTCG_op_st16_i32:
    case LIBTCG_op_st16_i64:
      StoreSize = 2;
      break;
    case LIBTCG_op_st_i32:
    case LIBTCG_op_st32_i64:
      StoreSize = 4;
      break;
    case LIBTCG_op_st_i64:
      StoreSize = 8;
      break;
    default:
      revng_unreachable("Unexpected opcode");
    }

    Value *Base = dyn_cast<LoadInst>(InArguments[1])->getPointerOperand();
    if (Base == nullptr or !Variables.isEnv(Base)) {
      // TODO: emit warning
      return std::errc::invalid_argument;
    }

    revng_assert(isa<ConstantInt>(InArguments[2]));
    auto Result = Variables.storeToEnvOffset(Builder,
                                             StoreSize,
                                             cast<ConstantInt>(InArguments[2])->getLimitedValue(),
                                             InArguments[0]);
    PCH->handleStore(Builder, *Result);

    return v{};
  }
  case LIBTCG_op_add_i32:
  case LIBTCG_op_sub_i32:
  case LIBTCG_op_mul_i32:
  case LIBTCG_op_div_i32:
  case LIBTCG_op_divu_i32:
  case LIBTCG_op_rem_i32:
  case LIBTCG_op_remu_i32:
  case LIBTCG_op_and_i32:
  case LIBTCG_op_or_i32:
  case LIBTCG_op_xor_i32:
  case LIBTCG_op_shl_i32:
  case LIBTCG_op_shr_i32:
  case LIBTCG_op_sar_i32:
  case LIBTCG_op_add_i64:
  case LIBTCG_op_sub_i64:
  case LIBTCG_op_mul_i64:
  case LIBTCG_op_div_i64:
  case LIBTCG_op_divu_i64:
  case LIBTCG_op_rem_i64:
  case LIBTCG_op_remu_i64:
  case LIBTCG_op_and_i64:
  case LIBTCG_op_or_i64:
  case LIBTCG_op_xor_i64:
  case LIBTCG_op_shl_i64:
  case LIBTCG_op_shr_i64:
  case LIBTCG_op_sar_i64: {
    // TODO: assert on sizes?
    Instruction::BinaryOps BinaryOp = opcodeToBinaryOp(Opcode);
    Value *Operation = Builder.CreateBinOp(BinaryOp,
                                           InArguments[0],
                                           InArguments[1]);
    return v{ Operation };
  }
  case LIBTCG_op_div2_i32:
  case LIBTCG_op_divu2_i32:
  case LIBTCG_op_div2_i64:
  case LIBTCG_op_divu2_i64: {
    Instruction::BinaryOps DivisionOp, RemainderOp;

    if (Opcode == LIBTCG_op_div2_i32
        or Opcode == LIBTCG_op_div2_i64) {
      DivisionOp = Instruction::SDiv;
      RemainderOp = Instruction::SRem;
    } else if (Opcode == LIBTCG_op_divu2_i32
               or Opcode == LIBTCG_op_divu2_i64) {
      DivisionOp = Instruction::UDiv;
      RemainderOp = Instruction::URem;
    } else {
      revng_unreachable("Unknown operation type");
    }

    // TODO: we're ignoring InArguments[1], which is the MSB
    // TODO: assert on sizes?
    Value *Division = Builder.CreateBinOp(DivisionOp,
                                          InArguments[0],
                                          InArguments[2]);
    Value *Remainder = Builder.CreateBinOp(RemainderOp,
                                           InArguments[0],
                                           InArguments[2]);
    return v{ Division, Remainder };
  }
  case LIBTCG_op_rotr_i32:
  case LIBTCG_op_rotr_i64:
  case LIBTCG_op_rotl_i32:
  case LIBTCG_op_rotl_i64: {
    Value *Bits = ConstantInt::get(RegisterType, RegisterSize);

    Instruction::BinaryOps FirstShiftOp, SecondShiftOp;
    if (Opcode == LIBTCG_op_rotl_i32
        or Opcode == LIBTCG_op_rotl_i64) {
      FirstShiftOp = Instruction::Shl;
      SecondShiftOp = Instruction::LShr;
    } else if (Opcode == LIBTCG_op_rotr_i32
               or Opcode == LIBTCG_op_rotr_i64) {
      FirstShiftOp = Instruction::LShr;
      SecondShiftOp = Instruction::Shl;
    } else {
      revng_unreachable("Unexpected opcode");
    }

    Value *FirstShift = Builder.CreateBinOp(FirstShiftOp,
                                            InArguments[0],
                                            InArguments[1]);
    Value *SecondShiftAmount = Builder.CreateSub(Bits, InArguments[1]);
    Value *SecondShift = Builder.CreateBinOp(SecondShiftOp,
                                             InArguments[0],
                                             SecondShiftAmount);

    return v{ Builder.CreateOr(FirstShift, SecondShift) };
  }
  case LIBTCG_op_deposit_i32:
  case LIBTCG_op_deposit_i64: {
    Value *Result = genDeposit(Builder, RegisterSize,
                               InArguments[0], InArguments[1],
                               InArguments[2], InArguments[3]);
    return v{ Result };
  }
  case LIBTCG_op_ext8s_i32:
  case LIBTCG_op_ext16s_i32:
  case LIBTCG_op_ext8u_i32:
  case LIBTCG_op_ext16u_i32:
  case LIBTCG_op_ext8s_i64:
  case LIBTCG_op_ext16s_i64:
  case LIBTCG_op_ext32s_i64:
  case LIBTCG_op_ext8u_i64:
  case LIBTCG_op_ext16u_i64:
  case LIBTCG_op_ext32u_i64:
  case LIBTCG_op_ext_i32_i64:
  case LIBTCG_op_extu_i32_i64: {
    Type *SourceType = nullptr;
    switch (Opcode) {
    case LIBTCG_op_ext8s_i32:
    case LIBTCG_op_ext8u_i32:
    case LIBTCG_op_ext8s_i64:
    case LIBTCG_op_ext8u_i64:
      SourceType = Builder.getInt8Ty();
      break;
    case LIBTCG_op_ext16s_i32:
    case LIBTCG_op_ext16u_i32:
    case LIBTCG_op_ext16s_i64:
    case LIBTCG_op_ext16u_i64:
      SourceType = Builder.getInt16Ty();
      break;
    case LIBTCG_op_ext32s_i64:
    case LIBTCG_op_ext32u_i64:
    case LIBTCG_op_ext_i32_i64:
    case LIBTCG_op_extu_i32_i64:
      SourceType = Builder.getInt32Ty();
      break;
    default:
      revng_unreachable("Unexpected opcode");
    }

    Value *Truncated = Builder.CreateTrunc(InArguments[0], SourceType);

    switch (Opcode) {
    case LIBTCG_op_ext8s_i32:
    case LIBTCG_op_ext8s_i64:
    case LIBTCG_op_ext16s_i32:
    case LIBTCG_op_ext16s_i64:
    case LIBTCG_op_ext32s_i64:
    case LIBTCG_op_ext_i32_i64:
      return v{ Builder.CreateSExt(Truncated, RegisterType) };
    case LIBTCG_op_ext8u_i32:
    case LIBTCG_op_ext8u_i64:
    case LIBTCG_op_ext16u_i32:
    case LIBTCG_op_ext16u_i64:
    case LIBTCG_op_ext32u_i64:
    case LIBTCG_op_extu_i32_i64:
      return v{ Builder.CreateZExt(Truncated, RegisterType) };
    default:
      revng_unreachable("Unexpected opcode");
    }
  }
  case LIBTCG_op_extrl_i64_i32: {
      return v{ Builder.CreateTrunc(InArguments[0], Builder.getInt32Ty()) };
  }
  case LIBTCG_op_extrh_i64_i32: {
      Value *Shifted = Builder.CreateAShr(InArguments[0], ConstantInt::get(Builder.getInt64Ty(), 32));
      return v{ Builder.CreateTrunc(Shifted, Builder.getInt32Ty()) };
  }
  case LIBTCG_op_not_i32:
  case LIBTCG_op_not_i64:
    return v{ Builder.CreateXor(InArguments[0], getMaxValue(RegisterSize)) };
  case LIBTCG_op_neg_i32:
  case LIBTCG_op_neg_i64: {
    auto *InitialValue = ConstantInt::get(RegisterType, 0);
    return v{ Builder.CreateSub(InitialValue, InArguments[0]) };
  }
  case LIBTCG_op_andc_i32:
  case LIBTCG_op_andc_i64:
  case LIBTCG_op_orc_i32:
  case LIBTCG_op_orc_i64:
  case LIBTCG_op_eqv_i32:
  case LIBTCG_op_eqv_i64: {
    Instruction::BinaryOps ExternalOp;
    switch (Opcode) {
    case LIBTCG_op_andc_i32:
    case LIBTCG_op_andc_i64:
      ExternalOp = Instruction::And;
      break;
    case LIBTCG_op_orc_i32:
    case LIBTCG_op_orc_i64:
      ExternalOp = Instruction::Or;
      break;
    case LIBTCG_op_eqv_i32:
    case LIBTCG_op_eqv_i64:
      ExternalOp = Instruction::Xor;
      break;
    default:
      revng_unreachable("Unexpected opcode");
    }

    Value *Negate = Builder.CreateXor(InArguments[1],
                                      getMaxValue(RegisterSize));
    Value *Result = Builder.CreateBinOp(ExternalOp, InArguments[0], Negate);
    return v{ Result };
  }
  case LIBTCG_op_nand_i32:
  case LIBTCG_op_nand_i64: {
    Value *AndValue = Builder.CreateAnd(InArguments[0], InArguments[1]);
    Value *Result = Builder.CreateXor(AndValue, getMaxValue(RegisterSize));
    return v{ Result };
  }
  case LIBTCG_op_nor_i32:
  case LIBTCG_op_nor_i64: {
    Value *OrValue = Builder.CreateOr(InArguments[0], InArguments[1]);
    Value *Result = Builder.CreateXor(OrValue, getMaxValue(RegisterSize));
    return v{ Result };
  }
  case LIBTCG_op_bswap16_i32:
  case LIBTCG_op_bswap32_i32:
  case LIBTCG_op_bswap16_i64:
  case LIBTCG_op_bswap32_i64:
  case LIBTCG_op_bswap64_i64: {
    Type *SwapType = nullptr;
    switch (Opcode) {
    case LIBTCG_op_bswap16_i32:
    case LIBTCG_op_bswap16_i64:
      SwapType = Builder.getInt16Ty();
      break;
    case LIBTCG_op_bswap32_i32:
    case LIBTCG_op_bswap32_i64:
      SwapType = Builder.getInt32Ty();
      break;
    case LIBTCG_op_bswap64_i64:
      SwapType = Builder.getInt64Ty();
      break;
    default:
      revng_unreachable("Unexpected opcode");
    }

    Value *Truncated = Builder.CreateTrunc(InArguments[0], SwapType);

    Function *BSwapFunction = Intrinsic::getDeclaration(&TheModule,
                                                        Intrinsic::bswap,
                                                        { SwapType });
    Value *Swapped = Builder.CreateCall(BSwapFunction, Truncated);

    return v{ Builder.CreateZExt(Swapped, RegisterType) };
  }
  case LIBTCG_op_set_label: {
    revng_assert(ConstArguments[0].kind == LIBTCG_ARG_LABEL);
    auto LabelId = ConstArguments[0].label->id;

    std::stringstream LabelSS;
    LabelSS << "bb." << JumpTargets.nameForAddress(LastPC);
    LabelSS << "_L" << std::dec << LabelId;
    std::string Label = LabelSS.str();

    BasicBlock *Fallthrough = nullptr;
    if (!LabeledBasicBlocks.contains(Label)) {
      Fallthrough = BasicBlock::Create(Context, Label, TheFunction);
      Fallthrough->moveAfter(Builder.GetInsertBlock());
      LabeledBasicBlocks[Label] = Fallthrough;
    } else {
      // A basic block with that label already exist
      Fallthrough = LabeledBasicBlocks[Label];

      // Ensure it's empty
      revng_assert(Fallthrough->begin() == Fallthrough->end());

      // Move it to the bottom
      Fallthrough->removeFromParent();
      TheFunction->insert(TheFunction->end(), Fallthrough);
    }

    Builder.CreateBr(Fallthrough);

    Blocks.push_back(Fallthrough);
    Builder.SetInsertPoint(Fallthrough);
    Variables.newBasicBlock();

    return v{};
  }
  case LIBTCG_op_br:
  case LIBTCG_op_brcond_i32:
  case LIBTCG_op_brcond2_i32:
  case LIBTCG_op_brcond_i64: {
    // We take the last constant arguments, which is the LabelId both in
    // conditional and unconditional jumps
    revng_assert(ConstArguments.back().kind == LIBTCG_ARG_LABEL);
    auto LabelId = ConstArguments.back().label->id;

    std::stringstream LabelSS;
    LabelSS << "bb." << JumpTargets.nameForAddress(LastPC);
    LabelSS << "_L" << std::dec << LabelId;
    std::string Label = LabelSS.str();

    BasicBlock *Fallthrough = BasicBlock::Create(Context,
                                                 Label + "_ft",
                                                 TheFunction);

    // Look for a matching label
    BasicBlock *Target = nullptr;
    if (!LabeledBasicBlocks.contains(Label)) {
      // No matching label, create a temporary block
      Target = BasicBlock::Create(Context, Label, TheFunction);
      LabeledBasicBlocks[Label] = Target;
    } else {
      Target = LabeledBasicBlocks[Label];
    }

    if (Opcode == LIBTCG_op_br) {
      // Unconditional jump
      Builder.CreateBr(Target);
    } else if (Opcode == LIBTCG_op_brcond_i32
               or Opcode == LIBTCG_op_brcond_i64) {
      // Conditional jump
      revng_assert(ConstArguments[0].kind == LIBTCG_ARG_COND);
      Value *Compare = createICmp(Builder,
                                  ConstArguments[0].cond,
                                  InArguments[0],
                                  InArguments[1]);
      Builder.CreateCondBr(Compare, Target, Fallthrough);
    } else {
      revng_unreachable("Unhandled opcode");
    }

    Blocks.push_back(Fallthrough);
    Builder.SetInsertPoint(Fallthrough);
    Variables.newBasicBlock();

    return v{};
  }
  case LIBTCG_op_exit_tb: {
    auto *Zero = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Builder.CreateCall(JumpTargets.exitTB(), { Zero });
    Builder.CreateUnreachable();

    ExitBlocks.push_back(Builder.GetInsertBlock());

    auto *NextBB = BasicBlock::Create(Context, "", TheFunction);
    Blocks.push_back(NextBB);
    Builder.SetInsertPoint(NextBB);
    Variables.newBasicBlock();

    return v{};
  }
  case LIBTCG_op_goto_tb:
  case LIBTCG_op_goto_ptr:
    // Nothing to do here
    return v{};
  case LIBTCG_op_add2_i32:
  case LIBTCG_op_sub2_i32:
  case LIBTCG_op_add2_i64:
  case LIBTCG_op_sub2_i64: {
    Value *FirstOpLow = nullptr;
    Value *FirstOpHigh = nullptr;
    Value *SecondOpLow = nullptr;
    Value *SecondOpHigh = nullptr;

    IntegerType *DestinationType = Builder.getIntNTy(RegisterSize * 2);

    FirstOpLow = Builder.CreateZExt(InArguments[0], DestinationType);
    FirstOpHigh = Builder.CreateZExt(InArguments[1], DestinationType);
    SecondOpLow = Builder.CreateZExt(InArguments[2], DestinationType);
    SecondOpHigh = Builder.CreateZExt(InArguments[3], DestinationType);

    FirstOpHigh = Builder.CreateShl(FirstOpHigh, RegisterSize);
    SecondOpHigh = Builder.CreateShl(SecondOpHigh, RegisterSize);

    Value *FirstOp = Builder.CreateOr(FirstOpHigh, FirstOpLow);
    Value *SecondOp = Builder.CreateOr(SecondOpHigh, SecondOpLow);

    Instruction::BinaryOps BinaryOp = opcodeToBinaryOp(Opcode);

    Value *Result = Builder.CreateBinOp(BinaryOp, FirstOp, SecondOp);

    Value *ResultLow = Builder.CreateTrunc(Result, RegisterType);
    Value *ShiftedResult = Builder.CreateLShr(Result, RegisterSize);
    Value *ResultHigh = Builder.CreateTrunc(ShiftedResult, RegisterType);

    return v{ ResultLow, ResultHigh };
  }
  case LIBTCG_op_mulu2_i32:
  case LIBTCG_op_mulu2_i64:
  case LIBTCG_op_muls2_i32:
  case LIBTCG_op_muls2_i64: {
    IntegerType *DestinationType = Builder.getIntNTy(RegisterSize * 2);

    Value *FirstOp = nullptr;
    Value *SecondOp = nullptr;

    if (Opcode == LIBTCG_op_mulu2_i32
        or Opcode == LIBTCG_op_mulu2_i64) {
      FirstOp = Builder.CreateZExt(InArguments[0], DestinationType);
      SecondOp = Builder.CreateZExt(InArguments[1], DestinationType);
    } else if (Opcode == LIBTCG_op_muls2_i32
               or Opcode == LIBTCG_op_muls2_i64) {
      FirstOp = Builder.CreateSExt(InArguments[0], DestinationType);
      SecondOp = Builder.CreateSExt(InArguments[1], DestinationType);
    } else {
      revng_unreachable("Unexpected opcode");
    }

    Value *Result = Builder.CreateMul(FirstOp, SecondOp);

    Value *ResultLow = Builder.CreateTrunc(Result, RegisterType);
    Value *ShiftedResult = Builder.CreateLShr(Result, RegisterSize);
    Value *ResultHigh = Builder.CreateTrunc(ShiftedResult, RegisterType);

    return v{ ResultLow, ResultHigh };
  }
  case LIBTCG_op_muluh_i32:
  case LIBTCG_op_mulsh_i32:
  case LIBTCG_op_muluh_i64:
  case LIBTCG_op_mulsh_i64:
  case LIBTCG_op_setcond2_i32:
    revng_unreachable("Instruction not implemented");
  case LIBTCG_op_extract_i32: {
    auto *Const32 = ConstantInt::get(Type::getInt32Ty(Context), 32);
    Value *Length = InArguments[1];
    Value *Offset = InArguments[2];
    Value *ShlAmount = Builder.CreateSub(Const32,
                                         Builder.CreateAdd(Offset, Length));
    Value *Shl = Builder.CreateShl(InArguments[0], ShlAmount);
    Value *LShr = Builder.CreateLShr(Shl, Builder.CreateSub(Const32, Length));
    return v{ LShr };
  }
  case LIBTCG_op_sextract_i32: {
    auto *Const32 = ConstantInt::get(Type::getInt32Ty(Context), 32);
    Value *Length = InArguments[1];
    Value *Offset = InArguments[2];
    Value *ShlAmount = Builder.CreateSub(Const32,
                                         Builder.CreateAdd(Offset, Length));
    Value *Shl = Builder.CreateShl(InArguments[0], ShlAmount);
    Value *AShr = Builder.CreateAShr(Shl, Builder.CreateSub(Const32, Length));
    return v{ AShr };
  }
  case LIBTCG_op_extract2_i32:
  case LIBTCG_op_extract2_i64: {
    Value *Low    = InArguments[0];
    Value *High   = InArguments[1];
    Value *Offset = InArguments[2];

    auto *ConstSize = ConstantInt::get(RegisterType, RegisterSize);
    Value *Shift = Builder.CreateLShr(Low, Offset);
    Value *Result = genDeposit(Builder, RegisterSize, Shift, High,
                               Builder.CreateSub(ConstSize, Offset), Offset);

    return v{ Result };
  }
  case LIBTCG_op_clz_i32: {
    Type *Int1Ty = Type::getInt1Ty(Context);
    auto *One = ConstantInt::get(Int1Ty, 1);
    auto *Zero = ConstantInt::get(RegisterType, 0);
    Value *Arg = InArguments[0];
    Value *ZeroVal = InArguments[1];
    CallInst *Ctlz = Builder.CreateBinaryIntrinsic(Intrinsic::ctlz, Arg, One);
    Value *ICmp = Builder.CreateICmp(CmpInst::ICMP_EQ, Arg, Zero);
    Value *Select = Builder.CreateSelect(ICmp, ZeroVal, Ctlz);
    return v{ Select };
  }
  case LIBTCG_op_clz_i64: {
    Type *Int1Ty = Type::getInt1Ty(Context);
    auto *One = ConstantInt::get(Int1Ty, 1);
    auto *Zero = ConstantInt::get(RegisterType, 0);
    Value *Arg = InArguments[0];
    Value *ZeroVal = InArguments[1];
    CallInst *Ctlz = Builder.CreateBinaryIntrinsic(Intrinsic::ctlz, Arg, One);
    Value *ICmp = Builder.CreateICmp(CmpInst::ICMP_EQ, Arg, Zero);
    Value *Select = Builder.CreateSelect(ICmp, ZeroVal, Ctlz);
    return v{ Select };
  }
  default:
    // For debugging purposes printing the actual opcode
    // really helps.
    std::stringstream ErrSS;
    ErrSS << "Unknown libtcg opcode ["
          << Opcode
          << "]: "
          << LibTcg.get_instruction_name(Opcode);
    revng_unreachable(ErrSS.str().c_str());
  }
}

void IT::handleExitTB() {
  auto &Context = TheModule.getContext();
  auto *Zero = ConstantInt::get(Type::getInt32Ty(Context), 0);
  Builder.CreateCall(JumpTargets.exitTB(), { Zero });
  Builder.CreateUnreachable();

  ExitBlocks.push_back(Builder.GetInsertBlock());

  auto *NextBB = BasicBlock::Create(Context, "", TheFunction);
  Blocks.push_back(NextBB);
  Builder.SetInsertPoint(NextBB);
  Variables.newBasicBlock();
}
