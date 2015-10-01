/// \file
/// \brief  This file handles the translation from QEMU's PTC to LLVM IR.

#include <cstdint>
#include <sstream>

// LLVM API
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Metadata.h"

#include "ptctollvmir.h"
#include "revamb.h"
#include "ptcinterface.h"
#include "ptcdump.h"

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
  VariableManager(llvm::Module& Module) : Module(Module),
                                          Builder(Module.getContext()) { }
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
    llvm::StringRef TemporaryName(Temporary->name);

    GlobalsMap::iterator it = Globals.find(TemporaryName);
    if (it != Globals.end()) {
      return it->second;
    } else {
      for (llvm::GlobalVariable& Variable : Module.globals())
        if (Variable.getName() == TemporaryName)
          return &Variable;

      llvm::Constant *Initializer = llvm::ConstantInt::get(VariableType, 0);
      llvm::GlobalVariable *NewVariable = nullptr;
      NewVariable = new llvm::GlobalVariable(Module,
                                             VariableType,
                                             false,
                                             llvm::GlobalValue::CommonLinkage,
                                             Initializer,
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
    return llvm::Instruction::Add;
  case PTC_INSTRUCTION_op_sub_i32:
  case PTC_INSTRUCTION_op_sub_i64:
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

int Translate(std::ostream& Output, llvm::ArrayRef<uint8_t> Code) {
  const uint8_t *CodePointer = Code.data();
  const uint8_t *CodeEnd = CodePointer + Code.size();

  llvm::LLVMContext& Context = llvm::getGlobalContext();
  std::unique_ptr<llvm::Module> Module(new llvm::Module("top", Context));
  llvm::IRBuilder<> Builder(Context);

  llvm::FunctionType *MainType = nullptr;
  MainType = llvm::FunctionType::get(Builder.getInt32Ty(), false);

  llvm::Function *MainFunction = nullptr;
  MainFunction = llvm::Function::Create(MainType,
                                        llvm::Function::ExternalLinkage,
                                        "main",
                                        Module.get());

  llvm::BasicBlock *Entry = llvm::BasicBlock::Create(Context,
                                                     "entrypoint",
                                                     MainFunction);
  Builder.SetInsertPoint(Entry);
  llvm::Instruction *Delimiter = Builder.CreateUnreachable();

  unsigned OriginalInstrMDKind = Context.getMDKindID("oi");
  unsigned PTCInstrMDKind = Context.getMDKindID("pi");

  VariableManager Variables(*Module);

  // TODO: move me somewhere where it makes sense
  Architecture SourceArchitecture;
  Architecture TargetArchitecture;

  while (CodePointer < CodeEnd) {
    printf("\nPTC for 0x%llx\n", (long long) (CodePointer - Code.data()));

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

    for (; j < InstructionList->instruction_count; j++) {
      PTCInstruction Instruction = InstructionList->instructions[j];
      PTCOpcode Opcode = Instruction.opc;

      if (Opcode == PTC_INSTRUCTION_op_call)
        continue;

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
          PTCCondition Condition = static_cast<PTCCondition>(ConstArguments[0]);
          llvm::Value *Compare = nullptr;
          Compare = Builder.CreateICmp(conditionToPredicate(Condition),
                                       InArguments[0],
                                       InArguments[1]);
          OutArguments.push_back(Compare);
          break;
        }
      case PTC_INSTRUCTION_op_movcond_i32: // Resist the fallthrough temptation
      case PTC_INSTRUCTION_op_movcond_i64:
        {
          PTCCondition Condition = static_cast<PTCCondition>(ConstArguments[0]);
          llvm::Value *Compare = nullptr;
          Compare = Builder.CreateICmp(conditionToPredicate(Condition),
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
            AccessAlignment = SourceArchitecture.DefaultAlignment();

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
            llvm_unreachable("Unknown load size");
          }

          bool SignExtend = ptc_is_sign_extended_load(MemoryAccess.type);

          // TODO: handle 64 on 32
          // TODO: handle endianess mismatch
          assert(SourceArchitecture.Endianess() ==
                 TargetArchitecture.Endianess() &&
                 "Different endianess between the source and the target is not "
                 "supported yet");

          llvm::Value *Pointer = nullptr;
          Pointer = Builder.CreateIntToPtr(InArguments[0],
                                           MemoryType->getPointerTo());

          if (Opcode == PTC_INSTRUCTION_op_qemu_ld_i32 ||
              Opcode == PTC_INSTRUCTION_op_qemu_ld_i64) {

            llvm::Value *Load = Builder.CreateAlignedLoad(Pointer,
                                                          AccessAlignment);
            llvm::Value *Result = nullptr;
            if (SignExtend) {
              Result = Builder.CreateSExt(Load, RegisterType);
            } else {
              Result = Builder.CreateZExt(Load, RegisterType);
            }
            OutArguments.push_back(Result);

          } else if (Opcode == PTC_INSTRUCTION_op_qemu_st_i32 ||
                     Opcode == PTC_INSTRUCTION_op_qemu_st_i64) {

            llvm::Value *Value = Builder.CreateTrunc(InArguments[1], MemoryType);
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
          llvm::BasicBlock *NewBasicBlock = nullptr;
          NewBasicBlock = llvm::BasicBlock::Create(Context, Label, MainFunction);
          Blocks.push_back(NewBasicBlock);
          Builder.CreateBr(NewBasicBlock);
          Builder.SetInsertPoint(NewBasicBlock);
          Variables.newBasicBlock();
          break;
        }
      case PTC_INSTRUCTION_op_debug_insn_start:
        {
          uint64_t PC = ConstArguments[0];

          // TODO: replace using a field in Architecture
          if (ConstArguments.size() > 1)
            PC |= ConstArguments[1] << 32;

          std::stringstream OriginalStringStream;
          disassembleOriginal(OriginalStringStream, PC);
          std::string OriginalString = OriginalStringStream.str();
          llvm::MDString *MDOriginalString = nullptr;
          MDOriginalString = llvm::MDString::get(Context, OriginalString);
          MDOriginalInstr = llvm::MDNode::get(Context, MDOriginalString);

          break;
        }
      case PTC_INSTRUCTION_op_call:
      case PTC_INSTRUCTION_op_br:
      case PTC_INSTRUCTION_op_brcond_i32:
      case PTC_INSTRUCTION_op_brcond2_i32:
      case PTC_INSTRUCTION_op_brcond_i64:

      case PTC_INSTRUCTION_op_exit_tb:
      case PTC_INSTRUCTION_op_goto_tb:

      case PTC_INSTRUCTION_op_add2_i32:
      case PTC_INSTRUCTION_op_sub2_i32:
      case PTC_INSTRUCTION_op_mulu2_i32:
      case PTC_INSTRUCTION_op_muls2_i32:
      case PTC_INSTRUCTION_op_muluh_i32:
      case PTC_INSTRUCTION_op_mulsh_i32:
      case PTC_INSTRUCTION_op_add2_i64:
      case PTC_INSTRUCTION_op_sub2_i64:
      case PTC_INSTRUCTION_op_mulu2_i64:
      case PTC_INSTRUCTION_op_muls2_i64:
      case PTC_INSTRUCTION_op_muluh_i64:
      case PTC_INSTRUCTION_op_mulsh_i64:

      case PTC_INSTRUCTION_op_setcond2_i32:

      case PTC_INSTRUCTION_op_trunc_shr_i32:

        continue;
      default:
        llvm_unreachable("Unknown opcode");
      }

      unsigned OutArgumentsCount = ptc_instruction_out_arg_count(&ptc,
                                                                 &Instruction);
      assert(OutArgumentsCount == OutArguments.size());
      for (unsigned i = 0; i < OutArgumentsCount; i++) {
        unsigned TemporaryId = ptc_instruction_out_arg(&ptc, &Instruction, i);
        Builder.CreateStore(OutArguments[i], Variables.getOrCreate(TemporaryId));
      }

      // Set metadata for all the new instructions
      std::stringstream PTCStringStream;
      dumpInstruction(PTCStringStream, InstructionList.get(), j);
      std::string PTCString = PTCStringStream.str();
      llvm::MDString *MDPTCString = llvm::MDString::get(Context, PTCString);
      llvm::MDNode* MDPTCInstr = llvm::MDNode::get(Context, MDPTCString);
      for (llvm::BasicBlock *Block : Blocks) {
        llvm::BasicBlock::iterator I = Block->end();
        while (I != Block->begin() && !(--I)->hasMetadata()) {
          I->setMetadata(OriginalInstrMDKind, MDOriginalInstr);
          I->setMetadata(PTCInstrMDKind, MDPTCInstr);
        }
      }

    }

    if (dumpTranslation(Output, InstructionList.get()) != EXIT_SUCCESS)
      return EXIT_FAILURE;

    // CodePointer += ConsumedSize;
    (void) ConsumedSize;
    CodePointer = CodeEnd;
  }

  Delimiter->eraseFromParent();

  Module->dump();

  return EXIT_SUCCESS;
}
