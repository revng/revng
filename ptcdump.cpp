/// \file
/// \brief This file handles dumping PTC to text

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include "ptcinterface.h"

static const int MAX_TEMP_NAME_LENGTH = 128;

static void getTemporaryName(char *Buffer, size_t BufferSize,
                             PTCInstructionList *Instructions,
                             unsigned TemporaryId) {
  PTCTemp *Temporary = ptc_temp_get(Instructions, TemporaryId);

  if (ptc_temp_is_global(Instructions, TemporaryId))
    strncpy(Buffer, Temporary->name, BufferSize);
  else if (Temporary->temp_local)
    snprintf(Buffer, BufferSize, "loc%u", 
             TemporaryId - Instructions->global_temps);
  else
    snprintf(Buffer, BufferSize, "tmp%u",
             TemporaryId - Instructions->global_temps);
}

int dumpTranslation(FILE *OutputFile, PTCInstructionList *Instructions) {
  size_t j = 0;
  size_t i = 0;
  int is64 = 0;

  for (j = 0; j < Instructions->instruction_count; j++) {
    PTCInstruction Instruction = Instructions->instructions[j];
    PTCOpcode Opcode = Instruction.opc;
    PTCOpcodeDef *Definition = ptc_instruction_opcode_def(&ptc, &Instruction);
    char TemporaryName[MAX_TEMP_NAME_LENGTH];

    if (Opcode == PTC_INSTRUCTION_op_debug_insn_start) {
      /* TODO: create accessors for PTC_INSTRUCTION_op_debug_insn_start */
      uint64_t PC = Instruction.args[0];

      if (is64) {
        PC |= Instruction.args[1] << 32;
      }

      if (j != 0)
        fprintf(OutputFile, "\n");

      fprintf(OutputFile, " ---- 0x%" PRIx64, PC);
    } else if (Opcode == PTC_INSTRUCTION_op_call) {
      // TODO: replace PRIx64 with PTC_PRIxARG
      PTCInstructionArg Arg0 = ptc_call_instruction_const_arg(&ptc,
                                                              &Instruction, 0);
      PTCInstructionArg Arg1 = ptc_call_instruction_const_arg(&ptc,
                                                              &Instruction, 1);
      size_t OutArgs = ptc_call_instruction_out_arg_count(&ptc, &Instruction);
      PTCHelperDef *Helper = ptc_find_helper(&ptc, Arg0);
      const char *HelperName = "unknown_helper";

      if (Helper != nullptr && Helper->name != nullptr)
        HelperName = Helper->name;

      fprintf(OutputFile, "%s %s,$0x%" PRIx64 ",$%jd", Definition->name,
              HelperName, Arg1, OutArgs);

      // Print out arguments
      size_t OutArgsCount = ptc_call_instruction_out_arg_count(&ptc,
                                                               &Instruction);
      for (i = 0; i < OutArgsCount; i++) {
        getTemporaryName(TemporaryName, MAX_TEMP_NAME_LENGTH, Instructions,
                         ptc_call_instruction_out_arg(&ptc, &Instruction, i));
        fprintf(OutputFile, "%s", TemporaryName);
      }

      // Print in arguments
      size_t InArgsCount = ptc_call_instruction_in_arg_count(&ptc, &Instruction);
      for (i = 0; i < InArgsCount; i++) {
        PTCInstructionArg InArg = ptc_call_instruction_in_arg(&ptc,
                                                              &Instruction, i);

        if (InArg != PTC_CALL_DUMMY_ARG) {
          getTemporaryName(TemporaryName, MAX_TEMP_NAME_LENGTH, Instructions,
                           InArg);
          fprintf(OutputFile, "%s", TemporaryName);
        } else
          fprintf(OutputFile, ",<dummy>");
      }

    } else {
      /* TODO: fix commas */
      fprintf(OutputFile, "%s ", Definition->name);

      /* Print out arguments */
      for (i = 0; i < ptc_instruction_out_arg_count(&ptc, &Instruction); i++) {
        if (i != 0)
          fprintf(OutputFile, ",");

        getTemporaryName(TemporaryName, MAX_TEMP_NAME_LENGTH, Instructions,
                         ptc_instruction_out_arg(&ptc, &Instruction, i));
        fprintf(OutputFile, "%s", TemporaryName);
      }

      if (i != 0)
        fprintf(OutputFile, ",");

      /* Print in arguments */
      for (i = 0; i < ptc_instruction_in_arg_count(&ptc, &Instruction); i++) {
        if (i != 0)
          fprintf(OutputFile, ",");

        getTemporaryName(TemporaryName, MAX_TEMP_NAME_LENGTH, Instructions,
                         ptc_instruction_in_arg(&ptc, &Instruction, i));
        fprintf(OutputFile, "%s", TemporaryName);
      }

      if (i != 0)
        fprintf(OutputFile, ",");

      /* Parse some special const arguments */
      i = 0;
      switch (Opcode) {
      case PTC_INSTRUCTION_op_brcond_i32:
      case PTC_INSTRUCTION_op_setcond_i32:
      case PTC_INSTRUCTION_op_movcond_i32:
      case PTC_INSTRUCTION_op_brcond2_i32:
      case PTC_INSTRUCTION_op_setcond2_i32:
      case PTC_INSTRUCTION_op_brcond_i64:
      case PTC_INSTRUCTION_op_setcond_i64:
      case PTC_INSTRUCTION_op_movcond_i64:
        {
          PTCInstructionArg Arg = ptc_instruction_const_arg(&ptc,
                                                            &Instruction,
                                                            0);
          PTCCondition ConditionId = static_cast<PTCCondition>(Arg);
          const char *ConditionName = ptc.get_condition_name(ConditionId);

          if (ConditionName != nullptr)
            fprintf(OutputFile, ",%s", ConditionName);
          else
            fprintf(OutputFile, ",$0x%" PRIx64, Arg);

          /* Consume one argument */
          i++;
          break;
        }
      case PTC_INSTRUCTION_op_qemu_ld_i32:
      case PTC_INSTRUCTION_op_qemu_st_i32:
      case PTC_INSTRUCTION_op_qemu_ld_i64:
      case PTC_INSTRUCTION_op_qemu_st_i64:
        {
          PTCInstructionArg Arg = ptc_instruction_const_arg(&ptc,
                                                            &Instruction, 0);
          PTCLoadStoreArg LoadStoreArg = {};
          LoadStoreArg = ptc.parse_load_store_arg(Arg);

          if (LoadStoreArg.access_type == PTC_MEMORY_ACCESS_UNKNOWN)
            fprintf(OutputFile, ",$0x%x", LoadStoreArg.raw_op);
          else {
            const char *Alignment = nullptr;
            const char *LoadStoreName = nullptr;
            LoadStoreName = ptc.get_load_store_name(LoadStoreArg.type);

            switch (LoadStoreArg.access_type) {
            case PTC_MEMORY_ACCESS_NORMAL:
              Alignment = "";
              break;
            case PTC_MEMORY_ACCESS_UNALIGNED:
              Alignment = "un+";
              break;
            case PTC_MEMORY_ACCESS_ALIGNED:
              Alignment = "al+";
              break;
            default:
              return EXIT_FAILURE;
            }

            if (LoadStoreName == nullptr)
              return EXIT_FAILURE;

            fprintf(OutputFile, ",%s%s", Alignment, LoadStoreName);
          }

          fprintf(OutputFile, ",%u", LoadStoreArg.mmu_index);

          /* Consume one argument */
          i++;
          break;
        }
      default:
        break;
      }

      switch (Opcode) {
      case PTC_INSTRUCTION_op_set_label:
      case PTC_INSTRUCTION_op_br:
      case PTC_INSTRUCTION_op_brcond_i32:
      case PTC_INSTRUCTION_op_brcond_i64:
      case PTC_INSTRUCTION_op_brcond2_i32:
        {
          PTCInstructionArg Arg = ptc_instruction_const_arg(&ptc,
                                                            &Instruction, i);
          fprintf(OutputFile, ",$L%u", ptc.get_arg_label_id(Arg));

          /* Consume one more argument */
          i++;
          break;
        }
      default:
        break;
      }

      /* Print remaining const arguments */
      for (; i < ptc_instruction_const_arg_count(&ptc, &Instruction); i++) {
        if (i != 0)
          fprintf(OutputFile, ",");

        fprintf(OutputFile, "0x%" PRIx64,
                ptc_instruction_const_arg(&ptc, &Instruction, i));
      }

    }

    fprintf(OutputFile, "\n");
  }

  return EXIT_SUCCESS;
}
