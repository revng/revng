/// \file LLVMDisassemblerInterface.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "revng/Support/Debug.h"
#include "revng/Yield/Assembly/Assembly.h"
#include "revng/Yield/Assembly/LLVMDisassemblerInterface.h"

namespace options {

static bool UseIntelSyntax = true;

enum class ImmediateStyles { Decimal, CHexadecimal, AsmHexadecimal };
static ImmediateStyles ImmediateStyle = ImmediateStyles::CHexadecimal;

static bool ShouldSymbolizeOperands = false;

} // namespace options

/// \note: this might cause multithreading problems.
static void ensureDisassemblersWereInitializedOnce() {
  static bool WereTheyInitialized = false;
  if (!WereTheyInitialized) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();

    WereTheyInitialized = true;
  }
}

using DI = LLVMDisassemblerInterface;
DI::LLVMDisassemblerInterface(MetaAddressType::Values AddrType) {
  ensureDisassemblersWereInitializedOnce();

  auto LLVMArchitecture = MetaAddressType::arch(AddrType);
  revng_assert(LLVMArchitecture.hasValue(),
               "Impossible to create a disassembler for a non-code section");
  auto Architecture = llvm::Triple::getArchTypeName(*LLVMArchitecture);

  // Workaround for ARM
  if (*LLVMArchitecture == llvm::Triple::ArchType::arm)
    Architecture = "armv7";

  std::string ErrorMessage;
  using Registry = llvm::TargetRegistry;
  auto *LLVMTarget = Registry::lookupTarget(Architecture.str(), ErrorMessage);
  revng_assert(LLVMTarget != nullptr, "Requested target is not available");

  llvm::StringRef CPUDefinition = "";
  llvm::StringRef CPUFeatures = MetaAddressType::getLLVMCPUFeatures(AddrType);
  SubtargetInformation.reset(LLVMTarget->createMCSubtargetInfo(Architecture,
                                                               CPUDefinition,
                                                               CPUFeatures));
  revng_assert(SubtargetInformation != nullptr,
               "Subtarget information object creation failed.");

  RegisterInformation.reset(LLVMTarget->createMCRegInfo(Architecture));
  revng_assert(RegisterInformation != nullptr,
               "Register information object creation failed.");

  llvm::MCTargetOptions TargetOptions;
  AssemblyInformation.reset(LLVMTarget->createMCAsmInfo(*RegisterInformation,
                                                        Architecture,
                                                        TargetOptions));
  revng_assert(AssemblyInformation != nullptr,
               "Assembly information object creation failed.");

  ObjectFileInformation = std::make_unique<llvm::MCObjectFileInfo>();
  Context = std::make_unique<llvm::MCContext>(AssemblyInformation.get(),
                                              RegisterInformation.get(),
                                              ObjectFileInformation.get());

  llvm::Triple Triple(Architecture);
  bool IsPIC = false;
  ObjectFileInformation->InitMCObjectFileInfo(Triple, IsPIC, *Context);

  auto &SI = *SubtargetInformation;
  Disassembler.reset(LLVMTarget->createMCDisassembler(SI, *Context));
  revng_assert(Disassembler != nullptr, "Disassembler object creation failed.");

  InstructionInformation.reset(LLVMTarget->createMCInstrInfo());

  unsigned AssemblyDialect = 0;
  if (*LLVMArchitecture == llvm::Triple::ArchType::x86
      || *LLVMArchitecture == llvm::Triple::ArchType::x86_64) {
    if (options::UseIntelSyntax == true)
      AssemblyDialect = 1;
  }

  Printer.reset(LLVMTarget->createMCInstPrinter(Triple,
                                                AssemblyDialect,
                                                *AssemblyInformation,
                                                *InstructionInformation,
                                                *RegisterInformation));
  revng_assert(Printer != nullptr, "Printer object creation failed.");

  using namespace options;
  if (ImmediateStyle == ImmediateStyles::Decimal) {
    Printer->setPrintImmHex(false);
    Printer->setPrintBranchImmAsAddress(false);
  } else {
    Printer->setPrintImmHex(true);
    Printer->setPrintBranchImmAsAddress(true);
  }

  if (ImmediateStyle == ImmediateStyles::CHexadecimal)
    Printer->setPrintHexStyle(llvm::HexStyle::C);
  else if (ImmediateStyle == ImmediateStyles::AsmHexadecimal)
    Printer->setPrintHexStyle(llvm::HexStyle::Asm);

  Printer->setSymbolizeOperands(ShouldSymbolizeOperands);
  Printer->setUseMarkup(true);
}

std::pair<std::optional<llvm::MCInst>, uint64_t>
DI::disassemble(const MetaAddress &Address,
                llvm::ArrayRef<uint8_t> RawBytes,
                const llvm::MCDisassembler &Disassembler) {
  llvm::MCInst Result;
  llvm::raw_null_ostream NullStream;
  uint64_t LocalSize = 0;
  auto ResultCode = Disassembler.getInstruction(Result,
                                                LocalSize,
                                                RawBytes,
                                                Address.asPC(),
                                                NullStream);
  if (LocalSize == 0)
    return { std::nullopt, 0 };

  switch (ResultCode) {
  case llvm::MCDisassembler::Success:
    return std::pair{ std::move(Result), LocalSize };
  case llvm::MCDisassembler::SoftFail:
    return std::pair{ std::nullopt, LocalSize };
  case llvm::MCDisassembler::Fail:
    return std::pair{ std::nullopt, 0 };
  default:
    revng_abort("Unsupported MCDisassembler::getInstruction result.");
  }
}

static assembly::Instruction::TagType parseMarkupTag(llvm::StringRef Input) {
  if (Input == "imm:")
    return assembly::Instruction::TagType::Immediate;
  else if (Input == "mem:")
    return assembly::Instruction::TagType::Memory;
  else if (Input == "reg:")
    return assembly::Instruction::TagType::Register;
  else
    revng_abort("Unknown llvm markup tag.");
}

/// Counts the number of consecutive characters satisfying \p Lambda predicate
/// in the \p String starting from \p StartFrom and going forwards.
template<typename LambdaType>
size_t getConsecutiveCount(llvm::StringRef String,
                           LambdaType Lambda,
                           size_t StartFrom = 0) {
  for (size_t Index = StartFrom; Index < String.size(); ++Index)
    if (!Lambda(String[Index]))
      return Index - StartFrom;

  return String.size() - StartFrom;
}

/// Counts the number of consecutive characters satisfying \p Lambda predicate
/// in the \p String starting from \p StartFrom and going backwards.
template<typename LambdaType>
size_t getBackwardsConsecutiveCount(llvm::StringRef String,
                                    LambdaType Lambda,
                                    size_t StartFrom) {
  for (size_t Index = StartFrom - 1; Index != size_t(-1); --Index)
    if (!Lambda(String[Index]))
      return StartFrom - Index - 1;

  return StartFrom;
}

static assembly::Instruction
makeInvalidInstruction(MetaAddress Where, size_t Size, std::string Reason) {
  assembly::Instruction Result{ .Address = Where };

  Result.Text = "(invalid)";
  Result.Tags.emplace_back(assembly::Instruction::TagType::Mnemonic, 0, 9);
  Result.Comment = std::to_string(Size) + " bytes";
  Result.Error = std::move(Reason);

  return Result;
}

static llvm::StringRef cleanStringUp(std::string &Input) {
  Input = llvm::StringRef(Input).trim().str();

  size_t Position = Input.find_first_of('\t');
  while (Position != std::string::npos) {
    Input.replace(Position, 1, " ");
    Position = Input.find_first_of('\t', Position);
  }

  return Input;
}

// TODO: this is but a temporary measure. LLVM MCInstPrinter needs to be
// patched.
constexpr std::array CommonlyMisdetectedMnemonics = {
  "mov",  "mvn",  "or",   "push", "pop", "cmp",   "cmn",  "asr",  "lsl",
  "lsr",  "ror",  "rrx",  "mul",  "neg", "sbfiz", "sbfx", "sxtb", "sxth",
  "sxtw", "cset", "cinc", "tst",  "nop", "b.",    "b"
};

struct DetectedMnemonic {
  size_t Position = llvm::StringRef::npos;
  size_t FullPosition = llvm::StringRef::npos;

  size_t Size = 0;
  size_t PrefixSize = 0;
  size_t SuffixSize = 0;
  size_t FullSize = 0;
};

static std::optional<DetectedMnemonic>
tryDetectMnemonic(llvm::StringRef Text, llvm::StringRef Mnemonic) {
  if (Mnemonic.empty())
    return std::nullopt;

  // Workaround for improper mnemonics being returned by the printer.
  // This explicitly limits them to only contain letters and numbers.
  auto AlphaNumCheck = [](char C) { return std::isalnum(C) || C == '.'; };
  size_t AlphaNumCount = getConsecutiveCount(Mnemonic, AlphaNumCheck);
  if (AlphaNumCount < Mnemonic.size())
    Mnemonic = Mnemonic.take_front(AlphaNumCount);
  if (Mnemonic.empty())
    return std::nullopt;

  DetectedMnemonic Result;

  Result.Position = Text.find(Mnemonic);
  bool WasMnemonicDetected = Result.Position != llvm::StringRef::npos;
  if (WasMnemonicDetected == false) {
    // Try to find one of the commonly misdetected mnemonics.
    // TODO: patch llvm's printers so that we no longer need such ugly solutions
    for (const auto &CommonMnemonic : CommonlyMisdetectedMnemonics) {
      Result.Position = Text.find(CommonMnemonic);
      if ((WasMnemonicDetected = (Result.Position != llvm::StringRef::npos))) {
        Mnemonic = CommonMnemonic;
        break;
      }
    }

    if (WasMnemonicDetected == false)
      return std::nullopt;
  }

  Result.Size = Mnemonic.size();
  Result.PrefixSize = getBackwardsConsecutiveCount(Text,
                                                   AlphaNumCheck,
                                                   Result.Position);
  Result.SuffixSize = getConsecutiveCount(Text,
                                          AlphaNumCheck,
                                          Result.Position + Result.Size);

  revng_assert(Result.Position >= Result.PrefixSize);
  Result.FullPosition = Result.Position - Result.PrefixSize;
  Result.FullSize = Result.Size + Result.PrefixSize + Result.SuffixSize;

  return Result;
}

assembly::Instruction DI::parse(const llvm::MCInst &Instruction,
                                const MetaAddress &Address,
                                size_t InstructionSize,
                                llvm::MCInstPrinter &Printer,
                                const llvm::MCSubtargetInfo &SI) {
  assembly::Instruction Result{ .Address = Address };

  // Save the opcode for future use.
  if (auto Opcode = Printer.getOpcodeName(Instruction.getOpcode());
      !Opcode.empty())
    Result.Opcode = Opcode.str();

  std::string MarkupStorage;
  llvm::raw_string_ostream MarkupStream(MarkupStorage);

  // Some special considerations might be needed for the second operand.
  // See the `MCInstPrinter::printInst()` docs.
  Printer.printInst(&Instruction, Address.asPC(), "", SI, MarkupStream);

  if (MarkupStorage.empty())
    return Result;

  llvm::StringRef Markup = cleanStringUp(MarkupStorage);
  auto Mnemonic = tryDetectMnemonic(Markup,
                                    Printer.getMnemonic(&Instruction).first);
  if (!Mnemonic.has_value())
    Result.Error = "Impossible to detect mnemonic.";

  auto WhitespaceCheck = [](char C) {
    constexpr llvm::StringRef Whitespaces = " \t\n\v\f\r";
    return Whitespaces.contains(C);
  };

  const auto &Info = InstructionInformation->get(Instruction.getOpcode());
  Result.HasDelayedSlot = Info.hasDelaySlot();

  // Investigate the llvm-provided tags.
  constexpr llvm::StringRef TagBoundaries = "<>";
  llvm::SmallVector<size_t, 8> OpenTagStack;
  for (size_t Position = 0; Position < Markup.size(); ++Position) {
    // Mark the whitespaces so that the client can easily remove them if needed.
    size_t WhitespaceCount = getConsecutiveCount(Markup,
                                                 WhitespaceCheck,
                                                 Position);
    if (WhitespaceCount != 0) {
      Result.Tags.emplace_back(assembly::Instruction::TagType::Whitespace,
                               Result.Text.size(),
                               Result.Text.size() + WhitespaceCount);
      Result.Text += Markup.substr(Position, WhitespaceCount);
      Position += WhitespaceCount - 1;
      continue;
    }

    if (Markup[Position] == '<') {
      // Opens a new markup tag.
      llvm::StringRef Tag = Markup.slice(Position + 1, Position + 5);
      assembly::Instruction::TagType TagType = parseMarkupTag(Tag);
      OpenTagStack.emplace_back(Result.Tags.size());
      Result.Tags.emplace_back(TagType, Result.Text.size());
      Position += 4;
    } else if (Markup[Position] == '>') {
      // Closes the current markup tag.
      if (OpenTagStack.empty() || OpenTagStack.back() >= Result.Tags.size()) {
        Result.Error = "Tag stack got corrupted, impossible to provide markup.";
        Result.Tags.clear();
        break;
      }

      assembly::Instruction::Tag &CurrentTag = Result.Tags[OpenTagStack.back()];
      CurrentTag.ToIndex = Result.Text.size();
      OpenTagStack.pop_back();
    } else if (Mnemonic.has_value() && Position == Mnemonic->FullPosition) {
      // Mnemonic
      if (!OpenTagStack.empty()) {
        Result.Error = "Mnemonic could not be detected correctly";
        Result.Text += Markup[Position];
        continue;
      }

      size_t MnemonicStart = Result.Text.size();
      size_t MnemonicPrefixEnd = MnemonicStart + Mnemonic->PrefixSize;
      size_t MnemonifSuffixStart = MnemonicPrefixEnd + Mnemonic->Size;
      size_t MnemonicEnd = MnemonifSuffixStart + Mnemonic->SuffixSize;

      Result.Tags.emplace_back(assembly::Instruction::TagType::Mnemonic,
                               MnemonicStart,
                               MnemonicEnd);
      if (Mnemonic->PrefixSize != 0)
        Result.Tags.emplace_back(assembly::Instruction::TagType::MnemonicPrefix,
                                 MnemonicStart,
                                 MnemonicPrefixEnd);
      if (Mnemonic->SuffixSize != 0)
        Result.Tags.emplace_back(assembly::Instruction::TagType::MnemonicSuffix,
                                 MnemonifSuffixStart,
                                 MnemonicEnd);

      Result.Text += Markup.substr(Mnemonic->FullPosition, Mnemonic->FullSize);
      Position += Mnemonic->FullSize - 1;
    } else {
      // Nothing special, just a character.
      Result.Text += Markup[Position];
    }
  }

  if (!OpenTagStack.empty())
    Result.Error = "A tag doesn't have a closing bracket.";

  return Result;
}

std::pair<assembly::Instruction, uint64_t>
DI::instruction(const MetaAddress &Where, llvm::ArrayRef<uint8_t> RawBytes) {
  revng_assert(Where.isValid() && !RawBytes.empty());

  auto [Instruction, Size] = disassemble(Where, RawBytes, *Disassembler);
  if (Instruction.has_value()) {
    revng_assert(Size != 0);
    auto P = parse(*Instruction, Where, Size, *Printer, *SubtargetInformation);
    return std::pair{ std::move(P), Size };
  } else {
    if (Size == 0)
      Size = RawBytes.size();
    return std::pair{
      makeInvalidInstruction(Where, Size, "MCDisassembler failed"), Size
    };
  }
}
