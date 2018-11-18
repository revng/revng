/// \file binaryfile.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <string>
#include <tuple>
#include <utility>

// LLVM includes
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"

// Local libraries includes
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"

// Local includes
#include "BinaryFile.h"

// using directives
using namespace llvm;

using std::make_pair;

using LabelList = BinaryFile::LabelList;

static Logger<> EhFrameLog("ehframe");
static Logger<> LabelsLog("labels");

const unsigned char R_MIPS_IMPLICIT_RELATIVE = 255;

BinaryFile::BinaryFile(std::string FilePath, uint64_t BaseAddress) :
  BaseAddress(0) {
  auto BinaryOrErr = object::createBinary(FilePath);
  revng_assert(BinaryOrErr, "Couldn't open the input file");

  BinaryHandle = std::move(BinaryOrErr.get());

  auto *TheBinary = cast<object::ObjectFile>(BinaryHandle.getBinary());

  // TODO: QEMU should provide this information
  uint32_t InstructionAlignment = 0;
  StringRef SyscallHelper = "";
  StringRef SyscallNumberRegister = "";
  StringRef StackPointerRegister = "";
  ArrayRef<uint64_t> NoReturnSyscalls = {};
  SmallVector<ABIRegister, 20> ABIRegisters;
  uint32_t DelaySlotSize = 0;
  unsigned PCMContextIndex = ABIRegister::NotInMContext;
  llvm::StringRef WriteRegisterAsm = "";
  llvm::StringRef ReadRegisterAsm = "";
  llvm::StringRef JumpAsm = "";
  bool HasRelocationAddend;

  using RD = RelocationDescription;
  using namespace llvm::ELF;
  Architecture::RelocationTypesMap RelocationTypes;

  switch (TheBinary->getArch()) {
  case Triple::x86:
    InstructionAlignment = 1;
    SyscallHelper = "helper_raise_interrupt";
    SyscallNumberRegister = "eax";
    StackPointerRegister = "esp";
    NoReturnSyscalls = {
      0xfc, // exit_group
      0x01, // exit
      0x0b // execve
    };

    HasRelocationAddend = false;

    RelocationTypes[R_386_RELATIVE] = RD(RD::BaseRelative, RD::TargetValue);
    RelocationTypes[R_386_JUMP_SLOT] = RD(RD::SymbolRelative);
    RelocationTypes[R_386_GLOB_DAT] = RD(RD::SymbolRelative);
    RelocationTypes[R_386_COPY] = RD(RD::LabelOnly, RD::TargetValue);
    break;

  case Triple::x86_64:
    InstructionAlignment = 1;
    SyscallHelper = "helper_syscall";
    SyscallNumberRegister = "rax";
    StackPointerRegister = "rsp";
    NoReturnSyscalls = {
      0xe7, // exit_group
      0x3c, // exit
      0x3b // execve
    };
    PCMContextIndex = 0x10;

    // The offsets associated to the registers have been obtained running the
    // following command:
    //
    // scripts/compile-time-constants.py gcc ucontext.c
    //
    // where `ucontext.c` is:
    //
    // #define _GNU_SOURCE
    // #include <sys/ucontext.h>
    // #include <stdint.h>
    //
    // static ucontext_t UContext;
    //
    // #define REGISTER_OFFSET(reg) const int MContextIndex ## reg = REG_ ## reg
    //
    // REGISTER_OFFSET(R8);
    // REGISTER_OFFSET(R9);
    // REGISTER_OFFSET(R10);
    // REGISTER_OFFSET(R11);
    // REGISTER_OFFSET(R12);
    // REGISTER_OFFSET(R13);
    // REGISTER_OFFSET(R14);
    // REGISTER_OFFSET(R15);
    // REGISTER_OFFSET(RDI);
    // REGISTER_OFFSET(RSI);
    // REGISTER_OFFSET(RBP);
    // REGISTER_OFFSET(RBX);
    // REGISTER_OFFSET(RDX);
    // REGISTER_OFFSET(RAX);
    // REGISTER_OFFSET(RCX);
    // REGISTER_OFFSET(RSP);
    // REGISTER_OFFSET(RIP);

    // TODO: here we're hardcoding the offsets in the QEMU struct
    ABIRegisters = { { "rax", 0xD },
                     { "rbx", 0xB },
                     { "rcx", 0xE },
                     { "rdx", 0xC },
                     { "rbp", 0xA },
                     { "rsp", 0xF },
                     { "rsi", 0x9 },
                     { "rdi", 0x8 },
                     { "r8", 0x0 },
                     { "r9", 0x1 },
                     { "r10", 0x2 },
                     { "r11", 0x3 },
                     { "r12", 0x4 },
                     { "r13", 0x5 },
                     { "r14", 0x6 },
                     { "r15", 0x7 },
                     { "xmm0", "state_0x8558" },
                     { "xmm1", "state_0x8598" },
                     { "xmm2", "state_0x85d8" },
                     { "xmm3", "state_0x8618" },
                     { "xmm4", "state_0x8658" },
                     { "xmm5", "state_0x8698" },
                     { "xmm6", "state_0x86d8" },
                     { "xmm7", "state_0x8718" } };
    WriteRegisterAsm = "movq $0, %REGISTER";
    ReadRegisterAsm = "movq %REGISTER, $0";
    JumpAsm = "movq $0, %r11; jmpq *%r11";

    HasRelocationAddend = true;

    RelocationTypes[R_X86_64_RELATIVE] = RD(RD::BaseRelative, RD::Addend);
    RelocationTypes[R_X86_64_JUMP_SLOT] = RD(RD::SymbolRelative);
    RelocationTypes[R_X86_64_GLOB_DAT] = RD(RD::SymbolRelative);
    RelocationTypes[R_X86_64_COPY] = RD(RD::LabelOnly, RD::TargetValue);
    break;

  case Triple::arm:
    InstructionAlignment = 4;
    SyscallHelper = "helper_exception_with_syndrome";
    SyscallNumberRegister = "r7";
    StackPointerRegister = "r13";
    NoReturnSyscalls = {
      0xf8, // exit_group
      0x1, // exit
      0xb // execve
    };
    ABIRegisters = { { "r0" },  { "r1" },  { "r2" },  { "r3" },  { "r4" },
                     { "r5" },  { "r6" },  { "r7" },  { "r8" },  { "r9" },
                     { "r10" }, { "r11" }, { "r12" }, { "r13" }, { "r14" } };

    HasRelocationAddend = false;

    RelocationTypes[R_ARM_RELATIVE] = RD(RD::BaseRelative, RD::TargetValue);
    RelocationTypes[R_ARM_JUMP_SLOT] = RD(RD::SymbolRelative);
    RelocationTypes[R_ARM_GLOB_DAT] = RD(RD::SymbolRelative);
    RelocationTypes[R_ARM_COPY] = RD(RD::LabelOnly, RD::TargetValue);
    break;

  case Triple::mips:
  case Triple::mipsel:
    InstructionAlignment = 4;
    SyscallHelper = "helper_raise_exception";
    SyscallNumberRegister = "v0";
    StackPointerRegister = "sp";
    NoReturnSyscalls = {
      0x1096, // exit_group
      0xfa1, // exit
      0xfab // execve
    };
    DelaySlotSize = 1;
    ABIRegisters = {
      { "v0" }, { "v1" }, { "a0" }, { "a1" }, { "a2" }, { "a3" },
      { "s0" }, { "s1" }, { "s2" }, { "s3" }, { "s4" }, { "s5" },
      { "s6" }, { "s7" }, { "gp" }, { "sp" }, { "fp" }, { "ra" }
    };

    HasRelocationAddend = false;

    // R_MIPS_RELATIVE does not exist since the GOT has implicit base-relative
    // relocations
    RelocationTypes[R_MIPS_IMPLICIT_RELATIVE] = RD(RD::BaseRelative,
                                                   RD::TargetValue);
    RelocationTypes[R_MIPS_JUMP_SLOT] = RD(RD::SymbolRelative);
    RelocationTypes[R_MIPS_GLOB_DAT] = RD(RD::SymbolRelative);
    RelocationTypes[R_MIPS_COPY] = RD(RD::LabelOnly, RD::TargetValue);
    break;

  case Triple::systemz:
    SyscallHelper = "helper_exception";
    SyscallNumberRegister = "r1";
    StackPointerRegister = "r15";
    InstructionAlignment = 2;
    NoReturnSyscalls = {
      0xf8, // exit_group
      0x1, // exit
      0xb, // execve
    };

    HasRelocationAddend = true;

    // TODO: investigate (R_390_RELATIVE does not exist)
    RelocationTypes[R_390_GLOB_DAT] = RD(RD::SymbolRelative);
    RelocationTypes[R_390_COPY] = RD(RD::LabelOnly, RD::TargetValue);
    break;

  default:
    revng_abort();
  }

  TheArchitecture = Architecture(TheBinary->getArch(),
                                 InstructionAlignment,
                                 1,
                                 TheBinary->isLittleEndian(),
                                 TheBinary->getBytesInAddress() * 8,
                                 SyscallHelper,
                                 SyscallNumberRegister,
                                 NoReturnSyscalls,
                                 DelaySlotSize,
                                 StackPointerRegister,
                                 ABIRegisters,
                                 PCMContextIndex,
                                 WriteRegisterAsm,
                                 ReadRegisterAsm,
                                 JumpAsm,
                                 HasRelocationAddend,
                                 std::move(RelocationTypes));

  revng_assert(TheBinary->getFileFormatName().startswith("ELF"),
               "Only the ELF file format is currently supported");

  if (TheArchitecture.pointerSize() == 32) {
    if (TheArchitecture.isLittleEndian()) {
      if (TheArchitecture.hasRelocationAddend()) {
        parseELF<object::ELF32LE, true>(TheBinary, BaseAddress);
      } else {
        parseELF<object::ELF32LE, false>(TheBinary, BaseAddress);
      }
    } else {
      if (TheArchitecture.hasRelocationAddend()) {
        parseELF<object::ELF32BE, true>(TheBinary, BaseAddress);
      } else {
        parseELF<object::ELF32BE, false>(TheBinary, BaseAddress);
      }
    }
  } else if (TheArchitecture.pointerSize() == 64) {
    if (TheArchitecture.isLittleEndian()) {
      if (TheArchitecture.hasRelocationAddend()) {
        parseELF<object::ELF64LE, true>(TheBinary, BaseAddress);
      } else {
        parseELF<object::ELF64LE, false>(TheBinary, BaseAddress);
      }
    } else {
      if (TheArchitecture.hasRelocationAddend()) {
        parseELF<object::ELF64BE, true>(TheBinary, BaseAddress);
      } else {
        parseELF<object::ELF64BE, false>(TheBinary, BaseAddress);
      }
    }
  } else {
    revng_assert("Unexpect address size");
  }

  rebuildLabelsMap();
}

class FilePortion {
private:
  bool HasAddress;
  bool HasSize;
  uint64_t Size;
  uint64_t Address;

public:
  FilePortion() : HasAddress(false), HasSize(false), Size(0), Address(0) {}

public:
  void setAddress(uint64_t Address) {
    HasAddress = true;
    this->Address = Address;
  }

  void setSize(uint64_t Size) {
    HasSize = true;
    this->Size = Size;
  }

  uint64_t addressAtOffset(uint64_t Offset) {
    revng_assert(HasAddress and HasSize);
    revng_assert(Offset <= Size);
    return Address + Offset;
  }

  template<typename T>
  uint64_t addressAtIndex(uint64_t Index) {
    revng_assert(HasAddress and HasSize);
    uint64_t Offset = Index * sizeof(T);
    revng_assert(Offset <= Size);
    return Address + Offset;
  }

  bool isAvailable() const { return HasAddress; }

  bool isExact() const {
    revng_assert(HasAddress);
    return HasSize;
  }

  StringRef extractString(const std::vector<SegmentInfo> &Segments) const {
    ArrayRef<uint8_t> Data = extractData(Segments);
    const char *AsChar = reinterpret_cast<const char *>(Data.data());
    return StringRef(AsChar, Data.size());
  }

  template<typename T>
  ArrayRef<T> extractAs(const std::vector<SegmentInfo> &Segments) const {
    ArrayRef<uint8_t> Data = extractData(Segments);
    const size_t TypeSize = sizeof(T);
    revng_assert(Data.size() % TypeSize == 0);
    return ArrayRef<T>(reinterpret_cast<const T *>(Data.data()),
                       Data.size() / TypeSize);
  }

  ArrayRef<uint8_t>
  extractData(const std::vector<SegmentInfo> &Segments) const {
    revng_assert(HasAddress);

    for (const SegmentInfo &Segment : Segments) {
      if (Segment.contains(Address)) {
        uint64_t Offset = Address - Segment.StartVirtualAddress;
        uint64_t AvailableSize = Segment.size() - Offset;
        uint64_t TheSize = AvailableSize;

        if (HasSize) {
          revng_assert(AvailableSize >= Size);
          TheSize = Size;
        }

        return { ArrayRef<uint8_t>(Segment.Data.data() + Offset, TheSize) };
      }
    }

    revng_abort();
  }
};

template<typename T, bool HasAddend>
struct RelocationHelper {
  static uint64_t getAddend(llvm::object::Elf_Rel_Impl<T, HasAddend>);
};

template<typename T>
struct RelocationHelper<T, true> {
  static uint64_t getAddend(llvm::object::Elf_Rel_Impl<T, true> Relocation) {
    return Relocation.r_addend;
  }
};

template<typename T>
struct RelocationHelper<T, false> {
  static uint64_t getAddend(llvm::object::Elf_Rel_Impl<T, false> Relocation) {
    return 0;
  }
};

static bool shouldIgnoreSymbol(StringRef Name) {
  return Name == "$a" or Name == "$d";
}

template<typename T, bool HasAddend>
void BinaryFile::parseELF(object::ObjectFile *TheBinary, uint64_t BaseAddress) {
  // Parse the ELF file
  auto TheELFOrErr = object::ELFFile<T>::create(TheBinary->getData());
  if (not TheELFOrErr) {
    logAllUnhandledErrors(std::move(TheELFOrErr.takeError()), errs(), "");
    revng_abort();
  }
  object::ELFFile<T> TheELF = *TheELFOrErr;

  // BaseAddress makes sense only for shared (relocatable, PIC) objects
  if (TheELF.getHeader()->e_type == ELF::ET_DYN)
    this->BaseAddress = BaseAddress;

  // Look for static or dynamic symbols and relocations
  using ConstElf_ShdrPtr = const typename object::ELFFile<T>::Elf_Shdr *;
  using Elf_PhdrPtr = const typename object::ELFFile<T>::Elf_Phdr *;
  ConstElf_ShdrPtr SymtabShdr = nullptr;
  Elf_PhdrPtr DynamicPhdr = nullptr;
  Optional<uint64_t> DynamicAddress;
  Optional<uint64_t> EHFrameAddress;
  Optional<uint64_t> EHFrameSize;
  Optional<uint64_t> EHFrameHdrAddress;

  auto Sections = TheELF.sections();
  if (not Sections) {
    logAllUnhandledErrors(std::move(Sections.takeError()), errs(), "");
  } else {
    for (auto &Section : *Sections) {
      auto NameOrErr = TheELF.getSectionName(&Section);
      if (NameOrErr) {
        auto &Name = *NameOrErr;
        if (Name == ".symtab") {
          // TODO: check dedicated field in section header
          revng_assert(SymtabShdr == nullptr, "Duplicate .symtab");
          SymtabShdr = &Section;
        } else if (Name == ".eh_frame") {
          revng_assert(not EHFrameAddress, "Duplicate .eh_frame");
          EHFrameAddress = relocate(static_cast<uint64_t>(Section.sh_addr));
          EHFrameSize = static_cast<uint64_t>(Section.sh_size);
        } else if (Name == ".dynamic") {
          revng_assert(not DynamicAddress, "Duplicate .dynamic");
          DynamicAddress = relocate(static_cast<uint64_t>(Section.sh_addr));
        }
      }
    }
  }

  // If we found a symbol table
  if (SymtabShdr != nullptr && SymtabShdr->sh_link != 0) {
    // Obtain a reference to the string table
    auto Strtab = TheELF.getSection(SymtabShdr->sh_link);
    if (not Strtab) {
      logAllUnhandledErrors(std::move(Strtab.takeError()), errs(), "");
      revng_abort();
    }
    auto StrtabArray = TheELF.getSectionContents(*Strtab);
    if (not StrtabArray) {
      logAllUnhandledErrors(std::move(StrtabArray.takeError()), errs(), "");
      revng_abort();
    }
    StringRef StrtabContent(reinterpret_cast<const char *>(StrtabArray->data()),
                            StrtabArray->size());

    // Collect symbol names
    auto ELFSymbols = TheELF.symbols(SymtabShdr);
    if (not ELFSymbols) {
      logAllUnhandledErrors(std::move(ELFSymbols.takeError()), errs(), "");
      revng_abort();
    }
    for (auto &Symbol : *ELFSymbols) {
      auto Name = Symbol.getName(StrtabContent);
      if (not Name) {
        logAllUnhandledErrors(std::move(Name.takeError()), errs(), "");
        revng_abort();
      }

      if (shouldIgnoreSymbol(*Name))
        continue;

      registerLabel(Label::createSymbol(LabelOrigin::StaticSymbol,
                                        Symbol.st_value,
                                        Symbol.st_size,
                                        *Name,
                                        SymbolType::fromELF(Symbol.getType())));
    }
  }

  const auto *ElfHeader = TheELF.getHeader();
  EntryPoint = relocate(static_cast<uint64_t>(ElfHeader->e_entry));
  ProgramHeaders.Count = ElfHeader->e_phnum;
  ProgramHeaders.Size = ElfHeader->e_phentsize;

  // Loop over the program headers looking for PT_LOAD segments, read them out
  // and create a global variable for each one of them (writable or read-only),
  // assign them a section and output information about them in the linking info
  // CSV
  using Elf_Phdr = const typename object::ELFFile<T>::Elf_Phdr;
  using Elf_Dyn = const typename object::ELFFile<T>::Elf_Dyn;
  using Elf_Addr = const typename object::ELFFile<T>::Elf_Addr;

  auto ProgHeaders = TheELF.program_headers();
  if (not ProgHeaders) {
    logAllUnhandledErrors(std::move(ProgHeaders.takeError()), errs(), "");
    revng_abort();
  }
  for (Elf_Phdr &ProgramHeader : *ProgHeaders) {
    switch (ProgramHeader.p_type) {
    case ELF::PT_LOAD: {
      SegmentInfo Segment;
      auto Start = relocate(ProgramHeader.p_vaddr);
      Segment.StartVirtualAddress = Start;
      Segment.EndVirtualAddress = Start + ProgramHeader.p_memsz;
      Segment.IsReadable = ProgramHeader.p_flags & ELF::PF_R;
      Segment.IsWriteable = ProgramHeader.p_flags & ELF::PF_W;
      Segment.IsExecutable = ProgramHeader.p_flags & ELF::PF_X;

      auto ActualAddress = TheELF.base() + ProgramHeader.p_offset;
      Segment.Data = ArrayRef<uint8_t>(ActualAddress, ProgramHeader.p_filesz);

      // If it's an executable segment, and we've been asked so, register
      // which sections actually contain code
      if (Sections and UseDebugSymbols and Segment.IsExecutable) {
        using Elf_Shdr = const typename object::ELFFile<T>::Elf_Shdr;
        auto Inserter = std::back_inserter(Segment.ExecutableSections);
        for (Elf_Shdr &SectionHeader : *Sections) {
          if (SectionHeader.sh_flags & ELF::SHF_EXECINSTR) {
            auto SectionStart = relocate(SectionHeader.sh_addr);
            auto SectionEnd = SectionStart + SectionHeader.sh_size;
            Inserter = make_pair(SectionStart, SectionEnd);
          }
        }
      }

      Segments.push_back(Segment);

      // Check if it's the segment containing the program headers
      auto ProgramHeaderStart = ProgramHeader.p_offset;
      auto ProgramHeaderEnd = ProgramHeader.p_offset + ProgramHeader.p_filesz;
      if (ProgramHeaderStart <= ElfHeader->e_phoff
          && ElfHeader->e_phoff < ProgramHeaderEnd) {
        uint64_t PhdrAddress = (relocate(ProgramHeader.p_vaddr)
                                + ElfHeader->e_phoff - ProgramHeader.p_offset);
        ProgramHeaders.Address = PhdrAddress;
      }
    } break;

    case ELF::PT_GNU_EH_FRAME:
      revng_assert(!EHFrameHdrAddress);
      EHFrameHdrAddress = relocate(ProgramHeader.p_vaddr);
      break;

    case ELF::PT_DYNAMIC:
      revng_assert(DynamicPhdr == nullptr, "Duplicate .dynamic program header");
      DynamicPhdr = &ProgramHeader;
      revng_assert(((not DynamicAddress)
                    or (relocate(DynamicPhdr->p_vaddr) == *DynamicAddress)),
                   ".dynamic and PT_DYNAMIC have different addresses");
      DynamicAddress = relocate(DynamicPhdr->p_vaddr);
      break;
    }
  }

  revng_assert((DynamicPhdr != nullptr) == (DynamicAddress.hasValue()));

  Optional<uint64_t> FDEsCount;
  if (EHFrameHdrAddress) {
    uint64_t Address;

    std::tie(Address, FDEsCount) = ehFrameFromEhFrameHdr<T>(*EHFrameHdrAddress);
    if (EHFrameAddress) {
      revng_assert(*EHFrameAddress == Address);
    }

    EHFrameAddress = Address;
  }

  if (EHFrameAddress)
    parseEHFrame<T>(*EHFrameAddress, FDEsCount, EHFrameSize);

  // Parse the .dynamic table
  if (DynamicPhdr != nullptr) {
    SmallVector<uint64_t, 10> NeededLibraryNameOffsets;

    FilePortion DynstrPortion;
    FilePortion DynsymPortion;
    FilePortion ReldynPortion;
    FilePortion RelpltPortion;
    FilePortion GotPortion;
    Optional<uint64_t> SymbolsCount;
    Optional<uint64_t> MIPSFirstGotSymbol;
    Optional<uint64_t> MIPSLocalGotEntries;
    bool IsMIPS = (TheArchitecture.type() == Triple::mips
                   or TheArchitecture.type() == Triple::mipsel);

    auto DynamicEntries = TheELF.dynamicEntries();
    if (not DynamicEntries) {
      logAllUnhandledErrors(std::move(DynamicEntries.takeError()), errs(), "");
      revng_abort();
    }
    for (Elf_Dyn &DynamicTag : *DynamicEntries) {

      auto TheTag = DynamicTag.getTag();
      switch (TheTag) {
      case ELF::DT_NEEDED:
        NeededLibraryNameOffsets.push_back(DynamicTag.getVal());
        break;

      case ELF::DT_STRTAB:
        DynstrPortion.setAddress(relocate(DynamicTag.getPtr()));
        break;

      case ELF::DT_STRSZ:
        DynstrPortion.setSize(DynamicTag.getVal());
        break;

      case ELF::DT_SYMTAB:
        DynsymPortion.setAddress(relocate(DynamicTag.getPtr()));
        break;

      case ELF::DT_JMPREL:
        RelpltPortion.setAddress(relocate(DynamicTag.getPtr()));
        break;

      case ELF::DT_PLTRELSZ:
        RelpltPortion.setSize(DynamicTag.getVal());
        break;

      case ELF::DT_REL:
      case ELF::DT_RELA:
        revng_assert(TheTag == (HasAddend ? ELF::DT_RELA : ELF::DT_REL));
        ReldynPortion.setAddress(relocate(DynamicTag.getPtr()));
        break;

      case ELF::DT_RELSZ:
      case ELF::DT_RELASZ:
        revng_assert(TheTag == (HasAddend ? ELF::DT_RELASZ : ELF::DT_RELSZ));
        ReldynPortion.setSize(DynamicTag.getVal());
        break;

      case ELF::DT_PLTGOT:
        GotPortion.setAddress(relocate(DynamicTag.getPtr()));

        // Obtaint the canonical value of the global pointer in MIPS
        if (IsMIPS)
          CanonicalValues["gp"] = relocate(DynamicTag.getPtr() + 0x7ff0);
        break;

      case ELF::DT_MIPS_SYMTABNO:
        if (IsMIPS)
          SymbolsCount = DynamicTag.getVal();
        break;

      case ELF::DT_MIPS_GOTSYM:
        if (IsMIPS)
          MIPSFirstGotSymbol = DynamicTag.getVal();
        break;

      case ELF::DT_MIPS_LOCAL_GOTNO:
        if (IsMIPS)
          MIPSLocalGotEntries = DynamicTag.getVal();
        break;
      }
    }

    if (NeededLibraryNames.size() > 0)
      revng_assert(DynstrPortion.isAvailable());

    // In MIPS the GOT has one entry per symbol
    if (IsMIPS and SymbolsCount and MIPSFirstGotSymbol
        and MIPSLocalGotEntries) {
      uint32_t GotEntries = (*MIPSLocalGotEntries
                             + (*SymbolsCount - *MIPSFirstGotSymbol));
      GotPortion.setSize(GotEntries * sizeof(Elf_Addr));
    }

    StringRef Dynstr;

    if (DynstrPortion.isAvailable()) {
      Dynstr = DynstrPortion.extractString(Segments);
      for (auto Offset : NeededLibraryNameOffsets) {
        StringRef LibraryName = Dynstr.slice(Offset, Dynstr.size());
        NeededLibraryNames.push_back(LibraryName.data());
      }
    }

    // Collect symbols count and code pointers in image base-relative
    // relocations

    if (not SymbolsCount) {
      SymbolsCount = std::max(symbolsCount<T, HasAddend>(ReldynPortion),
                              symbolsCount<T, HasAddend>(RelpltPortion));
    }

    // Collect function addresses contained in dynamic symbols
    if (SymbolsCount and *SymbolsCount > 0 and DynsymPortion.isAvailable()) {
      using Elf_Sym = llvm::object::Elf_Sym_Impl<T>;
      DynsymPortion.setSize(*SymbolsCount * sizeof(Elf_Sym));
      ArrayRef<Elf_Sym> Symbols = DynsymPortion.extractAs<Elf_Sym>(Segments);
      for (Elf_Sym Symbol : Symbols) {
        auto Name = Symbol.getName(Dynstr);
        if (not Name) {
          logAllUnhandledErrors(std::move(Name.takeError()), errs(), "");
          revng_abort();
        }

        if (shouldIgnoreSymbol(*Name))
          continue;

        auto SymbolType = SymbolType::fromELF(Symbol.getType());
        registerLabel(Label::createSymbol(LabelOrigin::DynamicSymbol,
                                          Symbol.st_value,
                                          Symbol.st_size,
                                          *Name,
                                          SymbolType));
      }

      using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;
      if (ReldynPortion.isAvailable()) {
        auto Relocations = ReldynPortion.extractAs<Elf_Rel>(Segments);
        registerRelocations<T, HasAddend>(Relocations,
                                          DynsymPortion,
                                          DynstrPortion);
      }

      if (RelpltPortion.isAvailable()) {
        auto Relocations = RelpltPortion.extractAs<Elf_Rel>(Segments);
        registerRelocations<T, HasAddend>(Relocations,
                                          DynsymPortion,
                                          DynstrPortion);
      }

      if (IsMIPS and GotPortion.isAvailable()) {
        std::vector<Elf_Rel> MIPSImplicitRelocations;
        uint32_t GotIndex = 0;

        // Perform local relocations on GOT
        if (MIPSLocalGotEntries) {
          for (; GotIndex < *MIPSLocalGotEntries; GotIndex++) {
            auto Address = GotPortion.addressAtIndex<Elf_Addr>(GotIndex);
            Elf_Rel NewRelocation;
            NewRelocation.r_offset = Address;
            NewRelocation.setSymbolAndType(0, R_MIPS_IMPLICIT_RELATIVE, false);
            MIPSImplicitRelocations.push_back(NewRelocation);
          }
        }

        // Relocate the remaining entries of the GOT with global symbols
        if (MIPSFirstGotSymbol and SymbolsCount and DynstrPortion.isAvailable()
            and DynsymPortion.isAvailable()) {
          for (uint32_t SymbolIndex = *MIPSFirstGotSymbol;
               SymbolIndex < *SymbolsCount;
               SymbolIndex++, GotIndex++) {
            auto Address = GotPortion.addressAtIndex<Elf_Addr>(GotIndex);

            Elf_Rel NewRelocation;
            NewRelocation.r_offset = Address;
            NewRelocation.setSymbolAndType(SymbolIndex,
                                           llvm::ELF::R_MIPS_JUMP_SLOT,
                                           false);
            MIPSImplicitRelocations.push_back(NewRelocation);
          }
        }

        auto Relocations = ArrayRef<Elf_Rel>(MIPSImplicitRelocations);
        registerRelocations<T, HasAddend>(Relocations,
                                          DynsymPortion,
                                          DynstrPortion);
      }

      for (Label &L : Labels) {
        if (L.isSymbol() and L.isCode())
          CodePointers.insert(relocate(L.address()));
        else if (L.isBaseRelativeValue())
          CodePointers.insert(relocate(L.value()));
      }
    }
  }
}

template<typename T, bool HasAddend>
uint64_t BinaryFile::symbolsCount(const FilePortion &Relocations) {
  using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;

  if (not Relocations.isAvailable())
    return 0;

  uint32_t SymbolsCount = 0;
  revng_assert(Relocations.isExact());
  for (Elf_Rel Relocation : Relocations.extractAs<Elf_Rel>(Segments))
    SymbolsCount = std::max(SymbolsCount, Relocation.getSymbol(false) + 1);

  return SymbolsCount;
}

Optional<uint64_t>
BinaryFile::readRawValue(uint64_t Address, unsigned Size, Endianess E) const {
  bool IsLittleEndian = ((E == OriginalEndianess) ?
                           architecture().isLittleEndian() :
                           E == LittleEndian);

  for (auto &Segment : segments()) {
    // Note: we also consider writeable memory areas because, despite being
    // modifiable, can contain useful information
    if (Segment.contains(Address, Size) && Segment.IsReadable) {
      uint64_t Offset = Address - Segment.StartVirtualAddress;
      // Handle the [p_filesz, p_memsz] portion of the segment
      if (Offset > Segment.Data.size())
        return 0;

      const unsigned char *Start = Segment.Data.data() + Offset;

      using support::endianness;
      using support::endian::read;
      switch (Size) {
      case 1:
        return read<uint8_t, endianness::little, 1>(Start);
      case 2:
        if (IsLittleEndian)
          return read<uint16_t, endianness::little, 1>(Start);
        else
          return read<uint16_t, endianness::big, 1>(Start);
      case 4:
        if (IsLittleEndian)
          return read<uint32_t, endianness::little, 1>(Start);
        else
          return read<uint32_t, endianness::big, 1>(Start);
      case 8:
        if (IsLittleEndian)
          return read<uint64_t, endianness::little, 1>(Start);
        else
          return read<uint64_t, endianness::big, 1>(Start);
      default:
        revng_abort("Unexpected read size");
      }
    }
  }

  return Optional<uint64_t>();
}

Label BinaryFile::parseRelocation(unsigned char RelocationType,
                                  uint64_t Target,
                                  uint64_t Addend,
                                  StringRef SymbolName,
                                  uint64_t SymbolSize,
                                  SymbolType::Values SymbolType) {

  const auto &RelocationTypes = TheArchitecture.relocationTypes();
  auto It = RelocationTypes.find(RelocationType);
  if (It == RelocationTypes.end()) {
    dbg << "Warning: unhandled relocation type " << (int) RelocationType
        << "\n";
    return Label::createInvalid();
  }

  uint64_t Offset;

  using RD = RelocationDescription;
  const RD &Description = It->second;
  uint64_t PointerSize = TheArchitecture.pointerSize() / 8;

  switch (Description.Offset) {
  case RD::None:
    Offset = 0;
    break;

  case RD::Addend:
    Offset = Addend;
    break;

  case RD::TargetValue:
    Optional<uint64_t> ReadResult = readRawValue(Target, PointerSize);
    if (not ReadResult)
      return Label::createInvalid();
    Offset = *ReadResult;
    break;
  }

  const auto Origin = LabelOrigin::DynamicRelocation;
  switch (Description.Type) {
  case RD::BaseRelative:
    return Label::createBaseRelativeValue(Origin, Target, PointerSize, Offset);

  case RD::LabelOnly:
    if (shouldIgnoreSymbol(SymbolName))
      return Label::createInvalid();
    return Label::createSymbol(Origin,
                               Target,
                               SymbolSize,
                               SymbolName,
                               SymbolType);

  case RD::SymbolRelative:
    if (shouldIgnoreSymbol(SymbolName))
      return Label::createInvalid();
    return Label::createSymbolRelativeValue(Origin,
                                            Target,
                                            PointerSize,
                                            SymbolName,
                                            SymbolType,
                                            Offset);

  case RD::Invalid:
  default:
    revng_abort("Invalid relocation type");
    break;
  }
}

template<typename T, bool HasAddend>
void BinaryFile::registerRelocations(Elf_Rel_Array<T, HasAddend> Relocations,
                                     const FilePortion &Dynsym,
                                     const FilePortion &Dynstr) {
  using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;
  using Elf_Sym = llvm::object::Elf_Sym_Impl<T>;

  ArrayRef<Elf_Sym> Symbols;
  if (Dynsym.isAvailable())
    Symbols = Dynsym.extractAs<Elf_Sym>(Segments);

  for (Elf_Rel Relocation : Relocations) {
    unsigned char Type = Relocation.getType(false);
    uint64_t Addend = RelocationHelper<T, HasAddend>::getAddend(Relocation);
    uint64_t Address = relocate(Relocation.r_offset);

    StringRef SymbolName;
    uint64_t SymbolSize = 0;
    unsigned char SymbolType = llvm::ELF::STT_NOTYPE;
    if (Dynsym.isAvailable() and Dynstr.isAvailable()) {
      uint32_t SymbolIndex = Relocation.getSymbol(false);
      revng_check(SymbolIndex < Symbols.size());
      const Elf_Sym &Symbol = Symbols[SymbolIndex];
      auto Result = Symbol.getName(Dynstr.extractString(Segments));
      if (Result)
        SymbolName = *Result;
      SymbolSize = Symbol.st_size;
      SymbolType = Symbol.getType();
    }

    registerLabel(parseRelocation(Type,
                                  Address,
                                  Addend,
                                  SymbolName,
                                  SymbolSize,
                                  SymbolType::fromELF(SymbolType)));
  }
}

static LabelList &operator+=(LabelList &This, const LabelList &Other) {
  This.insert(std::end(This), std::begin(Other), std::end(Other));
  return This;
}

void BinaryFile::rebuildLabelsMap() {
  using Interval = boost::icl::interval<uint64_t>;

  // Clear the map
  LabelsMap.clear();

  // Identify all the 0-sized labels
  std::vector<Label *> ZeroSizedLabels;
  for (Label &L : Labels)
    if (L.isSymbol() and L.size() == 0)
      ZeroSizedLabels.push_back(&L);

  // Sort the 0-sized labels
  auto Compare = [](Label *This, Label *Other) {
    return This->address() < Other->address();
  };
  std::sort(ZeroSizedLabels.begin(), ZeroSizedLabels.end(), Compare);

  // Create virtual terminator label
  uint64_t HighestAddress = 0;
  for (const SegmentInfo &Segment : Segments)
    HighestAddress = std::max(HighestAddress, Segment.EndVirtualAddress);
  Label EndLabel = Label::createSymbol(LabelOrigin::Unknown,
                                       HighestAddress,
                                       0,
                                       "",
                                       SymbolType::Unknown);
  ZeroSizedLabels.push_back(&EndLabel);

  // Insert the 0-sized labels in the map
  for (unsigned I = 0; I < ZeroSizedLabels.size() - 1; I++) {
    uint64_t Start = ZeroSizedLabels[I]->address();

    const SegmentInfo *Segment = findSegment(Start);
    if (Segment == nullptr)
      continue;

    // Limit the symbol to the end of the segment containing it
    uint64_t End = std::min(ZeroSizedLabels[I + 1]->address(),
                            Segment->EndVirtualAddress);
    revng_assert(Start <= End);

    // Register virtual size
    ZeroSizedLabels[I]->setVirtualSize(End - Start);
  }

  // Insert all the other labels in the map
  for (Label &L : Labels) {
    uint64_t Start = L.address();
    uint64_t End = L.address() + L.size();
    LabelsMap += make_pair(Interval::right_open(Start, End), LabelList{ &L });
  }

  // Dump the map out
  if (LabelsLog.isEnabled()) {
    for (auto &P : LabelsMap) {
      dbg << P.first << "\n";
      for (const Label *L : P.second) {
        dbg << "  ";
        L->dump(dbg);
        dbg << "\n";
      }
      dbg << "\n";
    }
  }
}

//
// .eh_frame-related functions
//

template<typename E>
class DwarfReader {
public:
  DwarfReader(ArrayRef<uint8_t> Buffer, uint64_t Address) :
    Address(Address),
    Start(Buffer.data()),
    Cursor(Buffer.data()),
    End(Buffer.data() + Buffer.size()) {}

  template<typename T>
  T readNext() {
    revng_assert(Cursor + sizeof(T) <= End);
    T Result = Endianess<T, E>::read(Cursor);
    Cursor += sizeof(T);
    return Result;
  }

  uint8_t readNextU8() { return readNext<uint8_t>(); }
  uint16_t readNextU16() { return readNext<uint16_t>(); }
  uint32_t readNextU32() { return readNext<uint32_t>(); }
  uint64_t readNextU64() { return readNext<uint64_t>(); }

  uint64_t readNextU() {
    if (is64())
      return readNextU64();
    else
      return readNextU32();
  }

  uint64_t readULEB128() {
    unsigned Length;
    uint64_t Result = decodeULEB128(Cursor, &Length);
    Cursor += Length;
    revng_assert(Cursor <= End);
    return Result;
  }

  int64_t readSLEB128() {
    unsigned Length;
    int64_t Result = decodeSLEB128(Cursor, &Length);
    Cursor += Length;
    revng_assert(Cursor <= End);
    return Result;
  }

  Pointer readPointer(unsigned Encoding, uint64_t Base = 0) {
    revng_assert((Encoding & ~(0x70 | 0x0F | dwarf::DW_EH_PE_indirect)) == 0);

    if ((Encoding & 0x70) == dwarf::DW_EH_PE_pcrel)
      Base = Address + (Cursor - Start);

    unsigned Format = Encoding & 0x0F;
    switch (Format) {
    case dwarf::DW_EH_PE_uleb128:
      return readPointerInternal(readULEB128(), Encoding, Base);
    case dwarf::DW_EH_PE_sleb128:
      return readPointerInternal(readSLEB128(), Encoding, Base);
    case dwarf::DW_EH_PE_absptr:
      if (is64())
        return readPointerInternal(readNext<uint64_t>(), Encoding, Base);
      else
        return readPointerInternal(readNext<uint32_t>(), Encoding, Base);
    case dwarf::DW_EH_PE_signed:
      if (is64())
        return readPointerInternal(readNext<int64_t>(), Encoding, Base);
      else
        return readPointerInternal(readNext<int32_t>(), Encoding, Base);
    case dwarf::DW_EH_PE_udata2:
      return readPointerInternal(readNext<uint16_t>(), Encoding, Base);
    case dwarf::DW_EH_PE_sdata2:
      return readPointerInternal(readNext<int16_t>(), Encoding, Base);
    case dwarf::DW_EH_PE_udata4:
      return readPointerInternal(readNext<uint32_t>(), Encoding, Base);
    case dwarf::DW_EH_PE_sdata4:
      return readPointerInternal(readNext<int32_t>(), Encoding, Base);
    case dwarf::DW_EH_PE_udata8:
      return readPointerInternal(readNext<uint64_t>(), Encoding, Base);
    case dwarf::DW_EH_PE_sdata8:
      return readPointerInternal(readNext<int64_t>(), Encoding, Base);
    default:
      revng_unreachable("Unknown Encoding");
    }
  }

  void moveTo(uint64_t Offset) {
    const uint8_t *NewCursor = Start + Offset;
    revng_assert(NewCursor >= Cursor && NewCursor <= End);
    Cursor = NewCursor;
  }

  bool eof() const { return Cursor >= End; }
  uint64_t offset() const { return Cursor - Start; }

private:
  template<typename T>
  Pointer readPointerInternal(T Value, unsigned Encoding, uint64_t Base) {
    uint64_t Result = Value;

    if (Value != 0) {
      int EncodingRelative = Encoding & 0x70;
      revng_assert(EncodingRelative == 0 || EncodingRelative == 0x10);

      Result = Base;
      if (std::numeric_limits<T>::is_signed)
        Result += static_cast<int64_t>(Value);
      else
        Result += static_cast<uint64_t>(Value);
    }

    return Pointer(Encoding & dwarf::DW_EH_PE_indirect, Result);
  }

  bool is64() const;

private:
  uint64_t Address;
  const uint8_t *Start;
  const uint8_t *Cursor;
  const uint8_t *End;
};

template<>
bool DwarfReader<object::ELF32BE>::is64() const {
  return false;
}
template<>
bool DwarfReader<object::ELF32LE>::is64() const {
  return false;
}
template<>
bool DwarfReader<object::ELF64BE>::is64() const {
  return true;
}
template<>
bool DwarfReader<object::ELF64LE>::is64() const {
  return true;
}

template<typename T>
std::pair<uint64_t, uint64_t>
BinaryFile::ehFrameFromEhFrameHdr(uint64_t EHFrameHdrAddress) {
  auto R = getAddressData(EHFrameHdrAddress);
  revng_assert(R, ".eh_frame_hdr section not available in any segment");
  llvm::ArrayRef<uint8_t> EHFrameHdr = *R;

  DwarfReader<T> EHFrameHdrReader(EHFrameHdr, EHFrameHdrAddress);

  uint64_t VersionNumber = EHFrameHdrReader.readNextU8();
  revng_assert(VersionNumber == 1);

  // ExceptionFrameEncoding
  uint64_t ExceptionFrameEncoding = EHFrameHdrReader.readNextU8();

  // FDEsCountEncoding
  unsigned FDEsCountEncoding = EHFrameHdrReader.readNextU8();

  // LookupTableEncoding
  EHFrameHdrReader.readNextU8();

  Pointer EHFramePointer = EHFrameHdrReader.readPointer(ExceptionFrameEncoding);
  Pointer FDEsCountPointer = EHFrameHdrReader.readPointer(FDEsCountEncoding);

  return { getPointer<T>(EHFramePointer), getPointer<T>(FDEsCountPointer) };
}

template<typename T>
void BinaryFile::parseEHFrame(uint64_t EHFrameAddress,
                              Optional<uint64_t> FDEsCount,
                              Optional<uint64_t> EHFrameSize) {
  revng_assert(FDEsCount || EHFrameSize);

  auto R = getAddressData(EHFrameAddress);

  // Sometimes the .eh_frame section is present but not mapped in memory. This
  // means it cannot be used at runtime, therefore we can ignore it.
  if (!R)
    return;
  llvm::ArrayRef<uint8_t> EHFrame = *R;

  DwarfReader<T> EHFrameReader(EHFrame, EHFrameAddress);

  // A few fields of the CIE are used when decoding the FDE's.  This struct
  // will cache those fields we need so that we don't have to decode it
  // repeatedly for each FDE that references it.
  struct DecodedCIE {
    Optional<uint32_t> FDEPointerEncoding;
    Optional<uint32_t> LSDAPointerEncoding;
    bool hasAugmentationLength;
  };

  // Map from the start offset of the CIE to the cached data for that CIE.
  DenseMap<uint64_t, DecodedCIE> CachedCIEs;
  unsigned FDEIndex = 0;

  while (!EHFrameReader.eof()
         && ((FDEsCount && FDEIndex < *FDEsCount)
             || (EHFrameSize && EHFrameReader.offset() < *EHFrameSize))) {

    uint64_t StartOffset = EHFrameReader.offset();

    // Read the length of the entry
    uint64_t Length = EHFrameReader.readNextU32();
    if (Length == 0xffffffff)
      Length = EHFrameReader.readNextU64();

    // Compute the end offset of the entry
    uint64_t OffsetAfterLength = EHFrameReader.offset();
    uint64_t EndOffset = OffsetAfterLength + Length;

    // Zero-sized entry, skip it
    if (Length == 0) {
      revng_assert(EHFrameReader.offset() == EndOffset);
      continue;
    }

    // Get the entry ID, 0 means it's a CIE, otherwise it's a FDE
    uint32_t ID = EHFrameReader.readNextU32();
    if (ID == 0) {
      // This is a CIE
      revng_log(EhFrameLog, "New CIE");

      // Ensure the version is the one we expect
      uint32_t Version = EHFrameReader.readNextU8();
      revng_assert(Version == 1);

      // Parse a null terminated augmentation string
      SmallString<8> AugmentationString;
      for (uint8_t Char = EHFrameReader.readNextU8(); Char != 0;
           Char = EHFrameReader.readNextU8())
        AugmentationString.push_back(Char);

      // Optionally parse the EH data if the augmentation string says it's
      // there
      if (StringRef(AugmentationString).count("eh") != 0)
        EHFrameReader.readNextU();

      // CodeAlignmentFactor
      EHFrameReader.readULEB128();

      // DataAlignmentFactor
      EHFrameReader.readULEB128();

      // ReturnAddressRegister
      EHFrameReader.readNextU8();

      Optional<uint64_t> AugmentationLength;
      Optional<uint32_t> LSDAPointerEncoding;
      Optional<uint32_t> PersonalityEncoding;
      Optional<uint32_t> FDEPointerEncoding;
      if (!AugmentationString.empty() && AugmentationString.front() == 'z') {
        AugmentationLength = EHFrameReader.readULEB128();

        // Walk the augmentation string to get all the augmentation data.
        for (unsigned I = 1, e = AugmentationString.size(); I != e; ++I) {
          char Char = AugmentationString[I];
          switch (Char) {
          case 'e':
            revng_assert((I + 1) != e && AugmentationString[I + 1] == 'h',
                         "Expected 'eh' in augmentation string");
            break;
          case 'L':
            // This is the only information we really care about, all the rest
            // is processed just so we can get here
            revng_assert(!LSDAPointerEncoding, "Duplicate LSDA encoding");
            LSDAPointerEncoding = EHFrameReader.readNextU8();
            break;
          case 'P': {
            revng_assert(!PersonalityEncoding, "Duplicate personality");
            PersonalityEncoding = EHFrameReader.readNextU8();
            // Personality
            Pointer Personality;
            Personality = EHFrameReader.readPointer(*PersonalityEncoding);
            uint64_t PersonalityPtr = getPointer<T>(Personality);
            revng_log(EhFrameLog, "Personality function: " << PersonalityPtr);
            // TODO: technically this is not a landing pad
            LandingPads.insert(PersonalityPtr);
            break;
          }
          case 'R':
            revng_assert(!FDEPointerEncoding, "Duplicate FDE encoding");
            FDEPointerEncoding = EHFrameReader.readNextU8();
            break;
          case 'z':
            revng_unreachable("'z' must be first in the augmentation string");
          }
        }
      }

      // Cache this entry
      CachedCIEs[StartOffset] = { FDEPointerEncoding,
                                  LSDAPointerEncoding,
                                  AugmentationLength.hasValue() };

    } else {
      // This is an FDE
      FDEIndex++;

      // The CIE pointer for an FDE is the same location as the ID which we
      // already read
      uint64_t CIEOffset = OffsetAfterLength - ID;

      // Ensure we already met this CIE
      auto CIEIt = CachedCIEs.find(CIEOffset);
      revng_assert(CIEIt != CachedCIEs.end(),
                   "Couldn't find CIE at offset in to __eh_frame section");

      // Ensure we have at least the pointer encoding
      const DecodedCIE &CIE = CIEIt->getSecond();
      revng_assert(CIE.FDEPointerEncoding,
                   "FDE references CIE which did not set pointer encoding");

      // PCBegin
      auto PCBeginPointer = EHFrameReader.readPointer(*CIE.FDEPointerEncoding);
      uint64_t PCBegin = getPointer<T>(PCBeginPointer);
      revng_log(EhFrameLog, "PCBegin: " << std::hex << PCBegin);

      // PCRange
      EHFrameReader.readPointer(*CIE.FDEPointerEncoding);

      if (CIE.hasAugmentationLength)
        EHFrameReader.readULEB128();

      // Decode the LSDA if the CIE augmentation string said we should.
      if (CIE.LSDAPointerEncoding) {
        auto LSDAPointer = EHFrameReader.readPointer(*CIE.LSDAPointerEncoding);
        parseLSDA<T>(PCBegin, getPointer<T>(LSDAPointer));
      }
    }

    // Skip all the remaining parts
    EHFrameReader.moveTo(EndOffset);
  }
}

template<typename T>
void BinaryFile::parseLSDA(uint64_t FDEStart, uint64_t LSDAAddress) {
  revng_log(EhFrameLog, "LSDAAddress: " << std::hex << LSDAAddress);

  auto R = getAddressData(LSDAAddress);
  revng_assert(R, "LSDA not available in any segment");
  llvm::ArrayRef<uint8_t> LSDA = *R;

  DwarfReader<T> LSDAReader(LSDA, LSDAAddress);

  uint32_t LandingPadBaseEncoding = LSDAReader.readNextU8();
  uint64_t LandingPadBase = 0;
  if (LandingPadBaseEncoding != dwarf::DW_EH_PE_omit) {
    auto LandingPadBasePointer = LSDAReader.readPointer(LandingPadBaseEncoding);
    LandingPadBase = getPointer<T>(LandingPadBasePointer);
  } else {
    LandingPadBase = FDEStart;
  }

  revng_log(EhFrameLog, "LandingPadBase: " << std::hex << LandingPadBase);

  uint32_t TypeTableEncoding = LSDAReader.readNextU8();
  if (TypeTableEncoding != dwarf::DW_EH_PE_omit)
    LSDAReader.readULEB128();

  uint32_t CallSiteTableEncoding = LSDAReader.readNextU8();
  uint64_t CallSiteTableLength = LSDAReader.readULEB128();
  uint64_t CallSiteTableEnd = LSDAReader.offset() + CallSiteTableLength;

  while (LSDAReader.offset() < CallSiteTableEnd) {
    // InstructionStart
    LSDAReader.readPointer(CallSiteTableEncoding);

    // InstructionEnd
    LSDAReader.readPointer(CallSiteTableEncoding);

    // LandingPad
    Pointer LandingPadPointer = LSDAReader.readPointer(CallSiteTableEncoding,
                                                       LandingPadBase);
    uint64_t LandingPad = getPointer<T>(LandingPadPointer);

    // Action
    LSDAReader.readULEB128();

    if (LandingPad != 0) {
      if (EhFrameLog.isEnabled() and LandingPads.count(LandingPad) == 0) {
        EhFrameLog << "New landing pad found: " << std::hex << LandingPad
                   << DoLog;
      }
      LandingPads.insert(LandingPad);
    }
  }
}
