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
#include "llvm/Object/ELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"

// Local includes
#include "binaryfile.h"
#include "debug.h"

// using directives
using namespace llvm;

using std::make_pair;

BinaryFile::BinaryFile(std::string FilePath, bool UseSections) {
  auto BinaryOrErr = object::createBinary(FilePath);
  assert(BinaryOrErr && "Couldn't open the input file");

  BinaryHandle = std::move(BinaryOrErr.get());

  auto *TheBinary = cast<object::ObjectFile>(BinaryHandle.getBinary());

  // TODO: QEMU should provide this information
  unsigned InstructionAlignment = 0;
  StringRef SyscallHelper = "";
  StringRef SyscallNumberRegister = "";
  StringRef StackPointerRegister = "";
  ArrayRef<uint64_t> NoReturnSyscalls = { };
  SmallVector<ABIRegister, 20> ABIRegisters;
  unsigned DelaySlotSize = 0;
  unsigned PCMContextIndex = ABIRegister::NotInMContext;
  llvm::StringRef WriteRegisterAsm = "";
  llvm::StringRef ReadRegisterAsm = "";
  llvm::StringRef JumpAsm = "";
  switch (TheBinary->getArch()) {
  case Triple::x86:
    InstructionAlignment = 1;
    SyscallHelper = "helper_syscall";
    SyscallNumberRegister = "eax";
    StackPointerRegister = "esp";
    NoReturnSyscalls = {
        0xfc, // exit_group
        0x01, // exit
        0x0b // execve
    };
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

    ABIRegisters = { { "rax", 0xD }, { "rbx", 0xB }, { "rcx", 0xE },
                     { "rdx", 0xC }, { "rbp", 0xA }, { "rsp", 0xF },
                     { "rsi", 0x9 }, { "rdi", 0x8 }, { "r8", 0x0 },
                     { "r9", 0x1 }, { "r10", 0x2 }, { "r11", 0x3 },
                     { "r12", 0x4 }, { "r13", 0x5 }, { "r14", 0x6 },
                     { "r15", 0x7 }, { "xmm0", "state_0x8558" },
                     { "xmm1", "state_0x8598" }, { "xmm2", "state_0x85d8" },
                     { "xmm3", "state_0x8618" }, { "xmm4", "state_0x8658" },
                     { "xmm5", "state_0x8698" }, { "xmm6", "state_0x86d8" },
                     { "xmm7", "state_0x8718" } };
    WriteRegisterAsm = "movq $0, %REGISTER";
    ReadRegisterAsm = "movq %REGISTER, $0";
    JumpAsm = "movq $0, %r11; jmpq *%r11";
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
    ABIRegisters = { { "r0" }, { "r1" }, { "r2" }, { "r3" }, { "r4" },
                     { "r5" }, { "r6" }, { "r7" }, { "r8" }, { "r9" },
                     { "r10" }, { "r11" }, { "r12" }, { "r13" }, { "r14" } };
    break;
  case Triple::mips:
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
    ABIRegisters = { {"v0" }, { "v1" }, { "a0" }, { "a1" }, { "a2" }, { "a3" },
                     { "s0" }, { "s1" }, { "s2", }, { "s3" }, { "s4" },
                     { "s5" }, { "s6" }, { "s7" }, { "gp" }, { "sp" },
                     { "fp" }, { "ra" } };
    break;
  default:
    assert(false);
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
                                 JumpAsm);

  assert(TheBinary->getFileFormatName().startswith("ELF")
         && "Only the ELF file format is currently supported");

  if (TheArchitecture.pointerSize() == 32) {
    if (TheArchitecture.isLittleEndian()) {
      parseELF<object::ELF32LE>(TheBinary, UseSections);
    } else {
      parseELF<object::ELF32BE>(TheBinary, UseSections);
    }
  } else if (TheArchitecture.pointerSize() == 64) {
    if (TheArchitecture.isLittleEndian()) {
      parseELF<object::ELF64LE>(TheBinary, UseSections);
    } else {
      parseELF<object::ELF64BE>(TheBinary, UseSections);
    }
  } else {
    assert("Unexpect address size");
  }
}

template<typename T>
void BinaryFile::parseELF(object::ObjectFile *TheBinary, bool UseSections) {
  // Parse the ELF file
  std::error_code EC;
  object::ELFFile<T> TheELF(TheBinary->getData(), EC);
  assert(!EC && "Error while loading the ELF file");

  // Look for static or dynamic symbols and relocations
  using Elf_ShdrPtr = decltype(&(*TheELF.sections().begin()));
  using Elf_PhdrPtr = decltype(&(*TheELF.program_headers().begin()));
  Elf_ShdrPtr SymtabShdr = nullptr;
  Elf_PhdrPtr DynamicPhdr = nullptr;
  Optional<uint64_t> DynamicAddress;
  Optional<uint64_t> EHFrameAddress;
  Optional<uint64_t> EHFrameSize;
  Optional<uint64_t> EHFrameHdrAddress;

  for (auto &Section : TheELF.sections()) {
    if (ErrorOr<StringRef> Name = TheELF.getSectionName(&Section)) {
      if (*Name == ".symtab") {
        assert(SymtabShdr == nullptr && "Duplicate .symtab");
        SymtabShdr = &Section;
      } else if (*Name == ".eh_frame") {
        assert(not EHFrameAddress && "Duplicate .eh_frame");
        EHFrameAddress = static_cast<uint64_t>(Section.sh_addr);
        EHFrameSize = static_cast<uint64_t>(Section.sh_size);
      } else if (*Name == ".dynamic") {
        assert(not DynamicAddress && "Duplicate .dynamic");
        DynamicAddress = static_cast<uint64_t>(Section.sh_addr);
      }
    }
  }

  // If we found a symbol table
  if (SymtabShdr != nullptr && SymtabShdr->sh_link != 0) {
    // Obtain a reference to the string table
    const Elf_ShdrPtr Strtab = TheELF.getSection(SymtabShdr->sh_link).get();
    ArrayRef<uint8_t> StrtabArray = TheELF.getSectionContents(Strtab).get();
    StringRef StrtabContent(reinterpret_cast<const char *>(StrtabArray.data()),
                            StrtabArray.size());

    // Collect symbol names
    for (auto &Symbol : TheELF.symbols(SymtabShdr)) {
      Symbols.push_back({
        Symbol.getName(StrtabContent).get(),
        Symbol.st_value,
        Symbol.st_size
      });
    }
  }

  const auto *ElfHeader = TheELF.getHeader();
  EntryPoint = static_cast<uint64_t>(ElfHeader->e_entry);
  ProgramHeaders.Count = ElfHeader->e_phnum;
  ProgramHeaders.Size = ElfHeader->e_phentsize;

  // Loop over the program headers looking for PT_LOAD segments, read them out
  // and create a global variable for each one of them (writable or read-only),
  // assign them a section and output information about them in the linking info
  // CSV
  using Elf_Phdr = const typename object::ELFFile<T>::Elf_Phdr;
  using Elf_Dyn = const typename object::ELFFile<T>::Elf_Dyn;
  for (Elf_Phdr &ProgramHeader : TheELF.program_headers()) {
    switch (ProgramHeader.p_type) {
    case ELF::PT_LOAD:
      {
        SegmentInfo Segment;
        auto Start = ProgramHeader.p_vaddr;
        Segment.StartVirtualAddress = Start;
        Segment.EndVirtualAddress = Start + ProgramHeader.p_memsz;
        Segment.IsReadable = ProgramHeader.p_flags & ELF::PF_R;
        Segment.IsWriteable = ProgramHeader.p_flags & ELF::PF_W;
        Segment.IsExecutable = ProgramHeader.p_flags & ELF::PF_X;

        auto ActualAddress = TheELF.base() + ProgramHeader.p_offset;
        Segment.Data = ArrayRef<uint8_t>(ActualAddress, ProgramHeader.p_filesz);

        // If it's an executable segment, and we've been asked so, register
        // which sections actually contain code
        if (UseSections && Segment.IsExecutable) {
          using Elf_Shdr = const typename object::ELFFile<T>::Elf_Shdr;
          auto Inserter = std::back_inserter(Segment.ExecutableSections);
          for (Elf_Shdr &SectionHeader : TheELF.sections()) {
            if (SectionHeader.sh_flags & ELF::SHF_EXECINSTR) {
              auto SectionStart = SectionHeader.sh_addr;
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
          auto PhdrAddress = static_cast<uint64_t>(ProgramHeader.p_vaddr
                                                   + ElfHeader->e_phoff
                                                   - ProgramHeader.p_offset);
          ProgramHeaders.Address = PhdrAddress;
        }
      }
      break;

    case ELF::PT_GNU_EH_FRAME:
      assert(!EHFrameHdrAddress);
      EHFrameHdrAddress = ProgramHeader.p_vaddr;
      break;

    case ELF::PT_DYNAMIC:
      assert(DynamicPhdr == nullptr && "Duplicate .dynamic program header");
      DynamicPhdr = &ProgramHeader;
      assert(((not DynamicAddress)
              or (DynamicPhdr->p_vaddr == *DynamicAddress))
             and ".dynamic and PT_DYNAMIC have different addresses");
      break;
    }

  }

  assert((DynamicPhdr != nullptr) == (DynamicAddress.hasValue()));

  Optional<uint64_t> FDEsCount;
  if (EHFrameHdrAddress) {
    uint64_t Address;

    std::tie(Address, FDEsCount) = ehFrameFromEhFrameHdr<T>(*EHFrameHdrAddress);
    if (EHFrameAddress) {
      assert(*EHFrameAddress == Address);
    }

    EHFrameAddress = Address;
  }

  if (EHFrameAddress)
    parseEHFrame<T>(*EHFrameAddress, FDEsCount, EHFrameSize);

  // Search for needed shared libraries in the .dynamic table
  if (DynamicPhdr != nullptr) {
    SmallVector<uint64_t, 10> NeededLibraryNameOffsets;
    StringRef Dynstr;
    Optional<uint64_t> DynstrSize;
    for (Elf_Dyn &DynamicTag : *TheELF.dynamic_table(DynamicPhdr)) {
      switch(DynamicTag.getTag()) {
        case ELF::DT_NEEDED:
          NeededLibraryNameOffsets.push_back(DynamicTag.getVal());
          break;

        case ELF::DT_STRTAB: {
          Optional<ArrayRef<uint8_t>> DynstrData;
          DynstrData = getAddressData(DynamicTag.getPtr());
          assert(DynstrData.hasValue() &&
                 ".dynamic string table not available in any segment");
          Dynstr = StringRef(reinterpret_cast<const char *>(DynstrData->data()),
                             DynstrData->size());
        } break;

        case ELF::DT_STRSZ:
          DynstrSize = DynamicTag.getVal();
          break;

      }
    }

    assert(DynstrSize.hasValue() && *DynstrSize < Dynstr.size());
    Dynstr = StringRef(Dynstr.data(), *DynstrSize);

    for(auto Offset : NeededLibraryNameOffsets)
      NeededLibraryNames.push_back(Dynstr.slice(Offset, *DynstrSize).data());

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
    End(Buffer.data() + Buffer.size()) { }

  template<typename T>
  T readNext() {
    assert(Cursor + sizeof(T) <= End);
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
    assert(Cursor <= End);
    return Result;
  }

  int64_t readSLEB128() {
    unsigned Length;
    int64_t Result = decodeSLEB128(Cursor, &Length);
    Cursor += Length;
    assert(Cursor <= End);
    return Result;
  }

  Pointer readPointer(unsigned Encoding, uint64_t Base=0) {
    assert((Encoding & ~(0x70 | 0x0F | dwarf::DW_EH_PE_indirect)) == 0);

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
      llvm_unreachable("Unknown Encoding");
    }
  }

  void moveTo(uint64_t Offset) {
    const uint8_t *NewCursor = Start + Offset;
    assert(NewCursor >= Cursor && NewCursor <= End);
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
      assert(EncodingRelative == 0 || EncodingRelative == 0x10);

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

template<> bool DwarfReader<object::ELF32BE>::is64() const { return false; }
template<> bool DwarfReader<object::ELF32LE>::is64() const { return false; }
template<> bool DwarfReader<object::ELF64BE>::is64() const { return true; }
template<> bool DwarfReader<object::ELF64LE>::is64() const { return true; }

template<typename T>
std::pair<uint64_t, uint64_t>
BinaryFile::ehFrameFromEhFrameHdr(uint64_t EHFrameHdrAddress) {
  auto R = getAddressData(EHFrameHdrAddress);
  assert(R && ".eh_frame_hdr section not available in any segment");
  llvm::ArrayRef<uint8_t> EHFrameHdr = *R;

  DwarfReader<T> EHFrameHdrReader(EHFrameHdr, EHFrameHdrAddress);

  uint64_t VersionNumber = EHFrameHdrReader.readNextU8();
  assert(VersionNumber == 1);

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
  assert(FDEsCount || EHFrameSize);

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
      assert(EHFrameReader.offset() == EndOffset);
      continue;
    }

    // Get the entry ID, 0 means it's a CIE, otherwise it's a FDE
    uint32_t ID = EHFrameReader.readNextU32();
    if (ID == 0) {
      // This is a CIE
      DBG("ehframe", dbg << "New CIE\n");

      // Ensure the version is the one we expect
      uint32_t Version = EHFrameReader.readNextU8();
      assert(Version == 1);

      // Parse a null terminated augmentation string
      SmallString<8> AugmentationString;
      for (uint8_t Char = EHFrameReader.readNextU8();
           Char != 0;
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
        for (unsigned i = 1, e = AugmentationString.size(); i != e; ++i) {
          char Char = AugmentationString[i];
          switch (Char) {
            case 'e':
              assert((i + 1) != e && AugmentationString[i + 1] == 'h' &&
                     "Expected 'eh' in augmentation string");
              break;
            case 'L':
              // This is the only information we really care about, all the rest
              // is processed just so we can get here
              assert(!LSDAPointerEncoding && "Duplicate LSDA encoding");
              LSDAPointerEncoding = EHFrameReader.readNextU8();
              break;
            case 'P': {
              assert(!PersonalityEncoding && "Duplicate personality");
              PersonalityEncoding = EHFrameReader.readNextU8();
              // Personality
              Pointer Personality;
              Personality = EHFrameReader.readPointer(*PersonalityEncoding);
              uint64_t PersonalityPtr = getPointer<T>(Personality);
              DBG("ehframe", {
                  dbg << "Personality function: " << PersonalityPtr << "\n";
                });
              // TODO: technically this is not a landing pad
              LandingPads.insert(PersonalityPtr);
              break;
            }
            case 'R':
              assert(!FDEPointerEncoding && "Duplicate FDE encoding");
              FDEPointerEncoding = EHFrameReader.readNextU8();
              break;
            case 'z':
              llvm_unreachable("'z' must be first in the augmentation string");
          }
        }
      }

      // Cache this entry
      CachedCIEs[StartOffset] = {
        FDEPointerEncoding,
        LSDAPointerEncoding,
        AugmentationLength.hasValue()
      };

    } else {
      // This is an FDE
      FDEIndex++;

      // The CIE pointer for an FDE is the same location as the ID which we
      // already read
      uint64_t CIEOffset = OffsetAfterLength - ID;

      // Ensure we already met this CIE
      auto CIEIt = CachedCIEs.find(CIEOffset);
      assert(CIEIt != CachedCIEs.end()
             && "Couldn't find CIE at offset in to __eh_frame section");

      // Ensure we have at least the pointer encoding
      const DecodedCIE &CIE = CIEIt->getSecond();
      assert(CIE.FDEPointerEncoding &&
             "FDE references CIE which did not set pointer encoding");

      // PCBegin
      auto PCBeginPointer = EHFrameReader.readPointer(*CIE.FDEPointerEncoding);
      uint64_t PCBegin = getPointer<T>(PCBeginPointer);
      DBG("ehframe", dbg << "PCBegin: " << std::hex << PCBegin << "\n");

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
  DBG("ehframe", dbg << "LSDAAddress: " << std::hex << LSDAAddress << "\n");

  auto R = getAddressData(LSDAAddress);
  assert(R && "LSDA not available in any segment");
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

  DBG("ehframe",
      dbg << "LandingPadBase: " << std::hex << LandingPadBase << "\n");

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
      DBG("ehframe", {
          if (LandingPads.count(LandingPad) == 0)
            dbg << "New landing pad found: " << std::hex << LandingPad << "\n";
        });
      LandingPads.insert(LandingPad);
    }
  }
}
