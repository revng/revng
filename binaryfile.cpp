/// \file binaryfile.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <string>

// LLVM includes
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ELF.h"

// Local includes
#include "binaryfile.h"

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
  ArrayRef<uint64_t> NoReturnSyscalls = { };
  switch (TheBinary->getArch()) {
  case Triple::x86_64:
    InstructionAlignment = 1;
    SyscallHelper = "helper_syscall";
    SyscallNumberRegister = "rax";
    NoReturnSyscalls = {
      0xe7, // exit_group
      0x3c, // exit
      0x3b // execve
    };
    break;
  case Triple::arm:
    InstructionAlignment = 4;
    SyscallHelper = "helper_exception_with_syndrome";
    SyscallNumberRegister = "r7";
    NoReturnSyscalls = {
      0xf8, // exit_group
      0x1, // exit
      0xb // execve
    };
    break;
  case Triple::mips:
    InstructionAlignment = 4;
    SyscallHelper = "helper_raise_exception";
    SyscallNumberRegister = "v0";
    NoReturnSyscalls = {
      0x1096, // exit_group
      0xfa1, // exit
      0xfab // execve
    };
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
                                 NoReturnSyscalls);

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

  // Look for static or dynamic symbols
  using Elf_ShdrPtr = decltype(&(*TheELF.sections().begin()));
  Elf_ShdrPtr Symtab = nullptr;
  for (auto &Section : TheELF.sections()){
    auto Name = TheELF.getSectionName(&Section);
    if (Name && Name.get() == ".symtab") {
      Symtab = &Section;
      break;
    } else if (Name && Name.get() == ".dynsym") {
      Symtab = &Section;
    }
  }

  // If we found a symbol table
  if (Symtab != nullptr && Symtab->sh_link != 0) {
    // Obtain a reference to the string table
    auto *Strtab = TheELF.getSection(Symtab->sh_link).get();
    auto StrtabArray = TheELF.getSectionContents(Strtab).get();
    StringRef StrtabContent(reinterpret_cast<const char *>(StrtabArray.data()),
                            StrtabArray.size());

    // Collect symbol names
    for (auto &Symbol : TheELF.symbols(Symtab)) {
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
  for (Elf_Phdr &ProgramHeader : TheELF.program_headers()) {
    if (ProgramHeader.p_type == ELF::PT_LOAD) {
      SegmentInfo Segment;
      Segment.StartVirtualAddress = ProgramHeader.p_vaddr;
      Segment.EndVirtualAddress = ProgramHeader.p_vaddr + ProgramHeader.p_memsz;
      Segment.IsReadable = ProgramHeader.p_flags & ELF::PF_R;
      Segment.IsWriteable = ProgramHeader.p_flags & ELF::PF_W;
      Segment.IsExecutable = ProgramHeader.p_flags & ELF::PF_X;

      auto ActualAddress = TheELF.base() + ProgramHeader.p_offset;
      Segment.Data = ArrayRef<uint8_t>(ActualAddress, ProgramHeader.p_filesz);

      // If it's an executable segment, and we've been asked so, register which
      // sections actually contain code
      if (UseSections && Segment.IsExecutable) {
        using Elf_Shdr = const typename object::ELFFile<T>::Elf_Shdr;
        auto Inserter = std::back_inserter(Segment.ExecutableSections);
        for (Elf_Shdr &SectionHeader : TheELF.sections())
          if (SectionHeader.sh_flags & ELF::SHF_EXECINSTR)
            Inserter = make_pair(SectionHeader.sh_addr,
                                 SectionHeader.sh_addr + SectionHeader.sh_size);
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

  }
}
