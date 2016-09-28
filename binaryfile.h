#ifndef _BINARYFILE_H
#define _BINARYFILE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <string>
#include <vector>

// LLVM includes
#include "llvm/Object/Binary.h"

// Local includes
#include "revamb.h"

namespace llvm {
namespace object {
class ObjectFile;
}
}

/// \brief Simple data structure to describe an ELF segment
// TODO: information hiding
struct SegmentInfo {
  /// Produce a name for this segment suitable for human understanding
  std::string generateName();

  llvm::GlobalVariable *Variable; ///< \brief LLVM variable containing this
                                  ///  segment's data
  uint64_t StartVirtualAddress;
  uint64_t EndVirtualAddress;
  bool IsWriteable;
  bool IsExecutable;
  bool IsReadable;
  std::vector<std::pair<uint64_t, uint64_t>> ExecutableSections;
  llvm::ArrayRef<uint8_t> Data;

  bool contains(uint64_t Address) const {
    return StartVirtualAddress <= Address && Address < EndVirtualAddress;
  }

  bool contains(uint64_t Start, uint64_t Size) const {
    return contains(Start) && contains(Start + Size - 1);
  }

  uint64_t size() const { return EndVirtualAddress - StartVirtualAddress; }

  template<class C>
  void insertExecutableRanges(std::back_insert_iterator<C> Inserter) const {
    if (!IsExecutable)
      return;

    if (ExecutableSections.size() > 0) {
      std::copy(ExecutableSections.begin(),
                ExecutableSections.end(),
                Inserter);
    } else {
      Inserter = std::make_pair(StartVirtualAddress, EndVirtualAddress);
    }
  }

};

/// \brief Simple data structure to describe a symbol in an image format
///        independent way
// TODO: information hiding
struct SymbolInfo {
  llvm::StringRef Name;
  uint64_t Address;
  uint64_t Size;

  bool operator<(const SymbolInfo &Other) const {
    return Address < Other.Address;
  }

  bool operator==(const SymbolInfo &Other) const {
    return Name == Other.Name && Address == Other.Address && Size == Other.Size;
  }
};

/// \brief BinaryFile describes an input image file in a semi-architecture
///        independent way
class BinaryFile {
public:
  /// \param FilePath the path to the input file.
  /// \param UseSections whether information in sections, if available, should
  ///        be employed or not. This is useful to precisely identify exeutable
  ///        code.
  BinaryFile(std::string FilePath, bool UseSections);

  // Accessors
  const Architecture &architecture() const { return TheArchitecture; }
  std::vector<SegmentInfo> &segments() { return Segments; }
  const std::vector<SegmentInfo> &segments() const { return Segments; }
  const std::vector<SymbolInfo> &symbols() const { return Symbols; }
  uint64_t entryPoint() const { return EntryPoint; }

  // ELF specific accessors
  uint64_t programHeadersAddress() const { return ProgramHeaders.Address; }
  unsigned programHeaderSize() const { return ProgramHeaders.Size; }
  unsigned programHeadersCount() const { return ProgramHeaders.Count; }

private:
  template<typename T>
  void parseELF(llvm::object::ObjectFile *TheBinary,
                bool UseSections);

private:
  llvm::object::OwningBinary<llvm::object::Binary> BinaryHandle;
  Architecture TheArchitecture;
  std::vector<SymbolInfo> Symbols;
  std::vector<SegmentInfo> Segments;

  uint64_t EntryPoint;

  // ELF specific fields
  struct {
    uint64_t Address;
    unsigned Count;
    unsigned Size;
  } ProgramHeaders;
};

#endif // _BINARYFILE_H
