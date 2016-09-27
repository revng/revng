#ifndef _REVAMB_H
#define _REVAMB_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

// LLVM includes
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class GlobalVariable;
};

/// \brief Type of debug information to produce
enum class DebugInfoType {
  None, ///< no debug information.
  OriginalAssembly, ///< produce a file containing the assembly code of the
                    ///  input binary.
  PTC, ///< produce the PTC as translated by libtinycode.
  LLVMIR ///< produce an LLVM IR with debug metadata referring to itself.
};

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

  bool contains(uint64_t Address) const {
    return StartVirtualAddress <= Address && Address < EndVirtualAddress;
  }

  bool contains(uint64_t Start, uint64_t Size) const {
    return contains(Start) && contains(Start + Size - 1);
  }

  std::vector<std::pair<uint64_t, uint64_t>> ExecutableSections;

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

struct BinaryInfo {
  std::vector<SegmentInfo> Segments;
  std::vector<SymbolInfo> Symbols;
};

/// \brief Basic information about an input/output architecture
class Architecture {
public:

  enum EndianessType {
    LittleEndian,
    BigEndian
  };

public:
 Architecture() :
    InstructionAlignment(1),
    DefaultAlignment(1),
    Endianess(LittleEndian),
    PointerSize(64) { }

 Architecture(unsigned InstructionAlignment,
              unsigned DefaultAlignment,
              bool IsLittleEndian,
              unsigned PointerSize,
              llvm::StringRef SyscallHelper,
              llvm::StringRef SyscallNumberRegister,
              llvm::ArrayRef<uint64_t> NoReturnSyscalls) :
    InstructionAlignment(InstructionAlignment),
    DefaultAlignment(DefaultAlignment),
    Endianess(IsLittleEndian ? LittleEndian : BigEndian),
    PointerSize(PointerSize),
    SyscallHelper(SyscallHelper),
    SyscallNumberRegister(SyscallNumberRegister),
    NoReturnSyscalls(NoReturnSyscalls) { }

  unsigned instructionAlignment() { return InstructionAlignment; }
  unsigned defaultAlignment() { return DefaultAlignment; }
  EndianessType endianess() { return Endianess; }
  unsigned pointerSize() { return PointerSize; }
  bool isLittleEndian() { return Endianess == LittleEndian; }
  llvm::StringRef syscallHelper() { return SyscallHelper; }
  llvm::StringRef syscallNumberRegister() { return SyscallNumberRegister; }
  llvm::ArrayRef<uint64_t> noReturnSyscalls() { return NoReturnSyscalls; }


private:
  unsigned InstructionAlignment;
  unsigned DefaultAlignment;
  EndianessType Endianess;
  unsigned PointerSize;

  llvm::StringRef SyscallHelper;
  llvm::StringRef SyscallNumberRegister;
  llvm::ArrayRef<uint64_t> NoReturnSyscalls;
};

// TODO: this requires C++14
template<typename FnTy, FnTy Ptr>
struct GenericFunctor {
  template<typename... Args>
  auto operator()(Args... args) -> decltype(Ptr(args...)) {
    return Ptr(args...);
  }
};

// TODO: move me somewhere more appropriate
static inline bool startsWith(std::string String, std::string Prefix) {
  return String.substr(0, Prefix.size()) == Prefix;
}

/// \brief Simple helper function asserting a pointer is not a `nullptr`
template<typename T>
static inline T *notNull(T *Pointer) {
  assert(Pointer != nullptr);
  return Pointer;
}

template<typename T>
static inline bool contains(T Range, typename T::value_type V) {
  return std::find(std::begin(Range), std::end(Range), V) != std::end(Range);
}

#endif // _REVAMB_H
