#ifndef _REVAMB_H
#define _REVAMB_H

// Standard includes
#include <string>

// Path to the QEMU libraries should be given by the build system
#ifndef QEMU_LIB_PATH
# define QEMU_LIB_PATH "/usr/lib"
#endif

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

  bool contains(uint64_t Address) {
    return StartVirtualAddress <= Address && Address < EndVirtualAddress;
  }

  bool contains(uint64_t Start, uint64_t Size) {
    return contains(Start) && contains(Start + Size - 1);
  }

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
              EndianessType Endianess,
              unsigned PointerSize) :
    InstructionAlignment(InstructionAlignment),
    DefaultAlignment(DefaultAlignment),
    Endianess(Endianess),
    PointerSize(PointerSize) { }

 Architecture(unsigned InstructionAlignment,
              unsigned DefaultAlignment,
              bool IsLittleEndian,
              unsigned PointerSize) :
    InstructionAlignment(InstructionAlignment),
    DefaultAlignment(DefaultAlignment),
    Endianess(IsLittleEndian ? LittleEndian : BigEndian),
    PointerSize(PointerSize) { }

  unsigned instructionAlignment() { return InstructionAlignment; }
  unsigned defaultAlignment() { return DefaultAlignment; }
  EndianessType endianess() { return Endianess; }
  unsigned pointerSize() { return PointerSize; }
  bool isLittleEndian() { return Endianess == LittleEndian; }

private:
  unsigned InstructionAlignment;
  unsigned DefaultAlignment;
  EndianessType Endianess;
  unsigned PointerSize;
};

// TODO: this requires C++14
template<typename FnTy, FnTy Ptr>
struct GenericFunctor {
  template<typename... Args>
  auto operator()(Args... args) {
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
