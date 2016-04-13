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

enum class DebugInfoType {
  None,
  OriginalAssembly,
  PTC,
  LLVMIR
};

struct SegmentInfo {
  std::string generateName();

  llvm::GlobalVariable *Variable;
  uint64_t StartVirtualAddress;
  uint64_t EndVirtualAddress;
  bool IsWriteable;
  bool IsExecutable;
  bool IsReadable;
};

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

template<typename T>
static inline T *notNull(T *Pointer) {
  assert(Pointer != nullptr);
  return Pointer;
}

#endif // _REVAMB_H
