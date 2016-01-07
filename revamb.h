#ifndef _REVAMB_H
#define _REVAMB_H

// Standard includes
#include <string>

// Path to the QEMU libraries should be given by the build system
#ifndef QEMU_LIB_PATH
# define QEMU_LIB_PATH "/usr/lib"
#endif

enum class DebugInfoType {
  None,
  OriginalAssembly,
  PTC,
  LLVMIR
};

class Architecture {
public:

  enum EndianessType {
    LittleEndian,
    BigEndian
  };

public:
 Architecture() :
    DefaultAlignment(1),
    Endianess(LittleEndian),
    PointerSize(64) { }

 Architecture(unsigned DefaultAlignment,
              EndianessType Endianess,
              unsigned PointerSize) :
    DefaultAlignment(DefaultAlignment),
    Endianess(Endianess),
    PointerSize(PointerSize) { }

 Architecture(unsigned DefaultAlignment,
              bool IsLittleEndian,
              unsigned PointerSize) :
    DefaultAlignment(DefaultAlignment),
    Endianess(IsLittleEndian ? LittleEndian : BigEndian),
    PointerSize(PointerSize) { }

  unsigned defaultAlignment() { return DefaultAlignment; }
  EndianessType endianess() { return Endianess; }
  unsigned pointerSize() { return PointerSize; }
  bool isLittleEndian() { return Endianess == LittleEndian; }

private:
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

#endif // _REVAMB_H
