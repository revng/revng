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
 Architecture() : PCReg("pc"),
    DefaultAlignment(1),
    Endianess(LittleEndian),
    PointerSize(64) { }

  enum EndianessType {
    LittleEndian,
    BigEndian
  };

  unsigned defaultAlignment() { return DefaultAlignment; }
  EndianessType endianess() { return Endianess; }
  unsigned pointerSize() { return PointerSize; }
  const std::string PCReg;

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

#endif // _REVAMB_H
