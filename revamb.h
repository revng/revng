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
#include "llvm/ADT/Triple.h"

// Local includes
#include "ir-helpers.h"
namespace llvm {
class GlobalVariable;
};

class ABIRegister {
private:
  llvm::StringRef Name;
  llvm::StringRef QemuName;
  unsigned MContextIndex;

public:
  static const unsigned NotInMContext = std::numeric_limits<unsigned>::max();

public:
  ABIRegister(llvm::StringRef Name,
              unsigned MContextIndex) :
    Name(Name),
    QemuName(Name),
    MContextIndex(MContextIndex) { }

  ABIRegister(llvm::StringRef Name) :
    Name(Name),
    QemuName(Name),
    MContextIndex(NotInMContext) { }

  ABIRegister(llvm::StringRef Name, llvm::StringRef QemuName) :
    Name(Name),
    QemuName(QemuName),
    MContextIndex(NotInMContext) { }

  llvm::StringRef name() const { return Name; }

  llvm::StringRef qemuName() const { return QemuName; }

  bool inMContext() const { return MContextIndex != NotInMContext; }
  unsigned mcontextIndex() const {
    assert(inMContext());
    return MContextIndex;
  }
};

/// \brief Type of debug information to produce
enum class DebugInfoType {
  None, ///< no debug information.
  OriginalAssembly, ///< produce a file containing the assembly code of the
                    ///  input binary.
  PTC, ///< produce the PTC as translated by libtinycode.
  LLVMIR ///< produce an LLVM IR with debug metadata referring to itself.
};

// TODO: move me to another header file
/// \brief Classification of the various basic blocks we are creating
enum BlockType {
  UntypedBlock, ///< A basic block generated during translation that it's not a
                ///  jump target.
  DispatcherBlock, ///< Basic block representing the dispatcher.
  AnyPCBlock, ///< Basic block used to handle an expectedly unknown jump target.
  UnexpectedPCBlock, ///< Basic block used to handle an unexpectedly unknown
                     ///  jump target.
  JumpTargetBlock, ///< A basic block generated during translation representing
                   ///  a jump target.
  DispatcherFailure ///< Basic block representing the default case of the
                    ///  dispatcher switch.
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
    PointerSize(64),
    DelaySlotSize(0) { }

  Architecture(unsigned Type,
               unsigned InstructionAlignment,
               unsigned DefaultAlignment,
               bool IsLittleEndian,
               unsigned PointerSize,
               llvm::StringRef SyscallHelper,
               llvm::StringRef SyscallNumberRegister,
               llvm::ArrayRef<uint64_t> NoReturnSyscalls,
               unsigned DelaySlotSize,
               llvm::StringRef StackPointerRegister,
               llvm::SmallVector<ABIRegister, 20> ABIRegisters,
               unsigned PCMContextIndex,
               llvm::StringRef WriteRegisterAsm,
               llvm::StringRef ReadRegisterAsm,
               llvm::StringRef JumpAsm) :
    Type(static_cast<llvm::Triple::ArchType>(Type)),
    InstructionAlignment(InstructionAlignment),
    DefaultAlignment(DefaultAlignment),
    Endianess(IsLittleEndian ? LittleEndian : BigEndian),
    PointerSize(PointerSize),
    SyscallHelper(SyscallHelper),
    SyscallNumberRegister(SyscallNumberRegister),
    NoReturnSyscalls(NoReturnSyscalls),
    DelaySlotSize(DelaySlotSize),
    StackPointerRegister(StackPointerRegister),
    ABIRegisters(ABIRegisters),
    PCMContextIndex(PCMContextIndex),
    WriteRegisterAsm(WriteRegisterAsm),
    ReadRegisterAsm(ReadRegisterAsm),
    JumpAsm(JumpAsm) { }

  unsigned instructionAlignment() const { return InstructionAlignment; }
  unsigned defaultAlignment() const { return DefaultAlignment; }
  EndianessType endianess() const { return Endianess; }
  unsigned pointerSize() const { return PointerSize; }
  bool isLittleEndian() const { return Endianess == LittleEndian; }
  llvm::StringRef syscallHelper() const { return SyscallHelper; }
  llvm::StringRef syscallNumberRegister() const {
    return SyscallNumberRegister;
  }
  llvm::StringRef stackPointerRegister() const {
    return StackPointerRegister;
  }
  llvm::ArrayRef<uint64_t> noReturnSyscalls() const { return NoReturnSyscalls; }
  unsigned delaySlotSize() const { return DelaySlotSize; }
  llvm::SmallVector<ABIRegister, 20> abiRegisters() const {
    return ABIRegisters;
  }
  const char *name() const { return llvm::Triple::getArchTypeName(Type); }
  unsigned pcMContextIndex() const { return PCMContextIndex; }

  llvm::StringRef writeRegisterAsm() const { return WriteRegisterAsm; }
  llvm::StringRef readRegisterAsm() const { return ReadRegisterAsm; }
  llvm::StringRef jumpAsm() const { return JumpAsm; }
  bool isJumpOutSupported() const {
    bool IsSupported = WriteRegisterAsm.size() != 0;
    assert(IsSupported == (ReadRegisterAsm.size() != 0)
           && IsSupported == (JumpAsm.size() != 0));
    return IsSupported;
  }

private:
  llvm::Triple::ArchType Type;

  unsigned InstructionAlignment;
  unsigned DefaultAlignment;
  EndianessType Endianess;
  unsigned PointerSize;

  llvm::StringRef SyscallHelper;
  llvm::StringRef SyscallNumberRegister;
  llvm::ArrayRef<uint64_t> NoReturnSyscalls;
  unsigned DelaySlotSize;
  llvm::StringRef StackPointerRegister;
  llvm::SmallVector<ABIRegister, 20> ABIRegisters;
  unsigned PCMContextIndex;
  llvm::StringRef WriteRegisterAsm;
  llvm::StringRef ReadRegisterAsm;
  llvm::StringRef JumpAsm;
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

static const std::array<llvm::StringRef, 3> MarkerFunctionNames = {
  "newpc",
  "function_call",
  "exitTB"
};

static inline bool isMarker(llvm::Instruction *I) {
  using namespace std::placeholders;
  using llvm::any_of;
  using std::bind;

  return any_of(MarkerFunctionNames, bind(isCallTo, I, _1));
}

static inline llvm::Instruction *nextNonMarker(llvm::Instruction *I) {
  auto It = I->getIterator();
  auto End = I->getParent()->end();
  do {
    It++;
    assert(It != End);
  } while (isMarker(&*It));

  assert(It != End);
  return &*It;
}

#endif // _REVAMB_H
