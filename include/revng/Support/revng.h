#ifndef REVAMB_H
#define REVAMB_H

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
#include "llvm/BinaryFormat/ELF.h"

// Local libraries includes
#include "revng/Support/IRHelpers.h"

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
  ABIRegister(llvm::StringRef Name, unsigned MContextIndex) :
    Name(Name),
    QemuName(Name),
    MContextIndex(MContextIndex) {}

  ABIRegister(llvm::StringRef Name) :
    Name(Name),
    QemuName(Name),
    MContextIndex(NotInMContext) {}

  ABIRegister(llvm::StringRef Name, llvm::StringRef QemuName) :
    Name(Name),
    QemuName(QemuName),
    MContextIndex(NotInMContext) {}

  llvm::StringRef name() const { return Name; }

  llvm::StringRef qemuName() const { return QemuName; }

  bool inMContext() const { return MContextIndex != NotInMContext; }
  unsigned mcontextIndex() const {
    revng_assert(inMContext());
    return MContextIndex;
  }
};

// TODO: move me to another header file
// TODO: create namespace for this enum
/// \brief Classification of the various basic blocks we are creating
enum BlockType {
  // TODO: UntypedBlock is a bad name
  /// A basic block generated during translation representing a jump target
  JumpTargetBlock,

  /// A basic block generated during translation that it's not a jump target
  UntypedBlock,

  /// Basic block representing the dispatcher
  DispatcherBlock,

  /// Basic block used to handle an expectedly unknown jump target
  AnyPCBlock,

  /// Basic block used to handle an unexpectedly unknown jump target
  UnexpectedPCBlock,

  /// Basic block representing the default case of the dispatcher switch
  DispatcherFailureBlock,

  /// Basic block to handle jumps to non-translated code
  ExternalJumpsHandlerBlock,

  /// The entry point of the root function
  EntryPoint
};

namespace JTReason {

// TODO: move me to another header file
/// \brief Reason for registering a jump target
enum Values {
  PostHelper = 1, ///< PC after an helper (e.g., a syscall)
  DirectJump = 2, ///< Obtained from a direct store to the PC
  GlobalData = 4, ///< Obtained digging in global data
  AmbigousInstruction = 8, ///< Fallthrough of multiple instructions in the
                           ///  immediately preceeding bytes
  SETToPC = 16, ///< Obtained from SET on a store to the PC
  SETNotToPC = 32, ///< Obtained from SET (but not from a PC-store)
  UnusedGlobalData = 64, ///< Obtained digging in global data, buf never used
                         ///  by SET. Likely a function pointer.
  Callee = 128, ///< This JT is the target of a call instruction.
  SumJump = 256, ///< Obtained from the "sumjump" heuristic
  LoadAddress = 512, ///< A load has been performed from this address
  ReturnAddress = 1024, ///< Obtained as the fallthrough of a function call
  FunctionSymbol = 2048, ///< Obtained from a function symbol
  SimpleLiteral = 4096, ///< Immediate value found before SET
  LastReason = SimpleLiteral
};

inline const char *getName(Values Reason) {
  switch (Reason) {
  case PostHelper:
    return "PostHelper";
  case DirectJump:
    return "DirectJump";
  case GlobalData:
    return "GlobalData";
  case AmbigousInstruction:
    return "AmbigousInstruction";
  case SETToPC:
    return "SETToPC";
  case SETNotToPC:
    return "SETNotToPC";
  case UnusedGlobalData:
    return "UnusedGlobalData";
  case Callee:
    return "Callee";
  case SumJump:
    return "SumJump";
  case LoadAddress:
    return "LoadAddress";
  case ReturnAddress:
    return "ReturnAddress";
  case FunctionSymbol:
    return "FunctionSymbol";
  case SimpleLiteral:
    return "SimpleLiteral";
  }

  revng_abort();
}

inline Values fromName(llvm::StringRef ReasonName) {
  if (ReasonName == "PostHelper")
    return PostHelper;
  else if (ReasonName == "DirectJump")
    return DirectJump;
  else if (ReasonName == "GlobalData")
    return GlobalData;
  else if (ReasonName == "AmbigousInstruction")
    return AmbigousInstruction;
  else if (ReasonName == "SETToPC")
    return SETToPC;
  else if (ReasonName == "SETNotToPC")
    return SETNotToPC;
  else if (ReasonName == "UnusedGlobalData")
    return UnusedGlobalData;
  else if (ReasonName == "Callee")
    return Callee;
  else if (ReasonName == "SumJump")
    return SumJump;
  else if (ReasonName == "LoadAddress")
    return LoadAddress;
  else if (ReasonName == "ReturnAddress")
    return ReturnAddress;
  else if (ReasonName == "FunctionSymbol")
    return FunctionSymbol;
  else if (ReasonName == "SimpleLiteral")
    return SimpleLiteral;
  else
    revng_abort();
}

inline bool hasReason(uint32_t Reasons, Values ToCheck) {
  return (Reasons & static_cast<uint32_t>(ToCheck)) != 0;
}

} // namespace JTReason

namespace KillReason {

enum Values { NonKiller, KillerSyscall, EndlessLoop, LeadsToKiller };

inline llvm::StringRef getName(Values Reason) {
  switch (Reason) {
  case NonKiller:
    return "NonKiller";
  case KillerSyscall:
    return "KillerSyscall";
  case EndlessLoop:
    return "EndlessLoop";
  case LeadsToKiller:
    return "LeadsToKiller";
  }

  revng_abort("Unexpected reason");
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "NonKiller")
    return NonKiller;
  if (Name == "KillerSyscall")
    return KillerSyscall;
  else if (Name == "EndlessLoop")
    return EndlessLoop;
  else if (Name == "LeadsToKiller")
    return LeadsToKiller;
  else
    revng_abort("Unexpected name");
}

} // namespace KillReason

class RelocationDescription {
public:
  enum RelocationType { Invalid, BaseRelative, LabelOnly, SymbolRelative };

  enum OffsetType { None, Addend, TargetValue };

public:
  RelocationType Type;
  OffsetType Offset;

public:
  RelocationDescription() : Type(Invalid), Offset(None) {}
  RelocationDescription(RelocationType Type) : Type(Type), Offset(None) {}
  RelocationDescription(RelocationType Type, OffsetType Offset) :
    Type(Type),
    Offset(Offset) {}
};

/// \brief Basic information about an input/output architecture
class Architecture {
public:
  enum EndianessType { LittleEndian, BigEndian };
  using RelocationTypesMap = std::map<unsigned char, RelocationDescription>;

public:
  Architecture() :
    InstructionAlignment(1),
    DefaultAlignment(1),
    Endianess(LittleEndian),
    PointerSize(64),
    DelaySlotSize(0) {}

  Architecture(unsigned Type,
               uint32_t InstructionAlignment,
               uint32_t DefaultAlignment,
               bool IsLittleEndian,
               unsigned PointerSize,
               llvm::StringRef SyscallHelper,
               llvm::StringRef SyscallNumberRegister,
               llvm::ArrayRef<uint64_t> NoReturnSyscalls,
               uint32_t DelaySlotSize,
               llvm::StringRef StackPointerRegister,
               llvm::SmallVector<ABIRegister, 20> ABIRegisters,
               unsigned PCMContextIndex,
               llvm::StringRef WriteRegisterAsm,
               llvm::StringRef ReadRegisterAsm,
               llvm::StringRef JumpAsm,
               bool HasRelocationAddend,
               RelocationTypesMap RelocationTypes) :
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
    JumpAsm(JumpAsm),
    HasRelocationAddend(HasRelocationAddend),
    RelocationTypes(RelocationTypes) {}

  uint32_t instructionAlignment() const { return InstructionAlignment; }
  uint32_t defaultAlignment() const { return DefaultAlignment; }
  EndianessType endianess() const { return Endianess; }
  unsigned pointerSize() const { return PointerSize; }
  bool isLittleEndian() const { return Endianess == LittleEndian; }
  llvm::StringRef syscallHelper() const { return SyscallHelper; }
  llvm::StringRef syscallNumberRegister() const {
    return SyscallNumberRegister;
  }
  llvm::StringRef stackPointerRegister() const { return StackPointerRegister; }
  llvm::ArrayRef<uint64_t> noReturnSyscalls() const { return NoReturnSyscalls; }
  uint32_t delaySlotSize() const { return DelaySlotSize; }
  llvm::SmallVector<ABIRegister, 20> abiRegisters() const {
    return ABIRegisters;
  }
  const char *name() const {
    return llvm::Triple::getArchTypeName(Type).data();
  }
  llvm::Triple::ArchType type() const { return Type; }
  unsigned pcMContextIndex() const { return PCMContextIndex; }

  llvm::StringRef writeRegisterAsm() const { return WriteRegisterAsm; }
  llvm::StringRef readRegisterAsm() const { return ReadRegisterAsm; }
  llvm::StringRef jumpAsm() const { return JumpAsm; }
  bool isJumpOutSupported() const {
    bool IsSupported = WriteRegisterAsm.size() != 0;
    revng_assert(IsSupported == (ReadRegisterAsm.size() != 0)
                 && IsSupported == (JumpAsm.size() != 0));
    return IsSupported;
  }
  bool hasRelocationAddend() const { return HasRelocationAddend; }
  const RelocationTypesMap &relocationTypes() const { return RelocationTypes; }

private:
  llvm::Triple::ArchType Type;

  uint32_t InstructionAlignment;
  uint32_t DefaultAlignment;
  EndianessType Endianess;
  unsigned PointerSize;

  llvm::StringRef SyscallHelper;
  llvm::StringRef SyscallNumberRegister;
  llvm::ArrayRef<uint64_t> NoReturnSyscalls;
  uint32_t DelaySlotSize;
  llvm::StringRef StackPointerRegister;
  llvm::SmallVector<ABIRegister, 20> ABIRegisters;
  unsigned PCMContextIndex;
  llvm::StringRef WriteRegisterAsm;
  llvm::StringRef ReadRegisterAsm;
  llvm::StringRef JumpAsm;
  bool HasRelocationAddend;
  RelocationTypesMap RelocationTypes;
};

// TODO: move me somewhere more appropriate
inline bool startsWith(std::string String, std::string Prefix) {
  return String.substr(0, Prefix.size()) == Prefix;
}

/// \brief Simple helper function asserting a pointer is not a `nullptr`
template<typename T>
inline T *notNull(T *Pointer) {
  revng_assert(Pointer != nullptr);
  return Pointer;
}

static const std::array<llvm::StringRef, 4> MarkerFunctionNames = {
  { "newpc", "function_call", "exitTB", "nodce" }
};

/// \brief Checks if \p I is a marker
///
/// A marker a function call to an empty function acting as meta-information,
/// for example the `function_call` marker.
inline bool isMarker(llvm::Instruction *I) {
  using namespace std::placeholders;
  using llvm::any_of;
  using std::bind;

  return any_of(MarkerFunctionNames, bind(isCallTo, I, _1));
}

inline llvm::Instruction *nextNonMarker(llvm::Instruction *I) {
  auto It = I->getIterator();
  auto End = I->getParent()->end();
  do {
    It++;
    revng_assert(It != End);
  } while (isMarker(&*It));

  revng_assert(It != End);
  return &*It;
}

/// \brief Given a BasicBlock representing a function call, returns the callee
inline llvm::BasicBlock *getFunctionCallCallee(llvm::BasicBlock *BB) {
  using namespace llvm;

  auto Last = BB->begin();
  auto It = BB->end();
  while (It != Last) {
    It--;
    if (isCallTo(&*It, "function_call")) {
      auto *Call = cast<CallInst>(&*It);

      if (auto *BA = dyn_cast<BlockAddress>(Call->getArgOperand(0)))
        return BA->getBasicBlock();
      else
        return nullptr;
    }
  }

  // TODO: is it OK to treat indirect function calls and non-calls the same way?
  return nullptr;
}

#endif // REVAMB_H
