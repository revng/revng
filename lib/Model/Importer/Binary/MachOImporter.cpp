/// \file MachO.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"

#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Model/Pass/FlattenPrimitiveTypedefs.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/Debug.h"
#include "revng/Support/OverflowSafeInt.h"

#include "ELFImporter.h"
#include "Importers.h"

using namespace llvm::MachO;
using namespace llvm::object;
using namespace llvm;
using namespace model;

static Logger<> Log("macho-importer");

template<typename R>
void swapBytes(R &Value) {
  swapStruct(Value);
}

template<>
void swapBytes<uint32_t>(uint32_t &Value) {
  sys::swapByteOrder(Value);
}

template<typename T>
bool contains(const ArrayRef<T> &Container, const ArrayRef<T> &Contained) {
  return (Container.begin() <= Contained.begin()
          and Container.end() >= Contained.end());
}

template<typename T>
class ArrayRefReader {
private:
  ArrayRef<T> Array;
  const T *Cursor = nullptr;
  bool Swap;

public:
  ArrayRefReader(ArrayRef<T> Array, bool Swap) :
    Array(Array), Cursor(Array.begin()), Swap(Swap) {}

  bool eof() const { return Cursor == Array.end(); }

  template<typename R>
  R read() {
    revng_check(Cursor + sizeof(R) > Cursor);
    revng_check(Cursor + sizeof(R) <= Array.end());

    R Result;
    memcpy(&Result, Cursor, sizeof(R));

    if (Swap)
      swapBytes<R>(Result);

    Cursor += sizeof(R);

    return Result;
  }
};

static MetaAddress getInitialPC(Architecture::Values Architecture,
                                bool Swap,
                                ArrayRef<uint8_t> Command) {

  ArrayRefReader<uint8_t> Reader(Command, Swap);
  uint32_t Flavor = Reader.read<uint32_t>();
  uint32_t Count = Reader.read<uint32_t>();
  std::optional<uint64_t> PC;

  switch (Architecture) {
  case Architecture::x86: {

    switch (Flavor) {
    case MachO::x86_THREAD_STATE32:
      if (Count == MachO::x86_THREAD_STATE32_COUNT)
        PC = Reader.read<x86_thread_state32_t>().eip;
      break;

    case MachO::x86_THREAD_STATE:
      if (Count == MachO::x86_THREAD_STATE_COUNT)
        PC = Reader.read<x86_thread_state_t>().uts.ts32.eip;
      break;

    default:
      revng_log(Log, "Unexpected command flavor");
      break;
    }

  } break;

  case Architecture::x86_64: {

    switch (Flavor) {
    case MachO::x86_THREAD_STATE64:
      if (Count == MachO::x86_THREAD_STATE64_COUNT)
        PC = Reader.read<x86_thread_state64_t>().rip;
      break;

    case MachO::x86_THREAD_STATE:
      if (Count == MachO::x86_THREAD_STATE_COUNT)
        PC = Reader.read<x86_thread_state_t>().uts.ts64.rip;
      break;

    default:
      revng_log(Log, "Unexpected command flavor");
      break;
    }

  } break;

  case Architecture::arm: {

    switch (Flavor) {
    case MachO::ARM_THREAD_STATE:
      if (Count == MachO::ARM_THREAD_STATE_COUNT)
        PC = Reader.read<arm_thread_state_t>().uts.ts32.pc;
      break;

    default:
      revng_log(Log, "Unexpected command flavor");
      break;
    }

  } break;

  case Architecture::aarch64: {

    switch (Flavor) {
    case MachO::ARM_THREAD_STATE64:
      if (Count == MachO::ARM_THREAD_STATE64_COUNT)
        PC = Reader.read<arm_thread_state64_t>().pc;
      break;

    default:
      revng_log(Log, "Unexpected command flavor");
      break;
    }

  } break;

  default:
    revng_log(Log, "Unexpected architecture for Mach-O");
    break;
  }

  if (Reader.eof() and PC) {
    return MetaAddress::fromPC(Architecture::toLLVMArchitecture(Architecture),
                               *PC);

  } else {
    // TODO: emit a diagnostic message for the user.
    return MetaAddress::invalid();
  }
}

class MachOImporter : public BinaryImporterHelper {
private:
  RawBinaryView File;
  TupleTree<model::Binary> &Model;
  object::MachOObjectFile &TheBinary;

public:
  MachOImporter(TupleTree<model::Binary> &Model,
                object::MachOObjectFile &TheBinary,
                uint64_t BaseAddress) :
    BinaryImporterHelper(*Model, BaseAddress, Log),
    File(*Model, toArrayRef(TheBinary.getData())),
    Model(Model),
    TheBinary(TheBinary) {}

  llvm::Error import();

  template<typename T>
  void parseMachOSegment(ArrayRef<uint8_t> RawDataRef, const T &SegmentCommand);

  void registerBindEntry(const object::MachOBindEntry *Entry);
};

Error MachOImporter::import() {
  using LoadCommandInfo = MachOObjectFile::LoadCommandInfo;

  auto &MachO = cast<object::MachOObjectFile>(TheBinary);
  revng_assert(Model->Architecture() != Architecture::Invalid);

  if (Model->DefaultABI() == model::ABI::Invalid) {
    if (auto ABI = model::ABI::getDefaultForMachO(Model->Architecture())) {
      Model->DefaultABI() = ABI.value();
    } else {
      auto ArchName = model::Architecture::getName(Model->Architecture()).str();
      return revng::createError("Unsupported architecture for PECOFF: "
                                + ArchName);
    }
  }

  bool IsLittleEndian = Architecture::isLittleEndian(Model->Architecture());
  bool MustSwap = IsLittleEndian != sys::IsLittleEndianHost;
  StringRef StringDataRef = TheBinary.getData();
  auto RawDataRef = ArrayRef<uint8_t>(StringDataRef.bytes_begin(),
                                      StringDataRef.size());

  // Process segments first
  for (const LoadCommandInfo &LCI : MachO.load_commands()) {
    switch (LCI.C.cmd) {

    case LC_SEGMENT:
      parseMachOSegment(RawDataRef, MachO.getSegmentLoadCommand(LCI));
      break;

    case LC_SEGMENT_64:
      parseMachOSegment(RawDataRef, MachO.getSegment64LoadCommand(LCI));
      break;
    }
  }

  processSegments();

  // Identify EntryPoint
  bool EntryPointFound = false;
  std::optional<uint64_t> EntryPointOffset;
  for (const LoadCommandInfo &LCI : MachO.load_commands()) {
    switch (LCI.C.cmd) {
    case LC_UNIXTHREAD: {
      if (EntryPointFound) {
        revng_log(Log, "Multiple entry points found. Ignoring.");
        break;
      }

      EntryPointFound = true;
      const uint8_t *Pointer = reinterpret_cast<const uint8_t *>(LCI.Ptr);
      ArrayRef<uint8_t> CommandBuffer(Pointer + sizeof(thread_command),
                                      LCI.C.cmdsize - sizeof(thread_command));

      if (contains(RawDataRef, CommandBuffer)) {
        auto EntryPoint = getInitialPC(Model->Architecture(),
                                       MustSwap,
                                       CommandBuffer);
        setEntryPoint(EntryPoint);
      } else {
        revng_log(Log, "LC_UNIXTHREAD Ptr is out of bounds. Ignoring.");
      }

    } break;

    case LC_MAIN:
      if (EntryPointFound) {
        revng_log(Log, "Multiple entry points found. Ignoring.");
        break;
      }
      EntryPointFound = true;

      // This is an offset, delay translation to code for later
      EntryPointOffset = MachO.getEntryPointCommand(LCI).entryoff;
      break;

    case LC_FUNCTION_STARTS:
    case LC_DATA_IN_CODE:
    case LC_SYMTAB:
    case LC_DYSYMTAB:
      // TODO: exploit these commands
      break;
    }
  }

  if (EntryPointOffset) {
    using namespace model::Architecture;
    auto LLVMArchitecture = toLLVMArchitecture(Model->Architecture());
    auto EntryPoint = File.offsetToAddress(*EntryPointOffset)
                        .toPC(LLVMArchitecture);
    setEntryPoint(EntryPoint);
  }

  // TODO: emit following errors as diagnostic messages for the user.

  Error TheError = Error::success();
  for (const MachOBindEntry &U : MachO.bindTable(TheError))
    registerBindEntry(&U);
  if (TheError)
    revng_log(Log, "Error while decoding bindTable: " << TheError);

  for (const MachOBindEntry &U : MachO.lazyBindTable(TheError))
    registerBindEntry(&U);
  if (TheError)
    revng_log(Log, "Error while decoding lazyBindTable: " << TheError);

  // TODO: we should handle weak symbols
  for (const MachOBindEntry &U : MachO.weakBindTable(TheError))
    registerBindEntry(&U);
  if (TheError)
    revng_log(Log, "Error while decoding weakBindTable: " << TheError);

  model::flattenPrimitiveTypedefs(Model);
  model::deduplicateCollidingNames(Model);
  return Error::success();
}

template<typename T>
void MachOImporter::parseMachOSegment(ArrayRef<uint8_t> RawDataRef,
                                      const T &SegmentCommand) {
  MetaAddress Start = fromGeneric(SegmentCommand.vmaddr);
  Segment Segment({ Start, SegmentCommand.vmsize });

  Segment.StartOffset() = SegmentCommand.fileoff;
  auto MaybeEndOffset = OverflowSafeInt<uint64_t>(SegmentCommand.fileoff)
                        + SegmentCommand.filesize;
  if (not MaybeEndOffset) {
    revng_log(Log,
              "Invalid MachO segment found: overflow in computing end offset");
    return;
  }

  Segment.Name() = SegmentCommand.segname;
  Segment.FileSize() = SegmentCommand.filesize;

  Segment.IsReadable() = SegmentCommand.initprot & VM_PROT_READ;
  Segment.IsWriteable() = SegmentCommand.initprot & VM_PROT_WRITE;
  Segment.IsExecutable() = SegmentCommand.initprot & VM_PROT_EXECUTE;

  // TODO: replace the following with `populateSegmentTypeStruct`, when
  // LC_SYMTAB and LC_DYSYMTAB parsing is available
  auto &&[Struct, Type] = Model->makeStructDefinition(Segment.VirtualSize());
  Struct.CanContainCode() = Segment.IsExecutable();
  Segment.Type() = std::move(Type);

  Segment.verify(true);

  Model->Segments().insert(std::move(Segment));

  // TODO: parse sections contained in segments LC_SEGMENT and LC_SEGMENT_64
}

void MachOImporter::registerBindEntry(const object::MachOBindEntry *Entry) {
  MetaAddress Target = fromGeneric(Entry->address());
  uint64_t Addend = static_cast<uint64_t>(Entry->addend());
  RelocationType::Values Type = RelocationType::Invalid;
  (void) Type;
  uint64_t PointerSize = Architecture::getPointerSize(Model->Architecture());

  switch (Entry->type()) {
  case BIND_TYPE_POINTER:
    if (PointerSize == 4) {
      Type = RelocationType::WriteAbsoluteAddress32;
    } else if (PointerSize == 8) {
      Type = RelocationType::WriteAbsoluteAddress64;
    } else {
      revng_abort();
    }
    break;

  case BIND_TYPE_TEXT_ABSOLUTE32:
    Type = RelocationType::WriteAbsoluteAddress32;
    break;

  case BIND_TYPE_TEXT_PCREL32:
    Type = RelocationType::WriteRelativeAddress32;
    Addend = Addend - 4;
    break;

  case BIND_TYPE_INVALID:
  default:
    revng_log(Log,
              "Ignoring unexpected bind entry with type " << Entry->type());
    break;
  }

  // TODO: record relocation on symbol
}

Error importMachO(TupleTree<model::Binary> &Model,
                  object::MachOObjectFile &TheBinary,
                  const ImporterOptions &Options) {
  MachOImporter Importer(Model, TheBinary, Options.BaseAddress);
  return Importer.import();
}
