//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <optional>
#include <queue>
#include <type_traits>

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/Concepts.h"
#include "revng/Model/Binary.h"
#include "revng/Model/BinaryIdentifier.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/CommandLine.h"

#include "NamespaceBuilder.h"

namespace {

// TODO: all this logic should be moved to lib/TupleTree
Logger FieldAccessedLogger("field-accessed");

constexpr const char *StructNameHelpText = "regex that will make the program "
                                           "assert when a model struct which "
                                           "name matches this option is "
                                           "accessed. NOTE: enable "
                                           "field-accessed logger, optionally "
                                           "break on onFieldAccess from gdb.";
llvm::cl::opt<std::string> StructNameRegex("tracking-debug-struct-name",
                                           llvm::cl::desc(StructNameHelpText),
                                           llvm::cl::init(""),
                                           llvm::cl::cat(MainCategory));
constexpr const char *FieldNameHelpText = "regex that will "
                                          "make the "
                                          "program assert when "
                                          "a field "
                                          "of a model struct "
                                          "which name "
                                          "matches this "
                                          "option accessed. NOTE: enable "
                                          "field-accessed logger, optionally "
                                          "break on onFieldAccess from gdb.";

llvm::cl::opt<std::string> FieldNameRegex("tracking-debug-field-name",
                                          llvm::cl::desc(FieldNameHelpText),
                                          llvm::cl::init(""),
                                          llvm::cl::cat(MainCategory));

} // namespace

/// This is disabled by default, so it's fine to use something like this
/// internally to make debugging easier.
void onFieldAccess(llvm::StringRef FieldName, llvm::StringRef StructName) {
  if (FieldAccessedLogger.isEnabled()) {
    FieldAccessedLogger << (StructName + "::" + FieldName + " accessed").str();
    {
      auto LLVMStream = FieldAccessedLogger.getAsLLVMStream();
      llvm::sys::PrintStackTrace(*LLVMStream);
    }

    FieldAccessedLogger << DoLog;
  }
}

void fieldAccessed(llvm::StringRef FieldName, llvm::StringRef StructName) {
  if (StructNameRegex == "" and FieldNameRegex == "")
    return;

  llvm::Regex Reg(StructNameRegex);
  if (StructNameRegex != "" and not Reg.match(StructName))
    return;

  llvm::Regex Reg2(FieldNameRegex);
  if (FieldNameRegex != "" and not Reg2.match(FieldName))
    return;

  onFieldAccess(FieldName, StructName);
}

std::pair<model::TypeDefinition &, model::UpcastableType>
model::Binary::recordNewType(model::UpcastableTypeDefinition &&T) {
  revng_assert(!T.isEmpty());

  // Assign progressive ID
  if (T->ID() != uint64_t(-1)) {
    std::string Error = "Types must not have an ID before they are a part of "
                        "a binary.\n"
                        + ::toString(T);
    revng_abort(Error.c_str());
  }

  T->ID() = getAvailableTypeID();

  auto &&[It, Success] = TypeDefinitions().insert(T);
  revng_assert(Success);

  return { **It, makeType((*It)->key()) };
}

uint64_t model::Binary::getAvailableTypeID() const {
  if (TypeDefinitions().empty())
    return 0;

  return TypeDefinitions().rbegin()->get()->ID() + 1;
}

model::TypeDefinitionReference
model::Binary::getTypeDefinitionReference(const model::TypeDefinition::Key
                                            &Key) {
  using Fields = TupleLikeTraits<model::Binary>::Fields;
  TupleTreePath BinaryPath;
  BinaryPath.push_back(static_cast<size_t>(Fields::TypeDefinitions));
  BinaryPath.push_back(Key);
  return model::TypeDefinitionReference{ this, BinaryPath };
}

model::TypeDefinitionReference
model::Binary::getTypeDefinitionReference(const model::TypeDefinition::Key &Key)
  const {
  using Fields = TupleLikeTraits<model::Binary>::Fields;
  TupleTreePath BinaryPath;
  BinaryPath.push_back(static_cast<size_t>(Fields::TypeDefinitions));
  BinaryPath.push_back(Key);
  return model::TypeDefinitionReference{ this, BinaryPath };
}

model::BinaryIdentifierReference
model::Binary::getBinaryIdentifierReference(const model::BinaryIdentifier::Key
                                              &Key) {
  using Fields = TupleLikeTraits<model::Binary>::Fields;
  TupleTreePath BinaryPath;
  BinaryPath.push_back(static_cast<size_t>(Fields::Binaries));
  BinaryPath.push_back(std::get<0>(Key));
  return model::BinaryIdentifierReference{ this, BinaryPath };
}

model::BinaryIdentifierReference
model::Binary::getBinaryIdentifierReference(const model::BinaryIdentifier::Key
                                              &Key) const {
  using Fields = TupleLikeTraits<model::Binary>::Fields;
  TupleTreePath BinaryPath;
  BinaryPath.push_back(static_cast<size_t>(Fields::Binaries));
  BinaryPath.push_back(std::get<0>(Key));
  return model::BinaryIdentifierReference{ this, BinaryPath };
}

model::ABI::Values model::Binary::targetABI() const {
  model::ABI::Values ABI = TargetABI();

  if (ABI == model::ABI::Invalid) {
    ABI = DefaultABI();

    // TODO: We should do something smarter here:
    //       * Pick a better fallback (maybe other model properties), and/or
    //       * after exhausting all fallbacks, return invalid and additionally
    //         check the availability of a valid ABI in `checkPrecondition`.
    if (ABI == model::ABI::Invalid)
      ABI = model::ABI::SystemV_x86_64;
  }

  return ABI;
}

namespace model {

MetaAddressRangeSet Binary::executableRanges() const {
  MetaAddressRangeSet ExecutableRanges;
  struct Entry {
    Entry(MetaAddress Start,
          MetaAddress End,
          const model::StructDefinition &Type) :
      Start(Start), End(End), Type(Type) {}
    MetaAddress Start;
    MetaAddress End;
    const model::StructDefinition &Type;
  };
  std::queue<Entry> Queue;

  for (const model::Segment &Segment : Segments()) {
    if (Segment.IsExecutable()) {
      if (const auto *SegmentType = Segment.type()) {
        Queue.emplace(Segment.StartAddress(),
                      Segment.endDataAddress(),
                      *SegmentType);
      } else {
        ExecutableRanges.add(Segment.StartAddress(), Segment.endDataAddress());
      }
    }
  }

  while (not Queue.empty()) {
    auto QueueEntry = Queue.front();
    Queue.pop();

    // This function record an entry in ExecutableRanges, keeping into account
    // what data is actually on disk. In practice, we avoid marking executable
    // .bss.
    auto Register = [&QueueEntry, &ExecutableRanges](const MetaAddress &Start,
                                                     const MetaAddress &End) {
      revng_assert(Start >= QueueEntry.Start);

      if (Start >= QueueEntry.End) {
        // Ignoring this range: it starts after the end of the data available on
        // disk
        return;
      }

      if (End > QueueEntry.End) {
        // The range we're trying to add ends *after* the data available on
        // disk. Limit the range accordingly.
        ExecutableRanges.add(Start, QueueEntry.End);
      } else {
        ExecutableRanges.add(Start, End);
      }
    };

    MetaAddress PaddingStart = QueueEntry.Start;
    MetaAddress PaddingEnd;
    model::VerifyHelper Helper;

    revng_assert(QueueEntry.Type.CanContainCode());
    for (const model::StructField &Field : QueueEntry.Type.Fields()) {
      // Record the start address of field
      MetaAddress FieldStart = QueueEntry.Start + Field.Offset();

      // Update the end of padding
      PaddingEnd = FieldStart;

      // Register the padding as an executable range
      if (PaddingStart != PaddingEnd)
        Register(PaddingStart, PaddingEnd);

      // Enqueue the field type for processing
      //
      // Note: this only considers struct fields, so if any other type is in
      //       the way, the traversal stops.
      if (const model::StructDefinition *Struct = Field.Type()->getStruct())
        if (Struct->CanContainCode())
          Queue.emplace(FieldStart, QueueEntry.End, *Struct);

      // Set the next padding start
      auto FieldSize = *rc_eval(Field.Type()->size(Helper));
      PaddingStart = FieldStart + FieldSize;
    }

    // Record the trailing padding, if any
    PaddingEnd = QueueEntry.Start + QueueEntry.Type.Size();
    if (PaddingStart != PaddingEnd)
      Register(PaddingStart, PaddingEnd);
  }

  return ExecutableRanges;
}

namespace RelocationType {

Values fromELFRelocation(model::Architecture::Values Architecture,
                         unsigned char ELFRelocation) {
  using namespace llvm::ELF;
  switch (Architecture) {
  case model::Architecture::x86:
    switch (ELFRelocation) {
    case R_386_RELATIVE:
    case R_386_32:
      return AddAbsoluteAddress32;

    case R_386_JUMP_SLOT:
    case R_386_GLOB_DAT:
      return WriteAbsoluteAddress32;

    case R_386_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::x86_64:
    switch (ELFRelocation) {
    case R_X86_64_RELATIVE:
      return AddAbsoluteAddress64;

    case R_X86_64_JUMP_SLOT:
    case R_X86_64_GLOB_DAT:
    case R_X86_64_64:
      return WriteAbsoluteAddress64;

    case R_X86_64_32:
      return WriteAbsoluteAddress32;

    case R_X86_64_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::arm:
    switch (ELFRelocation) {
    case R_ARM_RELATIVE:
      return AddAbsoluteAddress32;

    case R_ARM_JUMP_SLOT:
    case R_ARM_GLOB_DAT:
      return WriteAbsoluteAddress32;

    case R_ARM_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::aarch64:
    return Invalid;

  case model::Architecture::mips:
  case model::Architecture::mipsel:
    switch (ELFRelocation) {
    case R_MIPS_IMPLICIT_RELATIVE:
      return AddAbsoluteAddress32;

    case R_MIPS_JUMP_SLOT:
    case R_MIPS_GLOB_DAT:
      return WriteAbsoluteAddress32;

    case R_MIPS_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::systemz:
    switch (ELFRelocation) {
    case R_390_GLOB_DAT:
      return WriteAbsoluteAddress64;

    case R_390_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  default:
    revng_abort();
  }
}

bool isELFRelocationBaseRelative(model::Architecture::Values Architecture,
                                 unsigned char ELFRelocation) {
  using namespace llvm::ELF;
  switch (Architecture) {
  case model::Architecture::x86:
    switch (ELFRelocation) {
    case R_386_RELATIVE:
      return true;

    case R_386_32:
    case R_386_JUMP_SLOT:
    case R_386_GLOB_DAT:
      return false;

    case R_386_COPY:
      // TODO: use

    default:
      return Invalid;
    }

  case model::Architecture::x86_64:
    switch (ELFRelocation) {
    case R_X86_64_RELATIVE:
      return true;

    case R_X86_64_JUMP_SLOT:
    case R_X86_64_GLOB_DAT:
    case R_X86_64_64:
    case R_X86_64_32:
      return false;

    case R_X86_64_COPY:
      // TODO: use

    default:
      return Invalid;
    }

  case model::Architecture::arm:
    switch (ELFRelocation) {
    case R_ARM_RELATIVE:
      return true;

    case R_ARM_JUMP_SLOT:
    case R_ARM_GLOB_DAT:
      return false;

    case R_ARM_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::aarch64:
    return Invalid;

  case model::Architecture::mips:
  case model::Architecture::mipsel:
    switch (ELFRelocation) {
    case R_MIPS_IMPLICIT_RELATIVE:
      return true;

    case R_MIPS_JUMP_SLOT:
    case R_MIPS_GLOB_DAT:
      return false;

    case R_MIPS_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::systemz:
    switch (ELFRelocation) {
    case R_390_GLOB_DAT:
      return false;

    case R_390_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  default:
    revng_abort();
  }
}

Values formCOFFRelocation(model::Architecture::Values Architecture) {
  switch (Architecture) {
  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return WriteAbsoluteAddress32;

  case model::Architecture::x86_64:
  case model::Architecture::aarch64:
  case model::Architecture::systemz:
    return WriteAbsoluteAddress64;
  default:
    revng_abort();
  }
}

} // namespace RelocationType
} // namespace model

std::set<uint64_t> model::Binary::collectAllTypeSizes() const {
  // TODO: don't hardcode this set here. Share it with the other users!
  //       Important: this should already contain all the primitive sizes we
  //       support (which includes all the pointers sizes).
  std::set<uint64_t> ByteSizes = { 1, 2, 4, 8, 10, 12, 16 };

  VerifyHelper SizeCache;
  for (const model::UpcastableTypeDefinition &Type : this->TypeDefinitions()) {
    // This takes care of all the type definitions, meaning we don't have to
    // look at defined types anymore.
    if (std::optional<uint64_t> MaybeSize = Type->size(SizeCache))
      ByteSizes.insert(MaybeSize.value());

    for (const model::Type *Edge : Type->edges()) {
      // Since primitives, pointers and defined types are already taken care of,
      // we are only interested in arrays here.

      // IMPORTANT: do not forget to update this after new type kinds are added!

      while (!llvm::isa<model::PrimitiveType>(Edge)
             && !llvm::isa<model::DefinedType>(Edge)) {
        if (const auto *Pointer = llvm::dyn_cast<model::PointerType>(Edge)) {
          // Keep going deeper on a pointer in case it's a pointer to an array.
          Edge = Pointer->PointeeType().get();

        } else if (const auto *Array = llvm::dyn_cast<model::ArrayType>(Edge)) {
          if (std::optional<uint64_t> MaybeSize = Edge->trySize(SizeCache))
            ByteSizes.insert(MaybeSize.value());

          // Keep going deeper on an array in case it's a nested one.
          Edge = Array->ElementType().get();

        } else {
          revng_abort("Unsupported type kind!");
        }
      }
    }

    if (const auto *RFT = Type->getRawFunction()) {
      uint64_t ReturnTypeSize = 0;
      for (const auto &RV : RFT->ReturnValues()) {
        std::optional<uint64_t> MaybeSize = RV.Type()->trySize(SizeCache);
        revng_assert(MaybeSize.has_value());
        ReturnTypeSize += MaybeSize.value();
      }

      if (ReturnTypeSize)
        ByteSizes.insert(ReturnTypeSize);
    }
  }

  return ByteSizes;
}

void model::Binary::dumpTypeGraph(const char *Path) const {
  DisableTracking Guard(*this);

  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out, *this);
  TSPrinter.print();
}

void model::Function::dumpTypeGraph(const char *Path,
                                    const model::Binary &Binary) const {
  DisableTracking Guard(*this);
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out, Binary);
  TSPrinter.print(*this);
}

void model::TypeDefinition::dumpTypeGraph(const char *Path,
                                          const model::Binary &Binary) const {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out, Binary);
  TSPrinter.print(*this);
}

llvm::StringRef model::Architecture::getPCCSVName(Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return "_rip";

  case model::Architecture::x86:
    return "_eip";

  case model::Architecture::systemz:
    return "_psw_addr";

  case model::Architecture::arm:
  case model::Architecture::aarch64:
    return "_pc";

  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return "_PC";

  default:
    revng_abort();
  }
}

#define UnknownCSVPrefix "state_"

namespace {

// An x86-64 zmm register occupies a contiguous 64-byte slot in the CPU state,
// laid out as eight 8-byte lanes. The lifter materializes one i64 CSV per lane,
// named `_state_0x<cpu-state-offset>`. This table is the single source of truth
// for the base CPU-state offset of each register.
constexpr uint64_t ZMMLaneSize = 8;
constexpr uint64_t ZMMLaneCount = 8;

struct ZMMRegisterCSV {
  model::Register::Values Register;
  uint64_t BaseOffset;
};

constexpr std::array<ZMMRegisterCSV, 8> ZMMRegisters{ {
  { model::Register::zmm0_x86_64, 0x2b10 },
  { model::Register::zmm1_x86_64, 0x2b50 },
  { model::Register::zmm2_x86_64, 0x2b90 },
  { model::Register::zmm3_x86_64, 0x2bd0 },
  { model::Register::zmm4_x86_64, 0x2c10 },
  { model::Register::zmm5_x86_64, 0x2c50 },
  { model::Register::zmm6_x86_64, 0x2c90 },
  { model::Register::zmm7_x86_64, 0x2cd0 },
} };

std::optional<uint64_t> zmmBaseOffset(model::Register::Values V) {
  for (const ZMMRegisterCSV &Entry : ZMMRegisters)
    if (Entry.Register == V)
      return Entry.BaseOffset;
  return std::nullopt;
}

std::string stateCSVName(uint64_t Offset) {
  return "_" UnknownCSVPrefix "0x"
         + llvm::utohexstr(Offset, /* LowerCase */ true);
}

// If `Name` is the CSV of an x86-64 zmm lane, return the register it belongs to
// and the byte portion of it the CSV covers.
std::optional<model::Register::RegisterPortion>
zmmPortionFromCSVName(llvm::StringRef Name) {
  if (not Name.consume_front("_" UnknownCSVPrefix "0x"))
    return std::nullopt;

  uint64_t Offset = 0;
  if (Name.getAsInteger(/* Radix */ 16, Offset))
    return std::nullopt;

  for (const ZMMRegisterCSV &Entry : ZMMRegisters) {
    if (Entry.BaseOffset <= Offset
        and Offset < Entry.BaseOffset + ZMMLaneCount * ZMMLaneSize) {
      return model::Register::RegisterPortion{ Entry.Register,
                                               Offset - Entry.BaseOffset,
                                               ZMMLaneSize };
    }
  }

  return std::nullopt;
}

// Resolve a CSV name that does not denote an x86-64 zmm lane.
model::Register::Values
nonVectorRegisterFromCSVName(llvm::StringRef Name,
                             model::Architecture::Values Architecture) {
  if (not Name.consume_front("_"))
    return model::Register::Invalid;

  if (Architecture == model::Architecture::x86
      and Name == UnknownCSVPrefix "0x2960")
    return model::Register::st0_x86;

  return model::Register::fromRegisterName(Name, Architecture);
}

} // namespace

std::string model::Register::getCSVName(Values V) {
  if (std::optional<uint64_t> Base = zmmBaseOffset(V))
    return stateCSVName(*Base);

  if (V == st0_x86)
    return stateCSVName(0x2960);

  return "_" + model::Register::getRegisterName(V).str();
}

model::Register::Values
model::Register::fromCSVName(llvm::StringRef Name,
                             model::Architecture::Values Architecture) {
  return registerPortionFromCSVName(Name, Architecture).Register;
}

std::vector<model::Register::CSV> model::Register::getCSVs(Values V) {
  // A zmm register spans several CSVs, one per 8-byte lane; every other
  // register maps to a single CSV covering the whole register.
  if (std::optional<uint64_t> Base = zmmBaseOffset(V)) {
    std::vector<CSV> Result;
    Result.reserve(ZMMLaneCount);
    for (uint64_t Lane = 0; Lane < ZMMLaneCount; ++Lane)
      Result.push_back({ stateCSVName(*Base + Lane * ZMMLaneSize),
                         Lane * ZMMLaneSize,
                         ZMMLaneSize });
    return Result;
  }

  return { { getCSVName(V), 0, model::Register::getSize(V) } };
}

std::string model::Register::singleCSVName(Values V) {
  std::vector<CSV> CSVs = getCSVs(V);
  revng_assert(CSVs.size() == 1,
               "singleCSVName called on a register composed of multiple CSVs");
  return CSVs.front().Name;
}

model::Register::RegisterPortion
model::Register::registerPortionFromCSVName(llvm::StringRef Name,
                                            model::Architecture::Values
                                              Architecture) {
  if (Architecture == model::Architecture::x86_64)
    if (std::optional<RegisterPortion> Portion = zmmPortionFromCSVName(Name))
      return *Portion;

  // Non-zmm CSVs map to a single register covering its whole size.
  model::Register::Values Register = nonVectorRegisterFromCSVName(Name,
                                                                  Architecture);
  uint64_t Size = Register == model::Register::Invalid ?
                    0 :
                    model::Register::getSize(Register);
  return { Register, 0, Size };
}

#undef UnknownCSVPrefix

template<ConstOrNot<model::Binary> BinaryType>
static std::pair<ConstIf<std::is_const_v<BinaryType>, model::Segment> *,
                 uint64_t>
getSegmentForImpl(BinaryType &Binary, const MetaAddress &Address) {
  revng_assert(Address.isValid());
  revng_assert(Address.isGeneric());

  for (auto &TheSegment : Binary.Segments()) {
    if (TheSegment.contains(Address)) {
      auto MaybeOffset = Address - TheSegment.StartAddress();
      if (MaybeOffset.has_value())
        return { &TheSegment, MaybeOffset.value() };
    }
  }

  return { nullptr, 0 };
}

std::pair<const model::Segment *, uint64_t>
model::Binary::getSegmentFor(const MetaAddress &Address) const {
  return getSegmentForImpl(*this, Address);
}

std::pair<model::Segment *, uint64_t>
model::Binary::getSegmentFor(const MetaAddress &Address) {
  return getSegmentForImpl(*this, Address);
}
