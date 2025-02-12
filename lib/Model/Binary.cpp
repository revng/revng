/// \file Binary.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <queue>

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Binary.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/CommandLine.h"

namespace {

// TODO: all this logic should be moved to lib/TupleTree
Logger<> FieldAccessedLogger("field-accessed");

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
  if (T->ID() != 0) {
    std::string Error = "Newly recorded types must not have an ID.\n"
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

namespace model {

MetaAddressRangeSet Binary::executableRanges() const {
  MetaAddressRangeSet ExecutableRanges;

  struct Entry {
    Entry(MetaAddress Start, const model::StructDefinition &Type) :
      Start(Start), Type(Type) {}
    MetaAddress Start;
    const model::StructDefinition &Type;
  };
  std::queue<Entry> Queue;

  for (const model::Segment &Segment : Segments()) {
    if (Segment.IsExecutable()) {
      if (const auto *SegmentType = Segment.type()) {
        Queue.emplace(Segment.StartAddress(), *SegmentType);
      } else {
        ExecutableRanges.add(Segment.StartAddress(), Segment.endAddress());
      }
    }
  }

  while (not Queue.empty()) {
    auto QueueEntry = Queue.front();
    Queue.pop();

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
        ExecutableRanges.add(PaddingStart, PaddingEnd);

      // Enqueue the field type for processing
      //
      // Note: this only considers struct fields, so if any other type is in
      //       the way, the traversal stops.
      if (const model::StructDefinition *Struct = Field.Type()->getStruct())
        if (Struct->CanContainCode())
          Queue.emplace(FieldStart, *Struct);

      // Set the next padding start
      auto FieldSize = *Field.Type()->size(Helper);
      PaddingStart = FieldStart + FieldSize;
    }

    // Record the trailing padding, if any
    PaddingEnd = QueueEntry.Start + QueueEntry.Type.Size();
    if (PaddingStart != PaddingEnd)
      ExecutableRanges.add(PaddingStart, PaddingEnd);
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
