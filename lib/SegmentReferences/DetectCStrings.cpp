//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringExtras.h"

#include "revng/Model/RawBinaryView.h"
#include "revng/Pipebox/Helpers.h"
#include "revng/SegmentReferences/DetectCStrings.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Unicode.h"

using namespace llvm;

static Logger Log("detect-c-strings");

void DetectCStrings::run(llvm::Module &M, llvm::Function *LimitTo) {
  revng_log(Log, "Starting detection");
  LoggerIndent Indent(Log);

  for (auto &&SegmentUse : SegmentUses.getUses(M, LimitTo)) {
    revng_log(Log,
              "Considering segment use "
                << getName(SegmentUse.TheUse->getUser()) << ". Address is "
                << SegmentUse.Address.toString() << ".");
    LoggerIndent Indent(Log);
    auto MaybeData = BinaryView.getFromAddressOn(SegmentUse.Address);

    // Check if there's data at the given offset
    if (not MaybeData.has_value()) {
      revng_log(Log, "No data, bailing out");
      continue;
    }

    // Does it look like a unicode string?
    auto String = UnicodeCStringView::getPrintable(*MaybeData);
    if (not String.isValid() or String.codePointCount() <= 4) {
      revng_log(Log, "Doesn't look like a valid string");
      continue;
    }

    // Create a uint8_t array of the size of the string (including the NUL
    // byte)
    auto CharSize = String.charSize();
    auto UInt8 = model::PrimitiveType::makeConstUnsigned(CharSize);
    auto StringType = model::ArrayType::make(std::move(UInt8),
                                             String.data().size() / CharSize);
    bool Success = GlobalBuilder.insert(SegmentUse.Address,
                                        std::move(StringType));

    if (Log.isEnabled()) {
      Log << (Success ? "Added" : "Not added") << " \"";
      llvm::printEscapedString(String.data(), *Log.getAsLLVMStream());
      Log << "\" (" << String.codePointCount() << " code points) at "
          << SegmentUse.Address.toString();
      Log << DoLog;
    }
  }
}

namespace revng::pypeline::analyses {

llvm::Error DetectCStrings::run(Model &Model,
                                const Request &Incoming,
                                llvm::StringRef Configuration,
                                const BinariesContainer &Binaries,
                                LLVMFunctionContainer &ModuleContainer) {
  RawBinaryView BinaryView = makeBinaryView(Model, Binaries);
  ::DetectCStrings StringDetector(*Model.get().get(), BinaryView);

  for (const ObjectID *Object : Incoming[1])
    StringDetector.run(ModuleContainer.getModule(*Object));

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
