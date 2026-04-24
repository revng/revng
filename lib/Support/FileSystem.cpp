//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "revng/Support/Debug.h"
#include "revng/Support/FileSystem.h"

using namespace llvm;

static Logger Log("filesystem");

std::optional<std::string>
findPathCaseInsensitive(StringRef Root, StringRef CaseInsensitivePath) {
  using sys::fs::directory_iterator;

  revng_log(Log, "Find " << CaseInsensitivePath << " in " << Root);
  LoggerIndent Indent(Log);

  SmallVector<StringRef> Components;
  CaseInsensitivePath.split(Components, "/", -1, false);

  SmallString<16> Path(Root);
  for (StringRef Component : Components) {
    revng_log(Log, "Considering path component " << Component);

    std::error_code EC;
    bool Found = false;
    for (directory_iterator It = directory_iterator(Path, EC), End;
         It != End && !EC;
         It.increment(EC)) {

      StringRef EntryName = llvm::sys::path::filename(It->path());

      // See if this entry matches the expected request
      if (EntryName.equals_insensitive(Component)) {
        llvm::sys::path::append(Path, EntryName);
        revng_log(Log, "Found: " << EntryName);

        Found = true;

        // Stop on the first one
        break;
      }
    }

    if (EC or not Found)
      return std::nullopt;
  }

  revng_log(Log, "Result: " << Path.str().str());
  return Path.str().str();
}
