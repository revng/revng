//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/CliftEmitC/Configuration.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftPipes/EmitCAsDirectory.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/Tar.h"

void revng::pypeline::piperuns::EmitCAsDirectory::run() {
  std::unique_ptr<llvm::raw_pwrite_stream> Out = Output.getOStream(ObjectID());
  revng_assert(Out);

  revng::TarWriter TarWriter{ *Out, TarFormat::Gzip };

  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  Buffer = InputC.getMemoryBuffer(ObjectID{});
  TarWriter.addMember("decompiled/functions.c", *Buffer);

  Buffer = InputTypesAndGlobals.getMemoryBuffer(ObjectID{});
  TarWriter.addMember("decompiled/types-and-globals.h", *Buffer);

  Buffer = InputHelpers.getMemoryBuffer(ObjectID{});
  TarWriter.addMember("decompiled/helpers.h", *Buffer);

  auto AddHeader = [&](llvm::StringRef HeaderName) {
    std::string HeaderPath;
    {
      llvm::raw_string_ostream Out(HeaderPath);
      Out << "share/revng/include/" << HeaderName;
    }

    auto Path = revng::ResourceFinder.findFile(HeaderPath);
    if (not Path or Path->empty())
      revng_abort("cannot find header");

    auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*Path);
    auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));

    std::string TarHeaderPath;
    {
      llvm::raw_string_ostream Out(TarHeaderPath);
      Out << "decompiled/" << HeaderName;
    }

    TarWriter.addMember(TarHeaderPath, *Buffer);
  };

  AddHeader("attributes.h");
  AddHeader("primitive-types.h");
  AddHeader("runtime-library.h");
}
