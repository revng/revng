//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/CliftEmitC/Configuration.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftPipes/EmitCAsDirectory.h"
#include "revng/Support/GzipTarFile.h"
#include "revng/Support/ResourceFinder.h"

void revng::pypeline::piperuns::EmitCAsDirectory::run() {
  std::unique_ptr<llvm::raw_pwrite_stream> Out = Output.getOStream(ObjectID());
  revng_assert(Out);

  GzipTarWriter TarWriter{ *Out };

  llvm::StringRef Buffer = InputC.getMemoryBuffer(ObjectID{})->getBuffer();
  TarWriter.append("decompiled/functions.c",
                   llvm::ArrayRef<char>(Buffer.data(), Buffer.size()));

  Buffer = InputTypesAndGlobals.getMemoryBuffer(ObjectID{})->getBuffer();
  TarWriter.append("decompiled/types-and-globals.h",
                   llvm::ArrayRef<char>(Buffer.data(), Buffer.size()));

  Buffer = InputHelpers.getMemoryBuffer(ObjectID{})->getBuffer();
  TarWriter.append("decompiled/helpers.h",
                   llvm::ArrayRef<char>(Buffer.data(), Buffer.size()));

  {
    auto Path = revng::ResourceFinder.findFile("share/revng/include/"
                                               "attributes.h");

    if (not Path or Path->empty())
      revng_abort("can't find attributes.h");

    auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*Path);
    auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));

    TarWriter.append("decompiled/attributes.h",
                     { Buffer->getBufferStart(), Buffer->getBufferSize() });
  }

  {
    auto Path = revng::ResourceFinder.findFile("share/revng/include/"
                                               "primitive-types.h");

    if (not Path or Path->empty())
      revng_abort("can't find primitive-types.h");

    auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*Path);
    auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));

    TarWriter.append("decompiled/primitive-types.h",
                     { Buffer->getBufferStart(), Buffer->getBufferSize() });
  }

  TarWriter.close();
}
