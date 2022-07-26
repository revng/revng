//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Fake the 'labeled-block' for indentation purposes.
var labeledBlock = undefined;
for (block of document.querySelectorAll('[data-scope="asm.basic-block"]')) {
  var labels = block.querySelectorAll('[data-token="asm.label"]');
  if (labels.length == 0) {
    if (labeledBlock == undefined)
      throw new Error("The first block must always be labeled");

    block.id = block.dataset.locationDefinition.substring(1);
    labeledBlock.appendChild(block);
  } else if (labels.length == 1) {
    labeledBlock = document.createElement("div");
    labeledBlock.dataset.scope = "asm.labeled-block";
    block.parentNode.appendChild(labeledBlock);
    labeledBlock.appendChild(block);
    labeledBlock.id = block.dataset.locationDefinition.substring(1);

    link = document.createElement("a");
    link.href = "#" + labeledBlock.id;
    labels[0].appendChild(link).appendChild(link.previousSibling);
  } else {
    throw new Error("Multiple labels in a single block");
  }
}
