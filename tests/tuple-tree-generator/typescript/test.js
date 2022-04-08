//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
"use strict";

const fs = require("fs");
const process = require("process");
const model = require("revng-model");

const file = fs.readFileSync(process.argv[2], { encoding: "utf-8" });
const model_file = model.parseModel(file);
fs.writeFileSync(process.argv[3], model.dumpModel(model_file));
