/**
 * This file is distributed under the MIT License. See LICENSE.mit for details.
 */

import { genGuid } from "./tuple_tree";

const metaAddressTypes = [
    "Invalid",
    "Generic32",
    "Generic64",
    "Code_x86",
    "Code_x86_64",
    "Code_mips",
    "Code_mipsel",
    "Code_arm",
    "Code_arm_thumb",
    "Code_aarch64",
    "Code_systemz",
] as const;
type MetaAddressType = (typeof metaAddressTypes)[number];

function maxValue(type: MetaAddressType) {
    switch (type) {
        case "Invalid":
            throw new Error("Cannot get maxValue of Invalid");
        case "Generic32":
        case "Code_x86":
        case "Code_arm_thumb":
        case "Code_mips":
        case "Code_mipsel":
        case "Code_arm":
            return 2n ** 32n;
        case "Generic64":
        case "Code_x86_64":
        case "Code_systemz":
        case "Code_aarch64":
            return 2n ** 64n;
    }
}

const metaAddressRegex = new RegExp(
    "^(?<Address>(0x[0-9a-fA-F]+)|)" +
        `:(?<Type>${metaAddressTypes.join("|")})` +
        "(:(?<Epoch>\\d+))?" +
        "(:(?<AddressSpace>\\d+))?$"
);

export type IMetaAddress = string;

export class MetaAddress {
    address: bigint;
    type: MetaAddressType;
    epoch: bigint;
    addressSpace: bigint;

    constructor(metaAddress?: IMetaAddress);
    constructor(address: bigint, type: MetaAddressType, epoch: bigint, addressSpace: bigint);
    constructor(
        stringOrAddress?: string | bigint,
        type?: MetaAddressType,
        epoch?: bigint,
        addressSpace?: bigint
    ) {
        if (typeof stringOrAddress === "string") {
            const match = stringOrAddress.match(metaAddressRegex);
            if (match !== null && match.groups !== undefined) {
                const address = match.groups["Address"];
                if (address !== undefined) {
                    this.type = match.groups["Type"] as MetaAddressType;
                    this.address = BigInt(address);
                    this.epoch = BigInt(match.groups["Epoch"] || 0);
                    this.addressSpace = BigInt(match.groups["AddressSpace"] || 0);
                } else {
                    this.type = "Invalid";
                }
            } else {
                this.type = "Invalid";
            }
        } else if (stringOrAddress === undefined) {
            this.type = "Invalid";
        } else {
            this.type = type || "Invalid";
            this.address = stringOrAddress;
            this.epoch = epoch || 0n;
            this.addressSpace = addressSpace || 0n;
        }
        if (!this.isValid()) {
            this.type = "Invalid";
        }
    }

    isValid() {
        if (this.type === "Invalid") {
            return true;
        }
        return (
            this.address >= 0n &&
            this.address < maxValue(this.type) &&
            this.epoch >= 0n &&
            this.epoch < 2n ** 32n &&
            this.addressSpace >= 0n &&
            this.addressSpace < 2n ** 16n
        );
    }

    addOffset(offset: bigint): MetaAddress {
        if (this.type === "Invalid") {
            return new MetaAddress(":Invalid");
        }
        const newAddress = this.address + offset;
        if (newAddress > maxValue(this.type)) {
            return new MetaAddress(":Invalid");
        }
        return new MetaAddress(newAddress, this.type, this.epoch, this.addressSpace);
    }

    toString(): string {
        if (this.type === "Invalid") {
            return `:${this.type}`;
        }

        const cleanStr = (num: bigint) => (num !== 0n ? `:${num.toString(10)}` : "");
        return (
            `0x${this.address.toString(16)}:${this.type}` +
            `${cleanStr(this.epoch)}${cleanStr(this.addressSpace)}`
        );
    }

    toJSON(): string {
        return this.toString();
    }

    equals(other: unknown): boolean {
        if (other instanceof MetaAddress) {
            return this.toString() == other.toString();
        } else {
            return false;
        }
    }
}

export function genPrimitiveTypeGuid(rawObject: IType): bigint {
    const realObject = rawObject as IPrimitiveType;
    const index = PrimitiveTypeKindValues.findIndex((e) => e === realObject.PrimitiveKind);
    return BigInt(BigInt(index << 8) | realObject.Size);
}

const genEnumTypeGuid = genGuid;
const genTypedefTypeGuid = genGuid;
const genStructTypeGuid = genGuid;
const genUnionTypeGuid = genGuid;
const genCABIFunctionTypeGuid = genGuid;
const genRawFunctionTypeGuid = genGuid;
