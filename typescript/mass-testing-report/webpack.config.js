/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
    entry: {
        "main.js": "./src/main.ts",
    },
    mode: "production",
    module: {
        rules: [
            {
                test: /\.ts$/,
                exclude: /node_modules/,
                use: ["ts-loader"],
            },
            {
                test: /\.css$/,
                use: ["style-loader", "css-loader"],
            },
        ],
    },
    plugins: [
        new CopyPlugin({
            patterns: [{ from: "src/*.html", to: "[name][ext]" }],
        }),
    ],
    resolve: {
        fallback: {
            path: require.resolve("path-browserify"),
        },
    },
    output: {
        filename: "[name]",
        path: path.resolve(__dirname, "dist"),
    },
    devtool: undefined,
};
