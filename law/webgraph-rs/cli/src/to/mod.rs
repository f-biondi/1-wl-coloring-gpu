/*
 * SPDX-FileCopyrightText: 2024 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Subcommand;

use super::GlobalArgs;

pub mod arcs;
pub mod gdata;
pub mod arcscuda;
pub mod arcsvalmari;
pub mod ascii;
pub mod bvgraph;
pub mod endianness;

#[derive(Subcommand, Debug)]
#[command(name = "to")]
/// Converts graphs from a representation to another.
pub enum SubCommands {
    Ascii(ascii::CliArgs),
    Bvgraph(bvgraph::CliArgs),
    Arcs(arcs::CliArgs),
    Endianness(endianness::CliArgs),
    ArcsCuda(arcscuda::CliArgs),
    ArcsValmari(arcsvalmari::CliArgs),
    Gdata(gdata::CliArgs),
}

pub fn main(global_args: GlobalArgs, subcommand: SubCommands) -> Result<()> {
    match subcommand {
        SubCommands::Ascii(args) => ascii::main(global_args, args),
        SubCommands::Bvgraph(args) => bvgraph::main(global_args, args),
        SubCommands::Arcs(args) => arcs::main(global_args, args),
        SubCommands::Endianness(args) => endianness::main(global_args, args),
        SubCommands::ArcsCuda(args) => arcscuda::main(global_args, args),
        SubCommands::ArcsValmari(args) => arcsvalmari::main(global_args, args),
        SubCommands::Gdata(args) => gdata::main(global_args, args),
    }
}
