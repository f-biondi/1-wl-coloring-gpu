/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::GlobalArgs;
use anyhow::Result;
use clap::Parser;
use dsi_bitstream::dispatch::factory::CodesReaderFactoryHelper;
use dsi_bitstream::prelude::*;
use dsi_progress_logger::prelude::*;
use lender::*;
use std::io::Write;
use std::path::PathBuf;
use webgraph::graphs::bvgraph::get_endianness;
use webgraph::traits::SequentialLabeling;
use webgraph::utils::MmapHelper;

#[derive(Parser, Debug)]
#[command(name = "gdata", about = "Graph Data", long_about = None)]
pub struct CliArgs {
    /// The basename of the graph.
    pub src: PathBuf,
}

pub fn main(global_args: GlobalArgs, args: CliArgs) -> Result<()> {
    match get_endianness(&args.src)?.as_str() {
        #[cfg(feature = "be_bins")]
        BE::NAME => to_csv::<BE>(global_args, args),
        #[cfg(feature = "le_bins")]
        LE::NAME => to_csv::<LE>(global_args, args),
        e => panic!("Unknown endianness: {}", e),
    }
}

pub fn to_csv<E: Endianness + 'static>(global_args: GlobalArgs, args: CliArgs) -> Result<()>
where
    MmapHelper<u32>: CodesReaderFactoryHelper<E>,
{
    let graph = webgraph::graphs::bvgraph::sequential::BvGraphSeq::with_basename(args.src)
        .endianness::<E>()
        .load()?;
    let num_nodes = graph.num_nodes();
    let num_edges = graph.num_arcs_hint().expect("Arcs num not present");

    // read the csv and put it inside the sort pairs
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    let mut pl = ProgressLogger::default();
    pl.display_memory(true)
        .item_name("nodes")
        .expected_updates(Some(num_nodes));

    if let Some(duration) = global_args.log_interval {
        pl.log_interval(duration);
    }

    pl.start("Reading BvGraph");

    writeln!(stdout, "{}", num_nodes)?;
    writeln!(stdout, "{}", num_edges)?;
    pl.light_update();

    pl.done();
    Ok(())
}
