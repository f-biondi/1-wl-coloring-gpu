#!/bin/bash

mkdir -p src/batched_be/build
cd src/batched_be/build
cmake ..
cmake --build . --target batched_be
cmake --build . --target batched_be_aggressive
cd ../../../
mv src/batched_be/build/batched_be .
mv src/batched_be/build/batched_be_aggressive .

mkdir -p src/random_be/build
cd src/random_be/build
cmake ..
cmake --build . --target random_be
cd ../../../
mv src/random_be/build/random_be .

cd src/part_ref
g++ -std=c++11 -W -Wall -pedantic -O2 MDPmin.cc -o part_ref
cd ../../
mv src/part_ref/part_ref .

cd webgraph-rs
cargo build --release
cd ..
mv webgraph-rs/target/release/webgraph .
