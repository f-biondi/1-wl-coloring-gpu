/* MARKOV DECISION PROCESS MINIMIZER PROGRAM */
/*
  Copyright Antti Valmari 17.9.2010
*/
/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
/*
  This program reads a Markov decision process (MDP) from the standard input,
  minimizes it, and writes the result to the standard output. This file
  basically just puts the pieces in other files together:

    MDP_minimize.cc     minimization operations
    MDP_read.cc         reads an MDP, assuming services from other packages
    reader.cc           gets and recognises tokens from standard input
    MDP_write.cc        writes an MDP, assuming services from other packages
    writer.cc           sends tokens to standard output, adds white space
    err.cc              services for reporting run-time errors

  See the individual files for description of issues that belong to them. This
  comment discusses the overall design.

  I tend to compile this with one of the following commands:

    g++ -ansi -W -Wall -pedantic -O2 MDPmin.cc -o MDPmin.out
    g++ -ansi -W -Wall -pedantic -O2 -DMDP_weight_is_double MDPmin.cc
      -o MDPmin.out

  There are no .hh files. Instead, all the code is #included as such. This is
  mostly because I have been lazy and partly to maximize the possibilities of
  inlining time-critical reading and writing subroutines.

  The design aims at a high level of modularity, so that the components could
  also be used in other programs. In particular, MDP_minimize.cc has been made
  unaware of the syntax of MDPs. It is not even necessary that the MDPs come
  from and go to a file; they may as well come from and go to another
  subroutines. Furthermore, MDP_minimize.cc can produce different kinds of
  information about the result depending on what is needed. The goal is that
  MDP_minimize.cc can be used flexibly as a component in many programs.

  Lexical issues are the responsibility of reader.cc and writer.cc. They are
  not aware that the tokens belong to an MDP. MDP_read.cc and MDP_write.cc
  take care of the syntax, that is, the order of tokens in an MDP. They are
  not aware of how the MDP is stored into data structures. These components
  can be used in other programs that use the same lexical rules or MDP syntax.

  This design goal has significantly complicated some of the interfaces
  between the components. MDP_read.cc cannot directly store the parts of the
  MDP to the data structures of MDP_minimize.cc, because MDP_read.cc must be
  independent of the internals of MDP_minimize.cc. Therefore, for each piece
  of input data, MDP_read.cc calls a subroutine provided by MDP_minimize.cc to
  deliver the data. MDP_minimize.cc must feature these subroutines and not
  trust that they are called in any particular order, except that it was a
  design principle that size information is given first. Similar remarks apply
  to the interface between MDP_minimize.cc and MDP_write.cc.

  MDP_read.cc uses names of the form MDP_xxx for these subroutines, while
  MDP_minimize.cc calls them MDPmin::xxx. Therefore, there must be name
  transformation at the main program level. However, this convention makes it
  possible in the future to have both the MDP minimizer and some other MDP
  manipulation component in the same program and to direct the i/o to the
  right place.

  The type of the weights of the weighed transitions is a problematic issue in
  this design. MDP_minimize.cc only assumes that it is a defined type with
  certain operations. On the other hand, MDP_read.cc and MDP_write.cc must
  know what kind of tokens correspond to the weights. This knowledge does not
  belong to reader.cc and writer.cc, because they must be unaware of MDP
  concepts.

  Currently two types have been implemented, integer and double, the latter
  being represented as a decimal number. The choice between them is made by
  #defining or not #defining MDP_weight_is_double.

  If a new weight type is introduced, like rational numbers, a textual
  representation for it must be designed, and MDP_read.cc and MDP_write.cc
  must be updated. If new token types are introduced, also reader.cc and
  writer.cc must be updated. (New token types need not be introduced if
  rationals are represented as two integers.)

  The philosophy behind sanity checks and the processing of run-time errors is
  the following. The mechanism should provide informative error messages
  without significant additional cost in the absence of errors. Errors are
  reported to standard error and the program is aborted. Each component mostly
  trusts on its own correct operation, but not on the operation of the others.
  It makes those sanity checks that it needs to detect run-time errors, such
  as indexing out of bounds. To report errors as early as possible, it may
  also make checks that are easy to make there but which it does not
  personally need. Checks whose time consumption may be non-negligible can be
  switched off at compile-time.
*/


/* The type of weights and its related features */
/* See MDP_minimize.cc for a description of these. */
#ifdef MDP_weight_is_double
  #include <cfloat>
  typedef double MDP_weight_type;
  const MDP_weight_type MDP_zero_weight = 0.0, MDP_unused_weight = DBL_MAX;
  inline void MDP_round_weight( MDP_weight_type & ww ){
    ww = float( ww );   // is this set of standard values okay?
  }
#else   // weight is int
  #include <climits>
  typedef int MDP_weight_type;
  const MDP_weight_type MDP_zero_weight = 0, MDP_unused_weight = INT_MIN;
  inline void MDP_round_weight( MDP_weight_type & ){}   // trivial for int
#endif


/* Include the minimizer file. */
#include "MDP_minimize.cc"

/* Change the names of the input and output operations. */

inline void MDP_store_sizes(
  unsigned nr_states, unsigned nr_labels, unsigned nr_l_trans,
  unsigned nr_w_trans, unsigned nr_blocks
){
  MDPmin::store_sizes(
    nr_states, nr_labels, nr_l_trans, nr_w_trans, nr_blocks
  );
}

inline void MDP_store_l_transition(
  unsigned tail, unsigned label, unsigned head
){ MDPmin::store_l_transition( tail, label, head ); }

inline void MDP_store_w_transition(
  unsigned tail, const MDP_weight_type & weight, unsigned head
){ MDPmin::store_w_transition( tail, weight, head ); }

inline void MDP_store_block( unsigned state, unsigned block ){
  MDPmin::store_block( state, block );
}

inline void MDP_give_sizes(
  unsigned & nr_states, unsigned & nr_labels, unsigned & nr_l_trans,
  unsigned & nr_w_trans, unsigned & nr_blocks
){
  MDPmin::give_sizes(
    nr_states, nr_labels, nr_l_trans, nr_w_trans, nr_blocks
  );
}

inline void MDP_give_l_transition(
  unsigned & tail, unsigned & label, unsigned & head
){ MDPmin::give_l_transition( tail, label, head ); }

inline void MDP_give_w_transition(
  unsigned & tail, MDP_weight_type & weight, unsigned & head
){ MDPmin::give_w_transition( tail, weight, head ); }

inline unsigned MDP_give_block_first( unsigned block_nr ){
  return MDPmin::give_block_first( block_nr );
}

inline unsigned MDP_give_block_next(){ return MDPmin::give_block_next(); }



/* Include the input and output files. */
#include "MDP_read.cc"
#include "MDP_write.cc"


/* The main function is trivial. */
int main(){
  std::ios::sync_with_stdio(false);
  std::cin.tie(0);
  reader input; MDP_read( input );
  MDPmin::minimize();
  //writer output; MDP_write( output );
  MDPmin::terminate();      // not necessary
}





