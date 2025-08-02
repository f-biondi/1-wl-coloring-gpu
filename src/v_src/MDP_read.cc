#ifndef MDP_read_incl
#define MDP_read_incl

/* MDP READER FUNCTION */
/*
  Copyright Antti Valmari 9.9.2010
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
  This file contains the following function for reading a Markov Decision
  Process:

    MDP_read( reader & input )

  "input", defined in "reader.cc", is a class that represents the input.
  The user must provide the following functions, via which "MDP_read"
  delivers the found MDP information to the user:

    void MDP_store_sizes(
      unsigned nr_states, unsigned nr_labels, unsigned nr_l_trans,
      unsigned nr_w_trans, unsigned nr_blocks
    )

    void MDP_store_l_transition(
      unsigned tail, unsigned label, unsigned head
    )

    void MDP_store_w_transition(
      unsigned tail, const MDP_weight_type & weight, unsigned head
    )

    void MDP_store_block( unsigned state, unsigned block )

  States (including tail and head), labels, and blocks are numbered 1, 2, ...
  "MDP_weight_type" is the type of the weights. This file assumes that it has
  been defined as double or int.

  Syntax:
    MDP             ::= <sizes> <l_transitions> <w_transitions> <blocks>
    <sizes>         ::= nr_states nr_labels nr_l_trans nr_w_trans nr_blocks
    <l_transitions> ::= <l_transition>*     // repeated  nr_l_trans times
    <l_transition>  ::= tail_state label head_state
    <w_transitions> ::= <w_transition>*     // repeated  nr_w_trans times
    <w_transition>  ::= tail_state weight head_state
    <blocks>        ::= <block>*            // repeated nr_blocks-1 times
    <block>         ::= state* 0

  Tokens are natural numbers except "weight", which may be decimal numbers.
  Tokens are separated by spaces and/or newlines. States of one block must not
  be given, because they are the remaining states.
*/


#include "reader.cc"


void MDP_read( reader & input ){
  unsigned
    nr_states = 0, nr_labels = 0, nr_l_trans = 0, nr_w_trans = 0,
    nr_blocks = 0;
  err err_context1( "MDP reader" ), err_context2( "" );

  /* Read the number of states */
  err::change_context( "nr. of states" ); nr_states = input.nat();

  /* Read and check the numbers of labels and labelled transitions */
  err::change_context( "nr. of labels" ); nr_labels = input.nat();
  err::change_context( "nr. of labelled trans." ); nr_l_trans = input.nat();
  if( nr_l_trans > 0 &&     // tests nr_l_trans > nr_states^2 * nr_labels
    ( !nr_states            // complicated, to avoid arithmetic overflow
    || ( nr_l_trans - 1 ) / nr_states / nr_states >= nr_labels
    )
  ){ err::msg( "too many:" ); err::msg( nr_l_trans ); err::ready(); }

  /* Read and check the number of weighed transitions */
  err::change_context( "nr. of weighed trans." ); nr_w_trans = input.nat();
  if( nr_w_trans > 0 &&     // tests nr_w_trans > nr_states^2
    ( !nr_states || ( nr_w_trans - 1 ) / nr_states >= nr_states )
  ){ err::msg( "too many:" ); err::msg( nr_w_trans ); err::ready(); }

  /* Read and check the number of initial blocks */
  err::change_context( "nr. of initial blocks" ); nr_blocks = input.nat();
  if( nr_blocks > nr_states || nr_states && !nr_blocks ){
    err::msg( "illegal:" ); err::msg( nr_blocks ); err::ready();
  }

  /* Report the size numbers to the user */
  MDP_store_sizes( nr_states, nr_labels, nr_l_trans, nr_w_trans, nr_blocks );

  /* Read labelled transitions and report to the user */
  err::change_context( "labelled transitions" );
  for( unsigned tr = 0; tr < nr_l_trans; ++tr ){
    unsigned st1 = input.nat();
    if( !st1 || st1 > nr_states ){
      err::msg( "bad tail state number:" ); err::msg( st1 ); err::ready();
    }
    unsigned lb = input.nat();
    if( !lb || lb > nr_labels ){
      err::msg( "bad label number:" ); err::msg( lb ); err::ready();
    }
    unsigned st2 = input.nat();
    if( !st2 || st2 > nr_states ){
      err::msg( "bad head state number:" ); err::msg( st2 ); err::ready();
    }
    MDP_store_l_transition( st1, lb, st2 );
  }

  /* Read weighed transitions and report to the user */
  err::change_context( "weighed transitions" );
  for( unsigned tr = 0; tr < nr_w_trans; ++tr ){
    unsigned st1 = input.nat();
    if( !st1 || st1 > nr_states ){
      err::msg( "bad tail state number:" ); err::msg( st1 ); err::ready();
    }
    #ifdef MDP_weight_is_double
      MDP_weight_type wg = input.decimal();
    #else
      MDP_weight_type wg = input.intgr();
    #endif
    unsigned st2 = input.nat();
    if( !st2 || st2 > nr_states ){
      err::msg( "bad head state number:" ); err::msg( st2 ); err::ready();
    }
    MDP_store_w_transition( st1, wg, st2 );
  }

  /* Read initial blocks (skip block 1) and report to the user */
  err::change_context( "initial blocks" );
  for( unsigned bl = 2; bl <= nr_blocks; ++bl ){
    unsigned st = input.nat();
    while( st ){
      if( st > nr_states ){
        err::msg( "bad state number:" ); err::msg( st ); err::ready();
      }
      MDP_store_block( st, bl ); st = input.nat();
    }
  }

}

#endif





