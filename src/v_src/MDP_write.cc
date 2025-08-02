#ifndef MDP_write_incl
#define MDP_write_incl

/* MDP WRITER FUNCTION */
/*
  Copyright Antti Valmari 10.9.2010
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
  This file contains the following function for writing a Markov Decision
  Process:

    void MDP_write( writer & output )

  "writer", defined in "writer.cc", is a class that represents the output.
  The user must provide the following functions, via which "MDP_write"
  fetches the remaining MDP information:

    void MDP_give_sizes(
      unsigned & nr_states, unsigned & nr_labels, unsigned & nr_l_trans,
      unsigned & nr_w_trans, unsigned & nr_blocks
    )

    void MDP_give_l_transition(
      unsigned & tail, unsigned & label, unsigned & head
    )

    void MDP_give_w_transition(
      unsigned & tail, MDP_weight_type & weight, unsigned & head
    )

    unsigned MDP_give_block_first( unsigned block_nr )

    unsigned MDP_give_block_next()

  "l_trans" refers to labelled and "w_trans" to weighed transitions.
  "MDP_weight_type" is the type of the weights. This file assumes that it has
  been defined as double or int. "MDP_give_block_next" gives 0 when every
  state of the block has been delivered.

  See "MDP_read.cc" for MDP file syntax.
*/


#include "writer.cc"


void MDP_write( writer & output ){
  err err_context( "MDP writer" );

  /* Output the sizes. */
  unsigned nr_states, nr_labels, nr_l_trans, nr_w_trans, nr_blocks;
  MDP_give_sizes( nr_states, nr_labels, nr_l_trans, nr_w_trans, nr_blocks );
  output.nat( nr_states ); output.nat( nr_labels ); output.nat( nr_l_trans );
  output.nat( nr_w_trans ); output.nat( nr_blocks ); output.new_line();

  /* Output the labelled transitions. */
  for( unsigned tr = 0; tr < nr_l_trans; ++tr ){
    unsigned tail, head, label;
    MDP_give_l_transition( tail, label, head );
    output.nat( tail ); output.nat( label ); output.nat( head );
  }
  output.end_line();

  /* Output the weighed transitions. */
  #ifdef MDP_weight_is_double
    output.set_prec( 3 );
  #endif
  for( unsigned tr = 0; tr < nr_w_trans; ++tr ){
    unsigned tail, head; MDP_weight_type weight;
    MDP_give_w_transition( tail, weight, head );
    output.nat( tail );
    #ifdef MDP_weight_is_double
      output.decimal( weight );
    #else
      output.intgr( weight );
    #endif
    output.nat( head );
  }
  output.end_line();

  /* Output the blocks */
  for( unsigned bl = 2; bl <= nr_blocks; ++bl ){
    unsigned st = MDP_give_block_first( bl );
    while( st ){ output.nat( st ); st = MDP_give_block_next(); }
    output.nat( 0 ); output.end_line();
  }

}

#endif





