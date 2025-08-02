#ifndef writer_incl
#define writer_incl

/* A SIMPLE OUTPUT WRITER CLASS */
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
  This class contains services for writing to a textual output: writing
  integers and decimal numbers, and changing lines. Unless otherwise told, it
  writes as densely as possible onto lines of a given maximum length. The
  output file is given in the constructor of an object of the class. If
  nothing is given, standard output is used.
*/


#include <iostream>
#include "err.cc"


class writer{

  /* Instance data, etc. */
  std::ostream *os;     // file being written
  unsigned nr_on_line;  // number of characters on most recent line
  static const unsigned
    max_on_line = 72;   // Line length is kept at most this if possible.
  unsigned prec;        // number of digits after the decimal point
  double round;         // for rounding up, e.g., 0.996 to 1.00 when prec = 2

  /* Buffer for auxiliary storage. Size 20 suffices for 64-bit integers. */
  static const unsigned buf_size = 20;
  static char buffer[];
  static void buf_ovfl(){
    err::msg( "I have been prepared for" ); err::msg( buf_size );
    err::msf( "-digit numbers, but you used longer.\nThis problem can be" );
    err::msg( "fixed by increasing buffer size in \"writer.cc\" and\n" );
    err::msf( "re-compiling." ); err::ready();
  }

  /* Prepares for writing a next token of size "len" by writing nothing, a
    line feed, or a space. Tries to prevent lines from becoming overlong, and
    keeps "nr_on_line" up to date. */
  inline void advance( unsigned len ){
    if( !nr_on_line ){ nr_on_line = len; }
    else if( nr_on_line + len >= max_on_line ){ // remember the space
      os->put( '\n' ); nr_on_line = len;
    }else{ os->put( ' ' ); nr_on_line += len + 1; }
  }


public:


  /* Constructor */
  writer( std::ostream *os = &std::cout ):
    os( os ), nr_on_line( 0 ), prec( 0 ), round( 0.5 ) {}


  /* Forces a new line. */
  inline void new_line(){ os->put( '\n' ); nr_on_line = 0; }


  /* Ends a line if and only if it is not empty. */
  inline void end_line(){ if( nr_on_line ){ new_line(); } }


  /* Writes a natural number. */
  void nat( unsigned long nn ){
    unsigned len = 0;
    do{
      if( len >= buf_size ){ buf_ovfl(); }
      buffer[ len ] = nn % 10 + '0'; nn /= 10; ++len;
    }while( nn );
    advance( len );
    while( len ){ os->put( buffer[ --len ] ); }
  }


  /* Writes an integer. */
  void intgr( long nn ){
    bool is_neg = nn < 0; if( is_neg ){ nn = -nn; }
    unsigned len = 0;
    do{
      if( len >= buf_size ){ buf_ovfl(); }
      buffer[ len ] = nn % 10 + '0'; nn /= 10; ++len;
    }while( nn );
    advance( len + is_neg );
    if( is_neg ){ os->put( '-' ); }
    while( len ){ os->put( buffer[ --len ] ); }
  }


  /* Sets the precision for writing decimal numbers. Look-up table would have
    been faster, but printing a number takes so much time anyway that I think
    this solution is fast enough. */
  void set_prec( unsigned pr ){
    prec = pr; round = 0.5;
    for( unsigned i1 = 0; i1 < prec; ++i1 ){ round /= 10; }
  }


  /* Writes a decimal number. */
  void decimal( double xx ){

    /* Take the sign apart. */
    bool is_neg = xx < 0.0; if( is_neg ){ xx = -xx; }

    /* Round the value */
    xx += round;

    /* Extract the integer part. */
    unsigned long nn = (unsigned long) xx; xx -= nn;
    if( xx < 0.0 ){ xx += 1.0; --nn; }
    if( xx < 0.0 || xx >= 1.0 ){
      err::msg( "Too big or small decimal number" ); err::ready();
    }

    /* Reserve space for the output. */
    unsigned len = 0;
    while( nn ){
      if( len >= buf_size ){ buf_ovfl(); }
      buffer[ len ] = nn % 10 + '0'; nn /= 10; ++len;
    }
    if( !len && !prec ){ len = 1; buffer[ 0 ] = '0'; }
    advance( len + is_neg + prec + 1 );

    /* Write the sign and integer part. */
    if( is_neg ){ os->put( '-' ); }
    while( len ){ os->put( buffer[ --len ] ); }

    /* Write the decimal part. */
    os->put( '.' );
    for( unsigned i1 = 0; i1 < prec; ++i1 ){
      xx *= 10; unsigned ii = unsigned( xx ); if( ii > 9 ){ ii = 9; }
      xx -= ii; if( xx < 0 ){ xx = 0.0; }
      os->put( char( ii + '0' ) );
    }

  }

};
char writer::buffer[ writer::buf_size ];

#endif




