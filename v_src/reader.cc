#ifndef reader_incl
#define reader_incl

/* A SIMPLE INPUT READER CLASS */
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
  This class contains services for reading from a textual input: skipping
  white space, reading an unsigned integer, and so on. The input file is given
  in the constructor of an object of the class. If nothing is given, standard
  input is used. The class keeps track of line and column numbers.
*/


#include <iostream>
#include "err.cc"

class reader{

  std::istream *is;     // file which is being read
  char buffer;          // most recently read character
  unsigned line, column;// how far reading has advanced, first place is (1,1)

  /* Gets next character. If impossible, gets null character. */
  inline void advance(){
    is->get( buffer );
    if( !*is ){ buffer = '\0'; }
    else if( buffer == '\n' ){ ++line; column = 0; }else{ ++column; }
  }

public:

  /* Constructor */
  reader( std::istream *is = &std::cin ):
    is( is ), buffer( '\0' ), line( 1 ), column( 0 ) { advance(); }

  /* Adds the line and column numbers to an error message. */
  void err_loc(){
    err::msg( "line" ); err::msg( line );
    err::msg( "col" ); err::msg( column ); err::msf( ":" );
  }

  /* Skips white space, i.e., spaces and ends of lines. */
  inline void skip_spaces(){
    while( buffer == ' ' || buffer == '\n' ){ advance(); }
  }

  /* Reads a natural number (possibly preceded by white space) from input and
    returns it. Does not check arithmetic overflow. */
  unsigned nat(){
    skip_spaces();
    if( buffer < '0' || buffer > '9' ){
      err_loc(); err::msg( "digit expected" ); err::ready();
    }
    unsigned result = buffer - '0'; advance();
    while( buffer >= '0' && buffer <= '9' ){
      result *= 10; result += buffer - '0'; advance();
    };
    return result;
  }

  /* Reads an integer (possibly preceded by white space) from input and
    returns it. Does not check arithmetic overflow. */
  int intgr(){
    skip_spaces();

    /* Read signs. */
    bool is_neg = false;
    while( buffer == '+' || buffer == '-' ){
      if( buffer == '-' ){ is_neg = !is_neg; }
      advance();
    }

    /* Read digits. */
    if( buffer < '0' || buffer > '9' ){
      err_loc(); err::msg( "+, - or digit expected" ); err::ready();
    }
    int result = buffer - '0'; advance();
    while( buffer >= '0' && buffer <= '9' ){
      result *= 10; result += buffer - '0'; advance();
    };

    /* Return result. */
    if( is_neg ){ return -result; }else{ return result; }

  }

  /* Reads a decimal number (possibly preceded by white space) from input and
    returns it as a double. Does not check arithmetic overflow. The syntax is:

      ( "+" | "-" )* digit* [ "." digit* ]

    with the additional requirement of at least one digit.
  */
  double decimal(){
    skip_spaces();

    /* Read signs. */
    bool is_neg = false;
    while( buffer == '+' || buffer == '-' ){
      if( buffer == '-' ){ is_neg = !is_neg; }
      advance();
    }

    /* Read integer part. */
    double result = 0;
    bool had_digits = buffer >= '0' && buffer <= '9';
    while( buffer >= '0' && buffer <= '9' ){
      result *= 10; result += buffer - '0'; advance();
    };

    /* Read decimal part. */
    bool had_dot = buffer == '.';
    if( had_dot ){
      advance(); had_digits |= buffer >= '0' && buffer <= '9';
      double scale = 1.0;
      while( buffer >= '0' && buffer <= '9' ){
        scale /= 10; result += scale * ( buffer - '0' ); advance();
      };
    }

    /* Start syntax error message, if necessary. */
    if( !had_digits ){
      err_loc();
      if( !had_dot ){ err::msg( "+, -, . or" ); }
      err::msg( "digit expected" ); err::ready();
    }

    /* Return result if successful. */
    if( is_neg ){ return -result; }else{ return result; }

  }

};

#endif




