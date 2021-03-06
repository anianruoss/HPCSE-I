// File       : mpi_float_compression.cpp
// Created    : Tue Nov 27 2018 02:05:07 PM (+0100)
// Description: Distributed floating point compression with MPI
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include <cassert>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>

#include "compressor.h"
#include "helper.h"

using namespace std;

/**
 * @brief Header meta data to describe global parameter
 */
struct FileHeader {
  double tol;
  size_t Nx, Ny, Nb;
};

/**
 * @brief Header meta data to describe a compression block
 */
struct BlockHeader {
  size_t start, compressed_bytes, bufsize;
};

/**
 * @brief Function used to write our compressed MPI file format.
 *
 * @param fname Filename
 * @param cbuf Pointer to compressed data
 * @param cbytes Number of compressed bytes
 * @param bufsize Workbuffer size used by the compressor
 * @param Nx Global x-data dimension
 * @param Ny Global y-data dimension
 * @param tol Compression tolerance
 */
void write_data(const string &fname, const unsigned char *const cbuf,
                const size_t cbytes, const size_t bufsize, const size_t Nx,
                const size_t Ny, const double tol) {
  // File structure:
  // The file begins with one file header followed be p block headers, see
  // the struct FileHeader and struct BlockHeader above.  In this exercise we
  // map the p blocks to p MPI processes.  That is, each process works on one
  // compressed block only.  This could be generalized.  The following sketch
  // illustrates the file format (the numbers in brackets illustrate the byte
  // offset in the file, e.g., [0:b] means from byte 0 to byte b):
  //
  // [0:f]   FileHeader
  // [f:b0]  BlockHeader0
  // [b0:b1] BlockHeader1
  // [b1:b2] BlockHeader2
  // ...
  // [bp:c0] cbuf0
  // [c0:c1] cbuf1
  // [c1:c2] cbuf2
  // ...
  // EOF

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const bool isroot = (0 == rank);

  // fill file header structure
  FileHeader fheader;
  fheader.tol = tol;
  fheader.Nx = Nx;
  fheader.Ny = Ny;
  fheader.Nb = static_cast<size_t>(size);

  // compute block header start and end offsets (for convenience, that is the
  // start f and end bp in bytes [see above for meaning of f and bp])
  size_t f = sizeof(FileHeader);
  size_t bk = f + rank * sizeof(BlockHeader);
  size_t bp = f + size * sizeof(BlockHeader);

  // determine local offsets of compressed blocks, that is byte offset ck for
  // rank k. The number of bytes for the compressed blocks can be different on
  // each rank.
  size_t prev_cbytes = 0;
  MPI_Exscan(&cbytes, &prev_cbytes, 1, MPI_UNSIGNED_LONG, MPI_SUM,
             MPI_COMM_WORLD);
  size_t ck = bp + prev_cbytes;

  // fill block header
  BlockHeader bheader;
  bheader.start = ck;
  bheader.compressed_bytes = cbytes;
  bheader.bufsize = bufsize;

  // open MPI file for write operations
  MPI_File fh;
  MPI_Status st;
  MPI_File_open(MPI_COMM_WORLD, fname.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

  // write headers (root will also write the file header)
  if (isroot) {
    MPI_File_write_at(fh, 0, &fheader, sizeof(FileHeader), MPI_CHAR, &st);
  }
  MPI_File_write_at_all(fh, bk, &bheader, sizeof(BlockHeader), MPI_CHAR, &st);

  // write compressed data
  MPI_File_write_at_all(fh, ck, cbuf, cbytes, MPI_CHAR, &st);

  // close file
  MPI_File_close(&fh);
}

/**
 * @brief Function used to read our compressed MPI file format.
 *
 * @param fname Filename
 * @param cbytes Number of compressed bytes (out)
 * @param bufsize Workbuffer size used by the compressor (out)
 * @param Nx Global x-data dimension (out)
 * @param Ny Global y-data dimension (out)
 * @param tol Compression tolerance (out)
 *
 * @return internally allocated pointer to compressed file contents
 */
unsigned char *read_data(const string &fname, size_t &cbytes, size_t &bufsize,
                         size_t &Nx, size_t &Ny, double &tol) {
  // inverse operations relative to the write_data function.

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // open MPI file for read operations
  MPI_File fh;
  MPI_Status st;
  MPI_File_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL,
                &fh);

  // read headers (meta data)
  FileHeader fheader;  // to be filled with meta data read from file
  BlockHeader bheader; // to be filled with meta data read from file
  MPI_File_read_at_all(fh, 0, &fheader, sizeof(FileHeader), MPI_CHAR, &st);
  MPI_File_read_at_all(fh, sizeof(FileHeader) + rank * sizeof(BlockHeader),
                       &bheader, sizeof(BlockHeader), MPI_CHAR, &st);

  tol = fheader.tol;
  Nx = fheader.Nx;
  Ny = fheader.Ny;
  cbytes = bheader.compressed_bytes;
  bufsize = bheader.bufsize;

  // allocate data buffer and read compressed data from file
  auto *cbuf = new unsigned char[cbytes];
  MPI_File_read_at_all(fh, bheader.start, cbuf, cbytes, MPI_CHAR, &st);

  // close file
  MPI_File_close(&fh);

  return cbuf;
}

int main(int argc, char *argv[]) {
  Time t1, t2;

  ///////////////////////////////////////////////////////////////////////////
  // Compression to file:
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const bool isroot = (0 == rank);

  // read command line input
  if (4 != argc) {
    if (isroot)
      cout << "USAGE: " << argv[0]
           << " <compression_tolerance> <input_dimension> <input_file.bin.gz>"
           << endl;
    exit(1);
  }
  const double tol = atof(argv[1]); // floating point compression tolerance
  const size_t N = atoi(argv[2]);   // input file dimension (square pixel image)
  const string infile(argv[3]);     // input file path

  assert(0 == N % size);          // check that domain decomposition is sane
  const size_t myrows = N / size; // decompose domain into tiles
  t1 = Clock::now();
  auto *data = read_gzfile_tile<double>(infile, rank * myrows, N, myrows);
  t2 = Clock::now();
  report_time(t1, t2, "Read GZIP");

  // setup floating point compressor
  ZFPCompressor2D<double> compressor(data, N, myrows, tol);

  // compress local buffer
  size_t
      bufsize; // compressor work buffer size (required for decompression later)
  size_t compressed_bytes; // number of bytes in compressed data
                           // (compressed_bytes <= bufsize)
  unsigned char *cdata;    // pointer to compressed data buffer
  t1 = Clock::now();
  cdata = compressor.compress(bufsize, compressed_bytes);
  t2 = Clock::now();
  report_time(t1, t2, "Compression");

  // write distributed file (file format for our custom compressor [.zfp])
  ostringstream finout;
  string basename = infile;
  basename.erase(basename.end() - 7, basename.end());
  finout << basename << "_t" << setprecision(3) << scientific << tol << ".zfp";
  t1 = Clock::now();
  write_data(finout.str(), cdata, compressed_bytes, bufsize, N, N, tol);
  t2 = Clock::now();
  report_time(t1, t2, "Write .zfp");

  // the compressor code could end here and another application could make
  // use of the compressed data stored in the file.
  ///////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////
  // Decompression from file: (This code could be a different application.)

  // read distributed compressed file (format for our compressor, all
  // parameter are obtained from the file [prefixed with 'f'])
  double ftol;
  size_t fNx, fNy, fbufsize, fcompressed_bytes;
  t1 = Clock::now();
  unsigned char *fbuf =
      read_data(finout.str(), fcompressed_bytes, fbufsize, fNx, fNy, ftol);
  t2 = Clock::now();
  report_time(t1, t2, "Read .zfp");
  if (!fbuf) {
    cerr << "'read_data' returned nullptr" << endl;
    MPI_Finalize();
    return 0;
  }

  // check that what we read from the file is correct
  assert(N == fNy);
  assert(N == fNx);
  assert(bufsize == fbufsize);
  assert(compressed_bytes == fcompressed_bytes);
  assert(abs(tol - ftol) <= numeric_limits<double>::epsilon());

  // decompress file contents
  auto *fdata = new double[fNx * myrows]; // array for decompressed data (lossy)
  ZFPCompressor2D<double> fdecompressor(fdata, fNy, myrows, ftol);
  t1 = Clock::now();
  fdecompressor.decompress(fbuf, fbufsize); // decompress file content
  t2 = Clock::now();
  report_time(t1, t2, "Decompression");

#ifndef _NO_GZIP_OUT_
  // write lossy data (for python post-processing)
  ostringstream fname;
  fname << basename << "_t" << setprecision(3) << scientific << tol
        << ".bin.gz";
  t1 = Clock::now();
  write_gzfile_tile<double>(fdata, fname.str(), N, myrows);
  t2 = Clock::now();
  report_time(t1, t2, "Write GZIP (not optimized)");
#endif /* _NO_GZIP_OUT_ */

  // compute compression statistics
  show_stat(data, fdata, N, myrows, compressed_bytes);

  // clean up (cdata is managed by the compressor instance)
  delete[] data;
  delete[] fdata;
  delete[] fbuf;

  MPI_Finalize();
  return 0;
}
