#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "mmult.h"

// --------------------------------------------------------------------
// function to be accelerated in HW wrapped with AXI4-Stream interface
void mmult_hw (AXI_VAL in_stream[IS_SIZE], AXI_VAL out_stream[OS_SIZE])
{
#pragma HLS INTERFACE s_axilite port=return     bundle=CONTROL_BUS
#pragma HLS INTERFACE axis      port=in_stream
#pragma HLS INTERFACE axis      port=out_stream

	// Assertions (to avoid out of array bound writes)
	assert(BATCH%TILING==0);
	assert(FEAT%W_WIDTH_RATIO==0);
	assert(FEAT%IN_WIDTH_RATIO==0);
	assert((BATCH*CLASSES)%OUT_WIDTH_RATIO==0);

	// weight converter
	union
	{
		axi_T packet;
		w_bit_T i8_vals[8];
	} w_converter;

	// input converter
	union
	{
		axi_T packet;
		in_bit_T ui8_vals[8];
	} in_converter;

	// output and offset converter
	union
	{
		axi_T packet;
		out_bit_T i32_vals[2];
	} out_converter;

	// Hardware memory buffers
	out_T offset_buf[CLASSES];
	w_T weight_buf[CLASSES][FEAT];
	in_T in_buf[TILING][FEAT];
	out_T out_buf[TILING][CLASSES];

#pragma HLS ARRAY_PARTITION variable=in_buf block factor=72 dim=2
#pragma HLS ARRAY_PARTITION variable=weight_buf block factor=72 dim=2

	// Input and output AXI stream indices
	int is_idx = 0;
	int os_idx = 0;

	// Stream in offset vector
	LOAD_OFF_1: for (int i = 0; i < CLASSES; i+=OUT_WIDTH_RATIO) {
		#pragma HLS PIPELINE II=1 rewind
		out_converter.packet = pop_stream(in_stream[is_idx++]);
		for (size_t val = 0; val < 2; val++) {
			offset_buf[i + val] = out_converter.i32_vals[val];
		}
	}

	// Stream in weight matrix
	LOAD_W_1: for (int i = 0; i < CLASSES; i++) {
		LOAD_W_2: for (int j = 0; j < FEAT; j+=IN_WIDTH_RATIO) {
			#pragma HLS PIPELINE II=1 rewind
			// Pop AXI data packet
			w_converter.packet = pop_stream(in_stream[is_idx++]);
			for (size_t val = 0; val < 8; val++) {
				weight_buf[i][j + val] = w_converter.i8_vals[val];
			}
		}
	}

	// Iterate over tiles
	LT: for (int t = 0; t < BATCH; t+=TILING) {

		// Stream in input tile
		// CSE548 TODO
		TILE: for (int i = 0; i < TILING; ++i)
		{
			#pragma HLS PIPELINE II=1 rewind
			LOAD_I_2: for (int j = 0; j < FEAT; j+=IN_WIDTH_RATIO)
			{
				// Pop AXI data packet
				in_converter.packet = pop_stream(in_stream[is_idx++]);
				for (size_t val = 0; val < 8; val++) {
					in_buf[i][j + val] = in_converter.ui8_vals[val];
				}
			}
		}

		// Perform matrix multiplication
		L1: for (int i = 0; i < TILING; i++) {
			// Iterate over output classes
			L2: for (int j = 0; j < CLASSES; j++) {
				#pragma HLS PIPELINE II=1 rewind
				// Perform the dot product
				out_T tmp = offset_buf[j];
				L3: for(int k = 0; k < FEAT; k++) {
// 					in_bit_T l = in_buf[i][k];
// 					int8_t r = weight_buf[j][k];
// 					out_bit_T mult = l * r;
// #pragma HLS RESOURCE variable=mult core=Mul_LUT
					tmp += in_buf[i][k] * weight_buf[j][k];
				}
				out_buf[i][j] = tmp;
			}
		}

		// Stream out output matrix
		// CSE548 TODO
		TILE3: for (int i = 0; i < TILING; ++i)
		{
			#pragma HLS PIPELINE II=1 rewind
			STORE_O_2: for (int j = 0; j < CLASSES; j += OUT_WIDTH_RATIO)
			{
				// Push output element into AXI stream
				out_converter.i32_vals[0] = out_buf[i][j+0];
				out_converter.i32_vals[1] = out_buf[i][j+1];
				out_stream[os_idx++] = push_stream(out_converter.packet, os_idx == (OS_SIZE));
			}
		}
	}
}


// --------------------------------------------------------
// functions to insert and extract elements from an axi stream
// includes conversion to correct data type
axi_T pop_stream(AXI_VAL const &e)
{
#pragma HLS INLINE

	axi_T ret = e.data;

	volatile ap_uint<sizeof(axi_T)> strb = e.strb;
	volatile ap_uint<sizeof(axi_T)> keep = e.keep;
	volatile ap_uint<AXI_U> user = e.user;
	volatile ap_uint<1> last = e.last;
	volatile ap_uint<AXI_TI> id = e.id;
	volatile ap_uint<AXI_TD> dest = e.dest;

	return ret;
}

AXI_VAL push_stream(axi_T const &v, bool last = false)
{
#pragma HLS INLINE

	AXI_VAL e;

	e.data = v;
	e.strb = (1<<sizeof(axi_T))-1;
	e.keep = (1<<sizeof(axi_T))-1;
	e.user = 0;
	e.last = last ? 1 : 0;
	e.id = 0;
	e.dest = 0;
	return e;
}
