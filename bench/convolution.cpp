#include <cstddef>
#include <cstdlib>

#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>

#include <nnpack.h>
#include <nnpack/reference.h>

using namespace std;

enum mode {
	mode_output,
	mode_input_gradient,
	mode_kernel_gradient,
	mode_inference,
};

inline static float relativeError(float reference, float actual) {
		return std::abs(reference - actual) / std::max(FLT_MIN, std::abs(reference));
	}

struct options {
	enum mode mode;
	size_t batch_size;
	size_t input_channels;
	size_t output_channels;
	struct nnp_size input_size;
	size_t input_padding;
	struct nnp_size kernel_size;
	struct nnp_size output_subsampling;
	enum nnp_convolution_algorithm algorithm;
	enum nnp_convolution_transform_strategy transform_strategy;
	size_t threads;
	size_t iterations;
	bool threadpool;
};

static void print_options_help(const char* program_name) {
	printf(
"%s parameters...\n"
"Required parameters:\n"
"  -ic  --input-channels     The number of input channels\n"
"  -oc  --output-channels    The number of output channels\n"
"  -is  --input-size         Input height and width\n"
"  -os  --kernel-size        Kernel height and width\n"
"Optional parameters:\n"
"  -m   --mode               The convolution mode (output, inference)\n"
"  -a   --algorithm          The algorithm (auto, ft8x8, ft16x16, wt8x8, or implicit-gemm) for computing convolution (default: auto)\n"
"  -s   --strategy           The transform strategy (block, tuple, precompute) in inference mode (default: tuple)\n"
"  -b   --batch              The size of a minibatch (default: 1)\n"
"       --output-subsampling The size of a output subsampling region (default: 1x1)\n"
"  -ip  --padding            Implicit input padding (default: 0)\n"
"  -t   --threads            The number of threads (default: all; 0 to disable threadpool)\n"
"  -i   --iterations         # iterations (default: 3)\n",
		program_name);
}

static struct options parse_options(int argc, char** argv) {
	struct options options = {
		.mode = mode_output,
		.batch_size = 1,
		.input_channels = 0,
		.output_channels = 0,
		.input_size = { 0, 0 },
		.input_padding = 0,
		.kernel_size = { 0, 0 },
		.output_subsampling = { 1, 1 },
		.algorithm = nnp_convolution_algorithm_auto,
		.transform_strategy = nnp_convolution_transform_strategy_tuple_based,
		.threads = 0,
		.iterations = 3,
		.threadpool = true,
	};
	for (int argi = 1; argi < argc; argi += 1) {
		if ((strcmp(argv[argi], "--batch") == 0) || (strcmp(argv[argi], "-b") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected batch value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.batch_size) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.batch_size == 0) {
				fprintf(stderr, "Error: invalid value %s for the batch size: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--input-channels") == 0) || (strcmp(argv[argi], "-ic") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected input channels value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.input_channels) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_channels == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of input channels: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--output-channels") == 0) || (strcmp(argv[argi], "-oc") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected output channels value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.output_channels) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.output_channels == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of output channels: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--input-size") == 0) || (strcmp(argv[argi], "-is") == 0)) {
			if (argc - argi < 2) {
				fprintf(stderr, "Error: expected two input size values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.input_size.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_size.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the input height: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 2], "%zu", &options.input_size.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_size.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the input width: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
		} else if ((strcmp(argv[argi], "--kernel-size") == 0) || (strcmp(argv[argi], "-ks") == 0)) {
			if (argc - argi < 2) {
				fprintf(stderr, "Error: expected two kernel size values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.kernel_size.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.kernel_size.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the kernel height: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 2], "%zu", &options.kernel_size.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.kernel_size.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the kernel width: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
		} else if ((strcmp(argv[argi], "--input-padding") == 0) || (strcmp(argv[argi], "-ip") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected padding value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.input_padding) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if (strcmp(argv[argi], "--output-subsampling") == 0) {
			if (argc - argi < 2) {
				fprintf(stderr, "Error: expected two output subsampling values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.output_subsampling.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.output_subsampling.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the output subsampling height: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 2], "%zu", &options.output_subsampling.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.output_subsampling.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the output subsampling width: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 2;
		} else if ((strcmp(argv[argi], "--algorithm") == 0) || (strcmp(argv[argi], "-a") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected convolution algorithm name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "auto") == 0) {
				options.algorithm = nnp_convolution_algorithm_auto;
			} else if (strcmp(argv[argi + 1], "ft8x8") == 0) {
				options.algorithm = nnp_convolution_algorithm_ft8x8;
			} else if (strcmp(argv[argi + 1], "ft16x16") == 0) {
				options.algorithm = nnp_convolution_algorithm_ft16x16;
			} else if (strcmp(argv[argi + 1], "wt8x8") == 0) {
				options.algorithm = nnp_convolution_algorithm_wt8x8;
			} else if (strcmp(argv[argi + 1], "implicit-gemm") == 0) {
				options.algorithm = nnp_convolution_algorithm_implicit_gemm;
			} else {
				fprintf(stderr, "Error: invalid convolution algorithm name %s\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--transform-strategy") == 0) || (strcmp(argv[argi], "-s") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected transform strategy\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "block") == 0) {
				options.transform_strategy = nnp_convolution_transform_strategy_block_based;
			} else if (strcmp(argv[argi + 1], "tuple") == 0) {
				options.transform_strategy = nnp_convolution_transform_strategy_tuple_based;
			} else if (strcmp(argv[argi + 1], "precomputed") == 0) {
				options.transform_strategy = nnp_convolution_transform_strategy_precomputed;
			} else {
				fprintf(stderr, "Error: invalid kernel transform strategy %s\n", argv[argi]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--mode") == 0) || (strcmp(argv[argi], "-m") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected convolution mode name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[argi + 1], "output") == 0) {
				options.mode = mode_output;
			} else if (strcmp(argv[argi + 1], "input-gradient") == 0) {
				options.mode = mode_input_gradient;
			} else if (strcmp(argv[argi + 1], "kernel-gradient") == 0) {
				options.mode = mode_kernel_gradient;
			} else if (strcmp(argv[argi + 1], "inference") == 0) {
				options.mode = mode_inference;
			} else {
				fprintf(stderr, "Error: invalid value %s for the convolution mode\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--threads") == 0) || (strcmp(argv[argi], "-t") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected number of threads value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.threads) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.threads == 0) {
				options.threadpool = false;
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--iterations") == 0) || (strcmp(argv[argi], "-i") == 0)) {
			if (argi + 1 == argc) {
				fprintf(stderr, "Error: expected iterations value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[argi + 1], "%zu", &options.iterations) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.iterations == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of iterations: positive value expected\n", argv[argi + 1]);
				exit(EXIT_FAILURE);
			}
			argi += 1;
		} else if ((strcmp(argv[argi], "--help") == 0) || (strcmp(argv[argi], "-h") == 0)) {
			print_options_help(argv[0]);
			exit(EXIT_SUCCESS);
		} else {
			fprintf(stderr, "Error: unknown argument '%s'\n", argv[argi]);
			print_options_help(argv[0]);
			exit(EXIT_FAILURE);
		}
	}
	if ((options.mode == mode_inference) && (options.batch_size != 1)) {
		fprintf(stderr, "Error: inference requires unit batch size\n");
		exit(EXIT_FAILURE);
	}
	if (options.input_channels == 0) {
		fprintf(stderr, "Error: the number of input channels is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.output_channels == 0) {
		fprintf(stderr, "Error: the number of output channels is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.input_size.width == 0) {
		fprintf(stderr, "Error: the input size is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.kernel_size.width == 0) {
		fprintf(stderr, "Error: the kernel size is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	return options;
}

int main(int argc, char** argv) {
	enum nnp_status init_status = nnp_initialize();
	if (init_status != nnp_status_success) {
		fprintf(stderr, "NNPACK initialization failed: error code %d\n", init_status);
		exit(EXIT_FAILURE);
	}

	const struct options options = parse_options(argc, argv);

	const size_t batch_size = options.batch_size;
	const size_t input_channels = options.input_channels;
	const size_t output_channels = options.output_channels;
	const struct nnp_padding input_padding = { options.input_padding, options.input_padding, options.input_padding, options.input_padding };
	const struct nnp_size input_size = options.input_size;
	const struct nnp_size kernel_size = options.kernel_size;
	const struct nnp_size output_subsampling = options.output_subsampling;
	const struct nnp_size output_size = {
		.width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1,
		.height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1
	};
	struct nnp_size tile_size;
	double flops_per_element;

	printf("Batch size: %zu\n", batch_size);
	printf("Input channels: %zu\n", input_channels);
	printf("Output channels: %zu\n", output_channels);
	printf("Input: %zux%zu with implicit padding %zu\n", input_size.height, input_size.width, options.input_padding);
	printf("Kernel: %zux%zu\n", kernel_size.height, kernel_size.width);
	printf("Subsampling: %zux%zu\n", output_subsampling.height, output_subsampling.width);
	switch (options.algorithm) {
		case nnp_convolution_algorithm_auto:
			/* To avoid failure in the next phases */
			tile_size = kernel_size;
			printf("Algorithm: auto\n");
			break;
		case nnp_convolution_algorithm_ft8x8:
			tile_size = (struct nnp_size) { 8, 8 };
			flops_per_element = 4.0;
			printf("Algorithm: FT8x8\n");
			break;
		case nnp_convolution_algorithm_ft16x16:
			tile_size = (struct nnp_size) { 16, 16 };
			flops_per_element = 4.0;
			printf("Algorithm: FT16x16\n");
			break;
		case nnp_convolution_algorithm_wt8x8:
			tile_size = (struct nnp_size) { 8, 8 };
			flops_per_element = 2.0;
			printf("Algorithm: WT8x8\n");
			break;
		case nnp_convolution_algorithm_implicit_gemm:
			tile_size = (struct nnp_size) { 1, 1 };
			flops_per_element = 2.0 * kernel_size.height * kernel_size.width;
			printf("Algorithm: Implicit GEMM\n");
			break;
	}

        size_t memory_size = calc_scratch_memory_size( options.algorithm,
                                                       input_channels,
                                                       output_channels,
                                                       input_size,
                                                       input_padding,
                                                       kernel_size,
                                                       output_subsampling );

        printf("Scratch memory size: %d bytes\n", memory_size * sizeof(float));

        if( memory_size == 0 )
        {
            return nnp_status_out_of_memory;
        }

	vector<float> input(batch_size * input_channels * input_size.width * input_size.height);
	vector<float> kernel(input_channels * output_channels * kernel_size.width * kernel_size.height );
	vector<float> output(batch_size * output_channels * output_size.width * output_size.height);
	vector<float> referenceOutput(batch_size * output_channels * output_size.width * output_size.height);
	vector<float> bias(output_channels);
        vector<float> scratch_memory(memory_size);

	const uint_fast32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	auto rng = std::bind(std::uniform_real_distribution<float>(), std::mt19937(seed));
	std::generate(input.begin(), input.end(), std::ref(rng));
	std::generate(kernel.begin(), kernel.end(), std::ref(rng));
	std::generate(bias.begin(), bias.end(), std::ref(rng));
	std::fill(output.begin(), output.end(), std::nanf(""));
	std::fill(referenceOutput.begin(), referenceOutput.end(), std::nanf(""));
        std::fill( scratch_memory.begin(), scratch_memory.end(), 0 );

	pthreadpool_t threadpool = NULL;
	if (options.threadpool) {
		threadpool = pthreadpool_create(options.threads);
		printf("Threads: %zu\n", pthreadpool_get_threads_count(threadpool));
	}
	printf("Iterations: %zu\n", options.iterations);

	nnp_convolution_output__reference(
	1, input_channels, output_channels,
	input_size, input_padding, kernel_size, output_subsampling,
	input.data(), kernel.data(), bias.data(), referenceOutput.data(),
	threadpool);

	struct nnp_profile convolution_profile;
	enum nnp_status status = 
        nnp_convolution_inference_mem(
		options.algorithm,
		options.transform_strategy,
		input_channels,
		output_channels,
		input_size,
		input_padding,
		kernel_size,
		output_subsampling,
		input.data(),
		kernel.data(),
		bias.data(),
		output.data(),
                scratch_memory.data(),
		threadpool,
		&convolution_profile );	

	const float maxError = std::inner_product(referenceOutput.cbegin(), referenceOutput.cend(), output.cbegin(), 0.0f,
				[](float x, float y)->float { return std::max<float>(y, x); }, relativeError);


	printf("Inference status: %d\n", status);
	printf("Max Error: %f\n", maxError);

	const double convolution_time = convolution_profile.total;

	const struct nnp_size output_tile_size = {
		.height = tile_size.height - kernel_size.height + 1,
		.width = tile_size.width - kernel_size.width + 1
	};

	const size_t tile_count =
		(output_size.height / output_tile_size.height + !!(output_size.height % output_tile_size.height)) *
		(output_size.width / output_tile_size.width + !!(output_size.width % output_tile_size.width));

	const size_t input_transform_footprint = sizeof(float) * batch_size * input_channels *
		(input_size.height * input_size.width + tile_count * tile_size.height * tile_size.width);
	const size_t kernel_transform_footprint = sizeof(float) * output_channels * input_channels *
		(kernel_size.height * kernel_size.width + tile_size.height * tile_size.width);
	const size_t output_transform_footprint = sizeof(float)* batch_size * output_channels *
		(output_size.height * output_size.width + tile_count * tile_size.height * tile_size.width);

	printf("Time: %5.3f ms\n", convolution_time * 1.0e+3);
	printf("Input transform: %5.3f ms (%.1f%%) [%.1f GB/s]\n",
		convolution_profile.input_transform * 1.0e+3,
		(convolution_profile.input_transform / convolution_time) * 100.0,
		((double) input_transform_footprint) * 1.0e-9 / convolution_profile.input_transform);
	printf("Kernel transform: %5.3f ms (%.1f%%) [%.1f GB/s]\n",
		convolution_profile.kernel_transform * 1.0e+3,
		(convolution_profile.kernel_transform / convolution_time) * 100.0,
		((double) kernel_transform_footprint) * 1.0e-9 / convolution_profile.kernel_transform);
	printf("Output transform: %5.3f ms (%.1f%%) [%.1f GB/s]\n",
		convolution_profile.output_transform * 1.0e+3,
		(convolution_profile.output_transform / convolution_time) * 100.0,
		((double) output_transform_footprint) * 1.0e-9 / convolution_profile.output_transform);
	if (convolution_profile.block_multiplication != 0.0) {
		if (options.algorithm == nnp_convolution_algorithm_auto) {
			/* We don't know which algorithm is actually used, and thus can't compute FLOPS */
			printf("Block multiplication: %5.3f ms (%.1f%%)\n",
				convolution_profile.block_multiplication * 1.0e+3,
				(convolution_profile.block_multiplication / convolution_time) * 100.0);
		} else {
			printf("Block multiplication: %5.3f ms (%.1f%%) [%.1f GFLOPS]\n",
				convolution_profile.block_multiplication * 1.0e+3,
				(convolution_profile.block_multiplication / convolution_time) * 100.0,
				(flops_per_element * tile_size.height * tile_size.width * tile_count * batch_size * output_channels * input_channels * 1.0e-9) /
					convolution_profile.block_multiplication);
		}
	}
	const double overhead_time = convolution_profile.total -
		(convolution_profile.input_transform +
			convolution_profile.kernel_transform +
			convolution_profile.output_transform +
			convolution_profile.block_multiplication);
	printf("Overhead: %5.3f ms (%.1f%%)\n",
		overhead_time * 1.0e+3, (overhead_time / convolution_time) * 100.0);

	if (threadpool) {
		pthreadpool_destroy(threadpool);
	}

	return EXIT_SUCCESS;
}
