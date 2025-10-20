#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <curand_kernel.h>
#include <cmath>

#define TILE_WIDTH 32 // We'll use this later for tiling optimization
#define MAX_TIMESTEPS 252 // Trading days in a year
#define NUM_ASSETS 16 // Multiple assets for complexity

// Structure for option parameters
struct OptionParams {
    float strike;
    float spot;
    float rate;
    float volatility;
    float maturity;
    float dividend_yield;
    int timesteps;
};

// Structure for market data (correlation matrix, volatilities, etc.)
struct MarketData {
    float correlation_matrix[NUM_ASSETS * NUM_ASSETS];
    float volatilities[NUM_ASSETS];
    float spot_prices[NUM_ASSETS];
    float weights[NUM_ASSETS]; // Portfolio weights
};

// Complex Monte Carlo Asian Option Pricing Kernel
// This kernel prices Asian options with multiple underlying assets
// THE BOTTLENECK: Multiple global memory accesses to market data arrays
__global__ void monteCarloAsianOptionKernel(
    float* option_prices,           // Output: option prices for each simulation
    float* convergence_data,        // Output: convergence analysis data  
    OptionParams* params,           // Input: option parameters array
    MarketData* market_data,        // Input: market data arrays
    float* random_numbers,          // Input: pre-generated random numbers
    int num_simulations,
    int num_options,
    int seed_offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int option_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tid >= num_simulations || option_idx >= num_options) return;
    
    // Initialize random number generator per thread
    curandState state;
    curand_init(seed_offset + tid * num_options + option_idx, 0, 0, &state);
    
    // Cache frequently accessed data (THE BOTTLENECK - multiple global memory reads)
    OptionParams current_option = params[option_idx];
    MarketData current_market = market_data[option_idx % NUM_ASSETS];
    
    float payoff_sum = 0.0f;
    float variance_sum = 0.0f;
    
    // Monte Carlo simulation loop
    for (int sim = 0; sim < num_simulations / (gridDim.x * blockDim.x); sim++) {
        float asset_paths[NUM_ASSETS];
        float asian_sum[NUM_ASSETS] = {0.0f};
        
        // Initialize asset prices
        for (int asset = 0; asset < NUM_ASSETS; asset++) {
            asset_paths[asset] = current_market.spot_prices[asset]; // GLOBAL MEMORY ACCESS
        }
        
        // Time evolution with correlated random walks
        for (int t = 0; t < current_option.timesteps; t++) {
            float dt = current_option.maturity / current_option.timesteps;
            
            // Generate correlated random numbers using Cholesky decomposition
            float independent_randoms[NUM_ASSETS];
            for (int i = 0; i < NUM_ASSETS; i++) {
                independent_randoms[i] = curand_normal(&state);
            }
            
            // Apply correlation structure (matrix multiplication)
            for (int asset = 0; asset < NUM_ASSETS; asset++) {
                float correlated_random = 0.0f;
                for (int j = 0; j <= asset; j++) {
                    // THE BOTTLENECK: Accessing correlation matrix from global memory
                    correlated_random += current_market.correlation_matrix[asset * NUM_ASSETS + j] 
                                        * independent_randoms[j];
                }
                
                // Black-Scholes evolution with complex calculations
                float vol = current_market.volatilities[asset]; // GLOBAL MEMORY ACCESS
                float drift = (current_option.rate - current_option.dividend_yield - 0.5f * vol * vol) * dt;
                float diffusion = vol * sqrtf(dt) * correlated_random;
                
                // Complex mathematical operations
                float log_return = drift + diffusion;
                asset_paths[asset] *= expf(log_return);
                
                // Accumulate for Asian average
                asian_sum[asset] += asset_paths[asset];
                
                // Additional complex computations (Greeks calculation)
                float delta = asset_paths[asset] * expf(-current_option.dividend_yield * (current_option.maturity - t * dt));
                float gamma = delta / (asset_paths[asset] * vol * sqrtf(current_option.maturity - t * dt));
                
                // Memory bandwidth intensive operations
                convergence_data[tid * MAX_TIMESTEPS + t] += gamma * current_market.weights[asset]; // GLOBAL MEMORY ACCESS
            }
        }
        
        // Calculate Asian option payoff (arithmetic average)
        float portfolio_average = 0.0f;
        for (int asset = 0; asset < NUM_ASSETS; asset++) {
            float asian_price = asian_sum[asset] / current_option.timesteps;
            portfolio_average += asian_price * current_market.weights[asset]; // GLOBAL MEMORY ACCESS
        }
        
        // Complex payoff calculation with barrier features
        float payoff = 0.0f;
        if (portfolio_average > current_option.strike) {
            // Call option with complex payoff structure
            payoff = (portfolio_average - current_option.strike) * expf(-current_option.rate * current_option.maturity);
            
            // Add exotic features (lookback, barrier, etc.)
            float max_price = portfolio_average; // Simplified - would track actual max
            float barrier_level = current_option.strike * 1.1f;
            
            if (max_price > barrier_level) {
                payoff *= 1.2f; // Bonus for breaching barrier
            }
            
            // Asian geometric mean for comparison
            float geometric_mean = 1.0f;
            for (int asset = 0; asset < NUM_ASSETS; asset++) {
                geometric_mean *= powf(asian_sum[asset] / current_option.timesteps, 
                                      current_market.weights[asset]); // GLOBAL MEMORY ACCESS
            }
            
            if (geometric_mean > current_option.strike) {
                payoff += 0.1f * (geometric_mean - current_option.strike);
            }
        }
        
        payoff_sum += payoff;
        variance_sum += payoff * payoff;
    }
    
    // Final calculations and global memory writes
    float mean_payoff = payoff_sum / (num_simulations / (gridDim.x * blockDim.x));
    float variance = variance_sum / (num_simulations / (gridDim.x * blockDim.x)) - mean_payoff * mean_payoff;
    
    option_prices[tid * num_options + option_idx] = mean_payoff;
    convergence_data[tid * num_options + option_idx] = sqrtf(variance); // Standard error
}

// Host function to initialize option parameters
void initializeOptions(OptionParams* options, int num_options) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> strike_dist(80.0f, 120.0f);
    std::uniform_real_distribution<float> spot_dist(90.0f, 110.0f);
    std::uniform_real_distribution<float> vol_dist(0.15f, 0.45f);
    std::uniform_real_distribution<float> rate_dist(0.01f, 0.05f);
    std::uniform_real_distribution<float> maturity_dist(0.25f, 2.0f);
    std::uniform_int_distribution<int> timesteps_dist(50, MAX_TIMESTEPS);
    
    for (int i = 0; i < num_options; i++) {
        options[i].strike = strike_dist(gen);
        options[i].spot = spot_dist(gen);
        options[i].rate = rate_dist(gen);
        options[i].volatility = vol_dist(gen);
        options[i].maturity = maturity_dist(gen);
        options[i].dividend_yield = rate_dist(gen) * 0.5f;
        options[i].timesteps = timesteps_dist(gen);
    }
}

// Host function to initialize market data with realistic correlation structure
void initializeMarketData(MarketData* market_data, int num_datasets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> vol_dist(0.15f, 0.35f);
    std::uniform_real_distribution<float> price_dist(50.0f, 150.0f);
    std::uniform_real_distribution<float> weight_dist(0.05f, 0.15f);
    std::normal_distribution<float> corr_dist(0.0f, 0.3f);
    
    for (int dataset = 0; dataset < num_datasets; dataset++) {
        // Initialize spot prices and volatilities
        float weight_sum = 0.0f;
        for (int asset = 0; asset < NUM_ASSETS; asset++) {
            market_data[dataset].spot_prices[asset] = price_dist(gen);
            market_data[dataset].volatilities[asset] = vol_dist(gen);
            market_data[dataset].weights[asset] = weight_dist(gen);
            weight_sum += market_data[dataset].weights[asset];
        }
        
        // Normalize weights to sum to 1
        for (int asset = 0; asset < NUM_ASSETS; asset++) {
            market_data[dataset].weights[asset] /= weight_sum;
        }
        
        // Create positive definite correlation matrix (Cholesky factor)
        for (int i = 0; i < NUM_ASSETS; i++) {
            for (int j = 0; j < NUM_ASSETS; j++) {
                if (i == j) {
                    market_data[dataset].correlation_matrix[i * NUM_ASSETS + j] = 1.0f;
                } else if (i > j) {
                    float correlation = std::max(-0.8f, std::min(0.8f, corr_dist(gen)));
                    market_data[dataset].correlation_matrix[i * NUM_ASSETS + j] = correlation;
                } else {
                    market_data[dataset].correlation_matrix[i * NUM_ASSETS + j] = 0.0f;
                }
            }
        }
    }
}

// Host function to verify Monte Carlo results
bool verifyMonteCarloResults(float* option_prices, float* convergence_data, int num_simulations, int num_options) {
    bool valid = true;
    int invalid_count = 0;
    
    for (int i = 0; i < num_simulations * num_options; i++) {
        if (std::isnan(option_prices[i]) || std::isinf(option_prices[i]) || option_prices[i] < 0) {
            invalid_count++;
            valid = false;
            if (invalid_count < 10) { // Don't spam too many error messages
                std::cout << "Invalid option price at index " << i << ": " << option_prices[i] << std::endl;
            }
        }
        
        if (std::isnan(convergence_data[i]) || std::isinf(convergence_data[i])) {
            invalid_count++;
            valid = false;
            if (invalid_count < 10) {
                std::cout << "Invalid convergence data at index " << i << ": " << convergence_data[i] << std::endl;
            }
        }
    }
    
    if (invalid_count > 10) {
        std::cout << "... and " << (invalid_count - 10) << " more invalid results" << std::endl;
    }
    
    return valid;
}

// Calculate theoretical Black-Scholes price for comparison
float blackScholesCall(float S, float K, float r, float sigma, float T) {
    if (T <= 0) return std::max(S - K, 0.0f);
    
    float d1 = (logf(S/K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtf(T));
    float d2 = d1 - sigma * sqrtf(T);
    
    // Simplified normal CDF approximation
    auto norm_cdf = [](float x) -> float {
        return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
    };
    
    return S * norm_cdf(d1) - K * expf(-r * T) * norm_cdf(d2);
}

int main(int argc, char** argv) {
    // Default simulation parameters
    int num_simulations = 65536;  // 64K simulations
    int num_options = 1024;       // 1K different options
    
    // Parse command line arguments
    if (argc > 1) {
        num_simulations = std::atoi(argv[1]);
        if (num_simulations <= 0) {
            std::cerr << "Invalid number of simulations: " << num_simulations << std::endl;
            return 1;
        }
    }
    if (argc > 2) {
        num_options = std::atoi(argv[2]);
        if (num_options <= 0) {
            std::cerr << "Invalid number of options: " << num_options << std::endl;
            return 1;
        }
    }
    
    std::cout << "Monte Carlo Asian Option Pricing Simulation" << std::endl;
    std::cout << "Simulations: " << num_simulations << ", Options: " << num_options << std::endl;
    std::cout << "Assets per portfolio: " << NUM_ASSETS << std::endl;
    
    // Calculate sizes and memory requirements
    size_t option_prices_size = num_simulations * num_options * sizeof(float);
    size_t convergence_data_size = num_simulations * num_options * sizeof(float);
    size_t options_size = num_options * sizeof(OptionParams);
    size_t market_data_size = NUM_ASSETS * sizeof(MarketData);
    size_t random_numbers_size = num_simulations * MAX_TIMESTEPS * NUM_ASSETS * sizeof(float);
    
    std::cout << "Memory requirements:" << std::endl;
    std::cout << "  Option prices: " << option_prices_size / (1024*1024) << " MB" << std::endl;
    std::cout << "  Market data: " << market_data_size / 1024 << " KB" << std::endl;
    std::cout << "  Random numbers: " << random_numbers_size / (1024*1024) << " MB" << std::endl;
    
    // Allocate host memory
    float* h_option_prices = new float[num_simulations * num_options];
    float* h_convergence_data = new float[num_simulations * num_options];
    OptionParams* h_options = new OptionParams[num_options];
    MarketData* h_market_data = new MarketData[NUM_ASSETS];
    float* h_random_numbers = new float[num_simulations * MAX_TIMESTEPS * NUM_ASSETS];
    
    // Initialize data structures
    std::cout << "Initializing options and market data..." << std::endl;
    initializeOptions(h_options, num_options);
    initializeMarketData(h_market_data, NUM_ASSETS);
    
    // Generate random numbers on host (for reproducibility)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    for (size_t i = 0; i < num_simulations * MAX_TIMESTEPS * NUM_ASSETS; i++) {
        h_random_numbers[i] = normal_dist(gen);
    }
    
    // Allocate device memory
    float* d_option_prices, * d_convergence_data, * d_random_numbers;
    OptionParams* d_options;
    MarketData* d_market_data;
    
    cudaMalloc(&d_option_prices, option_prices_size);
    cudaMalloc(&d_convergence_data, convergence_data_size);
    cudaMalloc(&d_options, options_size);
    cudaMalloc(&d_market_data, market_data_size);
    cudaMalloc(&d_random_numbers, random_numbers_size);
    
    // Copy data to device
    std::cout << "Copying data to device..." << std::endl;
    cudaMemcpy(d_options, h_options, options_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_market_data, h_market_data, market_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_random_numbers, h_random_numbers, random_numbers_size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    dim3 blockSize(TILE_WIDTH, 4); // Optimized for Monte Carlo workload
    dim3 gridSize((num_simulations + blockSize.x - 1) / blockSize.x,
                  (num_options + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Grid Size: (" << gridSize.x << ", " << gridSize.y << ")" << std::endl;
    std::cout << "Block Size: (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    
    // Warm up kernel (not timed)
    monteCarloAsianOptionKernel<<<gridSize, blockSize>>>(
        d_option_prices, d_convergence_data, d_options, d_market_data, 
        d_random_numbers, num_simulations, num_options, 12345
    );
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }
    
    // Time the kernel execution
    std::cout << "Running Monte Carlo Asian option pricing kernel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch kernel multiple times for better timing accuracy
    const int num_iterations = 5;
    for (int i = 0; i < num_iterations; ++i) {
        monteCarloAsianOptionKernel<<<gridSize, blockSize>>>(
            d_option_prices, d_convergence_data, d_options, d_market_data, 
            d_random_numbers, num_simulations, num_options, 12345 + i * 1000
        );
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate timing
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    
    std::cout << "Average execution time: " << avg_time_ms << " ms" << std::endl;
    
    // Calculate throughput (approximate operations per simulation)
    double ops_per_sim = (double)MAX_TIMESTEPS * NUM_ASSETS * NUM_ASSETS * 20; // Rough estimate
    double total_ops = ops_per_sim * num_simulations * num_options;
    double gflops = (total_ops / 1e9) / (avg_time_ms / 1000.0);
    std::cout << "Performance: " << gflops << " GFLOPS (estimated)" << std::endl;
    
    // Copy results back to host
    cudaMemcpy(h_option_prices, d_option_prices, option_prices_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_convergence_data, d_convergence_data, convergence_data_size, cudaMemcpyDeviceToHost);
    
    // Verify results
    std::cout << "Verifying Monte Carlo results..." << std::endl;
    if (verifyMonteCarloResults(h_option_prices, h_convergence_data, num_simulations, num_options)) {
        std::cout << "Result verification: PASSED" << std::endl;
    } else {
        std::cout << "Result verification: FAILED" << std::endl;
    }
    
    // Calculate and display statistics
    float mean_price = 0.0f, min_price = h_option_prices[0], max_price = h_option_prices[0];
    for (int i = 0; i < num_simulations * num_options; i++) {
        mean_price += h_option_prices[i];
        min_price = std::min(min_price, h_option_prices[i]);
        max_price = std::max(max_price, h_option_prices[i]);
    }
    mean_price /= (num_simulations * num_options);
    
    std::cout << "\nOption Pricing Results:" << std::endl;
    std::cout << "Mean option price: $" << mean_price << std::endl;
    std::cout << "Price range: $" << min_price << " - $" << max_price << std::endl;
    
    // Compare with theoretical Black-Scholes for first few options
    std::cout << "\nComparison with Black-Scholes (first 5 options):" << std::endl;
    for (int i = 0; i < std::min(5, num_options); i++) {
        float bs_price = blackScholesCall(h_options[i].spot, h_options[i].strike, 
                                          h_options[i].rate, h_options[i].volatility, 
                                          h_options[i].maturity);
        float mc_price = 0.0f;
        for (int sim = 0; sim < num_simulations; sim++) {
            mc_price += h_option_prices[sim * num_options + i];
        }
        mc_price /= num_simulations;
        
        std::cout << "Option " << i << ": MC=$" << mc_price << ", BS=$" << bs_price 
                  << ", Diff=" << (mc_price - bs_price) << std::endl;
    }
    
    // Clean up
    delete[] h_option_prices;
    delete[] h_convergence_data;
    delete[] h_options;
    delete[] h_market_data;
    delete[] h_random_numbers;
    
    cudaFree(d_option_prices);
    cudaFree(d_convergence_data);
    cudaFree(d_options);
    cudaFree(d_market_data);
    cudaFree(d_random_numbers);
    
    return 0;
}
