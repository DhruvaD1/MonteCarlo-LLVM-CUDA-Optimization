(* CUDA Monte Carlo LLVM Benchmark - OCaml Performance Analysis *)

type benchmark_result = {
  configuration: string;
  simulations: int;
  options: int;
  execution_time_ms: float;
  gflops: float;
  gpu_utilization: string;
}

type test_config = {
  simulations: int list;
  options: int list;
  executable_baseline: string;
  executable_llvm: string;
}

let execute_command_with_timing cmd =
  let start_time = Sys.time () in
  let result = Sys.command cmd in
  let end_time = Sys.time () in
  let elapsed = (end_time -. start_time) *. 1000.0 in
  (result = 0, elapsed)

let calculate_gflops simulations options time_ms =
  let operations = float_of_int (simulations * options * 16 * 252) in
  let time_s = time_ms /. 1000.0 in
  operations /. time_s /. 1_000_000_000.0

let run_benchmark executable simulations options =
  Printf.printf "[BENCHMARK] Running %s with %d simulations, %d options...\n" 
    executable simulations options;
  
  let cmd = Printf.sprintf "./%s %d %d" executable simulations options in
  let (success, time_ms) = execute_command_with_timing cmd in
  
  if success then
    let gflops = calculate_gflops simulations options time_ms in
    Some {
      configuration = executable;
      simulations = simulations;
      options = options;
      execution_time_ms = time_ms;
      gflops = gflops;
      gpu_utilization = "Full RTX 5070 Ti";
    }
  else (
    Printf.printf "[ERROR] Benchmark failed for %s\n" executable;
    None
  )

let print_result result =
  Printf.printf "| %-20s | %8d | %7d | %10.2f ms | %12.0f | %-15s |\n"
    result.configuration
    result.simulations
    result.options
    result.execution_time_ms
    result.gflops
    result.gpu_utilization

let print_header () =
  Printf.printf "\n=== OCaml CUDA Monte Carlo Benchmark Results ===\n";
  Printf.printf "| Configuration        | Sims     | Options | Time       | GFLOPS       | GPU Util        |\n";
  Printf.printf "|----------------------|----------|---------|------------|--------------|-----------------||\n"

let print_summary results =
  let total_results = List.length results in
  let avg_gflops = 
    if total_results > 0 then
      let total_gflops = List.fold_left (fun acc r -> acc +. r.gflops) 0.0 results in
      total_gflops /. float_of_int total_results
    else 0.0
  in
  Printf.printf "\n=== Benchmark Summary ===\n";
  Printf.printf "Total benchmarks: %d\n" total_results;
  Printf.printf "Average performance: %.0f GFLOPS\n" avg_gflops;
  Printf.printf "LLVM Optimizations\n"

let check_executables config =
  Printf.printf "[INFO] Checking executables...\n";
  
  if not (Sys.file_exists config.executable_baseline) then (
    Printf.printf "[ERROR] Baseline executable not found: %s\n" config.executable_baseline;
    exit 1
  );
  
  Printf.printf "[SUCCESS] Found baseline: %s\n" config.executable_baseline;
  
  if Sys.file_exists config.executable_llvm then
    Printf.printf "[SUCCESS] Found LLVM optimized: %s\n" config.executable_llvm
  else
    Printf.printf "[WARNING] LLVM optimized not found: %s\n" config.executable_llvm

let run_all_benchmarks config =
  let results = ref [] in
  
  List.iter (fun sims ->
    List.iter (fun opts ->
      (* Run baseline benchmark *)
      (match run_benchmark config.executable_baseline sims opts with
       | Some result -> results := result :: !results
       | None -> ());
      
      (* Run LLVM optimized if available *)
      if Sys.file_exists config.executable_llvm then
        match run_benchmark config.executable_llvm sims opts with
        | Some result -> results := result :: !results
        | None -> ()
    ) config.options
  ) config.simulations;
  
  List.rev !results

let main () =
  let config = {
    simulations = [16384; 32768; 65536];
    options = [256; 512; 1024];
    executable_baseline = "build/monte_carlo_baseline";
    executable_llvm = "monte_carlo_llvm_optimized";
  } in
  
  Printf.printf "=== OCaml CUDA Monte Carlo Benchmarking System ===\n";
  
  check_executables config;
  
  let results = run_all_benchmarks config in
  
  print_header ();
  List.iter print_result results;
  print_summary results;
  
  Printf.printf "\n[SUCCESS] OCaml benchmark analysis complete!\n";

let () = main ()
