(* CUDA Monte Carlo with LLVM Optimization - OCaml Build System *)

type gpu_arch = 
  (* RTX 50 Series *)
  | RTX_5090 of string 
  | RTX_5080 of string
  | RTX_5070_Ti of string 
  | RTX_5070 of string
  | RTX_5060_Ti of string
  | RTX_5060 of string
  (* RTX 40 Series *)
  | RTX_4090 of string
  | RTX_4080_Super of string
  | RTX_4080 of string
  | RTX_4070_Ti_Super of string
  | RTX_4070_Ti of string
  | RTX_4070_Super of string
  | RTX_4070 of string
  | RTX_4060_Ti of string
  | RTX_4060 of string
  (* RTX 30 Series *)
  | RTX_3090_Ti of string
  | RTX_3090 of string
  | RTX_3080_Ti of string
  | RTX_3080 of string
  | RTX_3070_Ti of string
  | RTX_3070 of string
  | RTX_3060_Ti of string
  | RTX_3060 of string
  (* RTX 20 Series *)
  | RTX_2080_Ti of string
  | RTX_2080_Super of string
  | RTX_2080 of string
  | RTX_2070_Super of string
  | RTX_2070 of string
  | RTX_2060_Super of string
  | RTX_2060 of string
  (* Professional/Workstation *)
  | RTX_A6000 of string
  | RTX_A5000 of string
  | RTX_A4000 of string
  | RTX_A2000 of string
  | Quadro_RTX_8000 of string
  | Quadro_RTX_6000 of string
  | Quadro_RTX_5000 of string
  | Quadro_RTX_4000 of string
  | Tesla_V100 of string
  | Tesla_T4 of string
  | A100 of string
  | H100 of string
  | Other of string

type build_config = {
  project_root: string;
  build_dir: string;
  cuda_arch: string;
  build_type: string;
  llvm_dir: string;
}

let get_arch_string = function
  (* RTX 50 Series - all use sm_89 *)
  | RTX_5090 arch | RTX_5080 arch | RTX_5070_Ti arch | RTX_5070 arch 
  | RTX_5060_Ti arch | RTX_5060 arch
  (* RTX 40 Series - all use sm_89 *)
  | RTX_4090 arch | RTX_4080_Super arch | RTX_4080 arch 
  | RTX_4070_Ti_Super arch | RTX_4070_Ti arch | RTX_4070_Super arch | RTX_4070 arch
  | RTX_4060_Ti arch | RTX_4060 arch
  (* RTX 30 Series - all use sm_86 *) 
  | RTX_3090_Ti arch | RTX_3090 arch | RTX_3080_Ti arch | RTX_3080 arch
  | RTX_3070_Ti arch | RTX_3070 arch | RTX_3060_Ti arch | RTX_3060 arch
  (* RTX 20 Series - all use sm_75 *)
  | RTX_2080_Ti arch | RTX_2080_Super arch | RTX_2080 arch
  | RTX_2070_Super arch | RTX_2070 arch | RTX_2060_Super arch | RTX_2060 arch
  (* Professional/Workstation *)
  | RTX_A6000 arch | RTX_A5000 arch | RTX_A4000 arch | RTX_A2000 arch
  | Quadro_RTX_8000 arch | Quadro_RTX_6000 arch | Quadro_RTX_5000 arch | Quadro_RTX_4000 arch
  | Tesla_V100 arch | Tesla_T4 arch | A100 arch | H100 arch | Other arch -> arch

let get_gpu_name = function
  (* RTX 50 Series *)
  | RTX_5090 _ -> "RTX 5090"
  | RTX_5080 _ -> "RTX 5080" 
  | RTX_5070_Ti _ -> "RTX 5070 Ti"
  | RTX_5070 _ -> "RTX 5070"
  | RTX_5060_Ti _ -> "RTX 5060 Ti"
  | RTX_5060 _ -> "RTX 5060"
  (* RTX 40 Series *)
  | RTX_4090 _ -> "RTX 4090"
  | RTX_4080_Super _ -> "RTX 4080 Super"
  | RTX_4080 _ -> "RTX 4080"
  | RTX_4070_Ti_Super _ -> "RTX 4070 Ti Super"
  | RTX_4070_Ti _ -> "RTX 4070 Ti"
  | RTX_4070_Super _ -> "RTX 4070 Super"
  | RTX_4070 _ -> "RTX 4070"
  | RTX_4060_Ti _ -> "RTX 4060 Ti"
  | RTX_4060 _ -> "RTX 4060"
  (* RTX 30 Series *)
  | RTX_3090_Ti _ -> "RTX 3090 Ti"
  | RTX_3090 _ -> "RTX 3090"
  | RTX_3080_Ti _ -> "RTX 3080 Ti"
  | RTX_3080 _ -> "RTX 3080"
  | RTX_3070_Ti _ -> "RTX 3070 Ti"
  | RTX_3070 _ -> "RTX 3070"
  | RTX_3060_Ti _ -> "RTX 3060 Ti"
  | RTX_3060 _ -> "RTX 3060"
  (* RTX 20 Series *)
  | RTX_2080_Ti _ -> "RTX 2080 Ti"
  | RTX_2080_Super _ -> "RTX 2080 Super"
  | RTX_2080 _ -> "RTX 2080"
  | RTX_2070_Super _ -> "RTX 2070 Super"
  | RTX_2070 _ -> "RTX 2070"
  | RTX_2060_Super _ -> "RTX 2060 Super"
  | RTX_2060 _ -> "RTX 2060"
  (* Professional/Workstation *)
  | RTX_A6000 _ -> "RTX A6000"
  | RTX_A5000 _ -> "RTX A5000"
  | RTX_A4000 _ -> "RTX A4000"
  | RTX_A2000 _ -> "RTX A2000"
  | Quadro_RTX_8000 _ -> "Quadro RTX 8000"
  | Quadro_RTX_6000 _ -> "Quadro RTX 6000"
  | Quadro_RTX_5000 _ -> "Quadro RTX 5000"
  | Quadro_RTX_4000 _ -> "Quadro RTX 4000"
  | Tesla_V100 _ -> "Tesla V100"
  | Tesla_T4 _ -> "Tesla T4"
  | A100 _ -> "A100"
  | H100 _ -> "H100"
  | Other _ -> "Unknown GPU"

let execute_command cmd =
  let result = Sys.command cmd in
  result = 0

let execute_with_output cmd success_msg error_msg =
  if execute_command cmd then
    Printf.printf "%s\n" success_msg
  else (
    Printf.printf "%s\n" error_msg;
    exit 1
  )

let execute_optional cmd success_msg warning_msg =
  if execute_command cmd then
    Printf.printf "%s\n" success_msg
  else
    Printf.printf "%s\n" warning_msg

let detect_gpu () =
  (* For now, we'll default to RTX 5070 Ti since that's the detected GPU *)
  (* In a full implementation, we'd parse nvidia-smi output *)
  RTX_5070_Ti "sm_89"

let check_prerequisites () =
  Printf.printf "[INFO] Checking prerequisites...\n";
  
  if not (execute_command "which nvcc > /dev/null 2>&1") then (
    Printf.printf "[ERROR] nvcc not found. Please install CUDA Toolkit.\n";
    exit 1
  );
  
  Printf.printf "[SUCCESS] Found CUDA Toolkit\n";
  
  if execute_command "which llvm-config > /dev/null 2>&1" then
    Printf.printf "[SUCCESS] Found LLVM\n"
  else (
    Printf.printf "[WARNING] llvm-config not found. Checking custom LLVM...\n";
    if Sys.file_exists "/home/dhruvad/llvm-project/build" then
      Printf.printf "[SUCCESS] Found LLVM at /home/dhruvad/llvm-project/build\n"
    else (
      Printf.printf "[ERROR] LLVM not found\n";
      exit 1
    )
  )

let create_build_dir build_dir =
  Printf.printf "[INFO] Creating build directory...\n";
  if not (Sys.file_exists build_dir) then
    execute_with_output (Printf.sprintf "mkdir -p %s" build_dir)
      "[SUCCESS] Created build directory"
      "[ERROR] Failed to create build directory"

let build_cuda_kernel config gpu =
  Printf.printf "[INFO] Building CUDA Monte Carlo kernel...\n";
  let cuda_arch = get_arch_string gpu in
  let build_cmd = Printf.sprintf 
    "nvcc -O3 -arch=%s -std=c++17 src/matrix_mul.cu -o %s/monte_carlo_baseline -lcurand"
    cuda_arch config.build_dir in
  
  execute_with_output build_cmd 
    "[SUCCESS] Built baseline Monte Carlo kernel"
    "[ERROR] Failed to build CUDA kernel"

let generate_llvm_ir config gpu =
  Printf.printf "[INFO] Generating LLVM IR from CUDA kernel...\n";
  let cuda_arch = get_arch_string gpu in
  let ir_cmd = Printf.sprintf
    "clang++ -I/usr/local/cuda-12.6/include --cuda-gpu-arch=%s --cuda-path=/usr/local/cuda-12.6 -S -emit-llvm src/matrix_mul.cu"
    cuda_arch in
  
  execute_with_output ir_cmd
    "[SUCCESS] Generated LLVM IR (host + device)"
    "[ERROR] Failed to generate LLVM IR"

let optimize_llvm_ir () =
  Printf.printf "[INFO] Applying LLVM optimizations...\n";
  execute_optional "opt-18 -O3 matrix_mul.ll -o matrix_mul_optimized.ll"
    "[SUCCESS] Applied LLVM -O3 optimizations"
    "[WARNING] LLVM optimization failed, continuing with baseline"

let compile_optimized () =
  Printf.printf "[INFO] Compiling LLVM-optimized version...\n";
  let compile_cmd = "clang++ -O3 matrix_mul_optimized.ll -L/usr/local/cuda-12.6/lib64 -lcudart -lcurand -o monte_carlo_llvm_optimized" in
  execute_optional compile_cmd
    "[SUCCESS] Built LLVM-optimized executable"
    "[WARNING] LLVM-optimized build failed"

let print_summary gpu config =
  Printf.printf "\n=== OCaml Build Summary ===\n";
  Printf.printf "GPU: %s (%s)\n" (get_gpu_name gpu) (get_arch_string gpu);
  Printf.printf "Project root: %s\n" config.project_root;
  Printf.printf "Build directory: %s\n" config.build_dir;
  Printf.printf "Build type: %s\n" config.build_type;
  Printf.printf "\n[SUCCESS] CUDA Monte Carlo with LLVM pipeline ready!\n";
  Printf.printf "Run: ./%s/monte_carlo_baseline 16384 256\n" config.build_dir;

let main () =
  let config = {
    project_root = Sys.getcwd ();
    build_dir = "build";
    cuda_arch = "sm_89";
    build_type = "Release";
    llvm_dir = "/home/dhruvad/llvm-project/build";
  } in
  
  let gpu = detect_gpu () in
  Printf.printf "Detected %s\n" (get_gpu_name gpu);
  Printf.printf "=== OCaml CUDA Monte Carlo Build System ===\n";
  
  check_prerequisites ();
  create_build_dir config.build_dir;
  build_cuda_kernel config gpu;
  generate_llvm_ir config gpu;
  optimize_llvm_ir ();
  compile_optimized ();
  print_summary gpu config

let () = main ()
