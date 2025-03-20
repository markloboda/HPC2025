$width = $args[0]
$source = "./test_images/3840x2160.png"
$target = "./output_images/3840x2160_parallel.png"

# Compile and run
gcc -O2 -lm --openmp parallel_seam_carving.c -o parallel_seam_carving.exe
./parallel_seam_carving.exe $source $target $width