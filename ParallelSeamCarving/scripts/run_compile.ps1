$image_name = "3840x2160"
$program = "seam_carving.c"
$source = "./test_images/$image_name.png"
$target = "./output_images/$image_name.png"

$width = $args[0]

# Compile and run
gcc -O2 -lm --openmp $program -o out.exe
./out.exe $source $target $width