#!/bin/bash

d=./results/clevr_monet_pretrained/test_latest/images
out="$(dirname $d)"

for i in {0..9}; do
	montage $d/CLEVR_test_00000${i}_m{0..10}.png $d/CLEVR_test_00000${i}_x{0..10}.png $d/CLEVR_test_00000${i}_xm{0..10}.png $d/CLEVR_test_00000${i}_x.png $d/CLEVR_test_00000${i}_x_tilde.png -geometry +2+2 -tile 11x4 $out/CLEVR_test_montage_${i}.png
done
