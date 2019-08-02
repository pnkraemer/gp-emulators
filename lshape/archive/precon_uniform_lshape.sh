
# run as:
# bash precon_uniform_lshape <path_to_files> <rank_for_h2lib_approx> <gmres_acc>

cd ../gp-emulators/lshape/

echo Creating file for output...

rm output_uniform_lshape.txt
touch output_uniform_lshape.txt

echo done.

echo ---------------------------------------------------
echo                     21
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 21 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/clean_lshape_gmres $1 21 24 $2 $3 |& tee -a output_uniform_lshape.txt


echo ---------------------------------------------------
echo                     65
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 65

time ./../../H2Lib/lshape/clean_lshape_gmres $1 65 68 $2 $3 >> output_uniform_lshape.txt

echo ---------------------------------------------------
echo                     225
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 225 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/clean_lshape_gmres $1 225 85 $2 $3 |& tee -a output_uniform_lshape.txt

echo ---------------------------------------------------
echo                     833
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 833 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/clean_lshape_gmres $1 833 130 $2 $3 |& tee -a output_uniform_lshape.txt

echo ---------------------------------------------------
echo                     3201
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 3201 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/clean_lshape_gmres $1 3201 187 $2 $3 |& tee -a output_uniform_lshape.txt


echo ---------------------------------------------------
echo                     12545
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 12545 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/clean_lshape_gmres $1 12545 254 $2 $3 |& tee -a output_uniform_lshape.txt



echo ---------------------------------------------------
echo                     49665
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 49665 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/clean_lshape_gmres $1 49665 333 $2 $3 |& tee -a output_uniform_lshape.txt
