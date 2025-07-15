experimenttag=$1
export CUDA_VISIBLE_DEVICES=${2:-0}


basepath=$(pwd)
data="${basepath}/data/"

cd ${basepath}/src/gaussiansplatting

numiterations=5000
timestamp=$(date +%s)

if pip install submodules/diff-gaussian-rasterization submodules/simple-knn; then

    for rep in "01"
    # for rep in "01" "02" "03" "04"
    do
        for scene in "JAX_068"
        # for scene in "JAX_004" "JAX_068" "JAX_214" "JAX_260"
        #for scene in "IARPA_001" "IARPA_002" "IARPA_003" "JAX_004" "JAX_068" "JAX_214" "JAX_260"
        do
            path_json="${data}/affine_models/${scene}/affine_models.json"
            #echo "path_json : $path_json"
            if [ ! -e ${path_json} ]; then
                #echo "111"
                python ${basepath}/scripts/dataset_creation/to_affine.py --root_dir ${data}/rpcs --dataset_destination_path ${data}/affine_models --scene_name $scene
            fi

            #expname="test_${timestamp}_${scene}_NEW_${experimenttag}_rep${rep}"
            expname="JAX_068_250714"
: << END
            # Train the EOGS model
            python train.py \
                -s ${data}/affine_models/${scene} \
                --images ${data}/images/${scene} \
                --eval \
                -m ${basepath}/output/${expname} \
                --sh_degree 0 \
                --iterations ${numiterations}

            # Render the EOGS model after training
            python render.py -m ${basepath}/output/${expname}
END

#: << END
            # Evaluate the EOGS model using the ground truth DSM
            # get the last dsm file name, it corresponds to the top down view
            dsm_name=$(ls ${basepath}/output/${expname}/test_opNone/ours_${numiterations}/dsm/ | sort -V | tail -n 1)
            #echo "dsm_name : $dsm_name"
            #exit 1
            # If you are interested in reproducing the second part of the Table 1 in the paper,
            # add the following argument to the command line: --filter_tree
            python ${basepath}/scripts/eval/eval_dsm.py \
                --pred-dsm-path ${basepath}/output/${expname}/test_opNone/ours_${numiterations}/dsm/${dsm_name} \
                --gt-dir ${data}/truth/${scene} \
                --out-dir ${basepath}/output/${expname}/ \
                --aoi-id ${scene}
#END
        done
    done
fi
cd -
