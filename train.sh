experimenttag=$1


basepath=$(pwd)
#data="${basepath}/data/"
data="${basepath}/data/"

cd ${basepath}/src/gaussiansplatting

#numiterations=5000
#numiterations=15000
numiterations=20000
timestamp=$(date +%s)

ITERS_RENDER=-1 
#ITERS_RENDER="2000 7000 12000 15000" 

if pip install submodules/diff-gaussian-rasterization submodules/simple-knn; then

    #for scene in "JAX_068"
    #for scene in "JAX_068_john_img"
    #for scene in "JAX_068_selected_by_john"
    #for scene in "JAX_214_selected_by_john"
    #for scene in "JAX_214_john"
    for scene in "add_WV3"
    #for scene in "add_EROS"
    #for scene in "JAX_214"
    # for scene in "JAX_004" "JAX_068" "JAX_214" "JAX_260"
    #for scene in "IARPA_001" "IARPA_002" "IARPA_003" "JAX_004" "JAX_068" "JAX_214" "JAX_260"
    do

        expname=$scene

: << END

        path_json="${data}/affine_models/${scene}/affine_models.json"
        #echo "path_json : $path_json"
        #if [ ! -e ${path_json} ]; then
            python ${basepath}/scripts/dataset_creation/to_affine.py --root_dir ${data}/rpcs --dataset_destination_path ${data}/affine_models --dir_imgs ${data}/images --scene_name $scene
        #fi

        #expname="test_${timestamp}_${scene}_NEW_${experimenttag}_rep${rep}"
        #expname="JAX_068_250714"
        #expname="JAX_068_john_img"
        #expname="JAX_068_selected_by_john"
        
        if [[ "$scene" == "$add_"* ]]; then
            # Train the EOGS model
            export CUDA_VISIBLE_DEVICES=0
            export CUDA_LAUNCH_BLOCKING=1
            export TORCH_USE_CUDA_DSA=1
            python train.py \
                -s ${data}/affine_models/${scene} \
                --images ${data}/images/${scene} \
                -m ${basepath}/output/${expname} \
                --sh_degree 0 \
                --iterations ${numiterations}
        else
            # Train the EOGS model
            python train.py \
                -s ${data}/affine_models/${scene} \
                --images ${data}/images/${scene} \
                -m ${basepath}/output/${expname} \
                --sh_degree 0 \
                --iterations ${numiterations} \
                --eval
        fi

END

#: << END

        export CUDA_VISIBLE_DEVICES=1
        for iter_render in $ITERS_RENDER
        do
            echo "iter_render : $iter_render"
            # Render the EOGS model after training
            python render.py -m ${basepath}/output/${expname} --iteration $iter_render
            #python render.py -m "${basepath}/output/${expname}/density_0.13" --iteration $iter_render
        done

#END



: << END
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
END
    done

fi
cd -
