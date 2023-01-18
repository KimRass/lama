# What is LaMa?
- [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://github.com/saic-mdal/lama)

# Directory Structure
```
lama
├── LICENSE
├── README.md
├── bin
│   ├── analyze_errors.py
│   ├── blur_predicts.py
│   ├── calc_dataset_stats.py
│   ├── debug
│   │   └── analyze_overlapping_masks.sh
│   ├── evaluate_predicts.py
│   ├── evaluator_example.py
│   ├── extract_masks.py
│   ├── filter_sharded_dataset.py
│   ├── gen_debug_mask_dataset.py
│   ├── gen_mask_dataset.py
│   ├── gen_mask_dataset_hydra.py
│   ├── gen_outpainting_dataset.py
│   ├── make_checkpoint.py
│   ├── mask_example.py
│   ├── paper_runfiles
│   │   ├── blur_tests.sh
│   │   ├── env.sh
│   │   ├── find_best_checkpoint.py
│   │   ├── generate_test_celeba-hq.sh
│   │   ├── generate_test_ffhq.sh
│   │   ├── generate_test_paris.sh
│   │   ├── generate_test_paris_256.sh
│   │   ├── generate_val_test.sh
│   │   ├── predict_inner_features.sh
│   │   └── update_test_data_stats.sh
│   ├── predict.py
│   ├── predict_inner_features.py
│   ├── report_from_tb.py
│   ├── sample_from_dataset.py
│   ├── side_by_side.py
│   ├── split_tar.py
│   ├── to_jit.py
│   └── train.py
├── colab
│   └── LaMa_inpainting.ipynb
├── conda_env.yml
├── configs
│   ├── analyze_mask_errors.yaml
│   ├── data_gen
│   │   ├── random_medium_256.yaml
│   │   ├── random_medium_512.yaml
│   │   ├── random_thick_256.yaml
│   │   ├── random_thick_512.yaml
│   │   ├── random_thin_256.yaml
│   │   └── random_thin_512.yaml
│   ├── debug_mask_gen.yaml
│   ├── eval1.yaml
│   ├── eval2.yaml
│   ├── eval2_cpu.yaml
│   ├── eval2_gpu.yaml
│   ├── eval2_jpg.yaml
│   ├── eval2_segm.yaml
│   ├── eval2_segm_test.yaml
│   ├── eval2_test.yaml
│   ├── places2-categories_157.txt
│   ├── prediction
│   │   └── default.yaml
│   ├── test_large_30k.lst
│   └── training
│       ├── ablv2_work.yaml
│       ├── ablv2_work_ffc075.yaml
│       ├── ablv2_work_md.yaml
│       ├── ablv2_work_no_fm.yaml
│       ├── ablv2_work_no_segmpl.yaml
│       ├── ablv2_work_no_segmpl_csdilirpl.yaml
│       ├── ablv2_work_no_segmpl_csdilirpl_celeba_csdilirpl1_new.yaml
│       ├── ablv2_work_no_segmpl_csirpl.yaml
│       ├── ablv2_work_no_segmpl_csirpl_celeba_csirpl03_new.yaml
│       ├── ablv2_work_no_segmpl_vgg.yaml
│       ├── ablv2_work_no_segmpl_vgg_celeba_l2_vgg003_new.yaml
│       ├── ablv2_work_nodil_segmpl.yaml
│       ├── ablv2_work_small_holes.yaml
│       ├── big-lama-celeba.yaml
│       ├── big-lama-regular-celeba.yaml
│       ├── big-lama-regular.yaml
│       ├── big-lama.yaml
│       ├── data
│       │   ├── abl-02-thin-bb.yaml
│       │   ├── abl-04-256-mh-dist-celeba.yaml
│       │   ├── abl-04-256-mh-dist-web.yaml
│       │   └── abl-04-256-mh-dist.yaml
│       ├── discriminator
│       │   └── pix2pixhd_nlayer.yaml
│       ├── evaluator
│       │   └── default_inpainted.yaml
│       ├── generator
│       │   ├── ffc_resnet_075.yaml
│       │   ├── pix2pixhd_global.yaml
│       │   ├── pix2pixhd_global_sigmoid.yaml
│       │   └── pix2pixhd_multidilated_catin_4dil_9b.yaml
│       ├── hydra
│       │   ├── no_time.yaml
│       │   └── overrides.yaml
│       ├── lama-fourier-celeba.yaml
│       ├── lama-fourier.yaml
│       ├── lama-regular-celeba.yaml
│       ├── lama-regular.yaml
│       ├── lama_small_train_masks.yaml
│       ├── location
│       │   ├── celeba_example.yaml
│       │   ├── docker.yaml
│       │   └── places_example.yaml
│       ├── optimizers
│       │   └── default_optimizers.yaml
│       ├── trainer
│       │   ├── any_gpu_large_ssim_ddp_final.yaml
│       │   ├── any_gpu_large_ssim_ddp_final_benchmark.yaml
│       │   └── any_gpu_large_ssim_ddp_final_celeba.yaml
│       └── visualizer
│           └── directory.yaml
├── docker
│   ├── 1_generate_masks_from_raw_images.sh
│   ├── 2_predict.sh
│   ├── 3_evaluate.sh
│   ├── Dockerfile
│   ├── Dockerfile-cuda111
│   ├── build-cuda111.sh
│   ├── build.sh
│   └── entrypoint.sh
├── fetch_data
│   ├── celebahq_dataset_prepare.sh
│   ├── celebahq_gen_masks.sh
│   ├── eval_sampler.py
│   ├── places_challenge_train_download.sh
│   ├── places_standard_evaluation_prepare_data.sh
│   ├── places_standard_test_val_gen_masks.sh
│   ├── places_standard_test_val_prepare.sh
│   ├── places_standard_test_val_sample.sh
│   ├── places_standard_train_prepare.sh
│   ├── sampler.py
│   ├── train_shuffled.flist
│   └── val_shuffled.flist
├── models
│   ├── ade20k
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── color150.mat
│   │   ├── mobilenet.py
│   │   ├── object150_info.csv
│   │   ├── resnet.py
│   │   ├── segm_lib
│   │   │   ├── nn
│   │   │   │   ├── __init__.py
│   │   │   │   ├── modules
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── batchnorm.py
│   │   │   │   │   ├── comm.py
│   │   │   │   │   ├── replicate.py
│   │   │   │   │   ├── tests
│   │   │   │   │   │   ├── test_numeric_batchnorm.py
│   │   │   │   │   │   └── test_sync_batchnorm.py
│   │   │   │   │   └── unittest.py
│   │   │   │   └── parallel
│   │   │   │       ├── __init__.py
│   │   │   │       └── data_parallel.py
│   │   │   └── utils
│   │   │       ├── __init__.py
│   │   │       ├── data
│   │   │       │   ├── __init__.py
│   │   │       │   ├── dataloader.py
│   │   │       │   ├── dataset.py
│   │   │       │   ├── distributed.py
│   │   │       │   └── sampler.py
│   │   │       └── th.py
│   │   └── utils.py
│   └── lpips_models
│       ├── alex.pth
│       ├── squeeze.pth
│       └── vgg.pth
├── requirements.txt
└── saicinpainting
    ├── __init__.py
    ├── evaluation
    │   ├── __init__.py
    │   ├── data.py
    │   ├── evaluator.py
    │   ├── losses
    │   │   ├── __init__.py
    │   │   ├── base_loss.py
    │   │   ├── fid
    │   │   │   ├── __init__.py
    │   │   │   ├── fid_score.py
    │   │   │   └── inception.py
    │   │   ├── lpips.py
    │   │   └── ssim.py
    │   ├── masks
    │   │   ├── README.md
    │   │   ├── __init__.py
    │   │   ├── countless
    │   │   │   ├── README.md
    │   │   │   ├── __init__.py
    │   │   │   ├── countless2d.py
    │   │   │   ├── countless3d.py
    │   │   │   ├── images
    │   │   │   │   ├── gcim.jpg
    │   │   │   │   ├── gray_segmentation.png
    │   │   │   │   ├── segmentation.png
    │   │   │   │   └── sparse.png
    │   │   │   ├── memprof
    │   │   │   │   ├── countless2d_gcim_N_1000.png
    │   │   │   │   ├── countless2d_quick_gcim_N_1000.png
    │   │   │   │   ├── countless3d.png
    │   │   │   │   ├── countless3d_dynamic.png
    │   │   │   │   ├── countless3d_dynamic_generalized.png
    │   │   │   │   └── countless3d_generalized.png
    │   │   │   ├── requirements.txt
    │   │   │   └── test.py
    │   │   └── mask.py
    │   ├── refinement.py (o)
    │   ├── utils.py
    │   └── vis.py
    ├── training
    │   ├── __init__.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   ├── aug.py
    │   │   ├── datasets.py
    │   │   └── masks.py
    │   ├── losses
    │   │   ├── __init__.py
    │   │   ├── adversarial.py
    │   │   ├── constants.py
    │   │   ├── distance_weighting.py
    │   │   ├── feature_matching.py
    │   │   ├── perceptual.py
    │   │   ├── segmentation.py
    │   │   └── style_loss.py
    │   ├── modules
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── depthwise_sep_conv.py
    │   │   ├── fake_fakes.py
    │   │   ├── ffc.py
    │   │   ├── multidilated_conv.py
    │   │   ├── multiscale.py
    │   │   ├── pix2pixhd.py
    │   │   ├── spatial_transform.py
    │   │   └── squeeze_excitation.py
    │   ├── trainers
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   └── default.py
    │   └── visualizers
    │       ├── __init__.py
    │       ├── base.py
    │       ├── colors.py
    │       ├── directory.py
    │       └── noop.py
    └── utils.py
```

# Model Architecture
```
LaMa(
  (model): Sequential(
    (0): ReflectionPad2d((3, 3, 3, 3))
    (1): FFC_BN_ACT(
      (ffc): FFC(
        (convl2l): Conv2d(4, 64, kernel_size=(7, 7), stride=(1, 1), bias=False, padding_mode=reflect)
        (convl2g): Identity()
        (convg2l): Identity()
        (convg2g): Identity()
        (gate): Identity()
      )
      (bn_l): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn_g): Identity()
      (act_l): ReLU(inplace=True)
      (act_g): Identity()
    )
    (2): FFC_BN_ACT(
      (ffc): FFC(
        (convl2l): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, padding_mode=reflect)
        (convl2g): Identity()
        (convg2l): Identity()
        (convg2g): Identity()
        (gate): Identity()
      )
      (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn_g): Identity()
      (act_l): ReLU(inplace=True)
      (act_g): Identity()
    )
    (3): FFC_BN_ACT(
      (ffc): FFC(
        (convl2l): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, padding_mode=reflect)
        (convl2g): Identity()
        (convg2l): Identity()
        (convg2g): Identity()
        (gate): Identity()
      )
      (bn_l): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn_g): Identity()
      (act_l): ReLU(inplace=True)
      (act_g): Identity()
    )
    (4): FFC_BN_ACT(
      (ffc): FFC(
        (convl2l): Conv2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, padding_mode=reflect)
        (convl2g): Conv2d(256, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, padding_mode=reflect)
        (convg2l): Identity()
        (convg2g): Identity()
        (gate): Identity()
      )
      (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_l): ReLU(inplace=True)
      (act_g): ReLU(inplace=True)
    )
    (5): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (6): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (7): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (8): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (9): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (10): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (11): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (12): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (13): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (14): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (15): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (16): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (17): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (18): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (19): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (20): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (21): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (22): FFCResnetBlock(
      (conv1): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
      (conv2): FFC_BN_ACT(
        (ffc): FFC(
          (convl2l): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convl2g): Conv2d(128, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2l): Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, padding_mode=reflect)
          (convg2g): SpectralTransform(
            (downsample): Identity()
            (conv1): Sequential(
              (0): Conv2d(384, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU(inplace=True)
            )
            (fu): FourierUnit(
              (conv_layer): Conv2d(384, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (relu): ReLU(inplace=True)
            )
            (conv2): Conv2d(192, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
          )
          (gate): Identity()
        )
        (bn_l): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bn_g): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act_l): ReLU(inplace=True)
        (act_g): ReLU(inplace=True)
      )
    )
    (23): ConcatTupleLayer()
    (24): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (25): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (26): ReLU(inplace=True)
    (27): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (28): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace=True)
    (30): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (31): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace=True)
    (33): ReflectionPad2d((3, 3, 3, 3))
    (34): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
    (35): Sigmoid()
  )
)
```

# Paper Review
- Paper: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/pdf/2109.07161.pdf)
## Image Inpainting
- The inpainting problem is inherently ambiguous. There could be many plausible fillings for the same missing areas,
especially when the “holes” become wider.
## Train
- The usual practice is to train inpainting systems on a large automatically generated dataset, created by randomly
masking real images.
- The training is performed on a dataset of (image, mask) pairs obtained from real images and synthetically generated masks.
## Characteristics
- It’s common to use complicated two-stage models with intermediate predictions, such as smoothed images, edges, and segmentation maps. In this work, we achieve state-of-the-art results with a simple single-stage network.
- Generalizes surprisingly well to resolutions that are higher than those seen at train time.
- Can generalize to high-resolution images after training only on low-resolution data.
- Can capture and generate complex periodic structures, and is robust to large masks.
- Significantly less trainable parameters and inference time costs.
### Large Effective Receptive Field
- Essential for understanding the global structure of an image.
- In the case of a large mask, an even large yet limited receptive field may not be enough to access information necessary for generating a quality inpainting.
- Popular convolutional architectures might lack a sufficiently large effective receptive field.
### large training masks
## Method
- Our goal is to inpaint a color image 'x' masked by a binary mask of unknown pixels 'm'. The mask 'm' is stacked with the masked image, resulting in a four-channel input tensor 'x'.
  - 'lama'>'saicinpainting'>'evaluation'>'refinement.py'>`_infer` (121 ~ 122):
    ```python
    masked_image = image * (1 - mask)
    masked_image = torch.cat([masked_image, mask], dim=1)
    ```
  - 'lama'>'saicinpainting'>'training'>'trainers'>'default.py'>`DefaultInpaintingTrainingModule`>`forward`:
    ```python
    masked_img = img * (1 - mask)
    ...
    if self.concat_mask:
        masked_img = torch.cat([masked_img, mask], dim=1)
    ```

# Architecture
- Processes the input in a fully-convolutional manner.
- the generation of proper inpainting requires to consider global context. Thus, we argue that a good architecture should have units with as wide-as-possible receptive field as early as possible in the pipeline.
## ResNet
- The conventional fully convolutional models, e.g. ResNet, suffer from slow growth of effective receptive field.
- Receptive field might be insufficient, especially in the early layers of the network, due to the typically small (e.g. 3 × 3) convolutional kernels. Thus, many layers in the network will be lacking global context and will waste computations and parameters to create one.
- For wide masks, the whole receptive field of a generator at the specific position may be inside the mask, thus observing only missing pixels. The issue becomes especially pronounced for high-resolution images.
## Fast Fourier Convolutions (FCCs)
- Have image-wide receptive field
- Allow for a receptive field that covers an entire image even in the early layers of the network.
- The inductive bias of FFC allows the network to generalize to high resolutions that are never seen during training.
- Fast Fourier convolution (FFC) is the recently proposed operator that allows to use global context in early layers.
- FFC is based on a channel-wise fast Fourier transform (FFT) and has a receptive field that covers the entire image.
- FFC splits channels into two parallel branches: i) local branch uses conventional convolutions, and ii) global branch uses real FFT to account for global context. Real FFT can be applied only to real valued signals, and inverse real FFT ensures that the output is real valued. Real FFT uses only half of the spectrum compared to the FFT.
- the outputs of the local (i) and global (ii) branches are fused together
- FFCs are fully differentiable and easy-to-use drop-in replacement for conventional convolutions.
- Due to the image-wide receptive field, FFCs allow the generator to account for the global context starting from
the early layers, which is crucial for high-resolution image inpainting.

# Loss
- a multi-component loss that combines adversarial loss and a high receptive field perceptual loss
```python
"""l1 loss on src pixels, and downscaled predictions if on_pred=True"""
loss = torch.mean(torch.abs(pred[mask<1e-8] - image[mask<1e-8]))
if on_pred: 
    loss += torch.mean(torch.abs(pred_downscaled[mask_downscaled>=1e-8] - ref[mask_downscaled>=1e-8]))  
```
## High Receptive Field Perceptual Loss (HRF PL)
- Naive supervised losses require the generator to reconstruct the ground truth precisely. However, the visible parts of the image often do not contain enough information for the exact reconstruction of the masked part. Therefore, using naive supervision leads to blurry results due to the averaging of multiple plausible modes of the inpainted content.