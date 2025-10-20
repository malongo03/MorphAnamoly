torchrun --nproc_per_node=2 morphFM/train/train.py \
--config-file morphFM/configs/train/have_data_separate.yaml \
--output-dir ours_add_data_separate \
train.dataset_path=NeuronMorpho:split=TRAIN:root=sample_predata:extra=sample_predata

