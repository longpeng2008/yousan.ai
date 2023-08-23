# pre_train T_net 

python3 train.py \
	--dataDir='./data' \
	--saveDir='./ckpt' \
	--trainData='human_matting_data' \
	--trainList='./data/train.txt' \
	--load='human_matting' \
	--nThreads=4 \
	--patch_size=320 \
	--train_batch=8 \
	--lr=1e-3 \
	--lrdecayType='keep' \
	--nEpochs=100 \
	--save_epoch=1 \
	--train_phase='pre_train_t_net'

# train end to end
python3 train.py \
	--dataDir='./data' \
	--saveDir='./ckpt' \
	--trainData='human_matting_data' \
	--trainList='./data/train.txt' \
	--load='human_matting' \
	--nThreads=4 \
	--patch_size=320 \
	--train_batch=8 \
	--lr=1e-4 \
	--lrdecayType='keep' \
	--nEpochs=200 \
	--save_epoch=1 \
	--finetuning \
	--train_phase='end_to_end'
