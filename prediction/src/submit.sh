#print("Tg S2",txt2R2('./Tg_top_1_nor_Tg_S2.txt'))

!/bin/sh
#PBS -N representation
#PBS -q batch
#PBS -l nodes=1:ppn=2:gpus=1

cd $PBS_O_WORKDIR

NPROCS=`wc -l < $PBS_NODEFILE`

cd /home/Gabriella/polymagic/src

source ~/.bashrc
conda activate tensorflow-gpu

export CUDA_VISIBLE_DEVICES=0

n_gpu='2'
repre='S2'
ext='False'
# node_n='32'
# train='true'
ppty='exp'
python3 rnn_exp.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset Dc_dataset.csv --trainset Dc_genmix1 --trainset_n 1 --exten $ext  --train true --LR 1e-5 --top false
# python3 rnn_exp.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset Dc_dataset.csv --trainset Dc --trainset_n 1 --exten $ext  --train true --LR 1e-4 --top false

# repre='S0'
# python3 rnn_exp.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset Dc_dataset.csv --trainset Dc --trainset_n 1 --exten $ext  --train true --LR 1e-4 --top false




# for l in {2..4..2}
# do
# l = 4
# for i in {120..380..20}
# do
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split $i --name $l
# done
# done







# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split 300 --name 4
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split 20 --name 4
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split 320 --name 4
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split 240 --name 4
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split 340 --name 4
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split 260 --name 4
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split 360 --name 4
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split 280 --name 4
#
# for l in {2..4..2}
# do
# for i in {120..400..20}
# do
# name=4
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split $i --name $name
# done
# done
# for i in {120..280..20}
# do
# name=2
# python3 rnn_exp_split.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4 --split $i --name $name
# done
#







# done
# done
#python3 rnn_exp.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 2 --exten $ext  --train true --LR 1e-3
#python3 rnn_exp.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 3 --exten $ext  --train true --LR 1e-3
#python3 rnn_exp.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 4 --exten $ext  --train true --LR 1e-3
#python3 rnn_exp.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 5 --exten $ext  --train true --LR 1e-3
# ppty='Solu'
# python3 rnn4_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4
# ppty='Density'
# python3 rnn4_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4
#ppty='Tg'
#python3 rnn_Tg.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-3 >> tmp
#python3 rnn_Tg.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 2 --exten $ext  --train true --LR 1e-3 >> tmp
#python3 rnn_Tg.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 3 --exten $ext  --train true --LR 1e-3 >> tmp
#python3 rnn_Tg.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 4 --exten $ext  --train true --LR 1e-3 >> tmp
#python3 rnn_Tg.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 5 --exten $ext  --train true --LR 1e-3 >> tmp

# ppty='Tc'
# python3 rnn4_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 5 --exten $ext  --train true --LR 1e-4

# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 2 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 3 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 4 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 5 --exten $ext  --train true --LR 1e-4

# node_n='16'
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 2 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 3 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 4 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 5 --exten $ext  --train true --LR 1e-4
#
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 2 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 3 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 4 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 5 --exten $ext  --train true --LR 1e-4
#
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 1 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 2 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 3 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 4 --exten $ext  --train true --LR 1e-4
# python3 rnn_q.py --GPU $n_gpu --repre $repre --ppty $ppty --dataset data4DL.csv --trainset k-fold --trainset_n 5 --exten $ext  --train true --LR 1e-4
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
exit 0
