#############################
# screen 
#############################
screen -ls
screen -S training
screen -D -r training
screen -S training -X quit
screen -S 365811 -X quit

#############################
# path 
#############################
conda activate py396
cd /home/hy/project/

nvidia-smi

rm -rf analysis configs dynamics gendata integrator learner  logs/ outputs runner_shell task utils/

#############################
# shell 
#############################

sh runner_shell/sp.sh
sh runner_shell/dp.sh
sh runner_shell/slider_crank.sh
sh runner_shell/twolink.sh
sh runner_shell/scissor_space_deployable.sh

