for((i=1;i<100;i++))
do
	rm samples/* -f
	nohup   ./loops.sh  &
        nohup   ./loops.sh  &
	nohup   ./loops.sh  &
	nohup   ./loops.sh  &
	nohup   ./loops.sh  &
	./loops.sh 

	python f.py
	cp m m0.$i -r
done



