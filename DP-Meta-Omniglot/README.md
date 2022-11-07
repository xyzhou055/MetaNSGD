# Meta-NSGD

---

## Data preparation

Download the Omniglot dataset from [link1](https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip), [link2](https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip). 
Create a folder `omniglot` under the same directory of code files; unzip and merge the downloaded files into the `omniglot` folder shown as follows:

```
./train_omniglot(train_omniglot_cluster).py
...
./omniglot/Alphabet_of_the_Magi/
./omniglot/Angelic/
./omniglot/Anglo-Saxon_Futhorc/
...
./omniglot/ULOG/
```
---

## Training

For single cluster training, run with the following command:
```bash
python train_omniglot.py log/o55 --classes 5 --noise-multiplier 0.423 --meta-batch 25 --validate-every 5000 --shots 5 --train-shots 10 --meta-iterations 20000 --iterations 10 --test-iterations 50 --batch 10 --meta-lr 0.1 --lr 0.001
```

For multi cluster training, run with the following command:

```bash
python train_omniglot_cluster.py log/o55 --classes 5 --noise-multiplier 0.423 --meta-batch 25 --q 2 --validate-every 5000 --shots 5 --train-shots 10 --meta-iterations 20000 --iterations 5 --test-iterations 50 --batch 10 --meta-lr 0.1 --lr 0.001
```
