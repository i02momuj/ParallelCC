# ParallelCC: Speeding up Classifier Chains in Multi-Label Classification

Classifier Chains (CC) [[Rea11]](#Rea11) is one of the best-performing methods in multi-label classification.
CC is based on the idea of building a binary classifier for each of the labels but linked in such a way that each binary classifier includes the predictions of previous labels in the chain as extra input features.
Therefore, the original definition of CC made it inherently sequential and non-parallelizable, since each binary classifier needs the outputs of previous ones to be built.

In this project, we propose a modified definition of CC in order to make it parallelizable, called Parallel Classifier Chains (PCC). In PCC, *k* binary classifiers are built in parallel, each including as extra input features the predictions of all previous classifiers that have already finished.
As a result, each binary classifier will introduce less label information as input features, so the dependencies among labels might be modeled slightly worse. However, in high-dimensional label spaces, this difference is negligible, and also it will depend on how many classifiers are built in parallel.
PCC has demonstrated that it is able to significantly reduce the runtime of CC while the predictive performance is not statistically significantly harmed.

More information about this method can be find in the following article:
> Jose M. Moyano, Eva L. Gibaja, Sebastián Ventura, and Alberto Cano. "Speeding up Classifier Chains in Multi-Label Classification". Submitted to *4th International Conference on Internet of Things, Big Data and Security*. (2019).

<!---  If you use ParallelCC, please cite the paper. Further, a [bibtex citation file](https://github.com/i02momuj/ParallelCC) is also provided. -->

In this repository we provide the code of ParallelCC, distributed under the GPLv3 License. ParallelCC has been implemented using Mulan [[Tso11]](#Tso11), and Weka [[Hal09]](#Hal09) libraries. Besides, the latest release [(v 1.1)](https://github.com/i02momuj/ParallelCC/releases/tag/v1.1) provides the executable jar to execute ParallelCC (and other related algorithms) as well as the javadoc.

In order to directly execute the provided ParallelCC jar, the following command have to be executed in a console:
```sh
java -jar ParallelCC.jar [parameters]
```

The jar file needs several parameters to indicate the method to execute, the number of threads, the datasets, and so on. The different parameters are the following:
* With the ```-d``` parameter, we define the path of the file which includes the paths of the different datasets that we want to use in the experiment. This allows us to use many datasets or partitions with the same configuration. Each line of this file correspond to a different dataset, and it must include, in this order and separated by spaces: the path to the train file, path to the test file, and path to the *xml* file.
* With the ```-t``` parameter, we set the number of threads to execute in parallel. If 0, it executes over all available threads. Further, if it is not set, its default value is 0.
* With the ```-s``` parameter, we define the number of different seeds for random numbers to use in the experiments.
* With the ```-o``` parameter, we define the filename for the file storing the results.
* With the ```-a``` parameter, we choose the algorithm to execute. It includes many classic methods such as BR, CC, EBR and ECC, which are not executed in parallel (regardless of the value of -t parameter). Further, parallel methods such as PCC (Parallel Classifier Chains), PEBR (Parallel Ensemble of Binary Relevance) and EPCC (Ensemble of Parallel Classifier Chains) can be choosen.

Three multi-label datasets (*Emotions* [[Tso08]](#Tso08), *Yeast* [[Eli01]](#Eli01), and *Birds* [[Bri13]](#Bri13)) have been included in the repository as example; however, a wide variety of dataset are available at the [KDIS Research Group Repository](http://www.uco.es/kdis/mllresources/). Further, one example configuration file (*data.txt*) is also provided.

### References

<a name="Bri13"></a>**[Bri13]** F. Briggs et al. (2012). Acoustic classification of multiple simultaneous bird species: A multi-instance multi-label approach. *The Journal of the Acoustical Society of America*, 131(6), 4640-4650.

<a name="Eli01"></a>**[Eli01]** A. Elisseeff and J. Weston. (2002). A kernel method for multi-labelled classification. In *Advances in neural information processing systems*, 681-687.

<a name="Hal09"></a>**[Hal09]** M. Hall et al. (2009). The WEKA data mining software: an update. *ACM SIGKDD explorations newsletter*, 11(1), 10-18.

<a name="Rea11"></a>**[Rea11]** J. Read, B. Pfahringer, G. Holmes, and E. Frank. (2011). Classifier chains for multi-label classification. *Machine learning*, 85(3), 333.

<a name="Tso08"></a>**[Tso08]** G. Tsoumakas, I. Katakis, and I. Vlahavas. (2008). Effective and efficient multilabel classification in domains with large number of labels. In *Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD’08)*, Vol. 21, 53-59.

<a name="Tso11"></a>**[Tso11]** G. Tsoumakas, E. Spyromitros-Xioufis, J. Vilcek, and I. Vlahavas. (2011). Mulan: A java library for multi-label learning. *Journal of Machine Learning Research*, 12, 2411-2414.

