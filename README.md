# deep-learning-project-ell-881
Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning


Created by Ankur Sharma - 2015CS50278

This is the README file for the project.
1. Time taken to run 1 epoch on CPU v/s Time taken to run 1 epoch on GPU
	Number of examples:
		Task 1 : Training => 942,854 pairs (snli_train + multnli_train), Validation => 20,000 pairs (snli_dev + multnli_dev)
		Task 2 : Training => 250,000 pairs (as parsed by Anshul Mittal), Validation => 2,000 pairs
		Task 3 : Training => 155,000 pairs (training-parallel-europarl-v7), Validation => 2,000 pairs
	CPU 1 Epoch =>
		-> Task 1: ~7 Hours (actual)
		-> Task 2: ~18 Hours (actual)
		-> Task 3: ~15 Hours (estimate)
	GPU 1 Epoch =>
		-> Task 1: ~2 Hours (actual)
		-> Task 2: ~4 Hours (actual)
		-> Task 3: ~4 Hours (actual)

2. To speed up the training, methods adopted were as follows:
	=> While training locally, batch_size was increased to 128 so that more batches can be trained easily and faster.
	=> Very large sentences have been filtered in the dataset initally, only sentences with <= 100 words have been kept for training purposes.
	=> Models were trained on HPC with access to multiple GPUs:
		-> To fit the dataset of size (batch_size x max_word_length x vocab_size) into the GPU allocated memory, batch_size was reduced. Note that all the sentences with words greater > 100 have already been clipped off.
		-> For Task 2: batch_size of 20 was used, and for Task 3: batch_size of 10 was used. (after a number of experiments).


3. Scores for the evaluation tasks: 'bag_of_words_results.txt' , 'google_encoder_results.txt' and 'my_encoder_evaluation_results.txt'
	Please find the file 'evaluation_tasks.txt' which contains the evalutations scores for all the tasks whose data is available. Evaluation script has been coded by me from the scratch using the scripts available under the 'examples' folder.

	Comparison between Bag of Words v/s Google Encoder v/s My Encoder =>
	
	#-----------------------------------------------------------------------------------------------------------------------#
	Task MR:
		Bag of Words -> Dev acc : 58.1 Test acc : 58.11
		My Encoder -> Dev acc : 62.16 Test acc : 54.08
		Google Universal Sentence Encoder -> Dev acc : 74.75 Test acc : 74.61

	#-----------------------------------------------------------------------------------------------------------------------#
	Task CR:
		Bag of Words -> Dev acc : 63.76 Test acc : 63.76
		My Encoder -> Dev acc : 69.03 Test acc : 67.76
		Google Universal Sentence Encoder -> Dev acc : 81.11 Test acc : 81.19
	
	#-----------------------------------------------------------------------------------------------------------------------#
	Task SUBJ:
		Bag of Words -> Dev acc : 99.6 Test acc : 99.6
		My Encoder -> Dev acc : 99.6 Test acc : 99.58
		Google Universal Sentence Encoder -> Dev acc : 92.27 Test acc : 92.4

	#-----------------------------------------------------------------------------------------------------------------------#
	Task MPQA:
		Bag of Words -> Dev acc : 68.77 Test acc : 68.77
		My Encoder -> Dev acc : 73.47 Test acc : 73.88
		Google Universal Sentence Encoder -> Dev acc : 84.13 Test acc : 84.14

	#-----------------------------------------------------------------------------------------------------------------------#
	Task SST Binary classification:
		Bag of Words -> Dev acc : 50.92 Test acc : 49.92
		My Encoder -> Dev acc : 62.73 Test acc : 61.01
		Google Universal Sentence Encoder -> Dev acc : 80.5 Test acc : 79.79

	#-----------------------------------------------------------------------------------------------------------------------#
	Task SST Fine-Grained classification:
		Bag of Words -> Dev acc : 26.25 Test acc : 28.64
		My Encoder -> Dev acc : 30.52 Test acc : 29.37
		Google Universal Sentence Encoder -> Dev acc : 43.32 Test acc : 41.67

	#-----------------------------------------------------------------------------------------------------------------------#
	Task MRPC:
		Bag of Words -> Dev acc : 67.54 Test acc 66.49
		My Encoder -> Dev acc : 68.87 Test acc 67.77
		Google Universal Sentence Encoder -> Dev acc : 70.51 Test acc 70.72

	#-----------------------------------------------------------------------------------------------------------------------#
	Task SICK-Entailment:
		Bag of Words -> Dev acc : 56.4 Test acc : 56.69
		My Encoder -> Dev acc : 76.2 Test acc : 74.08
		Google Universal Sentence Encoder -> Dev acc : 80.8 Test acc : 80.35

	#-----------------------------------------------------------------------------------------------------------------------#
	Task SICK-Relatedness:
		Bag of Words -> Test : Pearson 0 Spearman 0 MSE 1.0178068645228793
		My Encoder -> Test : Pearson 0.6795868092606345 Spearman 0.6388651528486534 MSE 0.5476471141073296
		Google Universal Sentence Encoder -> -

	#-----------------------------------------------------------------------------------------------------------------------#
	Task STSBenchmark:
		Bag of Words -> Test : Pearson 0 Spearman 0 MSE 2.4952048630823445
		My Encoder -> Test : Pearson 0.46197351809197945 Spearman 0.46203683488127034 MSE 1.9680122855126314
		Google Universal Sentence Encoder -> -

	#-----------------------------------------------------------------------------------------------------------------------#
	Task STS12:
		Bag of Words -> -
		My Encoder -> ALL (average) : Pearson = 0.2604,             Spearman = 0.3180
		Google Universal Sentence Encoder -> ALL (average) : Pearson = 0.5904,             Spearman = 0.6077

	#-----------------------------------------------------------------------------------------------------------------------#
	Task STS13:
		Bag of Words -> -
		My Encoder -> ALL (average) : Pearson = 0.1567,             Spearman = 0.1779
		Google Universal Sentence Encoder -> ALL (average) : Pearson = 0.5941,             Spearman = 0.6005

	#-----------------------------------------------------------------------------------------------------------------------#
	Task STS14:
		Bag of Words -> -
		My Encoder -> ALL (average) : Pearson = 0.2347,             Spearman = 0.2590
		Google Universal Sentence Encoder -> ALL (average) : Pearson = 0.6780,             Spearman = 0.6444

	#-----------------------------------------------------------------------------------------------------------------------#
	Task STS15:
		Bag of Words -> -
		My Encoder -> ALL (average) : Pearson = 0.3520,             Spearman = 0.3605
		Google Universal Sentence Encoder -> ALL (average) : Pearson = 0.7245,             Spearman = 0.7455

	#-----------------------------------------------------------------------------------------------------------------------#
	Task STS16:
		Bag of Words -> -
		My Encoder -> ALL (average) : Pearson = 0.3243,             Spearman = 0.3437
		Google Universal Sentence Encoder -> ALL (average) : Pearson = 0.7021,             Spearman = 0.7297
	#-----------------------------------------------------------------------------------------------------------------------#
	

4. Nearest Neighbour words:
	Using the cosine similarity measure privided in sklearn, I have created the method 'cosine_knn' which returns the closest 'k' neighbouts using this similarity measure in a brute force manner. This can be further improved using KD Tree for faster query answering.
	We can check the file 'closest_neighbours.txt' to see the words, and the nearest 5 words.

5. Training Logs:
	'training_log.txt' has been created which containts logs for the training logs for the final task - NMT. I read the 'logging' part after I've completed training the encoder for NLI, and CP tasks. So, I put the training logs for the NMT part which I ran on HPC.

6. Models:
	Trained Models for each task can be loaded and easily used for evaluation. One can find my trained models in the folders 'encoder_nli', 'encoder_cp' and 'encoder_nmt' folders.

7. Source Vocab File :- 'english_vocab.txt' and 'german_vocab.txt'
