
# Sentimental analysis with fine tuning of a pre made model distilbert

The last two years have seen the rise of Transfer Learning approaches in Natural Language Processing (NLP) with large-scale pre-trained language models becoming a basic tool in many NLP tasks [Devlin et al., 2018].

This project is dedicated to developing a sentiment analysis model for Twitter messages, focusing on distinguishing between negative and positive sentiments. Leveraging the powerful DistilBERT model, pre-trained for natural language understanding, our goal is to achieve robust performance metrics. 

The emphasis is on showcasing the capability to create sophisticated models without the need for extensive computational resources typically associated with training models from scratch.

## Project structure

There are 2 jupyter notebook on this project, the first one  encompasses all the data cleaning steps, exploratory data analysis (EDA), the trainning of the model and the second one is for the inference.

The model jupyter notebook have all the data cleanning steps and the eda analysis the steps with the the model trainning, the model was trained using gpu with the assistance of pytorch library.

The name of the inference is Inference code.ipynb and the trainning model is Sentiment analysis.ipynb.

The trained model and the vocabulary generated are allocated in ./models/ with the name models.pt and vocab_distilbert_twitter.bin respectively.


## Data

The training data employed in this project was obtained by a team of Stanford students dedicated to constructing a Twitter sentiment classification model. The dataset was gathered using the Twitter API, allowing for a diverse and representative sample of tweets. The labeling process involved filtering tweets based on emoticons, with smiley faces denoting positive sentiment labels and sad faces indicating negative sentiment labels. For a more detailed understanding of the dataset and the labeling approach, we recommend consulting the details provided in the referenced research paper that can be found [here](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf).

### Data quantity and split 

In terms of data quantity, approximately 30% of the entire dataset, totaling 467,269 instances, was utilized for training purposes. This selective approach to data usage was  because of the limitations posed by the capacity of my personal computer. In the interest of resource efficiency, I opted to work with a subset of the data.

### Trainning and valid split

Within the chosen subset, 80% of the data was allocated for training the sentiment analysis model, while the remaining 20% was designated for evaluation in the test dataset. 

### Random data removal

The process of data reduction for the training subset involved a random selection mechanism. This approach was chosen to prevent bias and maintain a representative sample.

### Data processing 

For data preprocessing, the tweet messages underwent initial general data cleaning to address misspelled words and punctuation issues. Subsequently, misspelled words were manually corrected using the GloVe dictionary as a reference.

## Usage

To run the Jupyter notebooks you will need to download the dataset [here](https://www.kaggle.com/datasets/kazanova/sentiment140), and put on the same directory described on the notebook. You will have to download the glove dictionary too, it can be found [here](https://www.kaggle.com/datasets/authman/pickled-glove840b300d-for-10sec-loading). After that, you can run the the trainning model and get the output of the trained model. After that you can run the inference code.


## Model trainning

The model architecture comprises the DistilBERT structure integrated with additional layers designed to yield probability-based outputs. Subsequently, the model undergoes fine-tuning of its surface layers using PyTorch.

The loss function chosen for this project is nn.CrossEntropyLoss, and the optimization algorithm employed is the Adam optimizer.

The hyperparameters choosed for this model are:

Epochs : 1

Train batch size = 4

Valid batch size = 2

Learning rate = 1e-05

## Evaluation

In terms of performance evaluation for this classification problem, accuracy and the ROC curve serve as key metrics. The baseline comparison is made against the DistilBERT model without fine-tuning.
## Results

Overall, the model demonstrated good performance, achieving an accuracy of 85.21% and an AUC of 0.93 on valid data. These results significantly outperformed the baseline model, which recorded an accuracy of 50.44% and AUC of 0.56.

Bellow  are the imagens of roc curve of both model and baseline model.

![baseline ROC curve](images/ROCcurve2.png)

![Model ROC curve](images/ROCcurve1.png)

In terms of inference, the response time is relatively short, mitigating any concerns related to processing time.

In general the project does what was expected, which is evaluate sentiment analysis with a good accuracy, and most important, it shows the usage of pre-trained models and the possibility to reach high levels of accuracy using them.

## Future considerations

While the current project has implemented a random removal strategy for creating the training subset, future iterations may explore more sophisticated techniques for data selection. These could include stratified sampling or other methodologies to further enhance the representativeness and generalization capabilities of the model. We can too, explore better hyper parameters of the model to achieve better results.


## Contact info

If you have any question related to this Project, you can send to my email:
ryosato111@gmail.com

## Refererences

 - [ Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009)](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)
 - [Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  Fri, 24 May 2019](https://github.com/matiassingers/awesome-readme)
 - [NLP with Disaster Tweets - EDA, Cleaning and BERT](https://www.kaggle.com/code/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert)

