<center><h1>RIPS 2018 Readings</h1></center>

## Table of Contents
* General Machine Learning and Artificial Intelligence
* Knowledge Graphs and Computational Fact-Checking
* Natural Language Processing
  * General
  * word2vec and doc2vec
* Parsing to Logical Form
  * First-Order Logic
  * Subject-Predicate-Object Triples
* Relational Learning
  * General
  * Managing Relational Data
    * Resource Description Framework
    * SPARQL Protocol and RDF Query Language

<center><h2>General Machine Learning and Artificial Intelligence</h2></center>

##### [Andrej Karpathy Blog](http://karpathy.github.io/)

* [Hacker's guide to Neural Networks](http://karpathy.github.io/neuralnets/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

##### [i am trask](https://iamtrask.github.io/)

I recommend starting here for basic Neural Network concepts.

* [A Neural Network in 11 lines of Python (Part 1)](https://iamtrask.github.io/2015/07/12/basic-python-network/)
* [A Neural Network in 13 lines of Python (Part 2 - Gradient Descent)](http://iamtrask.github.io/2015/07/27/python-network-part2/)
* [Anyone Can Learn To Code an LSTM-RNN in Python (Part 1: RNN)](http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)

##### [WildML](http://www.wildml.com/)

* [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
* [Recurrent Neural Networks Tutorial, Part 2 – Implementing a RNN with Python, Numpy and Theano](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
* [Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
* [Recurrent Neural Networks Tutorial, Part 4 – Implementing a GRU/LSTM RNN with Python and Theano](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

##### Other Sources
* [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning#python-nlp)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

<center><h2>Knowledge Graphs and Computational Fact-Checking</h2></center>

Knowledge Graphs are an increasingly popular data structure for representing relational information. They assume a knowledge base in the form of relational triples
(subject, predicate, object) and model these triples using a graph an ordered pair G=(V,E) where V is a set of concept nodes and E is a set of predicate edges.

* [Automated Fact-Checking presentation by Joshua Chen](http://joshchen.io/Computational%20Fact-Checking/Automated%20fact-checking%20-%20Jul%2025.pdf)

##### Papers

* [Computational Fact Checking from Knowledge Networks](pdf/Computational_Fact_Checking_from_Knowledge_Networks.pdf)
* [Computational Fact Checking through Query Perturbations](pdf/Computational_Fact_Checking_through_Query_Perturbations.pdf)
* [Discriminative Predicate Path Mining for Fact Checking in Knowledge Graphs](pdf/Discriminative_Predicate_Path_Mining_for_fact_checking_in_knowledge_graphs.pdf)
* [Open Information Extraction: The Second Generation](pdf/Open_Information_Extraction_The_Second_Generation.pdf)
* [A Review of Relational Machine Learning for Knowledge Graphs](pdf/A_Review_of_Relational_Machine_Learning_for_Knowledge_Graphs.pdf)
* [Towards Computational Fact-Checking](pdf/Towards_Computational_Fact-Checking.pdf)

<center><h2>Natural Language Processing</h2></center>

Natural language processing (NLP) is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. I recommend starting with "Text Mining: the State of the Art and the Challenges" for an overview of text mining.

#### General

##### Articles, Blogposts, and Tutorials

* [edX Course on Natural Language Processing](https://courses.edx.org/courses/course-v1:Microsoft+DEV288x+1T2018/course/)
* [Oxford Deep NLP 2017 Course](https://github.com/oxford-cs-deepnlp-2017/lectures)
* [Regular Expressions 101](https://regex101.com/)

##### Papers

* [Evolving Better Stoplists for Document Clustering and Web Intelligence](https://pdfs.semanticscholar.org/c53f/17e9ae7ff1ba13aba902739f4df85054cb0a.pdf)
* [On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter](http://oro.open.ac.uk/40666/1/292_Paper.pdf)
* [Preprocessing Techniques for Text Mining - An Overview](https://pdfs.semanticscholar.org/1fa1/1c4de09b86a05062127c68a7662e3ba53251.pdf)
* [Retrieval Effectiveness on the Web](https://www.sciencedirect.com/science/article/pii/S030645730000039X)
* [Risk Information Extraction and Aggregation](pdf/Risk_Information_Extraction_and_Aggregation.pdf)
* [Text Mining: The State of the Art and the Challenges](http://www.ntu.edu.sg/home/asahtan/papers/tm_pakdd99.pdf)

#### word2vec and doc2vec

*Efficient Estimation of Word Representations in Vector Space* and *Distributed Representations of Words and Phrases and their Compositionality* started it all. Here are the links for documentation on [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) and [doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html).

##### Articles, Blogposts, and Tutorials

* [A Gentle Introduction to Doc2Vec](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)
* [Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec)
* [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
* [Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

##### Papers

* [An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation](pdf/An_Empirical_Evaluation_of_doc2vec_with_Practical_Insights_into_Document_Embedding_Generation.pdf)
* [Distributed Representations of Sentences and Documents](pdf/Distributed_Representations_of_Sentences_and_Documents.pdf)
* [Distributed Representations of Words and Phrases and their Compositionality](pdf/Distributed_Representations_of_Words_and_Phrases_and_their_Compositionality.pdf)
* [Efficient Estimation of Word Representations in Vector Space](pdf/Efficient_Estimation_of_Word_Representations_in_Vector_Space.pdf)
* [Neural Network Doc2vec in Automated Sentiment Analysis for Short Informal Texts](pdf/Neural_Network_Doc2vec_in_Automated_Sentiment_Analysis_for_Short_Informal_Texts.pdf)

<center><h2>Parsing to Logical Form</h2></center>

The problem of learning to parse sentences to lambda-calculus representations or Subject-Predicate-Object triples of their underlying semantics.

#### First-Order Logic

##### Papers

* [Online Learning of Relaxed CCG Grammars for Parsing to Logical Form](pdf/Online_Learning_of_Relaxed_CCG_Grammars_for_Parsing_to_Logical_Form.pdf)

#### Subject-Predicate-Object Triples

##### Articles, Blogposts, and Tutorials

* [Reverb](http://reverb.cs.washington.edu/)

##### Papers

* [From Information to Knowledge Harvesting Entities and Relationships from Web Sources](pdf/From_Information_to_Knowledge_Harvesting_Entities_and_Relationships_from_Web_Sources.pdf)


<center><h2>Relational Learning</h2></center>

Statistical relational learning (SRL) is a subdiscipline of artificial intelligence and machine learning that is concerned with domain models that exhibit both uncertainty (which can be dealt with using statistical methods) and complex, relational structure. Note that SRL is sometimes called Relational Machine Learning (RML) in the literature. Typically, the knowledge representation formalisms developed in SRL use (a subset of) first-order logic to describe relational properties of a domain in a general manner (universal quantification) and draw upon probabilistic graphical models (such as Bayesian networks or Markov networks) to model the uncertainty; some also build upon the methods of inductive logic programming.

#### General

##### Papers

* [Community Detection in Graphs](pdf/Community_Detection_in_Graphs.pdf)
* [Philosophers are Mortal: Inferring the Truth of Unseen Facts](pdf/Philosophers_are_Mortal_Inferring_the_Truth_of_Unseen_Facts.pdf)
* [A Review of Relational Machine Learning for Knowledge Graphs](pdf/A_Review_of_Relational_Machine_Learning_for_Knowledge_Graphs.pdf)

#### Managing Relational Data

###### Resource Description Framework

* [Getting Started with RDFLib](http://rdflib.readthedocs.io/en/stable/gettingstarted.html)
* [RDF 1.1 Concepts and Abstract Syntax](https://www.w3.org/TR/rdf11-concepts/)
* [Resource Description Framework](https://en.wikipedia.org/wiki/Resource_Description_Framework)

###### SPARQL Protocol and RDF Query Language

* [Querying with SPARQL](http://rdflib.readthedocs.io/en/stable/intro_to_sparql.html)
* [SPARQL Endpoint Interface to Python](https://rdflib.github.io/sparqlwrapper/)
* [SPARQL Protocol and RDF Query Language](https://en.wikipedia.org/wiki/SPARQL)
* [SPARQL Wrapper](https://github.com/RDFLib/sparqlwrapper)
