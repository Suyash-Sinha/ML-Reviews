#To run the program to train the models:
To run our program, place the dataset in the project directory as a text file
named "data.txt" and run the program. It should train the three models/classifier(Naives Bayes, Base Decision tree, Best Decision Tree) and produce a prediction in an output files for each model. The output files also analyses the different classifier. Each output file
should contain:
1) The index of the row of each test data instance from the original data set and the predicted value for that instance.
2) The confusion matrix
3) Precision score
4) Recall score
5) f1-measure
6) Accuracy score

#Testing new dataset on the model in console
1) Add the new dataset as text file to the project directory.
2) After the program has trained the model we can import the dataset from the console using the following command:\
docx,labelx=read_documents("filename.txt")
3) Reformat the data using the following command:\
   docx_tfidf=vectorizer.transform(docx)
4) Create a prediction on the reformatted data with all three classifiers using the following commands:\
predicted_docx_nb=clf_naive.predict(docx_tfidf)\
predicted_docx_base_dt=clf_base_dt.predict(docx_tfidf)\
predicted_docx_best_dt=clf_best_dt.predict(docx_tfidf)
5) Then print the  analysis of the prediction with the following command:\
   print("\nNaives Bayes:"+to_string(labelx,predicted_docx_nb)+"\n\nBase DT:"+to_string(labelx,predicted_docx_base_dt)+"\n\nBest DT:"+to_string(labelx, predicted_docx_best_dt))