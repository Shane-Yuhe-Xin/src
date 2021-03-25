
import sklearn
def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    precision, recall, fscore = sklearn.metrics.precision_recall_fscore_support(y_true = labels, y_pred = preds, average='binary')
    accuray = sklearn.metrics.accuracy_score(y_true = labels, y_pred = preds)
    my_dict =  {"accuracy": accuray, "f1": fscore, "precision": precision,"recall scores": recall}
    return my_dict

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    from transformers import RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    return model
    
def my_function(eval_dict):
  return eval_dict['eval_loss']
