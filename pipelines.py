"""
Implement trainable ensemble: XGBoost, random forest, Linear Regression
"""

from models import CharVDCNN, WordLSTM, GloveLSTM, GloveSCNN, GloveDPCNN
from steps.base import Step, Dummy, stack_inputs, hstack_inputs, sparse_hstack_inputs, to_tuple_inputs
from steps.keras.loaders import Tokenizer
from steps.keras.models import GloveEmbeddingsMatrix
from steps.postprocessing import PredictionAverage
from steps.preprocessing import XYSplit, TextCleaner, TfidfVectorizer
from steps.sklearn.models import LogisticRegressionMultilabel


def train_preprocessing(config):
    xy_train = Step(name='xy_train',
                    transformer=XYSplit(**config.xy_splitter),
                    input_data=['input'],
                    adapter={'meta': ([('input', 'meta')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    text_cleaner_train = Step(name='text_cleaner_train',
                              transformer=TextCleaner(**config.text_cleaner),
                              input_steps=[xy_train],
                              adapter={'X': ([('xy_train', 'X')])},
                              cache_dirpath=config.env.cache_dirpath)

    xy_valid = Step(name='xy_valid',
                    transformer=XYSplit(**config.xy_splitter),
                    input_data=['input'],
                    adapter={'meta': ([('input', 'meta_valid')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    text_cleaner_valid = Step(name='text_cleaner_valid',
                              transformer=TextCleaner(**config.text_cleaner),
                              input_steps=[xy_valid],
                              adapter={'X': ([('xy_valid', 'X')])},
                              cache_dirpath=config.env.cache_dirpath)

    cleaning_output = Step(name='cleaning_output',
                           transformer=Dummy(),
                           input_data=['input'],
                           input_steps=[xy_train, text_cleaner_train, xy_valid, text_cleaner_valid],
                           adapter={'X': ([('text_cleaner_train', 'X')]),
                                    'y': ([('xy_train', 'y')]),
                                    'train_mode': ([('input', 'train_mode')]),
                                    'X_valid': ([('text_cleaner_valid', 'X')]),
                                    'y_valid': ([('xy_valid', 'y')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath)
    return cleaning_output


def inference_preprocessing(config):
    xy_train = Step(name='xy_train',
                    transformer=XYSplit(**config.xy_splitter),
                    input_data=['input'],
                    adapter={'meta': ([('input', 'meta')]),
                             'train_mode': ([('input', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    text_cleaner_train = Step(name='text_cleaner_train',
                              transformer=TextCleaner(**config.text_cleaner),
                              input_steps=[xy_train],
                              adapter={'X': ([('xy_train', 'X')])},
                              cache_dirpath=config.env.cache_dirpath)

    cleaning_output = Step(name='cleaning_output',
                           transformer=Dummy(),
                           input_data=['input'],
                           input_steps=[xy_train, text_cleaner_train],
                           adapter={'X': ([('text_cleaner_train', 'X')]),
                                    'y': ([('xy_train', 'y')]),
                                    'train_mode': ([('input', 'train_mode')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath)
    return cleaning_output


def char_vdcnn_train(config):
    preprocessed_input = train_preprocessing(config)
    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'X_valid': ([('cleaning_output', 'X_valid')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    network = Step(name='char_vdcnn',
                   transformer=CharVDCNN(**config.char_vdcnn_network),
                   overwrite_transformer=True,
                   input_steps=[char_tokenizer, preprocessed_input],
                   adapter={'X': ([('char_tokenizer', 'X')]),
                            'y': ([('cleaning_output', 'y')]),
                            'validation_data': (
                                [('char_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                            },
                   cache_dirpath=config.env.cache_dirpath)
    char_output = Step(name='char_output',
                       transformer=Dummy(),
                       input_steps=[network],
                       adapter={'y_pred': ([('char_vdcnn', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return char_output


def char_vdcnn_inference(config):
    preprocessed_input = inference_preprocessing(config)
    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    network = Step(name='char_vdcnn',
                   transformer=CharVDCNN(**config.char_vdcnn_network),
                   input_steps=[char_tokenizer, preprocessed_input],
                   adapter={'X': ([('char_tokenizer', 'X')]),
                            'y': ([('cleaning_output', 'y')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)
    char_output = Step(name='char_output',
                       transformer=Dummy(),
                       input_steps=[network],
                       adapter={'y_pred': ([('char_vdcnn', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return char_output


def word_lstm_train(config):
    preprocessed_input = train_preprocessing(config)
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'X_valid': ([('cleaning_output', 'X_valid')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cacyhe_dirpath)

    word_lstm = Step(name='word_lstm',
                     transformer=WordLSTM(**config.word_lstm_network),
                     overwrite_transformer=True,
                     input_steps=[word_tokenizer, preprocessed_input],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('cleaning_output', 'y')]),
                              'validation_data': (
                                  [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    word_output = Step(name='word_output',
                       transformer=Dummy(),
                       input_steps=[word_lstm],
                       adapter={'y_pred': ([('word_lstm', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return word_output


def word_lstm_inference(config):
    preprocessed_input = inference_preprocessing(config)
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    word_lstm = Step(name='word_lstm',
                     transformer=WordLSTM(**config.word_lstm_network),
                     input_steps=[word_tokenizer, preprocessed_input],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('cleaning_output', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath)
    word_output = Step(name='word_output',
                       transformer=Dummy(),
                       input_steps=[word_lstm],
                       adapter={'y_pred': ([('word_lstm', 'prediction_probability')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    return word_output


def glove_preprocessing_train(config, preprocessed_input):
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')]),
                                   'X_valid': ([('cleaning_output', 'X_valid')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    return word_tokenizer, glove_embeddings


def glove_preprocessing_inference(config, preprocessed_input):
    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[preprocessed_input],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)
    return word_tokenizer, glove_embeddings


def glove_lstm_train(config):
    preprocessed_input = train_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_train(config, preprocessed_input)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.glove_lstm_network),
                      overwrite_transformer=True,
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               'validation_data': (
                                   [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_lstm],
                        adapter={'y_pred': ([('glove_lstm', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_lstm_inference(config):
    preprocessed_input = inference_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_inference(config, preprocessed_input)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.glove_lstm_network),
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_lstm],
                        adapter={'y_pred': ([('glove_lstm', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_scnn_train(config):
    preprocessed_input = train_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_train(config, preprocessed_input)
    glove_scnn = Step(name='glove_scnn',
                      transformer=GloveSCNN(**config.glove_scnn_network),
                      overwrite_transformer=True,
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               'validation_data': (
                                   [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_scnn],
                        adapter={'y_pred': ([('glove_scnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_scnn_inference(config):
    preprocessed_input = inference_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_inference(config, preprocessed_input)
    glove_scnn = Step(name='glove_scnn',
                      transformer=GloveSCNN(**config.glove_scnn_network),
                      input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_scnn],
                        adapter={'y_pred': ([('glove_scnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_dpcnn_train(config):
    preprocessed_input = train_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_train(config, preprocessed_input)
    glove_dpcnn = Step(name='glove_dpcnn',
                       transformer=GloveDPCNN(**config.glove_dpcnn_network),
                       overwrite_transformer=True,
                       input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                       adapter={'X': ([('word_tokenizer', 'X')]),
                                'y': ([('cleaning_output', 'y')]),
                                'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                'validation_data': (
                                    [('word_tokenizer', 'X_valid'), ('cleaning_output', 'y_valid')], to_tuple_inputs),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_dpcnn],
                        adapter={'y_pred': ([('glove_dpcnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def glove_dpcnn_inference(config):
    preprocessed_input = inference_preprocessing(config)
    word_tokenizer, glove_embeddings = glove_preprocessing_inference(config, preprocessed_input)
    glove_dpcnn = Step(name='glove_dpcnn',
                       transformer=GloveDPCNN(**config.glove_dpcnn_network),
                       input_steps=[word_tokenizer, preprocessed_input, glove_embeddings],
                       adapter={'X': ([('word_tokenizer', 'X')]),
                                'y': ([('cleaning_output', 'y')]),
                                'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                },
                       cache_dirpath=config.env.cache_dirpath)
    glove_output = Step(name='output_glove',
                        transformer=Dummy(),
                        input_steps=[glove_dpcnn],
                        adapter={'y_pred': ([('glove_dpcnn', 'prediction_probability')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath)
    return glove_output


def tfidf_log_reg(preprocessed_input, config):
    tfidf_char_vectorizer = Step(name='tfidf_char_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_char_vectorizer),
                                 input_steps=[preprocessed_input],
                                 adapter={'text': ([('cleaning_output', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    tfidf_word_vectorizer = Step(name='tfidf_word_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_word_vectorizer),
                                 input_steps=[preprocessed_input],
                                 adapter={'text': ([('cleaning_output', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    log_reg_multi = Step(name='log_reg_multi',
                         transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                         input_steps=[preprocessed_input, tfidf_char_vectorizer, tfidf_word_vectorizer],
                         adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                         ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                  'y': ([('cleaning_output', 'y')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)
    log_reg_output = Step(name='log_reg_output',
                          transformer=Dummy(),
                          input_steps=[log_reg_multi],
                          adapter={'y_pred': ([('log_reg_multi', 'prediction_probability')]),
                                   },
                          cache_dirpath=config.env.cache_dirpath)
    return log_reg_output


def tfidf_logreg_train(config):
    preprocessed_input = train_preprocessing(config)
    logreg_output = tfidf_log_reg(preprocessed_input, config)
    return logreg_output


def tfidf_logreg_inference(config):
    preprocessed_input = inference_preprocessing(config)
    logreg_output = tfidf_log_reg(preprocessed_input, config)
    return logreg_output


def ensemble_extraction(config):
    xy_train = Step(name='xy_train',
                    transformer=XYSplit(**config.xy_splitter),
                    input_data=['input_ensemble'],
                    adapter={'meta': ([('input_ensemble', 'meta')]),
                             'train_mode': ([('input_ensemble', 'train_mode')])
                             },
                    cache_dirpath=config.env.cache_dirpath)

    text_cleaner_train = Step(name='text_cleaner_train',
                              transformer=TextCleaner(**config.text_cleaner),
                              input_steps=[xy_train],
                              adapter={'X': ([('xy_train', 'X')])},
                              cache_dirpath=config.env.cache_dirpath)

    cleaning_output = Step(name='cleaning_output',
                           transformer=Dummy(),
                           input_data=['input_ensemble'],
                           input_steps=[xy_train, text_cleaner_train],
                           adapter={'X': ([('text_cleaner_train', 'X')]),
                                    'y': ([('xy_train', 'y')]),
                                    'train_mode': ([('input', 'train_mode')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath)

    char_tokenizer = Step(name='char_tokenizer',
                          transformer=Tokenizer(**config.char_tokenizer),
                          input_steps=[cleaning_output],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    word_tokenizer = Step(name='word_tokenizer',
                          transformer=Tokenizer(**config.word_tokenizer),
                          input_steps=[cleaning_output],
                          adapter={'X': ([('cleaning_output', 'X')]),
                                   'train_mode': ([('cleaning_output', 'train_mode')])
                                   },
                          cache_dirpath=config.env.cache_dirpath)

    tfidf_char_vectorizer = Step(name='tfidf_char_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_char_vectorizer),
                                 input_steps=[cleaning_output],
                                 adapter={'text': ([('cleaning_output', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)
    tfidf_word_vectorizer = Step(name='tfidf_word_vectorizer',
                                 transformer=TfidfVectorizer(**config.tfidf_word_vectorizer),
                                 input_steps=[cleaning_output],
                                 adapter={'text': ([('cleaning_output', 'X')]),
                                          },
                                 cache_dirpath=config.env.cache_dirpath)

    glove_embeddings = Step(name='glove_embeddings',
                            transformer=GloveEmbeddingsMatrix(**config.glove_embeddings),
                            input_steps=[word_tokenizer],
                            adapter={'tokenizer': ([('word_tokenizer', 'tokenizer')]),
                                     },
                            cache_dirpath=config.env.cache_dirpath)

    log_reg_multi = Step(name='log_reg_multi',
                         transformer=LogisticRegressionMultilabel(**config.logistic_regression_multilabel),
                         input_steps=[cleaning_output, tfidf_char_vectorizer, tfidf_word_vectorizer],
                         adapter={'X': ([('tfidf_char_vectorizer', 'features'),
                                         ('tfidf_word_vectorizer', 'features')], sparse_hstack_inputs),
                                  'y': ([('cleaning_output', 'y')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         cache_output=True)

    char_vdcnn = Step(name='char_vdcnn',
                      transformer=CharVDCNN(**config.char_vdcnn_network),
                      input_steps=[char_tokenizer, cleaning_output],
                      adapter={'X': ([('char_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               },
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=True)
    word_lstm = Step(name='word_lstm',
                     transformer=WordLSTM(**config.word_lstm_network),
                     input_steps=[word_tokenizer, cleaning_output],
                     adapter={'X': ([('word_tokenizer', 'X')]),
                              'y': ([('cleaning_output', 'y')]),
                              },
                     cache_dirpath=config.env.cache_dirpath,
                     cache_output=True)
    glove_lstm = Step(name='glove_lstm',
                      transformer=GloveLSTM(**config.glove_lstm_network),
                      input_steps=[word_tokenizer, cleaning_output, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=True)
    glove_scnn = Step(name='glove_scnn',
                      transformer=GloveSCNN(**config.glove_scnn_network),
                      input_steps=[word_tokenizer, cleaning_output, glove_embeddings],
                      adapter={'X': ([('word_tokenizer', 'X')]),
                               'y': ([('cleaning_output', 'y')]),
                               'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                               },
                      cache_dirpath=config.env.cache_dirpath,
                      cache_output=True)

    glove_dpcnn = Step(name='glove_dpcnn',
                       transformer=GloveDPCNN(**config.glove_dpcnn_network),
                       input_steps=[word_tokenizer, cleaning_output, glove_embeddings],
                       adapter={'X': ([('word_tokenizer', 'X')]),
                                'y': ([('cleaning_output', 'y')]),
                                'embedding_matrix': ([('glove_embeddings', 'embeddings_matrix')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       cache_output=True)

    return [log_reg_multi, char_vdcnn, word_lstm, glove_lstm, glove_scnn, glove_dpcnn]


def weighted_average_ensemble_train(config):
    model_outputs = ensemble_extraction(config)
    output_mappings = [(output_step.name, 'prediction_probability') for output_step in model_outputs]

    prediction_average = Step(name='prediction_average',
                              transformer=PredictionAverage(**config.prediction_average),
                              input_steps=model_outputs,
                              adapter={'prediction_proba_list': (output_mappings, stack_inputs)},
                              cache_dirpath=config.env.cache_dirpath)

    average_ensemble_output = Step(name='average_ensemble_output',
                                   transformer=Dummy(),
                                   input_steps=[prediction_average],
                                   adapter={'y_pred': ([('prediction_average', 'prediction_probability')])},
                                   cache_dirpath=config.env.cache_dirpath)
    return average_ensemble_output


def weighted_average_ensemble_inference(config):
    weighted_average_ensemble = weighted_average_ensemble_train(config)

    for step in weighted_average_ensemble.get_step('prediction_average').input_steps:
        step.cache_output = False

    return weighted_average_ensemble


def logistic_regression_ensemble_train(config):
    model_outputs = ensemble_extraction(config)
    output_mappings = [(output_step.name, 'prediction_probability') for output_step in model_outputs]

    label = model_outputs[0].get_step('cleaning_output')

    input_steps = model_outputs + [label]

    log_reg = Step(name='log_reg_ensemble',
                   transformer=LogisticRegressionMultilabel(**config.logistic_regression_ensemble),
                   overwrite_transformer=True,
                   input_steps=input_steps,
                   adapter={'X': (output_mappings, hstack_inputs),
                            'y': ([('cleaning_output', 'y')])},
                   cache_dirpath=config.env.cache_dirpath)

    log_reg_ensemble_output = Step(name='log_reg_ensemble_output',
                                   transformer=Dummy(),
                                   input_steps=[log_reg],
                                   adapter={'y_pred': ([('log_reg_ensemble', 'prediction_probability')])},
                                   cache_dirpath=config.env.cache_dirpath)
    return log_reg_ensemble_output


def logistic_regression_ensemble_inference(config):
    linear_regression_ensemble = logistic_regression_ensemble_train(config)
    linear_regression_ensemble.get_step('log_reg_ensemble').overwrite_transformer = False
    for step in linear_regression_ensemble.get_step('log_reg_ensemble').input_steps:
        step.cache_output = False
    return linear_regression_ensemble


PIPELINES = {'char_vdcnn': {'train': char_vdcnn_train,
                            'inference': char_vdcnn_inference},
             'word_lstm': {'train': word_lstm_train,
                           'inference': word_lstm_inference},
             'glove_lstm': {'train': glove_lstm_train,
                            'inference': glove_lstm_inference},
             'glove_scnn': {'train': glove_scnn_train,
                            'inference': glove_scnn_inference},
             'glove_dpcnn': {'train': glove_dpcnn_train,
                             'inference': glove_dpcnn_inference},
             'tfidf_logreg': {'train': tfidf_logreg_train,
                              'inference': tfidf_logreg_inference},
             'weighted_average_ensemble': {'train': weighted_average_ensemble_train,
                                           'inference': weighted_average_ensemble_inference},
             'log_reg_ensemble': {'train': logistic_regression_ensemble_train,
                                  'inference': logistic_regression_ensemble_inference},
             }
