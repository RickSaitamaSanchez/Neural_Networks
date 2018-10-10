import kfold
import speech_recognition as sr

# sr.lstm_model()
# kfold.lstm_model_kfold("lstm_arch3", 100)
# kfold.lstm_model_kfold("lstm_arch4", 175)
# kfold.lstm_model_kfold("lstm_arch5", 100, 100)
# kfold.lstm_model_kfold("lstm_arch6", 100, 125)
# kfold.lstm_model_kfold("lstm_arch7", 100, 150)
# kfold.lstm_model_kfold("lstm_arch8", 125, 100)
# kfold.lstm_model_kfold("lstm_arch9", 150, 100)
# kfold.lstm_model_kfold("lstm_arch10", 125, 125)
# kfold.lstm_model_kfold("lstm_arch11", 125, 150)
# kfold.lstm_model_kfold("lstm_arch12", 150, 125)
# kfold.lstm_model_kfold("lstm_arch13", 150, 150)
# kfold.lstm_model_kfold("lstm_arch14", 100, 100, 100)
# kfold.lstm_model_kfold("lstm_arch15", 100, 100, 125)
# kfold.lstm_model_kfold("lstm_arch16", 100, 100, 150)
# kfold.lstm_model_kfold("lstm_arch17", 100, 125, 100)
kfold.lstm_model_kfold("lstm_arch18", 100, 125, 125)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Separated training for graphic ploting:

# sr.lstm_model(True, "lstm_arch5", "train", 100, 100)
# sr.lstm_model(True, "lstm_arch6", "train", 100, 125)
# sr.lstm_model(True, "lstm_arch7", "train", 100, 150)
# sr.lstm_model(True, "lstm_arch8", "train", 125, 100)