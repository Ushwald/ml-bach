import oldmusic
from music21 import *

predicted = oldmusic.load_txt_to_stream(fp="predicted_score.txt")
#predicted.show('midi')
fp = predicted.write('midi', fp='linear_regression_output_voice_3.mid')