import numpy as np
import random
from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

np.random.seed(123)
random.seed(123)

model = DeepSpeakerModel()
model.m.load_weights('/Users/ethanzh/Code/deep-speaker/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)

ethan_mfcc_001 = sample_from_mfcc(read_mfcc('samples/EthanHouston/1.wav', SAMPLE_RATE), NUM_FRAMES)
ethan_mfcc_002 = sample_from_mfcc(read_mfcc('samples/EthanHouston/2.wav', SAMPLE_RATE), NUM_FRAMES)
ethan_mfcc_003 = sample_from_mfcc(read_mfcc('samples/EthanHouston/3.wav', SAMPLE_RATE), NUM_FRAMES)

ethan_predict_001 = model.m.predict(np.expand_dims(ethan_mfcc_001, axis=0))
ethan_predict_002 = model.m.predict(np.expand_dims(ethan_mfcc_002, axis=0))
ethan_predict_003 = model.m.predict(np.expand_dims(ethan_mfcc_003, axis=0))


philippe_mfcc_001 = sample_from_mfcc(read_mfcc('/Users/ethanzh/Code/deep-speaker/samples/PhilippeRemy/PhilippeRemy_001.wav', SAMPLE_RATE), NUM_FRAMES)
philippe_mfcc_002 = sample_from_mfcc(read_mfcc('/Users/ethanzh/Code/deep-speaker/samples/PhilippeRemy/PhilippeRemy_002.wav', SAMPLE_RATE), NUM_FRAMES)
philippe_mfcc_003 = sample_from_mfcc(read_mfcc('/Users/ethanzh/Code/deep-speaker/samples/PhilippeRemy/PhilippeRemy_003.wav', SAMPLE_RATE), NUM_FRAMES)
philippe_mfcc_004 = sample_from_mfcc(read_mfcc('/Users/ethanzh/Code/deep-speaker/samples/PhilippeRemy/PhilippeRemy_004.wav', SAMPLE_RATE), NUM_FRAMES)

philippe_predict_001 = model.m.predict(np.expand_dims(philippe_mfcc_001, axis=0))
philippe_predict_002 = model.m.predict(np.expand_dims(philippe_mfcc_002, axis=0))
philippe_predict_003 = model.m.predict(np.expand_dims(philippe_mfcc_003, axis=0))
philippe_predict_004 = model.m.predict(np.expand_dims(philippe_mfcc_004, axis=0))

legal_eagle_mfcc_001 = sample_from_mfcc(read_mfcc('samples/LegalEagle/1.wav', SAMPLE_RATE), NUM_FRAMES)
legal_eagle_mfcc_002 = sample_from_mfcc(read_mfcc('samples/LegalEagle/2.wav', SAMPLE_RATE), NUM_FRAMES)
legal_eagle_mfcc_003 = sample_from_mfcc(read_mfcc('samples/LegalEagle/3.wav', SAMPLE_RATE), NUM_FRAMES)

legal_eagle_predict_001 = model.m.predict(np.expand_dims(legal_eagle_mfcc_001, axis=0))
legal_eagle_predict_002 = model.m.predict(np.expand_dims(legal_eagle_mfcc_002, axis=0))
legal_eagle_predict_003 = model.m.predict(np.expand_dims(legal_eagle_mfcc_003, axis=0))

ethans = [ethan_predict_001, ethan_predict_002, ethan_predict_003]
philippes = [philippe_predict_001, philippe_predict_002, philippe_predict_003, philippe_predict_004]

print(batch_cosine_similarity(legal_eagle_predict_001, legal_eagle_predict_002))
print(batch_cosine_similarity(legal_eagle_predict_002, legal_eagle_predict_003))
print(batch_cosine_similarity(legal_eagle_predict_001, legal_eagle_predict_003))


print(batch_cosine_similarity(ethan_predict_001, legal_eagle_predict_001))
print(batch_cosine_similarity(ethan_predict_002, legal_eagle_predict_002))
print(batch_cosine_similarity(ethan_predict_003, legal_eagle_predict_003))





# compare each ethan to each philippe
#print("Comparing Ethan and Philippe:")
similarity_sum = 0
for i, ethan in enumerate(ethans):
    for philippe in philippes:
        current_sum = batch_cosine_similarity(ethan, philippe)
        similarity_sum += current_sum
    
#print(similarity_sum / (len(ethans) * len(philippes)))


#print("Comparing Ethan and Ethan:")
similarity_sum = 0
for i, ethan_one in enumerate(ethans):
    for j, ethan_two in enumerate(ethans[i + 1:]):
        current_sum = batch_cosine_similarity(ethan_one, ethan_two)
        similarity_sum += current_sum
   
#print(similarity_sum / (len(ethans)))