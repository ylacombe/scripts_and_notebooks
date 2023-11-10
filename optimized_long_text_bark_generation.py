######################## 
# INTRODUCTION:
# This script shows how to generate very long texts using very high speed, based on Flash Attention 2 and batching.
# It was run on a 40 GB A100 GPU with Pytorch 2.1 and saved me 3 times more time than using Better Transfomer (https://huggingface.co/blog/optimizing-bark#1-%F0%9F%A4%97-better-transformer) with the same batch size.
# My benchmark also showed a 20x higher throughput when generating 400 semantic tokens and a batch size of 32 compared to generating sentence one by one with the native implementation of attention (without optimization).
# In other words, with batch generation and Flash Attention2, you'll get the whole generation 20x faster.

# REQUIREMENTS:
# install flash attention: !pip install -U flash-attn --no-build-isolation
# install transformers from main: !pip install git+https://github.com/huggingface/transformers.git

import nltk  # we'll use this to split into sentences
import numpy as np
from transformers import BarkModel, AutoProcessor
import torch
import time
import scipy

nltk.download('punkt')
device = "cuda"


# JFK Berling Speech extract - 777 ChatGPT tokens
TEXT_TO_GENERATE = """
I am proud to come to this city as the guest of your distinguished Mayor, who has symbolized throughout the world the fighting spirit of West Berlin. And I am proud to visit the Federal Republic with your distinguished Chancellor who for so many years has committed Germany to democracy and freedom and progress, and to come here in the company of my fellow American, General Clay, who has been in this city during its great moments of crisis and will come again if ever needed.

Two thousand years ago the proudest boast was "civis Romanus sum." Today, in the world of freedom, the proudest boast is "Ich bin ein Berliner."

I appreciate my interpreter translating my German!

There are many people in the world who really don't understand, or say they don't, what is the great issue between the free world and the Communist world. Let them come to Berlin. There are some who say that communism is the wave of the future. Let them come to Berlin. And there are some who say in Europe and elsewhere we can work with the Communists. Let them come to Berlin. And there are even a few who say that it is true that communism is an evil system, but it permits us to make economic progress. Lass' sie nach Berlin kommen. Let them come to Berlin.

Freedom has many difficulties and democracy is not perfect, but we have never had to put a wall up to keep our people in, to prevent them from leaving us. I want to say, on behalf of my countrymen, who live many miles away on the other side of the Atlantic, who are far distant from you, that they take the greatest pride that they have been able to share with you, even from a distance, the story of the last 18 years. I know of no town, no city, that has been besieged for 18 years that still lives with the vitality and the force, and the hope and the determination of the city of West Berlin. While the wall is the most obvious and vivid demonstration of the failures of the Communist system, for all the world to see, we take no satisfaction in it, for it is, as your Mayor has said, an offense not only against history but an offense against humanity, separating families, dividing husbands and wives and brothers and sisters, and dividing a people who wish to be joined together.

What is true of this city is true of Germany--real, lasting peace in Europe can never be assured as long as one German out of four is denied the elementary right of free men, and that is to make a free choice. In 18 years of peace and good faith, this generation of Germans has earned the right to be free, including the right to unite their families and their nation in lasting peace, with good will to all people. You live in a defended island of freedom, but your life is part of the main. So let me ask you as I close, to lift your eyes beyond the dangers of today, to the hopes of tomorrow, beyond the freedom merely of this city of Berlin, or your country of Germany, to the advance of freedom everywhere, beyond the wall to the day of peace with justice, beyond yourselves and ourselves to all mankind.

Freedom is indivisible, and when one man is enslaved, all are not free. When all are free, then we can look forward to that day when this city will be joined as one and this country and this great Continent of Europe in a peaceful and hopeful globe. When that day finally comes, as it will, the people of West Berlin can take sober satisfaction in the fact that they were in the front lines for almost two decades.

All free men, wherever they may live, are citizens of Berlin, and, therefore, as a free man, I take pride in the words "Ich bin ein Berliner."
"""

# Load model with optimization
model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16, use_flash_attention_2=True).to(device)
processor = AutoProcessor.from_pretrained("suno/bark")
# uncomment the next line if you have optimum installed and are memory constraint
# model =  model.to_bettertransformer()

# some needed parameters
sampling_rate = model.generation_config.sample_rate
silence = np.zeros(int(0.20 * sampling_rate))  # one fifth second of silence
voice_preset = "v2/en_speaker_3"
BATCH_SIZE = 32 # we use a high 



# warming - to get a fairest estimation of generation time - can be removed
model.generate(**processor("salut").to(device), return_output_lengths=True, min_eos_p=0.3)
model.generate(**processor("salut").to(device), return_output_lengths=True, min_eos_p=0.3)



start = time.time()


# split into sentences
model_input = nltk.sent_tokenize(TEXT_TO_GENERATE.replace("\n", " ").strip())

pieces = []
for i in range(0, len(model_input), BATCH_SIZE):
    inputs = model_input[i:min(i + BATCH_SIZE, len(model_input))]
    
    if len(inputs) != 0:
        # tokenize input sentences
        inputs = processor(inputs, voice_preset=voice_preset)

        # generate with bark
        speech_output, output_lengths = model.generate(**inputs.to(device), return_output_lengths=True, min_eos_p=0.5)

        # postprocess each sample to get the right length
        speech_output = [output[:length].cpu().numpy() for (output,length) in zip(speech_output, output_lengths)]
        
        print(f"{i}-th part generated - {len(inputs)}")
        
        # you could already play `speech_output` or wait for the whole generation
        pieces += [*speech_output, silence.copy()]
        
end = time.time()
print("TIME", end - start)    
whole_output = np.concatenate(pieces)

# save in wav file
scipy.io.wavfile.write("jfk_speech.wav", rate=sampling_rate, data=whole_output)
