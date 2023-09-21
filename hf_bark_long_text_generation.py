import nltk  # we'll use this to split into sentences
import numpy as np
import os
from transformers import BarkModel, AutoProcessor
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


device = "cuda:0" if torch.cuda.is_available() else "cpu"
voice_preset = "v2/en_speaker_6"

batch_size = 1


script = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,' thought Alice `without pictures or conversation?'
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, `Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled `ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody, so managed to put it into one of the cupboards as she fell past it.

`Well!' thought Alice to herself, `after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)

Down, down, down. Would the fall never come to an end! `I wonder how many miles I've fallen by this time?' she said aloud. `I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think--' (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) `--yes, that's about the right distance--but then I wonder what Latitude or Longitude I've got to?' (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.)
""".replace("\n", " ").strip()


sentences = nltk.sent_tokenize(script)

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(device)
sampling_rate = model.generation_config.sample_rate

# prepare the inputs
text_prompt = "Let's try generating speech, with Bark, a text-to-speech model"
inputs = processor(text_prompt,voice_preset=voice_preset)

# generate speech


silence = np.zeros(int(0.25 * sampling_rate))  # quarter second of silence

pieces = []
for i in range(0, len(sentences), batch_size):
    inputs = sentences[batch_size*i:min(batch_size*(i+1), len(sentences))]
    if len(inputs) != 0:
        inputs = processor(inputs, voice_preset=voice_preset)
        
        speech_output = model.generate(**inputs.to(device)).cpu().numpy()
        
        pieces += [*speech_output, silence.copy()]

import scipy
scipy.io.wavfile.write("long_generation_alice_unbatched.wav", rate=sampling_rate, data=np.concatenate(pieces))
