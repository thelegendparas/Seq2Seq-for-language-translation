
# Sequence to Sequence learning for machine translation



## About project

In this notebook, we will be building a language translation modeling using Sequence to Sequence architecture. Translation will be from German to English.

## Introduction

The most common Sequence-to-Sequence models are encoder-decoder models, which use RNN to encode the variable input vector into a context vector of **fixed size** which can be feeded into the decoder.
 
We can think of the context vector as being an abstract representation of the entire input  sentence.

The context vector then is decoded by the second RNN model(Decoder) which learns to output the target sequence by generating it one word at a time.

![Seq2Seq.png](https://media.discordapp.net/attachments/1097561980128219161/1281676912724279316/1_Ismhi-muID5ooWf3ZIQFFg.png?ex=66dc9624&is=66db44a4&hm=011915a704b40e1ad5c13619ce9cfea4f327990647201d7d90a8d3b81da26ecc&=&format=webp&quality=lossless&width=1428&height=662)
## DataSet

Our dataset has been taken from [Kraggle Parallel Translation Corpus in 24 languages! (~5M)](https://www.kaggle.com/datasets/hgultekin/paralel-translation-corpus-in-22-languages?select=EN-DE)
. The dataset has ~5 million parrallel sentences of german and english.
```
CD = "./Seq2Seq Model/DataSet/EN-DE/"
#source language is english
SL = 'EN'
#translated language is german
TL = 'DE'
df = pd.read_csv("D:\projects\Seq2Seq Model\DataSet\EN-DE\EN-DE.txt", sep = "\t", header= None)[[0,1]].rename(columns = {0:SL, 1:TL})
```

If u want to load the dataset for yourself, check the instructions in the Kraggle dataset link.
## Tokenizer 

I wrote a custom function for word tokenization. It inputs a text of string and outputs a list of words.

```
def tokenize(text):
    text = text.lower()
    # removes punctuations and split into words
    tokens = re.findall(r'\b\w+\b', text)
    return tokens
```

note- similarly I wrote a custom function for Dictionary building, U can check it out in the notebook


# Working behind the scenes

![Seq2Seq.png](https://media.discordapp.net/attachments/1097561980128219161/1281672654595690589/seq2seq1.png?ex=66dc922d&is=66db40ad&hm=b9ff350f96ff84ad4211c38dadf86ebee470ce1c0ae02dd57c426359b39f7f8f&=&format=webp&quality=lossless&width=868&height=526)


The above image shows an example translation. The input/source sentence, "guten morgen", is passed through the embedding layer (yellow) and then input into the encoder (green). We also append a start of sequence (`<sos>`) and end of sequence (`<eos>`) token to the start and end of sentence, respectively. At each time-step, the input to the encoder RNN is both the embedding, `e`, of the current word, `e(xt)`, as well as the hidden state from the previous time-step, `ht-1`, and the encoder RNN outputs a new hidden state `ht`. We can think of the hidden state as a vector representation of the sentence so far. The RNN can be represented as a function of both `e(xt)` and `ht-1`:

ht = EncoderRNN(e(xt), ht-1)


We're using the term RNN generally here, it could be any recurrent architecture, such as an LSTM (Long Short-Term Memory) or a GRU (Gated Recurrent Unit).

Here, we have `X = {x1, x2, ..., xT}`, where `x1 = <sos>`, `x2 = guten`, etc. The initial hidden state, `h0`, is usually either initialized to zeros or a learned parameter.

Once the final word, `xT`, has been passed into the RNN via the embedding layer, we use the final hidden state, `hT`, as the context vector, i.e. `hT = z`. This is a vector representation of the entire source sentence.

Now we have our context vector, `z`, we can start decoding it to get the output/target sentence, "good morning". Again, we append start and end of sequence tokens to the target sentence. At each time-step, the input to the decoder RNN (blue) is the embedding, `d`, of the current word, `d(yt)`, as well as the hidden state from the previous time-step, `st-1`, where the initial decoder hidden state, `s0`, is the context vector, `s0 = z = hT`, i.e. the initial decoder hidden state is the final encoder hidden state. Thus, similar to the encoder, we can represent the decoder as:

st = DecoderRNN(d(yt), st-1)


Although the input/source embedding layer, `e`, and the output/target embedding layer, `d`, are both shown in yellow in the diagram, they are two different embedding layers with their own parameters.

---

In the decoder, we need to go from the hidden state to an actual word, therefore at each time-step we use `st` to predict (by passing it through a Linear layer, shown in purple) what we think is the next word in the sequence, `yt-hat`.

yt-hat = f(st)


The words in the decoder are always generated one after another, with one per time-step. We always use `<sos>` for the first input to the decoder, `y1`, but for subsequent inputs, `yt>1`, we will sometimes use the actual, ground truth next word in the sequence, `yt`, and sometimes use the word predicted by our decoder, `yt-hat-1`. This is called **teacher forcing**.

When training/testing our model, we always know how many words are in our target sentence, so we stop generating words once we hit that many. During inference, it is common to keep generating words until the model outputs an `<eos>` token or after a certain amount of words have been generated.

Once we have our predicted target sentence, `Y-hat = {y1-hat, y2-hat, ..., yT-hat}`, we compare it against our actual target sentence, `Y = {y1, y2, ..., yT}`, to calculate our loss. We then use this loss to update all of the parameters in our model.

