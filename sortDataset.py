from faster_whisper import WhisperModel
import tqdm
import os
import multiprocessing
import logging
model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2")

from sentence_transformers import SentenceTransformer, util

model2 = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def semantic_match(text, threshold=0.75):
    query_vec = model2.encode("hello lilio", convert_to_tensor=True)
    text_vec = model2.encode(text, convert_to_tensor=True)
    score = util.cos_sim(query_vec, text_vec).item()
    return score > threshold

def move_file(file1, path2):
    os.replace(file1, path2)


def main():
    dirs = os.listdir("./lilio" )
    # Print all the files and directories
    good_path = "./lilio_good/"
    wrong_path = "./lilio_wrong/"
    os.makedirs(good_path, exist_ok=True)
    os.makedirs(wrong_path, exist_ok=True)
    for file in tqdm.tqdm(dirs):
        try :
            path = './lilio/' + file
            segments, info = model.transcribe(path, beam_size=5)
            for segment in segments:
                if semantic_match(segment.text):
                    move_file(path,good_path + file)
                else :
                    move_file(path,wrong_path+file)
                continue
        except :
            continue

main()