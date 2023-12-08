# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from moviepy.editor import VideoFileClip, TextClip, CompositeAudioClip, AudioClip
from pydub import AudioSegment
import torch
import librosa
import torchaudio
from transformers import pipeline

def extract_audio(video_path, audio_output_path):
    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Extract the audio
    audio_clip = video_clip.audio

    # Save the audio to the specified output path
    audio_clip.write_audiofile(audio_output_path)

    # Close the clips
    video_clip.close()
    audio_clip.close()


# Load pre-trained model for speech recognition
transcription_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

def transcribe_audio(audio_path):
    # Load audio file using librosa
    print(f"Loading audio file: {audio_path}")
    waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    # Convert waveform to torch tensor
    waveform = torch.FloatTensor(waveform)

    # Perform transcription
    print("Performing transcription")
    result = transcription_pipeline(waveform.numpy())

    return result
# Press the green button in the gutter to run the script.

if __name__ == "__main__":
    # Replace 'input_video.mp4' with the path to your input video file
    input_video_path = 'input_video.mp4'

    # Replace 'output_audio.wav' with the desired output audio file path
    output_audio_path = 'input_audio.mp3'

    # Replace 'output_text.txt' with the desired output text file path
    output_text_path = 'output_text.txt'

    # Extract audio from video
    # extract_audio(input_video_path, output_audio_path)

    transcription_result = transcribe_audio(output_audio_path)

    print("Transcription:", transcription_result)
    print("Text:", transcription_result['text'])

    # Save the transcribed text to a file
    with open(output_text_path, 'w') as text_file:
        text_file.write(transcription_result['text'])
        text_file.close()
