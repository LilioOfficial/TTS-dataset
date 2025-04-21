from generator import load_csm_1b, generate_streaming_audio
import tqdm

# Load the model
generator = load_csm_1b("cuda")

def main():
    for i in range(1000):
        # Generate audio with streaming and real-time playback
        generate_streaming_audio(
            generator=generator,
            text="Hello Lili-o",
            speaker=0,
            max_audio_length_ms=2000,
            context=[],  # No context needed for basic generation
            output_file="".join(("outputWav/lilio_",str(i),".wav")),
        )

main()
